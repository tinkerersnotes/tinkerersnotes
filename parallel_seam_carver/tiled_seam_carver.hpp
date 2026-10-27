// Seam carver whose path cost propagation runs on several threads, tiled by triangles. Header-only:
// used by the WebAssembly page (../wasm_seam_carver/tiled_seam_carver_wasm.cpp) and checked against
// the core by tiled_seam_carver.cpp, which also has a write-once checker for the tiling.
//
// Dense row-major arrays instead of the core's linked mesh: per pixel its colour, the cost of the
// cheapest path from the top row, and the parent's column offset (-1, 0, +1). Every seam is a full
// recompute with the same rules as SeamCarver::propagate_low_cost_paths(): parents in the order
// up, up-left, up-right, a later one wins only if strictly cheaper; then the leftmost cheapest
// bottom pixel. So the result matches the core's shrink_horizontally(n, false) bit for bit.
//
// Tiling, per strip of h = base / 2 rows below a fully computed row r0:
//   phase A  each task owns a base segment [a, b) of row r0 and fills the downward triangle
//            [a + k, b - k) of row r0 + k; segments at the image border don't shrink on that side
//   phase B  each boundary c between two segments fills the upward triangle [c - k, c + k),
//            which reads the sides of both neighbouring triangles from phase A
// Two barriers per strip. Neighbouring upward triangles never overlap, since base >= 2.
#pragma once

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/threading.h>
#endif

namespace seam_carving {

// Barrier whose waiting threads spin briefly, then sleep until the last one arrives.
// std::barrier in WebAssembly spins for up to ~64 us per wait instead, asking JavaScript for the
// time as it goes: with more threads than free cores, the spinning threads take the CPU from the
// ones still working. In the browser, 8 threads on 2 cores took 2.5x as long as 2 threads with
// std::barrier, and as long as 2 threads with this one.
class SleepingBarrier {
public:
    explicit SleepingBarrier(int count) : count_(count) {}

    void arrive_and_wait() {
        const int phase = phase_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            // Last to arrive: the reset happens before the release, so it is seen by every thread
            // that sees the new phase
            arrived_.store(0, std::memory_order_relaxed);
            phase_.store(phase + 1, std::memory_order_release);
            wake_all();
            return;
        }
        for (int i = 0; i < kSpins; ++i) {
            if (phase_.load(std::memory_order_acquire) != phase) {
                return;
            }
        }
        while (phase_.load(std::memory_order_acquire) == phase) {
            sleep_while(phase);
        }
    }

private:
    static constexpr int kSpins = 1000;  // a few microseconds: the others are often about to arrive

#ifdef __EMSCRIPTEN__
    // Atomics.wait and Atomics.notify; a worker sleeps, a browser's main thread would spin
    void sleep_while(int phase) { emscripten_futex_wait(&phase_, static_cast<std::uint32_t>(phase), INFINITY); }
    void wake_all() { emscripten_futex_wake(&phase_, INT_MAX); }
#else
    void sleep_while(int phase) { phase_.wait(phase, std::memory_order_acquire); }
    void wake_all() { phase_.notify_all(); }
#endif

    std::atomic<int> arrived_{0};
    std::atomic<int> phase_{0};
    const int count_;
};

class TiledSeamCarver {
public:
    // rgb is a row-major buffer of width * height * 3 bytes
    TiledSeamCarver(const std::uint8_t* rgb, int width, int height, int threads, int base = 128)
        : stride_(width), width_(width), height_(height), threads_(threads), base_(base) {
        if (rgb == nullptr || width <= 0 || height <= 0) {
            throw std::invalid_argument("image must be non-empty");
        }
        if (threads < 1 || base < 2) {
            throw std::invalid_argument("threads must be >= 1 and base >= 2");
        }
        const std::size_t cells = static_cast<std::size_t>(width) * height;
        rgb_.assign(rgb, rgb + 3 * cells);
        cost_.resize(cells);
        prev_.resize(cells);
        seam_.resize(height);
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int threads() const { return threads_; }

    // All threads work on every seam: the calling thread and threads - 1 started for this call
    void shrink_horizontally(int pixels_to_shrink) {
        if (pixels_to_shrink < 0 || pixels_to_shrink >= width_) {
            throw std::out_of_range("pixels_to_shrink must be in [0, width)");
        }
        if (pixels_to_shrink == 0) {
            return;
        }
        SleepingBarrier sync(threads_);
        auto worker = [&](int tid) {
            for (int s = 0; s < pixels_to_shrink; ++s) {
                remove_seam(tid, sync);
            }
        };
        std::vector<std::thread> pool;
        for (int tid = 1; tid < threads_; ++tid) {
            pool.emplace_back(worker, tid);
        }
        worker(0);
        for (std::thread& thread : pool) {
            thread.join();
        }
    }

    // Row-major RGB bytes of the current (shrunk) image, width() * height() * 3
    std::vector<std::uint8_t> shrunk_rgb() const {
        std::vector<std::uint8_t> rgb;
        rgb.reserve(static_cast<std::size_t>(width_) * height_ * 3);
        for (int y = 0; y < height_; ++y) {
            const std::uint8_t* row = px(0, y);
            rgb.insert(rgb.end(), row, row + 3 * width_);
        }
        return rgb;
    }

private:
    const std::uint8_t* px(int x, int y) const { return &rgb_[3 * (static_cast<std::size_t>(y) * stride_ + x)]; }

    static double distance(const std::uint8_t* p, const std::uint8_t* c) {
        const double r = double(p[0]) - c[0], g = double(p[1]) - c[1], b = double(p[2]) - c[2];
        return std::sqrt(r * r + g * g + b * b);
    }

    // Cells [lo, hi) of row y from row y - 1. Everything the compiler would otherwise reload after
    // each int8_t store (a char type may alias anything) is held in __restrict locals.
    void relax_range(int lo, int hi, int y) {
        if (lo >= hi) {
            return;
        }
        const int width = width_;
        const std::size_t row = static_cast<std::size_t>(y) * stride_;
        const double* __restrict up_cost = cost_.data() + row - stride_;
        double* __restrict out_cost = cost_.data() + row;
        std::int8_t* __restrict out_prev = prev_.data() + row;
        const std::uint8_t* __restrict up_px = rgb_.data() + 3 * (row - stride_);
        const std::uint8_t* __restrict px = rgb_.data() + 3 * row;
        // Border pixels lack a left or right parent: done apart, so the hot loop has no edge tests
        auto edge = [&](int x) {
            const std::uint8_t* c = px + 3 * x;
            double best = up_cost[x] + distance(up_px + 3 * x, c);
            std::int8_t from = 0;
            if (x > 0) {
                const double v = up_cost[x - 1] + distance(up_px + 3 * (x - 1), c);
                if (v < best) { best = v; from = -1; }
            }
            if (x + 1 < width) {
                const double v = up_cost[x + 1] + distance(up_px + 3 * (x + 1), c);
                if (v < best) { best = v; from = 1; }
            }
            out_cost[x] = best;
            out_prev[x] = from;
        };
        int first = lo, last = hi;
        if (first == 0) edge(first++);
        if (last == width && last > first) edge(--last);
        // Interior: the same priority as selects instead of branches. On a textured photo those
        // comparisons are coin flips the branch predictor can't learn.
        for (int x = first; x < last; ++x) {
            const std::uint8_t* c = px + 3 * x;
            const double up = up_cost[x] + distance(up_px + 3 * x, c);
            const double left = up_cost[x - 1] + distance(up_px + 3 * (x - 1), c);
            const double right = up_cost[x + 1] + distance(up_px + 3 * (x + 1), c);
            const bool take_left = left < up;
            const double best_ul = take_left ? left : up;
            const bool take_right = right < best_ul;
            out_cost[x] = take_right ? right : best_ul;
            out_prev[x] = std::int8_t(take_right ? 1 : (take_left ? -1 : 0));
        }
    }

    // One seam, run by every thread. width_ only changes between barriers.
    void remove_seam(int tid, SleepingBarrier& sync) {
        const int w = width_;
        const int tasks = (w + base_ - 1) / base_;
        const int h = base_ / 2;
        if (tid == 0) {
            std::fill_n(cost_.begin(), w, 0.0);  // the core's -1 top row cost counts as 0
        }
        sync.arrive_and_wait();
        for (int r0 = 0; r0 < height_ - 1; r0 += h) {
            const int rows = std::min(h, height_ - 1 - r0);
            // Phase A: downward triangles over each base segment
            for (int i = tid; i < tasks; i += threads_) {
                const int a = i * base_, b = std::min(w, a + base_);
                for (int k = 1; k <= rows; ++k) {
                    relax_range(i == 0 ? 0 : a + k, i == tasks - 1 ? w : b - k, r0 + k);
                }
            }
            sync.arrive_and_wait();
            // Phase B: upward triangles at the boundaries between segments
            for (int j = 1 + tid; j < tasks; j += threads_) {
                const int boundary = j * base_;
                for (int k = 1; k <= rows; ++k) {
                    relax_range(std::max(0, boundary - k), std::min(w, boundary + k), r0 + k);
                }
            }
            sync.arrive_and_wait();
        }
        if (tid == 0) {
            trace_seam();
        }
        sync.arrive_and_wait();
        // Rows are independent: each thread shifts its share left past the seam
        const int per = (height_ + threads_ - 1) / threads_;
        for (int y = std::min(height_, tid * per); y < std::min(height_, (tid + 1) * per); ++y) {
            std::uint8_t* row = &rgb_[3 * static_cast<std::size_t>(y) * stride_];
            const int x = seam_[y];
            std::memmove(row + 3 * x, row + 3 * (x + 1), 3 * static_cast<std::size_t>(w - x - 1));
        }
        sync.arrive_and_wait();
        if (tid == 0) {
            --width_;
        }
        sync.arrive_and_wait();
    }

    // Leftmost cheapest bottom pixel, then the parents back to the top
    void trace_seam() {
        const std::size_t bottom = static_cast<std::size_t>(height_ - 1) * stride_;
        int x = 0;
        for (int i = 1; i < width_; ++i) {
            if (cost_[bottom + i] < cost_[bottom + x]) {
                x = i;
            }
        }
        for (int y = height_ - 1; y >= 0; --y) {
            seam_[y] = x;
            if (y > 0) {
                x += prev_[static_cast<std::size_t>(y) * stride_ + x];
            }
        }
    }

    int stride_;  // the original width: rows are compacted in place
    int width_;
    int height_;
    int threads_;
    int base_;
    std::vector<std::uint8_t> rgb_;
    std::vector<double> cost_;
    std::vector<std::int8_t> prev_;
    std::vector<int> seam_;
};

}  // namespace seam_carving
