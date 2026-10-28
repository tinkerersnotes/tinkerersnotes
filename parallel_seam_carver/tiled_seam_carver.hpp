// Seam carver whose path cost propagation runs on several threads, tiled by triangles. Header-only:
// used by the WebAssembly page (../wasm_seam_carver/tiled_seam_carver_wasm.cpp) and checked against
// the core by tiled_seam_carver.cpp, which also has a write-once checker for the tiling.
//
// Dense row-major arrays instead of the core's linked mesh: per pixel its colour, the cost of the
// cheapest path from the top row, and the parent's column offset (-1, 0, +1). The same rules as the
// core: parents in the order up, up-left, up-right, a later one wins only if strictly cheaper; then
// the leftmost cheapest bottom pixel. So the result matches the core bit for bit, with or without
// incremental updates.
//
// Tiling, per strip of h = base / 2 rows below a fully computed row r0:
//   phase A  each task owns a base segment [a, b) of row r0 and fills the downward triangle
//            [a + k, b - k) of row r0 + k; segments at the image border don't shrink on that side
//   phase B  each boundary c between two segments fills the upward triangle [c - k, c + k),
//            which reads the sides of both neighbouring triangles from phase A
// Two barriers per strip. Neighbouring upward triangles never overlap, since base >= 2.
//
// Incremental updates: removing a seam shifts every row's cells right of it one column left, with
// their costs and parents, so most of them stay right. A cell can only change if a parent's colour
// or position changed, which happens next to the removed seam ("seeds"), or if a parent's cost
// changed. Those changes spread downward in a cone around the seam. So after the first seam, only
// the tiles of a finer tiling that contain a seed or read a cell whose cost changed are recomputed
// ("dirty" tiles); recomputing marks each cell whose cost changed with the seam's epoch. Every
// clean tile's cells have unchanged inputs, so their stored values are still those of a full
// recompute. On tundra, tiles 32 wide recompute 1.21x the cells the core's incremental update
// evaluates, 16% of a full recompute. 600 seams on 4 cores (Linux VM on Apple silicon) took 650 ms
// with 4 threads and 1314 ms with one, against 1592 ms for the core's incremental update and
// 3063 ms for full recomputes on 4 threads. Tiles 16 wide took 772 ms, 8 wide 1436 ms: narrower
// tiles waste less work but need more barriers, two per strip of base / 2 rows.
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

// Barrier whose waiting threads spin a while, then sleep until the last one arrives.
// std::barrier in WebAssembly spins for up to ~64 us per wait, asking JavaScript for the time as
// it goes: with more threads than free cores, the spinning threads take the CPU from the ones
// still working. In Chromium, 8 threads on 2 cores took 2.5x as long as 2 threads with
// std::barrier, and about as long as 2 threads with this one.
//
// Spinning too briefly costs the other way: every barrier then pays for waking sleeping threads,
// some microseconds natively, possibly more in browsers (Safari got much slower with 1000 spins).
// With few dirty tiles a phase holds only microseconds of work, so the right spin depends on
// whether the threads have cores to themselves: default_spins(). One check takes about 0.5 ns.
// The page's ?spins=N overrides the default to compare browsers.
class SleepingBarrier {
public:
    // Long spins when every thread has a core: 4 threads on 4 cores updated 300 seams in 1775 ms
    // with 1000 spins, 506 ms with 20000 and 442 ms with 200000. Short ones when they don't, so
    // spinning threads don't take the CPU from working ones: 8 threads on 2 cores in Chromium
    // were 10% slower with 20000 spins, 30% with 100000 and 3.6x with a million.
    static int default_spins(int threads) {
        const unsigned cores = std::thread::hardware_concurrency();  // navigator.hardwareConcurrency in wasm
        return cores != 0 && static_cast<unsigned>(threads) <= cores ? 100000 : 1000;
    }

    // spins: checks of the phase before sleeping
    SleepingBarrier(int count, int spins) : count_(count), spins_(spins) {}

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
        for (int i = 0; i < spins_; ++i) {
            if (phase_.load(std::memory_order_acquire) != phase) {
                return;
            }
        }
        while (phase_.load(std::memory_order_acquire) == phase) {
            sleep_while(phase);
        }
    }

private:
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
    const int spins_;
};

class TiledSeamCarver {
public:
    // rgb is a row-major buffer of width * height * 3 bytes. base: tile width of full recomputes;
    // incremental_base: tile width of incremental updates, smaller since few tiles are dirty
    // spins: barrier checks before sleeping, negative for SleepingBarrier::default_spins()
    TiledSeamCarver(const std::uint8_t* rgb, int width, int height, int threads, int base = 128,
                    int spins = -1, int incremental_base = 32)
        : stride_(width), width_(width), height_(height), threads_(threads), base_(base),
          spins_(spins >= 0 ? spins : SleepingBarrier::default_spins(threads)), incremental_base_(incremental_base) {
        if (rgb == nullptr || width <= 0 || height <= 0) {
            throw std::invalid_argument("image must be non-empty");
        }
        if (threads < 1 || base < 2 || incremental_base < 2) {
            throw std::invalid_argument("threads must be >= 1 and tile widths >= 2");
        }
        const std::size_t cells = static_cast<std::size_t>(width) * height;
        rgb_.assign(rgb, rgb + 3 * cells);
        cost_.resize(cells);
        prev_.resize(cells);
        changed_.resize(cells);
        seam_.resize(height);
        removed_.resize(height);
        seed_lo_.resize(height);
        seed_hi_.resize(height);
        const int strips = (height - 1 + incremental_base / 2 - 1) / (incremental_base / 2) + 1;
        strip_seed_lo_.resize(strips);
        strip_seed_hi_.resize(strips);
        changed_lo_ = std::vector<std::atomic<int>>(strips);
        changed_hi_ = std::vector<std::atomic<int>>(strips);
        computed_.resize(threads);
    }

    int width() const { return width_; }
    int height() const { return height_; }
    int threads() const { return threads_; }
    int spins() const { return spins_; }
    // Cells computed so far, by full recomputes and incremental updates
    std::uint64_t cells_computed() const {
        std::uint64_t total = 0;
        for (const Counter& c : computed_) total += c.cells;
        return total;
    }

    // All threads work on every seam: the calling thread and threads - 1 started for this call.
    // With incremental updates, every seam after the carver's first only recomputes dirty tiles.
    void shrink_horizontally(int pixels_to_shrink, bool incremental_cost_updates = false) {
        if (pixels_to_shrink < 0 || pixels_to_shrink >= width_) {
            throw std::out_of_range("pixels_to_shrink must be in [0, width)");
        }
        if (pixels_to_shrink == 0) {
            return;
        }
        SleepingBarrier sync(threads_, spins_);
        auto worker = [&](int tid) {
            for (int s = 0; s < pixels_to_shrink; ++s) {
                remove_seam(tid, sync, incremental_cost_updates);
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
    struct alignas(64) Counter {  // per thread, a cache line each
        std::uint64_t cells = 0;
    };

    const std::uint8_t* px(int x, int y) const { return &rgb_[3 * (static_cast<std::size_t>(y) * stride_ + x)]; }

    static double distance(const std::uint8_t* p, const std::uint8_t* c) {
        const double r = double(p[0]) - c[0], g = double(p[1]) - c[1], b = double(p[2]) - c[2];
        return std::sqrt(r * r + g * g + b * b);
    }

    // Cells [lo, hi) of row y from row y - 1. Everything the compiler would otherwise reload after
    // each int8_t store (a char type may alias anything) is held in __restrict locals. Track marks
    // the cells whose cost changed with the current epoch.
    template <bool Track>
    void relax_range(int lo, int hi, int y, int tid) {
        if (lo >= hi) {
            return;
        }
        computed_[tid].cells += hi - lo;
        const int width = width_;
        const std::uint32_t epoch = epoch_;
        const std::size_t row = static_cast<std::size_t>(y) * stride_;
        const double* __restrict up_cost = cost_.data() + row - stride_;
        double* __restrict out_cost = cost_.data() + row;
        std::int8_t* __restrict out_prev = prev_.data() + row;
        std::uint32_t* __restrict out_changed = changed_.data() + row;
        const std::uint8_t* __restrict up_px = rgb_.data() + 3 * (row - stride_);
        const std::uint8_t* __restrict px = rgb_.data() + 3 * row;
        auto store = [&](int x, double best, std::int8_t from) {
            if constexpr (Track) {
                out_changed[x] = out_cost[x] != best ? epoch : out_changed[x];
            }
            out_cost[x] = best;
            out_prev[x] = from;
        };
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
            store(x, best, from);
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
            store(x, take_right ? right : best_ul, std::int8_t(take_right ? 1 : (take_left ? -1 : 0)));
        }
    }

    // Full recompute of every cell, tiled with base_
    void propagate_all(int tid, SleepingBarrier& sync, int w) {
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
                    relax_range<false>(i == 0 ? 0 : a + k, i == tasks - 1 ? w : b - k, r0 + k, tid);
                }
            }
            sync.arrive_and_wait();
            // Phase B: upward triangles at the boundaries between segments
            for (int j = 1 + tid; j < tasks; j += threads_) {
                const int boundary = j * base_;
                for (int k = 1; k <= rows; ++k) {
                    relax_range<false>(std::max(0, boundary - k), std::min(w, boundary + k), r0 + k, tid);
                }
            }
            sync.arrive_and_wait();
        }
    }

    bool changed_in(int y, int lo, int hi) const {  // any cost in [lo, hi] of row y changed this seam
        lo = std::max(lo, 0);
        hi = std::min(hi, width_ - 1);
        const std::uint32_t* row = changed_.data() + static_cast<std::size_t>(y) * stride_;
        bool any = false;
        for (int x = lo; x <= hi; ++x) any |= row[x] == epoch_;
        return any;
    }

    bool seeded(int y, int lo, int hi) const {  // cells [lo, hi) of row y include a seed
        return lo < hi && seed_lo_[y] < hi && seed_hi_[y] >= lo;
    }

    static void atomic_min(std::atomic<int>& a, int v) {
        for (int cur = a.load(std::memory_order_relaxed); v < cur && !a.compare_exchange_weak(cur, v);) {}
    }
    static void atomic_max(std::atomic<int>& a, int v) {
        for (int cur = a.load(std::memory_order_relaxed); v > cur && !a.compare_exchange_weak(cur, v);) {}
    }

    // Recomputes cells [lo(k), hi(k)) of rows r0 + 1 .. r0 + rows, and records which columns of
    // the strip's last row changed, for the next strip's dirty tests
    template <typename Lo, typename Hi>
    void recompute_tile(int s, int r0, int rows, Lo lo, Hi hi, int tid) {
        for (int k = 1; k <= rows; ++k) relax_range<true>(lo(k), hi(k), r0 + k, tid);
        const std::uint32_t* last = changed_.data() + static_cast<std::size_t>(r0 + rows) * stride_;
        int first = INT_MAX, final = -1;
        for (int x = lo(rows); x < hi(rows); ++x) {
            if (last[x] == epoch_) {
                first = std::min(first, x);
                final = x;
            }
        }
        if (final >= 0) {
            atomic_min(changed_lo_[s + 1], first);
            atomic_max(changed_hi_[s + 1], final);
        }
    }

    // Recomputes the dirty tiles of an incremental tiling, after the last removed seam's columns
    // (removed_) gave the seeds and the epoch was advanced. Changes spread from the seeds and from
    // changed cells of the row above, so per strip only tiles near those columns are tested.
    void propagate_dirty(int tid, SleepingBarrier& sync, int w) {
        const int base = incremental_base_, h = base / 2;
        const int tasks = (w + base - 1) / base;
        for (int s = 0, r0 = 0; r0 < height_ - 1; ++s, r0 += h) {
            const int rows = std::min(h, height_ - 1 - r0);
            const int near_lo = std::min(strip_seed_lo_[s], changed_lo_[s].load(std::memory_order_relaxed) - 1);
            const int near_hi = std::max(strip_seed_hi_[s], changed_hi_[s].load(std::memory_order_relaxed) + 1);
            const int first = std::max(0, near_lo / base - 1), last = std::min(tasks - 1, near_hi / base + 1);
            // Phase A: a downward triangle reads only row r0, under its first row
            for (int i = first + tid; i <= last; i += threads_) {
                const int a = i * base, b = std::min(w, a + base);
                auto lo = [&](int k) { return i == 0 ? 0 : a + k; };
                auto hi = [&](int k) { return i == tasks - 1 ? w : b - k; };
                bool dirty = changed_in(r0, lo(1) - 1, hi(1));
                for (int k = 1; k <= rows && !dirty; ++k) dirty = seeded(r0 + k, lo(k), hi(k));
                if (dirty) recompute_tile(s, r0, rows, lo, hi, tid);
            }
            sync.arrive_and_wait();
            // Phase B: an upward triangle reads row r0 under its tip and, below that, the two
            // columns on each side of its previous row, which belong to the downward triangles
            for (int j = std::max(1, first) + tid; j <= std::min(tasks - 1, last + 1); j += threads_) {
                const int c = j * base;
                auto lo = [&](int k) { return std::max(0, c - k); };
                auto hi = [&](int k) { return std::min(w, c + k); };
                bool dirty = changed_in(r0, c - 2, c + 1);
                for (int k = 1; k <= rows && !dirty; ++k) {
                    dirty = seeded(r0 + k, lo(k), hi(k)) ||
                            (k > 1 && (changed_in(r0 + k - 1, c - k - 1, c - k) || changed_in(r0 + k - 1, c + k - 1, c + k)));
                }
                if (dirty) recompute_tile(s, r0, rows, lo, hi, tid);
            }
            sync.arrive_and_wait();
        }
    }

    // One seam, run by every thread. width_ only changes between barriers.
    void remove_seam(int tid, SleepingBarrier& sync, bool incremental) {
        const int w = width_;
        if (incremental && has_removed_) {
            if (tid == 0) {
                ++epoch_;
                // Seeds: cells whose parents moved, around the removed seam in the new columns.
                // Wider than strictly needed; recomputing a few extra cells is harmless.
                seed_lo_[0] = 1;
                seed_hi_[0] = 0;  // the top row never changes
                for (int y = 1; y < height_; ++y) {
                    seed_lo_[y] = std::min(removed_[y], removed_[y - 1]) - 2;
                    seed_hi_[y] = std::max(removed_[y], removed_[y - 1]) + 1;
                }
                // Per strip of the incremental tiling: the seeds' columns over its rows, and none
                // of its top row's columns changed yet
                const int h = incremental_base_ / 2;
                for (std::size_t s = 0; s < strip_seed_lo_.size(); ++s) {
                    int lo = INT_MAX, hi = INT_MIN;
                    for (int y = static_cast<int>(s) * h + 1; y <= std::min(height_ - 1, static_cast<int>(s + 1) * h); ++y) {
                        lo = std::min(lo, seed_lo_[y]);
                        hi = std::max(hi, seed_hi_[y]);
                    }
                    strip_seed_lo_[s] = lo;
                    strip_seed_hi_[s] = hi;
                    changed_lo_[s].store(INT_MAX, std::memory_order_relaxed);
                    changed_hi_[s].store(INT_MIN + 1, std::memory_order_relaxed);
                }
            }
            sync.arrive_and_wait();
            propagate_dirty(tid, sync, w);
        } else {
            propagate_all(tid, sync, w);
        }
        if (tid == 0) {
            trace_seam();
        }
        sync.arrive_and_wait();
        // Rows are independent: each thread shifts its share left past the seam, with the costs
        // and parents when the next seam may update them incrementally
        const int per = (height_ + threads_ - 1) / threads_;
        for (int y = std::min(height_, tid * per); y < std::min(height_, (tid + 1) * per); ++y) {
            const std::size_t row = static_cast<std::size_t>(y) * stride_;
            const int x = seam_[y];
            const std::size_t n = static_cast<std::size_t>(w - x - 1);
            std::memmove(&rgb_[3 * (row + x)], &rgb_[3 * (row + x + 1)], 3 * n);
            if (incremental) {
                std::memmove(&cost_[row + x], &cost_[row + x + 1], sizeof(double) * n);
                std::memmove(&prev_[row + x], &prev_[row + x + 1], n);
            }
        }
        sync.arrive_and_wait();
        if (tid == 0) {
            --width_;
            removed_.swap(seam_);
            has_removed_ = incremental;
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
    int spins_;  // barrier checks before sleeping
    int incremental_base_;
    std::vector<std::uint8_t> rgb_;
    std::vector<double> cost_;
    std::vector<std::int8_t> prev_;
    std::vector<std::uint32_t> changed_;  // epoch of the last update that changed the cell's cost
    std::uint32_t epoch_ = 0;
    std::vector<int> seam_;
    std::vector<int> removed_;  // columns of the last removed seam, before its removal
    bool has_removed_ = false;  // the costs fit the picture after removed_, ready for an update
    std::vector<int> seed_lo_;  // per row, the seed columns [lo, hi]
    std::vector<int> seed_hi_;
    std::vector<int> strip_seed_lo_;  // per strip of the incremental tiling, the seeds' columns
    std::vector<int> strip_seed_hi_;
    std::vector<std::atomic<int>> changed_lo_;  // per strip, the changed columns of its top row
    std::vector<std::atomic<int>> changed_hi_;
    std::vector<Counter> computed_;
};

}  // namespace seam_carving
