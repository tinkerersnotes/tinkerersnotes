// Seam carver whose path cost propagation runs on several threads, tiled by diamonds. Header-only:
// used by the WebAssembly page (../wasm_seam_carver/tiled_seam_carver_wasm.cpp) and checked against
// the core by tiled_seam_carver.cpp, which also has a write-once checker for the tiling.
//
// Dense row-major arrays instead of the core's linked mesh: per pixel its colour, the cost of the
// cheapest path from the top row, and the parent's column offset (-1, 0, +1). The same rules as the
// core: parents in the order up, up-left, up-right, a later one wins only if strictly cheaper; then
// the leftmost cheapest bottom pixel. So the result matches the core bit for bit, with or without
// incremental updates.
//
// Diamond tiling. Rows below the top one are cut into strips of h = base / 2 rows, s = 0, 1, ...;
// strip s holds rows r0 + 1 .. r0 + h, r0 = s * h. Every wave is one row of diamonds, centred on
// columns c spaced p = 2h apart, and a barrier follows each wave. The diamond of wave t at c:
//   upper half  strip t - 1, the upward triangle [c - k, c + k) of row r0 + k: it reads row r0
//               under its tip and, at its sides, the lower halves of the previous wave
//   lower half  strip t, the downward triangle [c - h + k, c + h - k) of row r0 + k: it reads
//               only the upper half's last row [c - h, c + h), so the same thread goes on
//               without waiting
// Wave t's centres are offset by h from wave t - 1's, so its upper halves fill the gaps between
// the previous lower halves. Wave 0 has only lower halves (and zeroes the top row's costs under
// them), the last wave only upper halves. Halves are clipped to the picture; neighbouring halves
// never overlap. So strips + 1 waves per seam, where separate up and down triangles took two per
// strip: on tundra 31 instead of 60 with base 64. base is rounded down to even.
//
// Fewer barriers matter less than it seems: one costs about a microsecond natively, and most of
// the time threads wait at barriers they spend idle, waiting for the wave's slowest tile. On 4
// cores (Linux VM on Apple silicon), 600 seams of tundra with full recomputes took 2090 ms in
// diamonds 64 wide against 2160 ms in triangles; the best tile width also moved, from 128 for
// triangles (2460 ms) down to 64. With incremental updates in tiles 32 wide, 610 ms against
// 650 ms: 64 barriers per seam instead of 124, 190 ms of waiting per thread instead of 210.
//
// Incremental updates: removing a seam shifts every row's cells right of it one column left, with
// their costs and parents, so most of them stay right. A cell can only change if a parent's colour
// or position changed, which happens next to the removed seam ("seeds"), or if a parent's cost
// changed. Those changes spread downward in a cone around the seam. So after the first seam, only
// the tiles of a finer tiling that contain a seed or read a cell whose cost changed are recomputed
// ("dirty" tiles); recomputing marks each cell whose cost changed with the seam's epoch. Every
// clean tile's cells have unchanged inputs, so their stored values are still those of a full
// recompute. On tundra, tiles 32 wide recompute 1.21x the cells the core's incremental update
// evaluates, 16% of a full recompute. 600 seams on 4 cores took 610 ms with 4 threads and 1355 ms
// with one, against 1592 ms for the core's incremental update. Tiles 16 wide took 680 ms, 8 wide
// 1160 ms: narrower tiles waste less work but need more barriers, one per strip of base / 2 rows.
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
    // Tile widths: 600 seams of tundra on 4 threads took 2090 ms with full recomputes in tiles 64
    // wide (2310 ms 128 wide, 2160 ms 32 wide) and 610 ms with incremental updates in tiles 32 wide
    // (680 ms 16 wide, 650 ms 48 wide)
    static constexpr int default_base = 64;
    static constexpr int default_incremental_base = 32;

    // rgb is a row-major buffer of width * height * 3 bytes. base: tile width of full recomputes;
    // incremental_base: tile width of incremental updates, smaller since few tiles are dirty
    // spins: barrier checks before sleeping, negative for SleepingBarrier::default_spins()
    TiledSeamCarver(const std::uint8_t* rgb, int width, int height, int threads, int base = default_base,
                    int spins = -1, int incremental_base = default_incremental_base)
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
        const int strips = (height - 1 + incremental_base / 2 - 1) / (incremental_base / 2) + 1;  // and one more
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
        if (incremental_cost_updates && has_removed_) {
            prepare_update();  // the following seams' updates are prepared between seams
        }
        SleepingBarrier sync(threads_, spins_);
        auto worker = [&](int tid) {
            for (int s = 0; s < pixels_to_shrink; ++s) {
                remove_seam(tid, sync, incremental_cost_updates, s + 1 < pixels_to_shrink);
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

    // The diamond tiling of a seam with half height h (see the top of the file): strips, rows of
    // strip s, wave t's first centre, and the clipped cells of a diamond's halves in row r0 + k
    struct Diamonds {
        int h, p, w, height, strips;
        Diamonds(int base, int width, int height_)
            : h(base / 2), p(2 * (base / 2)), w(width), height(height_), strips((height_ - 2 + base / 2) / (base / 2)) {}
        int r0(int s) const { return s * h; }
        int rows(int s) const { return std::min(h, height - 1 - s * h); }
        int offset(int t) const { return (t + 1) % 2 * h; }  // wave t's centres: offset(t) + j * p
        int centres(int t) const { return w + h - 1 - offset(t) < 0 ? 0 : (w + h - 1 - offset(t)) / p + 1; }
        int up_lo(int c, int k) const { return std::max(0, c - k); }
        int up_hi(int c, int k) const { return std::min(w, c + k); }
        int down_lo(int c, int k) const { return std::max(0, c - h + k); }
        int down_hi(int c, int k) const { return std::min(w, c + h - k); }
    };

    // Full recompute of every cell, tiled with base_
    void propagate_all(int tid, SleepingBarrier& sync, int w) {
        const Diamonds d(base_, w, height_);
        for (int t = 0; t <= d.strips; ++t) {
            for (int j = tid; j < d.centres(t); j += threads_) {
                const int c = d.offset(t) + j * d.p;
                if (t == 0) {  // the core's -1 top row cost counts as 0
                    std::fill(cost_.begin() + std::max(0, c - d.h), cost_.begin() + std::min(w, c + d.h), 0.0);
                } else {
                    const int r0 = d.r0(t - 1);
                    for (int k = 1; k <= d.rows(t - 1); ++k) relax_range<false>(d.up_lo(c, k), d.up_hi(c, k), r0 + k, tid);
                }
                if (t < d.strips) {
                    const int r0 = d.r0(t);
                    for (int k = 1; k <= d.rows(t); ++k) relax_range<false>(d.down_lo(c, k), d.down_hi(c, k), r0 + k, tid);
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

    // Recomputes the dirty halves of an incremental diamond tiling, once prepare_update() gave the
    // seeds of the last removed seam and advanced the epoch. Each half is tested on its own: the
    // upper one by the cells it reads outside itself, the lower one by its upper half's last row,
    // which the same thread just recomputed or left alone. Changes spread from the seeds and from
    // changed cells of a strip's top row, by at most h columns per strip, so per wave only diamonds
    // near those columns are tested.
    void propagate_dirty(int tid, SleepingBarrier& sync, int w) {
        const Diamonds d(incremental_base_, w, height_);
        for (int t = 0; t <= d.strips; ++t) {
            int near_lo = INT_MAX, near_hi = INT_MIN;
            if (t > 0) {
                near_lo = std::min(strip_seed_lo_[t - 1], changed_lo_[t - 1].load(std::memory_order_relaxed) - 1);
                near_hi = std::max(strip_seed_hi_[t - 1], changed_hi_[t - 1].load(std::memory_order_relaxed) + 1);
            }
            if (t < d.strips) {
                near_lo = std::min(near_lo, strip_seed_lo_[t]);
                near_hi = std::max(near_hi, strip_seed_hi_[t]);
            }
            // An upper half reads the lower halves h columns either side, whose changes came from
            // up to h columns further out
            const int first = std::max(0, (near_lo - d.p - 2 - d.offset(t)) / d.p);
            const int last = std::min(d.centres(t) - 1, (near_hi + d.p + 2 - d.offset(t)) / d.p);
            for (int j = first + tid; near_lo <= near_hi && j <= last; j += threads_) {
                const int c = d.offset(t) + j * d.p;
                if (t > 0) {  // under its tip, then two columns either side of its previous row
                    const int s = t - 1, r0 = d.r0(s);
                    bool dirty = changed_in(r0, c - 2, c + 1);
                    for (int k = 1; k <= d.rows(s) && !dirty; ++k) {
                        dirty = seeded(r0 + k, d.up_lo(c, k), d.up_hi(c, k)) ||
                                (k > 1 && (changed_in(r0 + k - 1, c - k - 1, c - k) || changed_in(r0 + k - 1, c + k - 1, c + k)));
                    }
                    if (dirty) {
                        recompute_tile(s, r0, d.rows(s), [&](int k) { return d.up_lo(c, k); }, [&](int k) { return d.up_hi(c, k); }, tid);
                    }
                }
                if (t < d.strips) {  // the row above its first row: the upper half's last row
                    const int r0 = d.r0(t);
                    bool dirty = changed_in(r0, d.down_lo(c, 1) - 1, d.down_hi(c, 1));
                    for (int k = 1; k <= d.rows(t) && !dirty; ++k) dirty = seeded(r0 + k, d.down_lo(c, k), d.down_hi(c, k));
                    if (dirty) {
                        recompute_tile(t, r0, d.rows(t), [&](int k) { return d.down_lo(c, k); }, [&](int k) { return d.down_hi(c, k); }, tid);
                    }
                }
            }
            sync.arrive_and_wait();
        }
    }

    // Seeds of an incremental update after removed_, the last removed seam, and a new epoch.
    // Run by one thread between seams.
    void prepare_update() {
        ++epoch_;
        // Seeds: cells whose parents moved, around the removed seam in the new columns.
        // Wider than strictly needed; recomputing a few extra cells is harmless.
        seed_lo_[0] = 1;
        seed_hi_[0] = 0;  // the top row never changes
        for (int y = 1; y < height_; ++y) {
            seed_lo_[y] = std::min(removed_[y], removed_[y - 1]) - 2;
            seed_hi_[y] = std::max(removed_[y], removed_[y - 1]) + 1;
        }
        // Per strip of the incremental tiling: the seeds' columns over its rows, and none of its
        // top row's columns changed yet
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

    // One seam, run by every thread. width_ only changes between barriers.
    // Barriers per seam: one per wave, then after the trace, the shift and the width update.
    // prepare_update() for the next seam rides on the last one.
    void remove_seam(int tid, SleepingBarrier& sync, bool incremental, bool another) {
        const int w = width_;
        if (incremental && has_removed_) {
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
            if (incremental && another) {
                prepare_update();
            }
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
