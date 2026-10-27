// Prototype: seam carving with the path cost propagation parallelized by triangle tiling.
//
// Dense row-major arrays instead of the linked mesh: per pixel the cost of the cheapest path from
// the top row and the parent's column offset (-1, 0, +1). Every seam is a full recompute, the
// same rules as SeamCarver::propagate_low_cost_paths(): parents in the order up, up-left, up-right,
// a later one wins only if strictly cheaper, so results must match the core's
// shrink_horizontally(n, false) bit for bit.
//
// Tiling, per strip of h rows below a fully computed row r0:
//   phase A  each task owns a base segment [a, b) of row r0 and fills the downward triangle
//            [a + k, b - k) of row r0 + k; segments at the image border don't shrink on that side
//   phase B  each boundary c between two segments fills the upward triangle [c - k, c + k),
//            which reads the sides of both neighbouring triangles from phase A
// Two barriers per strip. h = base / 2, so neighbouring upward triangles never overlap.
//
// CHECK=1 records which (seam, phase) wrote each cell and asserts every cell is written exactly
// once, and that every parent is final before it is read.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>

#include "seam_carver_impl.hpp"

#ifndef CHECK
#define CHECK 0
#endif
#ifndef BRANCHY
#define BRANCHY 0
#endif

struct Image {
    int stride, width, height;   // stride: the original width, rows are compacted in place
    std::vector<std::uint8_t> rgb;
    const std::uint8_t* px(int x, int y) const { return &rgb[3 * (std::size_t(y) * stride + x)]; }
};

static inline double distance(const std::uint8_t* p, const std::uint8_t* c) {
    const double r = double(p[0]) - c[0], g = double(p[1]) - c[1], b = double(p[2]) - c[2];
    return std::sqrt(r * r + g * g + b * b);
}

struct Carver {
    Image& im;
    std::vector<double> cost;
    std::vector<std::int8_t> prev;
#if CHECK
    std::vector<std::uint32_t> stamp;    // seam number * 4 + 1 + writer phase, 0 = never written
    std::uint32_t seam_no = 0;
#endif
    explicit Carver(Image& image)
        : im(image), cost(std::size_t(image.stride) * image.height), prev(cost.size()) {
#if CHECK
        stamp.assign(cost.size(), 0);
#endif
    }

    // One cell of the recurrence, the same arithmetic and tie-breaking as the core
    inline void relax(int x, int y, [[maybe_unused]] int phase) {
        const std::size_t row = std::size_t(y) * im.stride, up = row - im.stride;
#if CHECK
        const std::uint32_t mine = seam_no * 4 + 1;
        if (stamp[row + x] >= mine) { std::fprintf(stderr, "cell (%d,%d) written twice\n", x, y); std::abort(); }
        for (int dx = -1; dx <= 1; ++dx) {
            const int px = x + dx;
            if (px < 0 || px >= im.width) continue;
            if (y - 1 > 0 && stamp[up + px] < mine) {
                std::fprintf(stderr, "cell (%d,%d) read parent (%d,%d) before it was computed\n", x, y, px, y - 1);
                std::abort();
            }
        }
#endif
        const std::uint8_t* c = im.px(x, y);
        double best = cost[up + x] + distance(im.px(x, y - 1), c);
        std::int8_t from = 0;
        if (x > 0) {
            const double v = cost[up + x - 1] + distance(im.px(x - 1, y - 1), c);
            if (v < best) { best = v; from = -1; }
        }
        if (x + 1 < im.width) {
            const double v = cost[up + x + 1] + distance(im.px(x + 1, y - 1), c);
            if (v < best) { best = v; from = 1; }
        }
        cost[row + x] = best;
        prev[row + x] = from;
#if CHECK
        stamp[row + x] = mine + phase;
#endif
    }

    // Cells [lo, hi) of row y. Same arithmetic as relax(), with everything the compiler would
    // otherwise reload after each int8_t store (a char type may alias anything) held in locals.
    void relax_range(int lo, int hi, int y, [[maybe_unused]] int phase) {
#if CHECK
        for (int x = lo; x < hi; ++x) relax(x, y, phase);
#else
        const int width = im.width;
        const std::size_t row = std::size_t(y) * im.stride;
        const double* __restrict up_cost = cost.data() + row - im.stride;
        double* __restrict out_cost = cost.data() + row;
        std::int8_t* __restrict out_prev = prev.data() + row;
        const std::uint8_t* __restrict up_px = im.rgb.data() + 3 * (row - im.stride);
        const std::uint8_t* __restrict px = im.rgb.data() + 3 * row;
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
        if (lo >= hi) return;
        int first = lo, last = hi;
        if (first == 0) edge(first++);
        if (last == width && last > first) edge(--last);
        // Interior: the same up, left, right priority with strict <, written as selects instead of
        // branches. On a textured photo those comparisons are coin flips the predictor can't learn.
#if BRANCHY
        for (int x = first; x < last; ++x) edge(x);   // A/B: the original branchy form
#else
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
#endif
#endif
    }

    void top_row() {
        std::fill_n(cost.begin(), im.width, 0.0);   // the core's -1 cost counts as 0 for children
    }

    void serial_propagate() {
        top_row();
        for (int y = 1; y < im.height; ++y)
            relax_range(0, im.width, y, 0);
    }

    // Leftmost cheapest bottom pixel, then the parents back to the top; removes the seam
    void remove_seam_rows(int y_begin, int y_end, const std::vector<int>& seam) {
        for (int y = y_begin; y < y_end; ++y) {
            std::uint8_t* row = &im.rgb[3 * std::size_t(y) * im.stride];
            const int x = seam[y];
            std::memmove(row + 3 * x, row + 3 * (x + 1), 3 * std::size_t(im.width - x - 1));
        }
    }
    void trace_seam(std::vector<int>& seam) const {
        const std::size_t bottom = std::size_t(im.height - 1) * im.stride;
        int x = 0;
        for (int i = 1; i < im.width; ++i)
            if (cost[bottom + i] < cost[bottom + x]) x = i;
        for (int y = im.height - 1; y >= 0; --y) {
            seam[y] = x;
            if (y > 0) x += prev[std::size_t(y) * im.stride + x];
        }
    }
};

static Image load(const char* path, int width, int height) {
    std::ifstream in(path, std::ios::binary);
    Image im{width, width, height, std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), {})};
    if (im.rgb.size() != std::size_t(width) * height * 3) { std::fprintf(stderr, "bad input\n"); std::exit(1); }
    return im;
}

static std::vector<std::uint8_t> shrunk(const Image& im) {
    std::vector<std::uint8_t> out;
    for (int y = 0; y < im.height; ++y)
        out.insert(out.end(), im.px(0, y), im.px(0, y) + 3 * im.width);
    return out;
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// --- serial dense reference -------------------------------------------------------------------
static std::vector<std::uint8_t> run_serial(Image im, int seams, double& ms) {
    Carver c(im);
    std::vector<int> seam(im.height);
    const auto t = std::chrono::steady_clock::now();
    for (int s = 0; s < seams; ++s) {
#if CHECK
        ++c.seam_no;
#endif
        c.serial_propagate();
        c.trace_seam(seam);
        c.remove_seam_rows(0, im.height, seam);
        --im.width;
    }
    ms = ms_since(t);
    return shrunk(im);
}

// --- triangle-tiled, persistent threads ---------------------------------------------------------
static std::vector<std::uint8_t> run_tiled(Image im, int seams, int threads, int base, double& ms) {
    Carver c(im);
    std::vector<int> seam(im.height);
    std::barrier sync(threads);
    const int h = base / 2;  // strip height; base >= 2 is checked in main()

    auto worker = [&](int tid) {
        for (int s = 0; s < seams; ++s) {
            const int w = im.width;                          // stable: changes only between barriers
            const int tasks = (w + base - 1) / base;
            if (tid == 0) {
#if CHECK
                ++c.seam_no;
#endif
                c.top_row();
            }
            sync.arrive_and_wait();
            for (int r0 = 0; r0 < im.height - 1; r0 += h) {
                const int rows = std::min(h, im.height - 1 - r0);
                // Phase A: downward triangles over each base segment
                for (int i = tid; i < tasks; i += threads) {
                    const int a = i * base, b = std::min(w, a + base);
                    for (int k = 1; k <= rows; ++k) {
                        const int lo = i == 0 ? 0 : a + k, hi = i == tasks - 1 ? w : b - k;
                        c.relax_range(lo, hi, r0 + k, 1);
                    }
                }
                sync.arrive_and_wait();
                // Phase B: upward triangles at the boundaries between segments
                for (int j = 1 + tid; j < tasks; j += threads) {
                    const int edge = j * base;
                    for (int k = 1; k <= rows; ++k) {
                        const int lo = std::max(0, edge - k), hi = std::min(w, edge + k);
                        c.relax_range(lo, hi, r0 + k, 2);
                    }
                }
                sync.arrive_and_wait();
            }
            if (tid == 0) c.trace_seam(seam);
            sync.arrive_and_wait();
            const int per = (im.height + threads - 1) / threads;   // rows are independent
            c.remove_seam_rows(std::min(im.height, tid * per), std::min(im.height, (tid + 1) * per), seam);
            sync.arrive_and_wait();
            if (tid == 0) --im.width;
            sync.arrive_and_wait();
        }
    };
    const auto t = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int tid = 1; tid < threads; ++tid) pool.emplace_back(worker, tid);
    worker(0);
    for (auto& th : pool) th.join();
    ms = ms_since(t);
    return shrunk(im);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s image.rgb width height seams threads base [mode...]\n", argv[0]);
        return 2;
    }
    const Image im = load(argv[1], std::atoi(argv[2]), std::atoi(argv[3]));
    const int seams = std::atoi(argv[4]), threads = std::atoi(argv[5]), base = std::atoi(argv[6]);
    // Upward triangles at neighbouring boundaries grow 2k wide by row k <= base / 2: they only
    // stay apart if every base segment is at least 2 pixels wide
    if (base < 2 || threads < 1) {
        std::fprintf(stderr, "base must be >= 2 and threads >= 1\n");
        return 2;
    }
    double ms = 0;

    std::vector<std::uint8_t> reference;
    for (int i = 7; i < argc; ++i) {
        const std::string mode = argv[i];
        std::vector<std::uint8_t> out;
        if (mode == "core-full" || mode == "core-incremental") {
            const auto t = std::chrono::steady_clock::now();
            seam_carving::SeamCarver core(im.rgb.data(), im.width, im.height);
            const auto t_shrink = std::chrono::steady_clock::now();
            core.shrink_horizontally(seams, mode == "core-incremental");
            ms = ms_since(t_shrink);
            (void)t;
            out = core.shrunk_rgb();
        } else if (mode == "serial") {
            out = run_serial(im, seams, ms);
        } else if (mode == "tiled") {
            out = run_tiled(im, seams, threads, base, ms);
        } else {
            std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
            return 2;
        }
        if (const char* dir = std::getenv("DUMP")) {   // DUMP=dir writes each result as raw RGB
            std::ofstream(std::string(dir) + "/" + mode + ".rgb", std::ios::binary)
                .write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
        }
        if (reference.empty()) reference = out;
        std::printf("%-17s threads=%-2d base=%-4d %8.0f ms   identical to %s: %s\n", mode.c_str(),
                    mode == "tiled" ? threads : 1, mode == "tiled" ? base : 0, ms, argv[7],
                    out == reference ? "yes" : "NO");
    }
}
