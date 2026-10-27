// Embind bindings: exposes seam_carving::TiledSeamCarver from ../parallel_seam_carver to
// JavaScript, with the same interface as seam_carver_wasm.cpp. Built with -pthread, so its
// memory is a SharedArrayBuffer: browsers only allow that on cross-origin isolated pages.
//
// The thread pool is started with the module (PTHREAD_POOL_SIZE in build.sh), since a thread
// started on demand needs the main thread's event loop, which is blocked while seams are removed.
// The calling thread works too, so a carver uses one thread more than the pool has workers.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "tiled_seam_carver.hpp"

namespace {

using seam_carving::TiledSeamCarver;

// Must match PTHREAD_POOL_SIZE in build.sh: at most this many threads, the pool has one fewer
constexpr int kMaxThreads = 8;

int thread_count() {
    return std::clamp(static_cast<int>(std::thread::hardware_concurrency()), 1, kMaxThreads);
}

// JavaScript passes the pixels as a Uint8Array of row-major RGB bytes
TiledSeamCarver* create_tiled_seam_carver(const emscripten::val& rgb, int width, int height) {
    const std::vector<std::uint8_t> bytes = emscripten::convertJSArrayToNumberVector<std::uint8_t>(rgb);
    if (width <= 0 || height <= 0 || bytes.size() != static_cast<std::size_t>(width) * height * 3) {
        throw std::invalid_argument("rgb must hold width * height * 3 bytes");
    }
    return new TiledSeamCarver(bytes.data(), width, height, thread_count());
}

// Every seam is a full recompute: the flag the other engines take is accepted and ignored
void shrink_horizontally(TiledSeamCarver& carver, int pixels_to_shrink, bool /*incremental_cost_updates*/) {
    carver.shrink_horizontally(pixels_to_shrink);
}

// Copies the shrunk image's RGB bytes into a new JavaScript Uint8Array: a view into the
// module's memory would dangle once this vector is freed or the memory grows
emscripten::val shrunk_rgb(const TiledSeamCarver& carver) {
    const std::vector<std::uint8_t> rgb = carver.shrunk_rgb();
    return emscripten::val::global("Uint8Array").new_(emscripten::typed_memory_view(rgb.size(), rgb.data()));
}

}  // namespace

EMSCRIPTEN_BINDINGS(tiled_seam_carver) {
    emscripten::class_<TiledSeamCarver>("SeamCarver")
        // JavaScript owns the object: it must call delete(), WebAssembly memory isn't garbage collected
        .constructor(&create_tiled_seam_carver, emscripten::allow_raw_pointers())
        .function("shrink_horizontally", &shrink_horizontally)
        .function("shrunk_rgb", &shrunk_rgb)
        .property("width", &TiledSeamCarver::width)
        .property("height", &TiledSeamCarver::height)
        .property("threads", &TiledSeamCarver::threads);
}
