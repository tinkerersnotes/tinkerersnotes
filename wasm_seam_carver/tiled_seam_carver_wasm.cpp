// Embind bindings: exposes seam_carving::TiledSeamCarver from ../parallel_seam_carver to
// JavaScript, with the same interface as seam_carver_wasm.cpp plus a thread count. Built with
// -pthread, so its memory is a SharedArrayBuffer: browsers only allow that on cross-origin
// isolated pages.
//
// The page runs this module in a dedicated Web Worker (tiled_seam_carver_worker.js), never on
// its main thread: a browser's main thread may not sleep, so Emscripten makes it spin at every
// barrier, which burns a core the other threads need and freezes the page.
//
// The thread pool is started with the module (PTHREAD_POOL_SIZE in build.sh), since a thread
// started on demand needs the event loop of the thread that started it, which is busy while
// seams are removed. The calling thread works too, so a carver may use one thread more than the
// pool has workers.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "tiled_seam_carver.hpp"

namespace {

using seam_carving::TiledSeamCarver;

// JavaScript passes the pixels as a Uint8Array of row-major RGB bytes. threads must not exceed
// the pool's workers plus one, or starting a thread fails (PTHREAD_POOL_SIZE_STRICT in build.sh).
// spins: how often a thread checks a barrier before sleeping; tile: the tile width of incremental
// updates; negative for the defaults.
TiledSeamCarver* create_tiled_seam_carver(const emscripten::val& rgb, int width, int height, int threads, int spins,
                                          int tile) {
    const std::vector<std::uint8_t> bytes = emscripten::convertJSArrayToNumberVector<std::uint8_t>(rgb);
    if (width <= 0 || height <= 0 || bytes.size() != static_cast<std::size_t>(width) * height * 3) {
        throw std::invalid_argument("rgb must hold width * height * 3 bytes");
    }
    return new TiledSeamCarver(bytes.data(), width, height, threads, TiledSeamCarver::default_base, spins,
                               tile < 0 ? TiledSeamCarver::default_incremental_base : tile);
}

void shrink_horizontally(TiledSeamCarver& carver, int pixels_to_shrink, bool incremental_cost_updates) {
    carver.shrink_horizontally(pixels_to_shrink, incremental_cost_updates);
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
