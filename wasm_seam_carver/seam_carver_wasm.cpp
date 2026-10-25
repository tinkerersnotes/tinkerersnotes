// Embind bindings: exposes seam_carving::SeamCarver from ../seam_carver_core, unchanged, to JavaScript.
// Methods and properties bind straight to the C++ members. Only the constructor and shrunk_rgb()
// need small adapters, since raw pointers and std::vector don't cross into JavaScript as such.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "seam_carver_impl.hpp"

namespace {

using seam_carving::SeamCarver;

// JavaScript passes the pixels as a Uint8Array of row-major RGB bytes
SeamCarver* create_seam_carver(const emscripten::val& rgb, int width, int height) {
    const std::vector<std::uint8_t> bytes = emscripten::convertJSArrayToNumberVector<std::uint8_t>(rgb);
    if (width <= 0 || height <= 0 || bytes.size() != static_cast<std::size_t>(width) * height * 3) {
        throw std::invalid_argument("rgb must hold width * height * 3 bytes");
    }
    // The C++ side copies the pixels into its mesh, bytes may be freed afterwards
    return new SeamCarver(bytes.data(), width, height);
}

// Copies the shrunk image's RGB bytes into a new JavaScript Uint8Array: a view into the
// module's memory would dangle once this vector is freed or the memory grows
emscripten::val shrunk_rgb(const SeamCarver& carver) {
    const std::vector<std::uint8_t> rgb = carver.shrunk_rgb();
    return emscripten::val::global("Uint8Array").new_(emscripten::typed_memory_view(rgb.size(), rgb.data()));
}

}  // namespace

EMSCRIPTEN_BINDINGS(seam_carver) {
    emscripten::class_<SeamCarver>("SeamCarver")
        // JavaScript owns the object: it must call delete(), WebAssembly memory isn't garbage collected
        .constructor(&create_seam_carver, emscripten::allow_raw_pointers())
        .function("shrink_horizontally", &SeamCarver::shrink_horizontally)
        .function("shrunk_rgb", &shrunk_rgb)
        .property("width", &SeamCarver::width)
        .property("height", &SeamCarver::height);
}
