#include "seam_carver_c_api.h"

#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "seam_carver_impl.hpp"

struct sc_seam_carver {
    template <class... Args>
    explicit sc_seam_carver(Args&&... args) : carver(std::forward<Args>(args)...) {}

    seam_carving::SeamCarver carver;
};

namespace {

thread_local std::string last_error;

void set_last_error(const char* message) noexcept {
    try {
        last_error = message;
    } catch (...) {
        last_error.clear();
    }
}

sc_status fail(sc_status status, const char* message) noexcept {
    set_last_error(message);
    return status;
}

// C++ exceptions must not cross the C boundary, so turn them into status codes.
template <class Function>
sc_status guarded(Function&& function) noexcept {
    last_error.clear();
    try {
        std::forward<Function>(function)();
        return SC_OK;
    } catch (const std::bad_alloc&) {
        return fail(SC_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& e) {
        return fail(SC_INVALID_ARGUMENT, e.what());
    } catch (const std::out_of_range& e) {
        return fail(SC_OUT_OF_RANGE, e.what());
    } catch (const std::exception& e) {
        return fail(SC_ERROR, e.what());
    } catch (...) {
        return fail(SC_ERROR, "unknown C++ exception");
    }
}

}  // namespace

sc_status sc_create(const uint8_t* rgb, size_t rgb_size, int width, int height, sc_seam_carver** carver) {
    if (carver == nullptr) {
        return fail(SC_INVALID_ARGUMENT, "carver must not be NULL");
    }
    *carver = nullptr;
    if (width > 0 && height > 0 && rgb_size != static_cast<size_t>(width) * height * 3) {
        return fail(SC_INVALID_ARGUMENT, "rgb_size must be width * height * 3");
    }
    return guarded([&] { *carver = new sc_seam_carver(rgb, width, height); });
}

void sc_destroy(sc_seam_carver* carver) {
    delete carver;
}

sc_status sc_shrink_horizontally(sc_seam_carver* carver, int pixels_to_shrink, int incremental_cost_updates) {
    if (carver == nullptr) {
        return fail(SC_INVALID_ARGUMENT, "carver must not be NULL");
    }
    return guarded([&] { carver->carver.shrink_horizontally(pixels_to_shrink, incremental_cost_updates != 0); });
}

int sc_width(const sc_seam_carver* carver) {
    return carver == nullptr ? 0 : carver->carver.width();
}

int sc_height(const sc_seam_carver* carver) {
    return carver == nullptr ? 0 : carver->carver.height();
}

sc_status sc_copy_shrunk_rgb(const sc_seam_carver* carver, uint8_t* rgb, size_t rgb_size) {
    if (carver == nullptr || rgb == nullptr) {
        return fail(SC_INVALID_ARGUMENT, "carver and rgb must not be NULL");
    }
    return guarded([&] {
        const std::vector<uint8_t> shrunk = carver->carver.shrunk_rgb();
        if (shrunk.size() != rgb_size) {
            throw std::invalid_argument("rgb_size must be width * height * 3");
        }
        std::memcpy(rgb, shrunk.data(), shrunk.size());
    });
}

const char* sc_last_error(void) {
    return last_error.c_str();
}
