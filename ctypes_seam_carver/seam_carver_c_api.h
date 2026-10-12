/* C-style API over seam_carving::SeamCarver, for callers without C++
 * linkage such as Python ctypes. */
#ifndef SEAM_CARVER_C_API_H
#define SEAM_CARVER_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define SC_API __declspec(dllexport)
#else
#define SC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle, owned by the caller between sc_create and sc_destroy. */
typedef struct sc_seam_carver sc_seam_carver;

typedef enum sc_status {
    SC_OK = 0,
    SC_INVALID_ARGUMENT = 1,
    SC_OUT_OF_RANGE = 2,
    SC_OUT_OF_MEMORY = 3,
    SC_ERROR = 4,
} sc_status;

/* Copies rgb (row-major, rgb_size == width * height * 3 bytes) into a new
 * seam carver stored to *carver. */
SC_API sc_status sc_create(const uint8_t* rgb, size_t rgb_size, int width, int height, sc_seam_carver** carver);

/* Accepts NULL. */
SC_API void sc_destroy(sc_seam_carver* carver);

/* Removes pixels_to_shrink vertical seams, 0 <= pixels_to_shrink < width. */
SC_API sc_status sc_shrink_horizontally(sc_seam_carver* carver, int pixels_to_shrink, int incremental_cost_updates);

/* Current image size, 0 for NULL. */
SC_API int sc_width(const sc_seam_carver* carver);
SC_API int sc_height(const sc_seam_carver* carver);

/* Copies the current image into rgb, rgb_size must be width * height * 3. */
SC_API sc_status sc_copy_shrunk_rgb(const sc_seam_carver* carver, uint8_t* rgb, size_t rgb_size);

/* Message of the last failed call on the calling thread, "" if none. */
SC_API const char* sc_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* SEAM_CARVER_C_API_H */
