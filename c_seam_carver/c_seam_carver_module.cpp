// CSeamCarver: the same public interface as cpp_seam_carver.CppSeamCarver,
// written against the classic Python C API instead of Cython.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "seam_carver_impl.hpp"

namespace {

using seam_carving::SeamCarver;

// Owning reference to a Python object, released when it goes out of scope.
class PyRef {
public:
    explicit PyRef(PyObject* object = nullptr) : object_(object) {}
    ~PyRef() { Py_XDECREF(object_); }
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;

    PyObject* get() const { return object_; }
    explicit operator bool() const { return object_ != nullptr; }

private:
    PyObject* object_;
};

// Read-only view of a bytes-like object's memory, released when it goes out of scope.
class BufferView {
public:
    BufferView() = default;
    ~BufferView() {
        if (acquired_) {
            PyBuffer_Release(&view_);
        }
    }
    BufferView(const BufferView&) = delete;
    BufferView& operator=(const BufferView&) = delete;

    bool acquire(PyObject* object) {
        acquired_ = PyObject_GetBuffer(object, &view_, PyBUF_SIMPLE) == 0;
        return acquired_;
    }
    const std::uint8_t* data() const { return static_cast<const std::uint8_t*>(view_.buf); }
    Py_ssize_t size() const { return view_.len; }

private:
    Py_buffer view_{};
    bool acquired_ = false;
};

// Runs native code and turns C++ exceptions into Python exceptions, using
// the same mapping as Cython's "except +". Returns false if one was raised.
template <class Function>
bool call_native(Function&& function) {
    try {
        std::forward<Function>(function)();
        return true;
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
    } catch (const std::invalid_argument& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const std::out_of_range& e) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown C++ exception");
    }
    return false;
}

struct CSeamCarverObject {
    PyObject_HEAD
    // Never null, created together with the Python object in tp_new
    SeamCarver* carver;
    // Set while the GIL is released around native code, so other threads
    // can't use the carver at the same time.
    bool busy;
};

SeamCarver* checked_carver(CSeamCarverObject* self) {
    if (self->busy) {
        PyErr_SetString(PyExc_RuntimeError, "CSeamCarver is in use by another thread");
        return nullptr;
    }
    return self->carver;
}

void CSeamCarver_dealloc(CSeamCarverObject* self) {
    delete self->carver;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

// Construction happens exactly once in tp_new, like in immutable built-in
// types, so calling __init__ again can't replace the carver.
PyObject* CSeamCarver_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"image", nullptr};
    PyObject* image = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O:CSeamCarver", const_cast<char**>(keywords), &image)) {
        return nullptr;
    }

    PyRef mode(PyObject_GetAttrString(image, "mode"));
    if (!mode) {
        return nullptr;
    }
    const int is_rgb = PyUnicode_Check(mode.get()) && PyUnicode_CompareWithASCIIString(mode.get(), "RGB") == 0;
    PyRef rgb_image(is_rgb ? Py_NewRef(image) : PyObject_CallMethod(image, "convert", "s", "RGB"));
    if (!rgb_image) {
        return nullptr;
    }

    PyRef size(PyObject_GetAttrString(rgb_image.get(), "size"));
    int width = 0;
    int height = 0;
    if (!size || !PyArg_ParseTuple(size.get(), "ii", &width, &height)) {
        return nullptr;
    }

    PyRef rgb(PyObject_CallMethod(rgb_image.get(), "tobytes", nullptr));
    BufferView rgb_view;
    if (!rgb || !rgb_view.acquire(rgb.get())) {
        return nullptr;
    }
    if (width <= 0 || height <= 0) {
        PyErr_SetString(PyExc_ValueError, "image must be non-empty");
        return nullptr;
    }
    if (rgb_view.size() != Py_ssize_t(width) * height * 3) {
        PyErr_Format(PyExc_ValueError, "expected %d x %d x 3 RGB bytes, got %zd", width, height, rgb_view.size());
        return nullptr;
    }

    // The C++ side copies the pixels into its mesh, rgb may be freed afterwards
    std::unique_ptr<SeamCarver> carver;
    if (!call_native([&] { carver = std::make_unique<SeamCarver>(rgb_view.data(), width, height); })) {
        return nullptr;
    }
    auto* self = reinterpret_cast<CSeamCarverObject*>(type->tp_alloc(type, 0));
    if (self == nullptr) {
        return nullptr;
    }
    self->carver = carver.release();
    return reinterpret_cast<PyObject*>(self);
}

// Intentionally a no-op. It lets subclasses call super().__init__(image),
// which object.__init__ would reject once a subclass defines __init__.
int CSeamCarver_init(PyObject* Py_UNUSED(self), PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"image", nullptr};
    PyObject* image = nullptr;
    return PyArg_ParseTupleAndKeywords(args, kwargs, "O:CSeamCarver", const_cast<char**>(keywords), &image) ? 0 : -1;
}

PyObject* CSeamCarver_shrink_horizontally(CSeamCarverObject* self, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"pixels_to_shrink", "incremental_cost_updates", nullptr};
    int pixels_to_shrink = 0;
    int incremental_cost_updates = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|p:shrink_horizontally", const_cast<char**>(keywords),
                                     &pixels_to_shrink, &incremental_cost_updates)) {
        return nullptr;
    }
    SeamCarver* carver = checked_carver(self);
    if (carver == nullptr) {
        return nullptr;
    }

    // C++ exceptions must not cross the GIL release block, so carry them over
    std::exception_ptr error;
    self->busy = true;
    Py_BEGIN_ALLOW_THREADS
    try {
        carver->shrink_horizontally(pixels_to_shrink, incremental_cost_updates != 0);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    self->busy = false;

    if (!call_native([&] {
            if (error) {
                std::rethrow_exception(error);
            }
        })) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* CSeamCarver_get_shrunk_image(CSeamCarverObject* self, PyObject* Py_UNUSED(ignored)) {
    SeamCarver* carver = checked_carver(self);
    if (carver == nullptr) {
        return nullptr;
    }
    std::vector<std::uint8_t> rgb;
    if (!call_native([&] { rgb = carver->shrunk_rgb(); })) {
        return nullptr;
    }
    PyRef rgb_bytes(PyBytes_FromStringAndSize(reinterpret_cast<const char*>(rgb.data()), Py_ssize_t(rgb.size())));
    PyRef image_module(PyImport_ImportModule("PIL.Image"));
    if (!rgb_bytes || !image_module) {
        return nullptr;
    }
    return PyObject_CallMethod(image_module.get(), "frombytes", "s(ii)O", "RGB", carver->width(), carver->height(),
                               rgb_bytes.get());
}

PyObject* CSeamCarver_get_width(CSeamCarverObject* self, void* Py_UNUSED(closure)) {
    SeamCarver* carver = checked_carver(self);
    return carver == nullptr ? nullptr : PyLong_FromLong(carver->width());
}

PyObject* CSeamCarver_get_height(CSeamCarverObject* self, void* Py_UNUSED(closure)) {
    SeamCarver* carver = checked_carver(self);
    return carver == nullptr ? nullptr : PyLong_FromLong(carver->height());
}

PyMethodDef CSeamCarver_methods[] = {
    {"shrink_horizontally", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(CSeamCarver_shrink_horizontally)),
     METH_VARARGS | METH_KEYWORDS,
     "shrink_horizontally(pixels_to_shrink, incremental_cost_updates=True)\n--\n\n"
     "Remove pixels_to_shrink vertical seams of the lowest cost."},
    {"get_shrunk_image", reinterpret_cast<PyCFunction>(CSeamCarver_get_shrunk_image), METH_NOARGS,
     "get_shrunk_image()\n--\n\n"
     "Return the current image as a new RGB PIL.Image.Image."},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef CSeamCarver_getset[] = {
    {"width", reinterpret_cast<getter>(CSeamCarver_get_width), nullptr, "Current image width.", nullptr},
    {"height", reinterpret_cast<getter>(CSeamCarver_get_height), nullptr, "Current image height.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyTypeObject CSeamCarverType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "c_seam_carver.CSeamCarver",  // tp_name
};

PyModuleDef c_seam_carver_module = {
    PyModuleDef_HEAD_INIT,
    "c_seam_carver",  // m_name
    "Seam carver written with the classic Python C API over a C++ core.",  // m_doc
    -1,  // m_size
};

}  // namespace

PyMODINIT_FUNC PyInit_c_seam_carver() {
    // C++ has no designated initializers before C++20, so fill the slots here
    CSeamCarverType.tp_basicsize = sizeof(CSeamCarverObject);
    CSeamCarverType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    CSeamCarverType.tp_doc = PyDoc_STR(
        "CSeamCarver(image)\n--\n\n"
        "Seam carver over a copy of the RGB pixels of a PIL image.");
    CSeamCarverType.tp_new = CSeamCarver_new;
    CSeamCarverType.tp_init = CSeamCarver_init;
    CSeamCarverType.tp_dealloc = reinterpret_cast<destructor>(CSeamCarver_dealloc);
    CSeamCarverType.tp_methods = CSeamCarver_methods;
    CSeamCarverType.tp_getset = CSeamCarver_getset;
    if (PyType_Ready(&CSeamCarverType) < 0) {
        return nullptr;
    }

    PyObject* module = PyModule_Create(&c_seam_carver_module);
    if (module == nullptr) {
        return nullptr;
    }
    if (PyModule_AddObjectRef(module, "CSeamCarver", reinterpret_cast<PyObject*>(&CSeamCarverType)) < 0) {
        Py_DECREF(module);
        return nullptr;
    }
    return module;
}
