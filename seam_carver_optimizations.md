# Seam carver: an optimization journey across Python, Cython and native code

One algorithm, five Python-importable flavors, and a series of measured optimizations.
Every step was verified to produce pixel-identical output across all modules.

## Setup

- Linux VM (aarch64, Parallels), Python 3.12.3, Cython 3.3.0, g++ 13.3, Pillow 12.3.
- Test image: `tundra_1280x958.jpeg`. A run removes 50 or 600 vertical seams
  (1280 → 680 px wide) with incremental cost updates unless stated otherwise.
- Timings are wall-clock, best of 3 or median of 3–5 alternating runs. Runs on this VM vary by
  a few percent, so differences under ~3% are noise.

## The algorithm in one paragraph

Each pixel is a node in a 4-way linked mesh (`up`, `down`, `left`, `right`) with the cost of
the cheapest path from the top row and a `prev` link to the parent on that path. A seam is the
cheapest top-to-bottom path. Removing it only relinks pointers. After the first seam, costs are
updated incrementally: only nodes downstream of the removed seam are re-evaluated, row by row.

## Module flavors

All modules share one public API: `Carver(image)`, `shrink_horizontally(pixels_to_shrink,
incremental_cost_updates=True)`, `get_shrunk_image()` → RGB `PIL.Image`, `width`, `height`.

| Module | Class | Flavor |
|---|---|---|
| `py_seam_carver` | `PySeamCarver` | Pure Python. Nodes are `__slots__` objects. Mirrors the Cython version method for method. |
| `cy_seam_carver` | `CySeamCarver` | "Python-ish" Cython: `cdef class` nodes linked by Python object references. |
| `cpp_seam_carver` | `CppSeamCarver` | Cython wrapper (`cdef extern … cppclass`, `unique_ptr`, `nogil`) over the shared C++ core `seam_carver_impl.{hpp,cpp}`. The mesh is a single reserved `std::vector<MeshNode>`. |
| `c_seam_carver` | `CSeamCarver` | Hand-written classic Python C API extension (static `PyTypeObject`, `tp_new`, RAII helpers) over the same C++ core. |
| `ctypes_seam_carver` | `CtypesSeamCarver` | Pure Python `ctypes` wrapper over an `extern "C"` dynamic library (`libseam_carver.so`) built from the same C++ core. |

The three native flavors construct their object once, in `__cinit__`/`tp_new`/`__new__`. They
reject concurrent use while the GIL is released, and their `__init__` is a no-op that still
supports `super().__init__(image)`.

## Results at a glance

600 seams, total time including construction and export:

| Module | First measurement | Final | Overall gain |
|---|---|---|---|
| `PySeamCarver` | 136.5 s | **52.8 s** | ×2.6 |
| `CySeamCarver` | ~72 s ¹ | **3.55 s** | ~×20 |
| `CppSeamCarver` | 3.87 s | **1.49 s** | ×2.6 |
| `CSeamCarver` | 3.80 s | **1.48 s** | ×2.6 |
| `CtypesSeamCarver` | 2.81 s ² | **1.52 s** | ×1.8 |

¹ The original `def`-method Cython code, taken from the notebook's saved output and not re-measured.
² First measured after it was already built with hidden symbol visibility (see step 2).

Final 50-seam run:

| Module | Construct | Shrink | Export | Total |
|---|---|---|---|---|
| `PySeamCarver` | 1.39 s | 4.99 s | 0.22 s | 6.60 s |
| `CySeamCarver` | 0.35 s | 0.28 s | 0.28 s | 0.91 s |
| `CppSeamCarver` / `CSeamCarver` / `CtypesSeamCarver` | 0.02 s | 0.14–0.15 s | 0.00 s | 0.17 s |

---

## 1. Native core: C++ instead of Python objects

**Change.** The same algorithm ported to C++, with all nodes in one contiguous
`std::vector<MeshNode>`, linked by raw pointers. Capacity is reserved once and never exceeded,
so pointers stay valid. The first wrapper was Cython (`CppSeamCarver`).

**Gain** against the original `def`-method Cython code:

| Run | Original Cython | C++ core | Gain |
|---|---|---|---|
| 50 seams, incremental | 8.14 s | 0.39 s | ×21 |
| 50 seams, full recompute | 20.67 s | 1.69 s | ×12 |
| 10 seams, incremental | 2.86 s | 0.12 s | ×24 |

## 2. Binding flavor does not matter, symbol visibility does

**Observation.** The C API and Cython wrappers ran at the same speed (0.38 s vs 0.40 s for
50 seams), since each call does a lot of work. The new `ctypes` flavor was unexpectedly about
**25% faster**: 1.40 s against about 1.87 s for 300 seams.

**Root cause.** Not ctypes. Its `build.sh` compiled with `-fvisibility=hidden`; setuptools does
not. In a `-fPIC` shared library every default-visibility function can be replaced by another
library at load time, so GCC will not inline the hot helpers into the seam loops.

| Build, 300 seams | Shrink time |
|---|---|
| ctypes library with `-fvisibility=hidden` | 1.37 s |
| ctypes library without it | 1.96 s |
| `CSeamCarver` rebuilt with it | 1.33 s |

**Change.** Added `-fvisibility=hidden` to `cpp_seam_carver.pyx` and `c_seam_carver/setup.py`.

**Gain.** All three native flavors now run 300 seams in 1.36–1.41 s, about **×1.35**. Exported
symbols dropped from 14 to 1 (`PyInit_*`). Other compiler flags (`-O3`, `-DNDEBUG`, `-g`,
`-fno-strict-overflow`) made no difference (1.35–1.41 s).

## 3. Profiling the Python-ish Cython code

**Tooling.**
- `cProfile` on Python 3.12 does not see Cython functions: it uses `sys.monitoring`, which
  Cython supports only on 3.13+.
- The pure-Python `profile` module crashes on Cython frames.
- `perf` is blocked on this VM (`perf_event_paranoid = 4`).
- What worked: a 30-line `sys.setprofile` hook, plus the per-line scores of `cython -a`.

**Observations.**
- **All the hot time is in calls between internal `def` methods.** Per seam on a 640×479 copy,
  an incremental shrink makes about 400k `_distance` and 118k `_update_cost_if_needed` calls; a
  full recompute makes about 914k each of `_update_cost` and `_distance`; the constructor makes
  1.8M `_node_index` calls.
- **What each call costs:** Python method lookup, argument parsing and type checks, plus boxing
  and unboxing of every `double` result.
- **Other Python overhead:** a temporary Python `list` per `_update_cost_if_needed` call, and a
  `dict[int, set]` work queue that boxes an `int` and hashes a node per operation.
- **Pixel I/O:** pixels move one at a time through `PixelAccess`.
- **Not a problem:** attribute access on typed nodes already compiles to plain C, and Cython
  infers types for untyped locals. Adding more annotations would gain nothing.
- **Garbage collector:** about 60% of construction time goes to the cyclic garbage collector
  scanning 1.2M linked nodes.

**Measured candidates**, applied cumulatively in scratch copies:

| Variant | Construct | 50 incremental seams | Full seam | Export |
|---|---|---|---|---|
| Original (`def` methods) | 1.02 s | 6.60 s | 0.39 s | 0.28 s |
| + `cdef` internal methods | 0.34 s | 2.27 s | 0.04 s | 0.29 s |
| + no temporary parent list | 0.36 s | 0.82 s | 0.04 s | 0.28 s |
| + list-per-row queue instead of `dict`/`set` | 0.37 s | 0.39 s | 0.04 s | 0.28 s |
| + byte buffers in and out | 0.18 s | 0.40 s | 0.04 s | 0.01 s |
| + `@cython.final`, no bounds checks | 0.17 s | 0.40 s | 0.04 s | 0.01 s |

`@cython.final` and the directives brought no measurable gain, and `final` would block
subclassing, so that step was dropped.

## 4. Optimization 1 (applied): internal methods become `cdef`

**Change.** Ten internal methods became `cdef` with C argument and return types (`cdef inline
double _distance(...) noexcept`, …). Method bodies were unchanged.

**Gain** (`CySeamCarver`):

| Run | Before | After | Gain |
|---|---|---|---|
| 50 incremental seams | 6.66 s | 2.28 s | ×2.9 |
| 5 full-recompute seams | 1.98 s | 0.18 s | ×10.7 |
| Construction | 1.02 s | 0.35 s | ×2.9 |
| 600 seams total | ~72 s (old notebook) | 27.35 s | ~×2.6 |

## 5. The pure Python flavor and two "trick" experiments

**`PySeamCarver`** is a line-by-line port of `CySeamCarver` with every Cython construct
replaced by plain Python. 600 seams took **136.5 s**: about 5× slower than the Cython version
on incremental seams (11.9 s vs 2.26 s for 50 seams) and 17× slower on a full recompute.

**Compiling the pure Python file with Cython is 2.6–3.2× slower** than interpreting it:

| Run | Interpreted | Compiled by Cython |
|---|---|---|
| 50 incremental seams | 11.9 s | 32.6 s |
| One full-recompute seam | 0.63 s | 1.99 s |
| 600 seams total | 136.5 s | 355.6 s |

- Cython used only the `float` (→ C `double`) and `bool` annotations. `int` and plain-class
  annotations stay Python objects.
- Turning annotation typing off barely changed anything (31.2 s vs 32.4 s), so the annotations
  are not the cause.
- Most likely cause: CPython 3.12 specializes attribute access on `__slots__` classes and
  Python-to-Python calls at run time. Cython-compiled code instead does a generic attribute
  lookup each time (21 reads and 5 writes in `_update_cost_if_needed` alone) and a generic call
  for each method.
- **Lesson:** Cython speeds code up through `cdef class` and `cdef` methods, not by compiling
  unchanged Python.

**Removing type annotations from the interpreted module had no effect.** Every method's
bytecode is identical with and without them: local annotations are never executed, and
parameter annotations are evaluated once at `def` time.

| Run | With annotations | Without annotations |
|---|---|---|
| 30 incremental seams | 7.49 s | 7.60 s |
| One full-recompute seam | 0.64 s | 0.64 s |

## 6. Optimization 2 (applied): no temporary parent list

**Change.** `_update_cost_if_needed` checks `up`, then `up.left` and `up.right` directly instead
of building a list. The repeated path-cost expression became a helper, `_path_cost(parent,
to_node)`, which `_update_cost` also uses.

**Gain:**

| Module | Run | Before | After |
|---|---|---|---|
| `CySeamCarver` | 50 incremental seams | 2.26 s | **0.82 s** (×2.8) |
| `CySeamCarver` | 5 full-recompute seams | 0.18 s | 0.18 s (that path never built the list) |
| `PySeamCarver` | 30 incremental seams, code repeated inline | 7.47 s | 6.24 s (×1.2) |
| `PySeamCarver` | 20 incremental seams, inline vs helper | 4.46 s | 4.95 s |
| `PySeamCarver` | 2 full-recompute seams, with helper | 1.27 s | 1.43 s |

**Design choices.**
- **Helper vs. code repeated inline.** In Cython the `cdef inline` helper is free (0.81 s either
  way). In pure Python each helper call costs, about 10–12%. It was **kept on purpose**: it
  shows where Cython and native code win.
- **Loop over a tuple.** Looping over `(up.left, up.right)` was rejected: Cython builds a Python
  tuple on every call (`PyTuple_New(2)`), making it 1.8× slower (1.43 s).
- **C++ needed nothing.** It already loops over a fixed three-element array on the stack. The
  compiled `update_cost_if_needed` is 120 instructions with no calls: the loop is unrolled and
  `distance` inlined three times.

## 7. Reserving C++ queue capacity: no gain

**Idea.** Pre-reserve the per-row update queues: (B) to `min(previous row size + 2, width)`
before each row, or (C) once per row to the arithmetic-progression bound `min(2y + 3, width)`.

| Variant | 600 seams | Reallocations |
|---|---|---|
| A: current | 2.723 s | 8,896 |
| B: next row, `prev + 2` | 2.721 s | 1,939 |
| C: all rows once, `2y + 3` | 2.721 s | 0 |

**Observation.** `clear()` already keeps capacity, so reallocations are rare: about 15 per seam
across 958 rows. Variant C proves the `2y + 3` bound always holds, but it changes nothing,
because the time is spent elsewhere.

## 8. Profiling the native core: redundant work

**Tooling.** A built-in `SIGPROF` sampler (1 kHz, about 13.5k samples), resolved to source lines
with `addr2line`, plus phase timers and work counters in a scratch copy.

**Observations** (600 seams, 2.71 s):

| Phase | Share |
|---|---|
| Incremental cost updates | **92.4%** |
| Walking down to the bottom-left node every seam | 3.3% |
| Unlinking the seam and queueing its neighbours | 3.2% |
| First seam's full cost propagation | 1.0% |

- 85% of samples are in `update_cost_if_needed`, at about 12 ns per call.
- **Per seam: 344k evaluations, but only 117k distinct nodes and 115k changed costs.** Each
  queued node evaluates its three children, and neighbouring nodes share children. Rows are
  processed top to bottom, so a child's parents are already final when it is first evaluated.
  The 228k repeated evaluations per seam **never changed a cost**.

## 9. Optimization (applied to C++): skip already-evaluated children

**Change.** The existing `queued_epoch` field, which only deduplicated the queue, was repurposed
as `evaluated_epoch`. A child already evaluated in the current seam is skipped, and queue dedup
was dropped: a node queued twice is cheap on its second visit. `MeshNode` stays 64 bytes, one
cache line; fields take 63 bytes and the rest is alignment padding.

| Variant, 600 seams | Time | Gain |
|---|---|---|
| Current: queue dedup via `queued_epoch` | 2.731 s | ×1.00 |
| `queued_epoch` removed, no dedup | 2.784 s | ×0.98 |
| "Evaluated" marker in a separate array | 1.611 s | ×1.70 |
| **`queued_epoch` reused as the "evaluated" marker** | **1.521 s** | **×1.80** |

Keeping the marker in the node beats a separate array, because the node's cache line is
already loaded. **Applied gain:** all three native flavors went from 2.73–2.79 s to 1.55–1.57 s
for 600 seams.

## 10. Optimization 3 (applied to Cython and Python): list queues with the evaluated marker

**Change.** The `dict[int, set]` work queue became one list per row with an `_epoch` counter, the
same algorithm as the C++ core. Nodes gained `evaluated_epoch`. A helper, `_evaluate_child`,
skips children already evaluated in the current seam; it avoids building a tuple per call in
Cython.

**Gain** (600 seams total):

| Module | Before | After | Gain |
|---|---|---|---|
| `CySeamCarver` | 27.35 s ³ | **3.55 s** | ×7.7 |
| `PySeamCarver` | 136.5 s ³ | **52.8 s** | ×2.6 |

³ Measured before optimization 2, so the gain covers optimizations 2 and 3 together.

Shrinking alone, Cython is now about 2× behind the native modules (3.04 s vs 1.46 s), and pure
Python about 35× behind.

---

## Lessons

1. **Algorithmic redundancy beats micro-optimization.** Skipping repeated evaluations gave ×1.8
   in C++. Reserving capacity, which removed every reallocation, gave nothing.
2. **The binding layer is irrelevant when calls are coarse-grained.** Cython, the C API and
   ctypes all run the same C++ core at the same speed.
3. **Build flags can impersonate binding differences.** `-fvisibility=hidden` alone was worth
   ×1.35 in `-fPIC` extension modules.
4. **Cython speedups come from C-level types, not from compilation.** `cdef` methods gave
   ×3–×11. Compiling unchanged Python made it ~3× *slower* than CPython 3.12.
5. **Type annotations do nothing in CPython.** The bytecode is identical, so the timings are
   too.
6. **Abstractions cost differently per flavor.** A small helper function is free in Cython
   (`cdef inline`) and C++ (inlined), but costs ~10% in pure Python.
7. **Watch for hidden allocations in Cython.** A temporary `list`, a `dict` of `set`s, or a loop
   over a tuple literal each allocate Python objects in the hot path.
8. **Profiling needs care on this stack.** Python 3.12's `cProfile` cannot see Cython code, and
   `perf` may be disabled. A small `sys.setprofile` hook or `SIGPROF` sampler fills the gap.

## Measured but not applied

| Idea | Expected gain | Note |
|---|---|---|
| Cython optimization 4: read and write pixels as raw bytes | Construction 0.35 → 0.18 s, export 0.28 → 0.01 s | Now the largest part of `CySeamCarver`'s 50-seam time |
| `@cython.no_gc` on `CyMeshNode` | Construction about 0.18 → 0.07 s | Nodes reference each other in cycles, so the carver must break the links on deallocation or leak ~78 MB |
| Track the bottom-left node directly in C++ | About 3% | Avoids walking 958 rows every seam |
| `@cython.final` and bounds-check directives | None | `final` also blocks subclassing |
| Remove the unused `MeshNode::x` in C++ | No speed change | Tidier code; the node stays 64 bytes |
