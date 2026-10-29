// Runs seam_carver.wasm, the module wasmtime_seam_carver.py loads, in a JavaScript engine: the
// same C API calls and the same timing as run_wasmtime.py, so only the runtime differs. Used by
// run_node.mjs and, inlined, by the page make_page.py writes.
'use strict';

// FNV-1a over the shrunk picture: equal hashes mean every runtime removed the same seams
function fnv1a(bytes) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    hash = Math.imul(hash ^ bytes[i], 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

// The imports Emscripten's standalone runtime asks for: WASI output (messages before an abort)
// and a memory growth notice
function imports(state) {
  return {
    wasi_snapshot_preview1: {
      fd_write(fd, iovs, count, written) {
        const view = new DataView(state.memory.buffer);
        let total = 0;
        for (let i = 0; i < count; i++) {
          const address = view.getUint32(iovs + 8 * i, true);
          const length = view.getUint32(iovs + 8 * i + 4, true);
          state.output += new TextDecoder().decode(new Uint8Array(state.memory.buffer, address, length));
          total += length;
        }
        view.setUint32(written, total, true);
        return 0;
      },
      fd_close: () => 0,
      fd_seek: () => 70,  // ESPIPE: no seekable files here
    },
    env: { emscripten_notify_memory_growth() {} },
  };
}

// One run in a fresh instance, as WasmtimeSeamCarver does per carver: construct, then time
// shrink_horizontally alone
async function runOnce(module, rgb, width, height, seams, incremental) {
  const state = { memory: null, output: '' };
  const instance = await WebAssembly.instantiate(module, imports(state));
  const e = instance.exports;
  state.memory = e.memory;
  e._initialize();
  const check = (status) => {
    if (status !== 0) {
      const address = e.sc_last_error();
      const bytes = new Uint8Array(e.memory.buffer, address, 256);
      throw new Error(new TextDecoder().decode(bytes.subarray(0, bytes.indexOf(0))) + state.output);
    }
  };
  const rgbAddress = e.malloc(rgb.length);
  const handleAddress = e.malloc(4);
  new Uint8Array(e.memory.buffer, rgbAddress, rgb.length).set(rgb);
  check(e.sc_create(rgbAddress, rgb.length, width, height, handleAddress));
  const handle = new DataView(e.memory.buffer).getUint32(handleAddress, true);
  e.free(rgbAddress);
  e.free(handleAddress);

  const start = performance.now();
  check(e.sc_shrink_horizontally(handle, seams, incremental ? 1 : 0));
  const ms = performance.now() - start;

  const w = e.sc_width(handle), h = e.sc_height(handle), size = w * h * 3;
  const out = e.malloc(size);
  check(e.sc_copy_shrunk_rgb(handle, out, size));
  const hash = fnv1a(new Uint8Array(e.memory.buffer, out, size));
  return { ms, hash, width: w, height: h };
}

// The workloads: incremental updates as the notebook and the page run them, and full recomputes,
// which spend their time in another loop of the core
const WORKLOADS = [
  { name: '600 seams, incremental', seams: 600, incremental: true },
  { name: '50 seams, full recompute', seams: 50, incremental: false },
];
const REPEATS = 5;

// Median and best of REPEATS fresh runs per workload; report(line) gets progress as it goes
async function compare(wasmBytes, rgb, width, height, report) {
  const module = await WebAssembly.compile(wasmBytes);
  const results = [];
  for (const workload of WORKLOADS) {
    const times = [];
    let hash;
    for (let i = 0; i < REPEATS; i++) {
      const r = await runOnce(module, rgb, width, height, workload.seams, workload.incremental);
      times.push(r.ms);
      hash = r.hash;
      report?.(`  ${workload.name}: run ${i + 1} ${r.ms.toFixed(0)} ms`);
    }
    times.sort((a, b) => a - b);
    results.push({ workload: workload.name, median: times[REPEATS >> 1], best: times[0], hash });
  }
  return results;
}

if (typeof module === 'object' && module.exports) {
  module.exports = { compare, WORKLOADS, REPEATS };
}
