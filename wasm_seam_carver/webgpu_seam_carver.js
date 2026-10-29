// Seam carver on the GPU with WebGPU: the path costs are computed in compute passes, in full for
// the first seam and, with incremental updates, only where the last removed seam changed them for
// the others. The picture, the costs and the seams stay on the GPU; only the final picture is
// read back. Same interface as the other engines (seam_carver.js, seam_carver_wasm.cpp), except
// that shrink_horizontally() and shrunk_rgb() return promises, since the GPU works asynchronously.
//
// The same rules as the core's full recompute, SeamCarver::propagate_low_cost_paths(): parents
// in the order up, up-left, up-right, a later one wins only if strictly cheaper, then the leftmost
// cheapest bottom pixel and its parents back to the top. One difference: WGSL has no f64, so costs
// are f32. The colour distance comes from a table of f32 square roots indexed by the integer
// dr² + dg² + db², and f32 additions are correctly rounded, so the result is exact for f32
// arithmetic: a CPU running the same steps in f32 gets the same seams. Compared with the f64
// engines, rounding and extra ties make the seams differ sooner or later.
//
// Rows depend on the row above, so the costs are computed in strips of HALF rows tiled by
// diamonds, as in ../parallel_seam_carver/tiled_seam_carver.hpp: one dispatch per wave, one
// workgroup per diamond. A diamond is the upward triangle of strip wave - 1, widening by one
// column on each side per row, above the downward triangle of strip wave, narrowing again; the
// lower half reads only the upper half's last row, so the workgroup goes straight on. Each wave's
// diamonds sit between the previous wave's, offset by HALF columns. Within a workgroup each
// invocation owns a column and a barrier separates the rows. strips + 1 dispatches per seam,
// where separate triangles took two per strip.
//
// Incremental updates work as in ../parallel_seam_carver/tiled_seam_carver.hpp: removing a seam
// shifts the costs and parents with the pixels, so only cells next to the removed seam ("seeds")
// and below cells whose cost changed can change. An incremental tiling (BASE = the tile width)
// launches every tile; each workgroup first tests whether its tile holds a seed or reads a cell
// changed in this seam's epoch, and returns at once if not. Recomputed cells whose cost
// changed get the epoch. The same arithmetic in the same order gives the same f32 results, so the
// seams are exactly those of full recomputes.
//
// Unlike threads, a GPU gains little from narrow tiles: it computes all tiles of a wave at once,
// and every seam still takes all rows in sequence, one barrier per row, however the rows are
// tiled. Narrow tiles only add dispatches (tiles 32 wide: 64 per seam instead of 12). So the
// default tile is as wide as the full recompute's, 256, and clean tiles just return early, which
// saves GPU work without adding latency. ?tile=N on the page tries other widths.
'use strict';

const WEBGPU_SEAM_CARVER_WGSL = /* wgsl */ `
override BASE: u32 = 256u;      // a diamond's width and the distance between centres (tile width)
override HALF: u32 = BASE / 2u;  // rows per strip, and per half of a diamond

struct Params {
  stride: u32,  // the original width: rows keep their place, the picture shrinks within them
  height: u32,
  wave: u32,    // upper halves in strip wave - 1, lower halves in strip wave
  unused: u32,
}

struct State {
  width: u32,  // current width
  epoch: u32,  // advanced once per seam
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> state: State;
@group(0) @binding(2) var<storage, read_write> pixels: array<u32>;  // 0x00BBGGRR
@group(0) @binding(3) var<storage, read_write> cost: array<f32>;
@group(0) @binding(4) var<storage, read_write> prev: array<i32>;    // parent's column offset
@group(0) @binding(5) var<storage, read> distance_table: array<f32>;
@group(0) @binding(6) var<storage, read_write> seam: array<u32>;    // seam column per row
@group(0) @binding(7) var<storage, read_write> changed: array<u32>; // epoch of the cost's last change

var<workgroup> shared_width: u32;
var<workgroup> shared_column: u32;
var<workgroup> best_cost: array<f32, BASE>;
var<workgroup> best_column: array<u32, BASE>;
var<workgroup> dirty: array<atomic<u32>, 2>;  // per half; zero at the start of every workgroup
var<workgroup> shared_dirty: u32;

// The width as a value WGSL knows is the same for the whole workgroup, so barriers may depend on it
fn uniform_width(local: u32) -> u32 {
  if (local == 0u) {
    shared_width = state.width;
  }
  return workgroupUniformLoad(&shared_width);
}

// Whether any invocation found half 0 (upper) or 1 (lower) dirty, as a value the whole
// workgroup agrees on
fn uniform_dirty(local: u32, half: u32, found: bool) -> bool {
  if (found) {
    atomicStore(&dirty[half], 1u);
  }
  workgroupBarrier();
  if (local == 0u) {
    shared_dirty = atomicLoad(&dirty[half]);
  }
  return workgroupUniformLoad(&shared_dirty) != 0u;
}

// The wave's diamonds: their centres, offset(t) + group * BASE, alternate between multiples of
// BASE and the columns halfway between; the strips of their halves, rows r0 + 1 .. r0 + rows
fn centre(group: u32) -> u32 {
  return (params.wave + 1u) % 2u * HALF + group * BASE;
}

fn upper_r0() -> u32 {
  return (params.wave - 1u) * HALF;
}

fn upper_rows() -> u32 {
  return select(min(HALF, params.height - 1u - upper_r0()), 0u, params.wave == 0u);
}

fn lower_r0() -> u32 {
  return params.wave * HALF;
}

fn lower_rows() -> u32 {
  return select(0u, min(HALF, params.height - 1u - lower_r0()), lower_r0() + 1u < params.height);
}

// Cells of a half in its row r0 + k: the upper widens by a column on each side per row, the lower
// narrows; clipped to the picture. x and c may lie left of it, so signed.
fn in_upper(x: i32, c: i32, k: u32, w: u32) -> bool {
  return x >= max(0, c - i32(k)) && x < min(i32(w), c + i32(k));
}

fn in_lower(x: i32, c: i32, k: u32, w: u32) -> bool {
  return x >= max(0, c - i32(HALF) + i32(k)) && x < min(i32(w), c + i32(HALF) - i32(k));
}

fn distance(a: u32, b: u32) -> f32 {
  let dr = i32(a & 0xffu) - i32(b & 0xffu);
  let dg = i32((a >> 8u) & 0xffu) - i32((b >> 8u) & 0xffu);
  let db = i32((a >> 16u) & 0xffu) - i32((b >> 16u) & 0xffu);
  return distance_table[u32(dr * dr + dg * dg + db * db)];
}

// Cheapest path cost of pixel (x, y) from its three parents in row y - 1
fn relax(x: u32, y: u32, w: u32) {
  let row = y * params.stride;
  let up = row - params.stride;
  let pixel = pixels[row + x];
  var best = cost[up + x] + distance(pixels[up + x], pixel);
  var parent = 0;
  if (x > 0u) {
    let left = cost[up + x - 1u] + distance(pixels[up + x - 1u], pixel);
    if (left < best) {
      best = left;
      parent = -1;
    }
  }
  if (x + 1u < w) {
    let right = cost[up + x + 1u] + distance(pixels[up + x + 1u], pixel);
    if (right < best) {
      best = right;
      parent = 1;
    }
  }
  cost[row + x] = best;
  prev[row + x] = parent;
}

// relax() that also marks the cell with the epoch if its cost changed
fn relax_tracked(x: u32, y: u32, w: u32, epoch: u32) {
  let i = y * params.stride + x;
  let before = cost[i];
  relax(x, y, w);
  if (cost[i] != before) {
    changed[i] = epoch;
  }
}

// Whether cells [lo, hi) of row y >= 1 hold a seed: a cell whose parents moved, around the last
// removed seam's columns in rows y - 1 and y (in its columns before the removal)
fn seeded(y: u32, lo: u32, hi: u32) -> bool {
  let a = i32(seam[y]);
  let b = i32(seam[y - 1u]);
  return lo < hi && min(a, b) - 2 < i32(hi) && max(a, b) + 1 >= i32(lo);
}

fn changed_at(y: u32, x: i32, w: u32) -> bool {
  return x >= 0 && x < i32(w) && changed[y * params.stride + u32(x)] == state.epoch;
}

// A full recompute's wave. Wave 0 also zeroes the top row's costs under its diamonds (the core's
// -1 counts as 0 for the children).
@compute @workgroup_size(BASE)
fn diamonds(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let w = uniform_width(local);
  let c = centre(group.x);
  if (c >= w + HALF) {
    return;
  }
  let x = i32(c) - i32(HALF) + i32(local);
  if (params.wave == 0u) {
    if (x >= 0 && x < i32(w)) {
      cost[u32(x)] = 0.0;
    }
    storageBarrier();
  }
  let r0 = upper_r0();
  for (var k = 1u; k <= upper_rows(); k++) {
    if (in_upper(x, i32(c), k, w)) {
      relax(u32(x), r0 + k, w);
    }
    storageBarrier();
  }
  let r1 = lower_r0();
  for (var k = 1u; k <= lower_rows(); k++) {
    if (in_lower(x, i32(c), k, w)) {
      relax(u32(x), r1 + k, w);
    }
    storageBarrier();
  }
}

// An incremental update's wave: each half is tested apart and returns at once if clean. The
// upper half reads row r0 under its tip and, at its sides, two columns either side of its
// previous row, from the previous wave's lower halves; the lower half reads only the upper half's
// last row. Each test spreads its cells over the invocations, one row each for the seeds.
@compute @workgroup_size(BASE)
fn dirty_diamonds(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let w = uniform_width(local);
  let c = centre(group.x);
  if (c >= w + HALF) {
    return;
  }
  let ci = i32(c);
  let x = ci - i32(HALF) + i32(local);
  let epoch = state.epoch;
  let k = local + 1u;
  if (params.wave > 0u) {
    let r0 = upper_r0();
    let rows = upper_rows();
    var found = false;
    for (var d = local; d < 4u; d += BASE) {  // under the tip: columns c - 2 .. c + 1
      found = found || changed_at(r0, ci - 2 + i32(d), w);
    }
    if (k <= rows) {
      found = found || seeded(r0 + k, u32(max(0, ci - i32(k))), u32(min(i32(w), ci + i32(k))));
      if (k > 1u) {
        let y = r0 + k - 1u;
        let d = i32(k);
        found = found || changed_at(y, ci - d - 1, w) || changed_at(y, ci - d, w) ||
                changed_at(y, ci + d - 1, w) || changed_at(y, ci + d, w);
      }
    }
    if (uniform_dirty(local, 0u, found)) {
      for (var j = 1u; j <= rows; j++) {
        if (in_upper(x, ci, j, w)) {
          relax_tracked(u32(x), r0 + j, w, epoch);
        }
        storageBarrier();
      }
    }
  }
  let rows = lower_rows();
  if (rows > 0u) {
    let r0 = lower_r0();
    // Row r0 under the first row: the upper half's last row, one column per invocation
    var found = changed_at(r0, x, w);
    if (k <= rows) {
      found = found || seeded(r0 + k, u32(max(0, ci - i32(HALF) + i32(k))), u32(min(i32(w), ci + i32(HALF) - i32(k))));
    }
    if (uniform_dirty(local, 1u, found)) {
      for (var j = 1u; j <= rows; j++) {
        if (in_lower(x, ci, j, w)) {
          relax_tracked(u32(x), r0 + j, w, epoch);
        }
        storageBarrier();
      }
    }
  }
}

// Leftmost cheapest pixel of the bottom row, then the parents back to the top
@compute @workgroup_size(BASE)
fn trace_seam(@builtin(local_invocation_index) local: u32) {
  let w = uniform_width(local);
  let bottom = (params.height - 1u) * params.stride;
  var c = 0.0;
  var column = 0xffffffffu;
  for (var x = local; x < w; x += BASE) {
    let v = cost[bottom + x];
    if (column == 0xffffffffu || v < c) {
      c = v;
      column = x;
    }
  }
  best_cost[local] = c;
  best_column[local] = column;
  for (var span = BASE / 2u; span > 0u; span /= 2u) {
    workgroupBarrier();
    if (local < span) {
      let other_cost = best_cost[local + span];
      let other_column = best_column[local + span];
      let mine = best_column[local];
      if (other_column != 0xffffffffu &&
          (mine == 0xffffffffu || other_cost < best_cost[local] ||
           (other_cost == best_cost[local] && other_column < mine))) {
        best_cost[local] = other_cost;
        best_column[local] = other_column;
      }
    }
  }
  if (local == 0u) {
    var x = best_column[0];
    for (var y = params.height - 1u; ; y--) {
      seam[y] = x;
      if (y == 0u) {
        break;
      }
      x = u32(i32(x) + prev[y * params.stride + x]);
    }
  }
}

// One workgroup per row shifts the pixels right of the seam one column left, with their costs and
// parents for incremental updates, a chunk at a time: all of a chunk's invocations read before any
// of them writes
@compute @workgroup_size(BASE)
fn remove_seam(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let w = uniform_width(local);
  if (local == 0u) {
    shared_column = seam[group.x];
  }
  let column = workgroupUniformLoad(&shared_column);
  let row = group.x * params.stride;
  for (var start = column - column % BASE; start + 1u < w; start += BASE) {
    let x = start + local;
    let moves = x >= column && x + 1u < w;
    var pixel = 0u;
    var c = 0.0;
    var parent = 0;
    if (moves) {
      pixel = pixels[row + x + 1u];
      c = cost[row + x + 1u];
      parent = prev[row + x + 1u];
    }
    storageBarrier();
    if (moves) {
      pixels[row + x] = pixel;
      cost[row + x] = c;
      prev[row + x] = parent;
    }
    storageBarrier();
  }
}

@compute @workgroup_size(1)
fn narrow() {
  state.width -= 1u;
  state.epoch += 1u;
}
`;

class WebGPUSeamCarver {
  static tile = 256;  // tile width of incremental updates for carvers created from now on

  // Resolves to a GPUDevice, or rejects if this browser or machine has no WebGPU. Needed once
  // before the first constructor call.
  static async init() {
    if (WebGPUSeamCarver.device) {
      return WebGPUSeamCarver.device;
    }
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      throw new Error('this browser has no WebGPU');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('no WebGPU adapter available');
    }
    // The largest buffers hold one 32-bit value per pixel: ask for what the adapter can do
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message);
      WebGPUSeamCarver.device = null;
    });
    const module = device.createShaderModule({ code: WEBGPU_SEAM_CARVER_WGSL });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) {
      throw new Error(`WGSL: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
    }
    const storage = (type) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        { binding: 1, ...storage('storage') },
        { binding: 2, ...storage('storage') },
        { binding: 3, ...storage('storage') },
        { binding: 4, ...storage('storage') },
        { binding: 5, ...storage('read-only-storage') },
        { binding: 6, ...storage('storage') },
        { binding: 7, ...storage('storage') },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = {};
    for (const entryPoint of ['diamonds', 'trace_seam', 'remove_seam', 'narrow']) {
      pipelines[entryPoint] = await device.createComputePipelineAsync({
        layout: pipelineLayout, compute: { module, entryPoint, constants: { BASE: 256 } },
      });
    }
    // Distances between two pixels: the f32 square root of every possible integer dr² + dg² + db²
    const table = new Float32Array(3 * 255 * 255 + 1);
    for (let i = 0; i < table.length; i++) {
      table[i] = Math.sqrt(i);
    }
    const distanceTable = device.createBuffer({ size: table.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(distanceTable, 0, table);
    Object.assign(WebGPUSeamCarver, { device, module, pipelineLayout, layout, pipelines, distanceTable, tilePipelines: new Map() });
    WebGPUSeamCarver.incrementalPipelines(WebGPUSeamCarver.tile);
    return device;
  }

  // The incremental update's pipeline for a tile width, compiled on first use
  static incrementalPipelines(tile) {
    const { device, module, pipelineLayout, tilePipelines } = WebGPUSeamCarver;
    if (!Number.isInteger(tile) || tile < 2 || tile % 2 || tile > device.limits.maxComputeInvocationsPerWorkgroup) {
      throw new RangeError(`the tile width must be even, from 2 to ${device.limits.maxComputeInvocationsPerWorkgroup}`);
    }
    if (!tilePipelines.has(tile)) {
      tilePipelines.set(tile, device.createComputePipeline({
        layout: pipelineLayout, compute: { module, entryPoint: 'dirty_diamonds', constants: { BASE: tile } },
      }));
    }
    return tilePipelines.get(tile);
  }

  // rgb is a row-major Uint8Array of width * height * 3 bytes
  constructor(rgb, width, height) {
    const device = WebGPUSeamCarver.device;
    if (!device) {
      throw new Error('WebGPUSeamCarver.init() must succeed first');
    }
    if (!rgb || !Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
      throw new RangeError('image must be non-empty');
    }
    if (rgb.length !== width * height * 3) {
      throw new RangeError('rgb must hold width * height * 3 bytes');
    }
    const cellBytes = width * height * 4;
    if (cellBytes > device.limits.maxStorageBufferBindingSize || height > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new RangeError(`${width} × ${height} pixels are too many for this GPU`);
    }
    this._width = width;
    this._height = height;
    this.stride = width;  // rows keep their place in the buffers while the picture narrows
    this.tile = WebGPUSeamCarver.tile;
    this.tilePipeline = WebGPUSeamCarver.incrementalPipelines(this.tile);
    this.hasRemoved = false;  // costs fit the picture after the last removed seam (WebGPU zeroes new buffers)

    const packed = new Uint32Array(width * height);
    for (let i = 0, j = 0; i < packed.length; i++, j += 3) {
      packed[i] = rgb[j] | (rgb[j + 1] << 8) | (rgb[j + 2] << 16);
    }
    const buffer = (size, usage) => device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | usage });
    this.pixels = buffer(cellBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    device.queue.writeBuffer(this.pixels, 0, packed);
    this.cost = buffer(cellBytes, 0);
    this.prev = buffer(cellBytes, 0);
    this.changed = buffer(cellBytes, 0);
    this.seam = buffer(height * 4, 0);
    this.state = buffer(8, GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(this.state, 0, new Uint32Array([width, 0]));

    // Parameters per wave, at the offsets dynamic uniform bindings require, for the full
    // recompute's diamonds 256 wide and the incremental update's tile wide. Entry 0 also serves
    // the passes outside the waves, which only read stride and height.
    this.align = device.limits.minUniformBufferOffsetAlignment;
    const tiling = (half) => {
      const waves = Math.ceil((height - 1) / half) + 1;
      const params = new Uint32Array((waves * this.align) / 4);
      for (let t = 0; t < waves; t++) {
        params.set([width, height, t, 0], (t * this.align) / 4);
      }
      const uniforms = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uniforms, 0, params);
      const bindGroup = device.createBindGroup({
        layout: WebGPUSeamCarver.layout,
        entries: [
          { binding: 0, resource: { buffer: uniforms, size: 16 } },
          ...[this.state, this.pixels, this.cost, this.prev, WebGPUSeamCarver.distanceTable, this.seam, this.changed]
            .map((b, i) => ({ binding: i + 1, resource: { buffer: b } })),
        ],
      });
      // Dispatch sizes cover the original width; workgroups beyond the current width do nothing
      return { waves, uniforms, bindGroup, diamonds: Math.floor((width + half - 1) / (2 * half)) + 1 };
    };
    this.full = tiling(128);
    this.incremental = tiling(this.tile / 2);
  }

  get width() { return this._width; }
  get height() { return this._height; }

  // Removes pixelsToShrink seams in one submission; resolves once the GPU is done. With
  // incremental updates, every seam after the carver's first only recomputes dirty tiles.
  shrink_horizontally(pixelsToShrink, incrementalCostUpdates = false) {
    if (!Number.isInteger(pixelsToShrink) || pixelsToShrink < 0 || pixelsToShrink >= this._width) {
      return Promise.reject(new RangeError('pixels_to_shrink must be in [0, width)'));
    }
    const { device, pipelines } = WebGPUSeamCarver;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const run = (pipeline, tiling, groups, wave = 0) => {
      if (groups > 0) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, tiling.bindGroup, [wave * this.align]);
        pass.dispatchWorkgroups(groups);
      }
    };
    const { full, incremental } = this;
    for (let n = 0; n < pixelsToShrink; n++) {
      const [tiling, pipeline] = incrementalCostUpdates && this.hasRemoved ?
        [incremental, this.tilePipeline] : [full, pipelines.diamonds];
      for (let t = 0; t < tiling.waves; t++) {
        run(pipeline, tiling, tiling.diamonds, t);
      }
      run(pipelines.trace_seam, full, 1);
      run(pipelines.remove_seam, full, this._height);
      run(pipelines.narrow, full, 1);
      this.hasRemoved = incrementalCostUpdates;
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
    this._width -= pixelsToShrink;
    return device.queue.onSubmittedWorkDone();
  }

  // Resolves to the shrunk picture as row-major RGB bytes
  async shrunk_rgb() {
    const { device } = WebGPUSeamCarver;
    const stride = this.stride;
    const readback = device.createBuffer({ size: this.pixels.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.pixels, 0, readback, 0, this.pixels.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const packed = new Uint32Array(readback.getMappedRange());
    const rgb = new Uint8Array(this._width * this._height * 3);
    let j = 0;
    for (let y = 0; y < this._height; y++) {
      for (let x = 0, i = y * stride; x < this._width; x++, i++) {
        const p = packed[i];
        rgb[j++] = p & 0xff;
        rgb[j++] = (p >> 8) & 0xff;
        rgb[j++] = (p >> 16) & 0xff;
      }
    }
    readback.destroy();
    return rgb;
  }

  delete() {
    for (const b of [this.pixels, this.cost, this.prev, this.changed, this.seam, this.state,
                     this.full?.uniforms, this.incremental?.uniforms]) {
      b?.destroy();
    }
    this.pixels = this.cost = this.prev = this.changed = this.seam = this.state = this.full = this.incremental = null;
  }
}

if (typeof module === 'object' && module.exports) {
  module.exports = { WebGPUSeamCarver, WEBGPU_SEAM_CARVER_WGSL };
}
