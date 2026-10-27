// Seam carver on the GPU with WebGPU: every seam is a full recompute of the path costs, run as
// compute passes. The picture, the costs and the seams stay on the GPU; only the final picture is
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
// triangles, as in ../parallel_seam_carver/tiled_seam_carver.cpp:
//   phase A  one workgroup per base segment of BASE columns fills the downward triangle below it,
//            narrowing by one column on each side per row (not at the picture's borders)
//   phase B  one workgroup per boundary between segments fills the upward triangle in between,
//            widening by one column on each side per row, from both neighbours' phase A results
// Within a workgroup each invocation owns a column and a barrier separates the rows.
'use strict';

const WEBGPU_SEAM_CARVER_WGSL = /* wgsl */ `
const BASE: u32 = 256u;  // columns per base segment, one invocation each
const HALF: u32 = 128u;  // rows per strip: upward triangles reach HALF columns to each side

struct Params {
  stride: u32,  // the original width: rows keep their place, the picture shrinks within them
  height: u32,
  r0: u32,      // the strip's first row, already final
  rows: u32,    // rows computed below r0 in this strip
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> width: u32;          // current width
@group(0) @binding(2) var<storage, read_write> pixels: array<u32>;  // 0x00BBGGRR
@group(0) @binding(3) var<storage, read_write> cost: array<f32>;
@group(0) @binding(4) var<storage, read_write> prev: array<i32>;    // parent's column offset
@group(0) @binding(5) var<storage, read> distance_table: array<f32>;
@group(0) @binding(6) var<storage, read_write> seam: array<u32>;    // seam column per row

var<workgroup> shared_width: u32;
var<workgroup> shared_column: u32;
var<workgroup> best_cost: array<f32, BASE>;
var<workgroup> best_column: array<u32, BASE>;

// The width as a value WGSL knows is the same for the whole workgroup, so barriers may depend on it
fn uniform_width(local: u32) -> u32 {
  if (local == 0u) {
    shared_width = width;
  }
  return workgroupUniformLoad(&shared_width);
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

// The top row costs nothing (the core's -1 counts as 0 for the children)
@compute @workgroup_size(BASE)
fn top_row(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < width) {
    cost[id.x] = 0.0;
  }
}

@compute @workgroup_size(BASE)
fn downward_triangles(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let w = uniform_width(local);
  let segments = (w + BASE - 1u) / BASE;
  let i = group.x;
  if (i >= segments) {
    return;
  }
  let a = i * BASE;
  let x = a + local;
  for (var k = 1u; k <= params.rows; k++) {
    // Narrower by one column per row on each side, except at the picture's borders
    let lo = select(a + k, 0u, i == 0u);
    let hi = select(a + BASE - k, w, i == segments - 1u);
    if (x >= lo && x < hi) {
      relax(x, params.r0 + k, w);
    }
    storageBarrier();
  }
}

@compute @workgroup_size(BASE)
fn upward_triangles(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
  let w = uniform_width(local);
  let segments = (w + BASE - 1u) / BASE;
  let boundary = (group.x + 1u) * BASE;
  if (group.x + 1u >= segments) {
    return;
  }
  let x = boundary - HALF + local;
  for (var k = 1u; k <= params.rows; k++) {
    // Columns [boundary - k, boundary + k): what both neighbours' downward triangles left out
    if (x + k >= boundary && x < boundary + k && x < w) {
      relax(x, params.r0 + k, w);
    }
    storageBarrier();
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

// One workgroup per row shifts the pixels right of the seam one column left, a chunk at a time:
// all of a chunk's invocations read before any of them writes
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
    if (moves) {
      pixel = pixels[row + x + 1u];
    }
    storageBarrier();
    if (moves) {
      pixels[row + x] = pixel;
    }
    storageBarrier();
  }
}

@compute @workgroup_size(1)
fn narrow() {
  width -= 1u;
}
`;

class WebGPUSeamCarver {
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
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = {};
    for (const entryPoint of ['top_row', 'downward_triangles', 'upward_triangles', 'trace_seam', 'remove_seam', 'narrow']) {
      pipelines[entryPoint] = await device.createComputePipelineAsync({
        layout: pipelineLayout, compute: { module, entryPoint },
      });
    }
    // Distances between two pixels: the f32 square root of every possible integer dr² + dg² + db²
    const table = new Float32Array(3 * 255 * 255 + 1);
    for (let i = 0; i < table.length; i++) {
      table[i] = Math.sqrt(i);
    }
    const distanceTable = device.createBuffer({ size: table.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(distanceTable, 0, table);
    Object.assign(WebGPUSeamCarver, { device, layout, pipelines, distanceTable });
    return device;
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

    const packed = new Uint32Array(width * height);
    for (let i = 0, j = 0; i < packed.length; i++, j += 3) {
      packed[i] = rgb[j] | (rgb[j + 1] << 8) | (rgb[j + 2] << 16);
    }
    const buffer = (size, usage) => device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | usage });
    this.pixels = buffer(cellBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    device.queue.writeBuffer(this.pixels, 0, packed);
    this.cost = buffer(cellBytes, 0);
    this.prev = buffer(cellBytes, 0);
    this.seam = buffer(height * 4, 0);
    this.widthBuffer = buffer(4, GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(this.widthBuffer, 0, new Uint32Array([width]));

    // Parameters per strip, at the offsets dynamic uniform bindings require. Entry 0 also serves
    // the passes outside the strips, which only read stride and height.
    const align = device.limits.minUniformBufferOffsetAlignment;
    const half = 128;
    this.strips = Math.ceil((height - 1) / half);
    const params = new Uint32Array((Math.max(this.strips, 1) * align) / 4);
    for (let s = 0; s < Math.max(this.strips, 1); s++) {
      const r0 = s * half;
      params.set([width, height, r0, Math.max(0, Math.min(half, height - 1 - r0))], (s * align) / 4);
    }
    this.params = device.createBuffer({ size: params.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, params);
    this.align = align;

    this.bindGroup = device.createBindGroup({
      layout: WebGPUSeamCarver.layout,
      entries: [
        { binding: 0, resource: { buffer: this.params, size: 16 } },
        ...[this.widthBuffer, this.pixels, this.cost, this.prev, WebGPUSeamCarver.distanceTable, this.seam]
          .map((b, i) => ({ binding: i + 1, resource: { buffer: b } })),
      ],
    });
    // Dispatch sizes cover the original width; workgroups beyond the current width do nothing
    this.segments = Math.ceil(width / 256);
  }

  get width() { return this._width; }
  get height() { return this._height; }

  // Removes pixelsToShrink seams in one submission; resolves once the GPU is done
  shrink_horizontally(pixelsToShrink) {
    if (!Number.isInteger(pixelsToShrink) || pixelsToShrink < 0 || pixelsToShrink >= this._width) {
      return Promise.reject(new RangeError('pixels_to_shrink must be in [0, width)'));
    }
    const { device, pipelines } = WebGPUSeamCarver;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const run = (entryPoint, groups, strip = 0) => {
      if (groups > 0) {
        pass.setPipeline(pipelines[entryPoint]);
        pass.setBindGroup(0, this.bindGroup, [strip * this.align]);
        pass.dispatchWorkgroups(groups);
      }
    };
    for (let n = 0; n < pixelsToShrink; n++) {
      run('top_row', this.segments);
      for (let s = 0; s < this.strips; s++) {
        run('downward_triangles', this.segments, s);
        run('upward_triangles', this.segments - 1, s);
      }
      run('trace_seam', 1);
      run('remove_seam', this._height);
      run('narrow', 1);
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
    for (const b of [this.pixels, this.cost, this.prev, this.seam, this.widthBuffer, this.params]) {
      b?.destroy();
    }
    this.pixels = this.cost = this.prev = this.seam = this.widthBuffer = this.params = null;
  }
}

if (typeof module === 'object' && module.exports) {
  module.exports = { WebGPUSeamCarver, WEBGPU_SEAM_CARVER_WGSL };
}
