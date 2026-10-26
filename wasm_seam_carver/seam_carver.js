// JavaScript port of seam_carving::SeamCarver (../seam_carver_core/seam_carver_impl.{hpp,cpp}):
// the same algorithm, iteration order and tie-breaking, so it removes exactly the same seams.
// Same interface as the WebAssembly binding in seam_carver_wasm.cpp: a constructor taking
// row-major RGB bytes, shrink_horizontally(), shrunk_rgb(), width, height and delete().
//
// One deliberate difference: the mesh is a struct of arrays. Each MeshNode field is a typed
// array indexed by node, and nodes link to each other by index instead of by pointer, with NONE
// for nullptr. Millions of small JavaScript objects would be far slower and larger.
'use strict';

const NONE = -1;

class SeamCarver {
  // rgb is a row-major Uint8Array of width * height * 3 bytes
  constructor(rgb, width, height) {
    if (!rgb || !Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
      throw new RangeError('image must be non-empty');
    }
    if (rgb.length !== width * height * 3) {
      throw new RangeError('rgb must hold width * height * 3 bytes');
    }
    this._width = width;
    this._height = height;
    const nodeCount = width * height;
    // MeshNode fields, one typed array each (x is never read, so it isn't kept)
    this.up = new Int32Array(nodeCount).fill(NONE);
    this.down = new Int32Array(nodeCount).fill(NONE);
    this.left = new Int32Array(nodeCount).fill(NONE);
    this.right = new Int32Array(nodeCount).fill(NONE);
    this.prev = new Int32Array(nodeCount).fill(NONE);  // Parent on the cheapest path from the top row
    this.cost = new Float64Array(nodeCount).fill(-1.0);
    this.y = new Int32Array(nodeCount);
    this.evaluated_epoch = new Uint32Array(nodeCount);  // Last incremental update that evaluated this node
    this.r = new Uint8Array(nodeCount);
    this.g = new Uint8Array(nodeCount);
    this.b = new Uint8Array(nodeCount);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const node = y * width + x;
        this.y[node] = y;
        this.r[node] = rgb[3 * node];
        this.g[node] = rgb[3 * node + 1];
        this.b[node] = rgb[3 * node + 2];
      }
    }
    this.link_mesh_nodes();
    this.top_left = 0;
    // Nodes whose cost may be stale, per row
    this.update_heads = Array.from({ length: height }, () => []);
    this.has_update_heads = false;
    // Incremented once per removed seam, so evaluated_epoch never needs resetting
    this.epoch = 0;
  }

  get width() { return this._width; }
  get height() { return this._height; }

  // Nothing to free: the mesh is garbage collected. Kept for the WebAssembly binding's interface.
  delete() {}

  link_mesh_nodes() {
    for (let y = 0; y < this._height; y++) {
      for (let x = 0; x < this._width; x++) {
        const node = y * this._width + x;
        if (x > 0) {
          const left = node - 1;
          this.right[left] = node;
          this.left[node] = left;
        }
        if (y > 0) {
          const up = node - this._width;
          this.down[up] = node;
          this.up[node] = up;
        }
      }
    }
  }

  distance(n1, n2) {
    const r = this.r[n1] - this.r[n2];
    const g = this.g[n1] - this.g[n2];
    const b = this.b[n1] - this.b[n2];
    return Math.sqrt(r * r + g * g + b * b);
  }

  update_cost(from_node, to_node) {
    const cost = this.cost;
    let c = cost[from_node] >= 0.0 ? cost[from_node] : 0.0;
    c += this.distance(from_node, to_node);

    if (cost[to_node] < 0.0 || cost[to_node] > c) {
      cost[to_node] = c;
      this.prev[to_node] = from_node;
    }
  }

  propagate_low_cost_paths() {
    for (let node = this.top_left; node !== NONE; node = this.right[node]) {
      this.cost[node] = -1.0;
      this.prev[node] = NONE;
    }
    let layer_top = this.top_left;
    while (this.down[layer_top] !== NONE) {
      for (let n = layer_top; n !== NONE; n = this.right[n]) {
        this.cost[this.down[n]] = -1.0;
        this.prev[this.down[n]] = NONE;
      }

      layer_top = this.down[layer_top];
      for (let n = layer_top; n !== NONE; n = this.right[n]) {
        const up = this.up[n];
        this.update_cost(up, n);
        if (this.left[up] !== NONE) {
          this.update_cost(this.left[up], n);
        }
        if (this.right[up] !== NONE) {
          this.update_cost(this.right[up], n);
        }
      }
    }
    return layer_top;
  }

  update_cost_if_needed(to_node) {
    const cost = this.cost;
    const prev = this.prev;
    const up = this.up[to_node];
    if (prev[to_node] !== NONE) {
      if (prev[to_node] !== up && prev[to_node] !== this.left[up] && prev[to_node] !== this.right[up]) {
        // Previous lowest cost path node was in removed seam
        cost[to_node] = -1.0;
        prev[to_node] = NONE;
      }
    } else {
      cost[to_node] = -1.0;
    }

    // Parents in the order up, up-left, up-right, like the C++ parent_nodes array
    let achievable_cost = -1.0;
    let best_parent = NONE;
    for (let k = 0; k < 3; k++) {
      const parent_node = k === 0 ? up : k === 1 ? this.left[up] : this.right[up];
      if (parent_node === NONE) {
        continue;
      }
      const c = (cost[parent_node] >= 0.0 ? cost[parent_node] : 0.0) + this.distance(parent_node, to_node);
      if (achievable_cost < 0.0 || c < achievable_cost) {
        achievable_cost = c;
        best_parent = parent_node;
      }
    }

    if (!(achievable_cost * 0.9999 <= cost[to_node] && cost[to_node] <= achievable_cost * 1.0001)) {
      cost[to_node] = achievable_cost;
      prev[to_node] = best_parent;
      return true;
    }
    return false;
  }

  // A node may be queued twice (as a seam neighbour and after its cost changed),
  // the second visit is cheap since all its children are already evaluated.
  enqueue_update(node) {
    if (node !== NONE) {
      this.update_heads[this.y[node]].push(node);
    }
  }

  update_cost_incrementally() {
    for (let depth = 0; depth < this._height; depth++) {
      // Only update_heads[depth + 1] grows here, so indexing stays valid
      const layer = this.update_heads[depth];
      for (let i = 0; i < layer.length; i++) {
        const down = this.down[layer[i]];
        if (down === NONE) {
          continue;
        }
        // Children in the order down, down-left, down-right, like the C++ initializer list
        for (let k = 0; k < 3; k++) {
          const child = k === 0 ? down : k === 1 ? this.left[down] : this.right[down];
          // Rows are processed top to bottom, so all parents of child are final
          // by now and evaluating it again could not change its cost.
          if (child === NONE || this.evaluated_epoch[child] === this.epoch) {
            continue;
          }
          this.evaluated_epoch[child] = this.epoch;
          if (this.update_cost_if_needed(child)) {
            this.enqueue_update(child);
          }
        }
      }
    }

    let layer_top = this.top_left;
    while (this.down[layer_top] !== NONE) {
      layer_top = this.down[layer_top];
    }
    return layer_top;
  }

  find_cheapest_seam(bottom_left) {
    let cheapest = bottom_left;
    for (let n = bottom_left; n !== NONE; n = this.right[n]) {
      if (this.cost[n] < this.cost[cheapest]) {
        cheapest = n;
      }
    }
    return cheapest;
  }

  remove_vertical_seam(incremental_cost_updates) {
    const bottom_left = incremental_cost_updates && this.has_update_heads
      ? this.update_cost_incrementally()
      : this.propagate_low_cost_paths();

    for (const layer of this.update_heads) {
      layer.length = 0;
    }
    this.has_update_heads = false;
    // The next incremental update marks evaluated nodes with a new epoch.
    this.epoch++;

    const { up, down, left, right } = this;
    let seam_node = this.find_cheapest_seam(bottom_left);
    let prev_removed_node = NONE;
    while (seam_node !== NONE) {
      const l = left[seam_node];
      const r = right[seam_node];
      if (l !== NONE) {
        right[l] = r;
      }
      if (r !== NONE) {
        left[r] = l;
      }

      if (seam_node === this.top_left) {
        // Special node tracking top left corner
        this.top_left = right[this.top_left];
      }

      if (prev_removed_node !== NONE) {
        if (up[prev_removed_node] === seam_node) {
          // Seam node is removed vertically
          const d = down[seam_node];
          if (left[d] !== NONE) {
            this.prev[left[d]] = NONE;
          }
          if (right[d] !== NONE) {
            this.prev[right[d]] = NONE;
          }
        } else if (up[prev_removed_node] === right[seam_node]) {
          // Left diagonal seam
          up[down[seam_node]] = right[seam_node];
          down[right[seam_node]] = down[seam_node];
        } else {
          // Right diagonal seam
          up[down[seam_node]] = left[seam_node];
          down[left[seam_node]] = down[seam_node];
        }
      }
      prev_removed_node = seam_node;
      seam_node = this.prev[seam_node];

      // Head of incremental low cost propagation
      if (incremental_cost_updates) {
        this.enqueue_update(left[prev_removed_node]);
        this.enqueue_update(right[prev_removed_node]);
        this.has_update_heads = true;
      }
    }
    this._width--;
  }

  shrink_horizontally(pixels_to_shrink, incremental_cost_updates) {
    if (!Number.isInteger(pixels_to_shrink) || pixels_to_shrink < 0 || pixels_to_shrink >= this._width) {
      throw new RangeError('pixels_to_shrink must be in [0, width)');
    }
    for (let i = 0; i < pixels_to_shrink; i++) {
      this.remove_vertical_seam(Boolean(incremental_cost_updates));
    }
  }

  // Row-major RGB bytes of the current (shrunk) image, width * height * 3
  shrunk_rgb() {
    const rgb = new Uint8Array(this._width * this._height * 3);
    let i = 0;
    for (let row = this.top_left; row !== NONE; row = this.down[row]) {
      for (let column = row; column !== NONE; column = this.right[column]) {
        rgb[i++] = this.r[column];
        rgb[i++] = this.g[column];
        rgb[i++] = this.b[column];
      }
    }
    return rgb;
  }
}

if (typeof module === 'object' && module.exports) {
  module.exports = { SeamCarver };  // for Node.js tests
}
