// Weighted Delaunay (regular) triangulation by divide and conquer, ported from
// weighted_divide_and_conquer.py on top of delaunay.js, which must load first. The triangulation of
// points (x, y) with heights h is their lifted points' lower convex hull seen from above; heights z and
// -z of 3D points give its lower and upper hulls, which convexHull3d() glues into the 3D convex hull.
//
// The merge is delaunay.js's zip, with flips testing the lifted plane. An illegal edge that is reflex
// at a corner proves that corner's point redundant, above the lower hull, so the point is removed and
// its hole triangulated again ear by ear. A point on the zip's open edge has no closed hole yet: its
// edge is retried after the zip.

class WeightedTriangulation extends Triangulation {
  constructor(x, y, n, h) {
    super(x, y, n, {h, redundant: new Set(), deferred: [], edgeFrom: new Map(), removals: 0});
  }

  addTriangle(a, b, c) {
    super.addTriangle(a, b, c);
    // Every triangle a flip or a removal takes away is replaced by triangles at the same corners, the
    // removed point's apart, so each remaining vertex keeps an existing edge to start its ring from
    this.edgeFrom.set(a, b);
    this.edgeFrom.set(b, c);
    this.edgeFrom.set(c, a);
  }

  inside(a, b, c, d) {
    // d's lifted point clearly below the plane through those of the counterclockwise triangle a, b,
    // c: the determinant beats its rounding noise, a small fraction of the sum of its terms' sizes
    const x = this.x, y = this.y, h = this.h;
    const ax = x[a] - x[d], ay = y[a] - y[d], ah = h[a] - h[d];
    const bx = x[b] - x[d], by = y[b] - y[d], bh = h[b] - h[d];
    const cx = x[c] - x[d], cy = y[c] - y[d], ch = h[c] - h[d];
    const det = ax * (by * ch - bh * cy) - ay * (bx * ch - bh * cx) + ah * (bx * cy - by * cx);
    const size = Math.abs(ax) * (Math.abs(by * ch) + Math.abs(bh * cy))
      + Math.abs(ay) * (Math.abs(bx * ch) + Math.abs(bh * cx))
      + Math.abs(ah) * (Math.abs(bx * cy) + Math.abs(by * cx));
    return det > 1e-12 * size;
  }

  legalize(edges) {
    const stack = edges;
    while (stack.length) {
      const [a, b] = stack.pop();
      const c = this.getApex(a, b), d = this.getApex(b, a);
      if (c < 0 || d < 0 || !this.inside(a, b, c, d)) continue;
      // Triangles a, b, c and b, a, d: the quadrilateral a, d, b, c is convex at c and d
      const convexA = this.orient(c, a, d) > 0, convexB = this.orient(d, b, c) > 0;
      if (convexA && convexB) {
        this.removeTriangle(a, b, c);
        this.removeTriangle(b, a, d);
        this.addTriangle(a, d, c);
        this.addTriangle(d, b, c);
        if (++this.flips > this.flipLimit) throw new Error('Too many flips');
        stack.push([a, d], [d, b], [b, c], [c, a]);
      } else {
        // The reflex corner is redundant
        const changed = this.removeVertex(convexA ? b : a);
        if (changed.length) stack.push(...changed);
        else this.deferred.push([a, b]);
      }
    }
  }

  removeVertex(v) {
    // Remove redundant point v and triangulate its hole again; returns the edges to check, or none
    // when v's triangles don't close around it yet
    const x = this.x, y = this.y, h = this.h;
    const ring = [this.edgeFrom.get(v)];
    for (;;) {
      const next = this.getApex(v, ring[ring.length - 1]);
      if (next < 0) return [];
      if (next === ring[0]) break;
      ring.push(next);
      if (ring.length > this.n) throw new Error('Ring lost');
    }
    ring.forEach((p, i) => this.removeTriangle(v, p, ring[(i + 1) % ring.length]));

    const heightOverV = (a, b, c) => {
      // The plane through the lifted a, b and c, at v's place
      const ux = x[b] - x[a], uy = y[b] - y[a], uh = h[b] - h[a];
      const wx = x[c] - x[a], wy = y[c] - y[a], wh = h[c] - h[a];
      const nx = uy * wh - uh * wy, ny = uh * wx - ux * wh, nz = ux * wy - uy * wx;
      return h[a] - (nx * (x[v] - x[a]) + ny * (y[v] - y[a])) / nz;
    };
    // The hole is star-shaped around v, so it has ears: corners turning counterclockwise with no other
    // corner of the hole inside. Cut off the one whose plane is lowest over v, which leaves fewer flips
    // to do than the highest first.
    const hole = ring.slice(), edges = [];
    while (hole.length > 3) {
      let best = -1, bestHeight = Infinity;
      for (let i = 0; i < hole.length; i++) {
        const a = hole[(i - 1 + hole.length) % hole.length], b = hole[i], c = hole[(i + 1) % hole.length];
        if (this.orient(a, b, c) <= 0) continue;
        if (hole.some(p => p !== a && p !== b && p !== c
            && this.orient(a, b, p) > 0 && this.orient(b, c, p) > 0 && this.orient(c, a, p) > 0)) continue;
        const height = heightOverV(a, b, c);
        if (height < bestHeight) { bestHeight = height; best = i; }
      }
      if (best < 0) throw new Error('No ear');
      const a = hole[(best - 1 + hole.length) % hole.length], b = hole[best], c = hole[(best + 1) % hole.length];
      this.addTriangle(a, b, c);
      edges.push([a, b], [b, c], [c, a]);
      hole.splice(best, 1);
    }
    this.addTriangle(hole[0], hole[1], hole[2]);
    edges.push([hole[0], hole[1]], [hole[1], hole[2]], [hole[2], hole[0]]);
    this.redundant.add(v);
    this.removals++;
    return edges;
  }

  merge(left, right, mx, my, sx, sy) {
    const hull = super.merge(left, right, mx, my, sx, sy);
    // Only deferred edges can still be illegal; with the zip done, their corners' holes are closed
    while (this.deferred.length) {
      const before = this.flips + this.removals, deferred = this.deferred;
      this.deferred = [];
      this.legalize(deferred);
      if (this.flips + this.removals === before) break;
    }
    this.deferred = [];
    return hull;
  }

  triangles() {
    // Every triangle once, as [a, b, c] counterclockwise, from its smallest corner
    const n = this.n, triangles = [];
    for (const [key, c] of this.apex) {
      const a = Math.floor(key / n), b = key - a * n;
      if (a < b && a < c) triangles.push([a, b, c]);
    }
    return triangles;
  }
}

function convexHull3d(x, y, z, n) {
  // The convex hull of 3D points as faces [a, b, c], counterclockwise seen from outside: the lower hull,
  // turned to face down, and the upper hull, the lower hull of heights -z. They meet along the outline
  // seen from above.
  const lower = new WeightedTriangulation(x, y, n, z);
  const upper = new WeightedTriangulation(x, y, n, z.map(v => -v));
  const faces = [...lower.triangles().map(([a, b, c]) => [a, c, b]), ...upper.triangles()];
  return {faces, lower, upper};
}
