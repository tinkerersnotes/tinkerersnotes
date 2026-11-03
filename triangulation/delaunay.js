// Divide and conquer Delaunay triangulation and its dual, the Voronoi diagram, ported from
// divide_and_conquer.py. Shared by the triangulation pages, which load it with a plain <script src>,
// so they still open straight from disk.
//
// Points are indices into the coordinate arrays x and y. Assumes no three points on a line, which
// moving random points satisfy with probability 1; a degenerate case throws, and a page skips that
// frame.
class Triangulation {
  constructor(x, y, n, fields = {}) {
    this.x = x;
    this.y = y;
    this.n = n;
    // A subclass's own fields, set here because the triangulation is built before its constructor goes on
    Object.assign(this, fields);
    // Directed edge a -> b, as the key a * n + b, -> apex c of the counterclockwise triangle a, b, c.
    // The triangle on the other side of the edge, if any, is under b -> a.
    this.apex = new Map();
    this.flips = 0;
    this.flipLimit = 50 * n + 100;  // Flips always end; this only guards against rounding gone wrong
    this.hull = this.triangulate(Array.from({length: n}, (_, i) => i));
  }

  orient(a, b, c) {
    // Twice the signed area of triangle a, b, c: positive when they turn counterclockwise
    const x = this.x, y = this.y;
    return (x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a]);
  }

  inCircle(a, b, c, d) {
    // Positive when d lies inside the circle through the counterclockwise triangle a, b, c
    const x = this.x, y = this.y;
    const ax = x[a] - x[d], ay = y[a] - y[d], bx = x[b] - x[d], by = y[b] - y[d];
    const cx = x[c] - x[d], cy = y[c] - y[d];
    const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
    return ax * (by * c2 - b2 * cy) - ay * (bx * c2 - b2 * cx) + a2 * (bx * cy - by * cx);
  }

  inside(a, b, c, d) {
    // d clearly inside: the circle test beats its rounding noise, a small fraction of the sum of its
    // terms' sizes, so points on one circle never flip back and forth
    const x = this.x, y = this.y;
    const ax = Math.abs(x[a] - x[d]), ay = Math.abs(y[a] - y[d]), bx = Math.abs(x[b] - x[d]);
    const by = Math.abs(y[b] - y[d]), cx = Math.abs(x[c] - x[d]), cy = Math.abs(y[c] - y[d]);
    const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
    const size = ax * (by * c2 + b2 * cy) + ay * (bx * c2 + b2 * cx) + a2 * (bx * cy + by * cx);
    return this.inCircle(a, b, c, d) > 1e-12 * size;
  }

  insideAngle(o, a, b, p) {
    // p strictly inside the angle at o, less than pi, turning counterclockwise from a to b
    return this.orient(o, a, p) > 0 && this.orient(o, p, b) > 0;
  }

  anglesOverlap(o, a1, a2, b1, b2) {
    // Two angles at o, each less than pi, overlap when one starts inside the other
    return this.insideAngle(o, a1, a2, b1) || this.insideAngle(o, b1, b2, a1);
  }

  getApex(a, b) {
    const c = this.apex.get(a * this.n + b);
    return c === undefined ? -1 : c;
  }

  addTriangle(a, b, c) {
    const n = this.n;
    this.apex.set(a * n + b, c);
    this.apex.set(b * n + c, a);
    this.apex.set(c * n + a, b);
  }

  removeTriangle(a, b, c) {
    const n = this.n;
    this.apex.delete(a * n + b);
    this.apex.delete(b * n + c);
    this.apex.delete(c * n + a);
  }

  legalize(edges) {
    // Flip edges until none of them, nor any edge a flip touches, breaks the empty circle rule
    const stack = edges;
    while (stack.length) {
      const [a, b] = stack.pop();
      const c = this.getApex(a, b), d = this.getApex(b, a);
      if (c < 0 || d < 0 || !this.inside(a, b, c, d)) continue;
      // Triangles a, b, c and b, a, d make the convex quadrilateral a, d, b, c: swap diagonals
      this.removeTriangle(a, b, c);
      this.removeTriangle(b, a, d);
      this.addTriangle(a, d, c);
      this.addTriangle(d, b, c);
      if (++this.flips > this.flipLimit) throw new Error('Too many flips');
      stack.push([a, d], [d, b], [b, c], [c, a]);
    }
  }

  triangulate(ids) {
    // Triangulate the points ids, return their convex hull in counterclockwise order
    const x = this.x, y = this.y, m = ids.length;
    if (m <= 3) {
      if (m === 3) {
        let [a, b, c] = ids;
        if (this.orient(a, b, c) < 0) [b, c] = [c, b];
        this.addTriangle(a, b, c);
        return [a, b, c];
      }
      return ids.slice();  // A point or a segment is its own hull
    }

    // The split line goes through the centre of mass along the direction the points spread the
    // least: the eigenvector of their covariance with the smaller eigenvalue
    let mx = 0, my = 0;
    for (const v of ids) { mx += x[v]; my += y[v]; }
    mx /= m; my /= m;
    let sxx = 0, syy = 0, sxy = 0;
    for (const v of ids) {
      const dx = x[v] - mx, dy = y[v] - my;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    const low = (sxx + syy) / 2 - Math.hypot((sxx - syy) / 2, sxy);
    // Either row of the matrix minus low gives the eigenvector; take the better conditioned one
    let sx = sxy, sy = low - sxx;
    if (Math.hypot(low - syy, sxy) > Math.hypot(sx, sy)) { sx = low - syy; sy = sxy; }
    const length = Math.hypot(sx, sy);
    if (length > 0) { sx /= length; sy /= length; } else { sx = 1; sy = 0; }
    if (sx < 0 || (sx === 0 && sy < 0)) { sx = -sx; sy = -sy; }  // Along +x: left is the upper half

    let left = [], right = [];
    for (const v of ids) {
      const dx = x[v] - mx, dy = y[v] - my;
      const height = sx * dy - sy * dx, along = sx * dx + sy * dy;
      // A point right on the line goes by which way along the line it lies
      (height > 0 || (height === 0 && along <= 0) ? left : right).push(v);
    }
    if (!left.length || !right.length) throw new Error('Points on one line');

    const leftHull = this.triangulate(left);
    const rightHull = this.triangulate(right);
    return this.merge(leftHull, rightHull, mx, my, sx, sy);
  }

  merge(left, right, mx, my, sx, sy) {
    // Join the triangulations inside the counterclockwise convex hulls left and right (on the left
    // and right of the line through mx, my along sx, sy), return the joined convex hull. With the
    // split line as the x axis, left is above. The low bridge lLow -> rLow closes the joined hull
    // at small x, rHigh -> lHigh at large x, and both have every point on their left.
    const neighbours = (hull, step) => {
      const map = new Map();
      hull.forEach((v, i) => map.set(v, hull[(i + step + hull.length) % hull.length]));
      return map;
    };
    const leftNext = neighbours(left, 1), leftPrev = neighbours(left, -1);
    const rightNext = neighbours(right, 1), rightPrev = neighbours(right, -1);
    const height = v => sx * (this.y[v] - my) - sy * (this.x[v] - mx);
    let lStart = left[0], rStart = right[0];
    for (const v of left) if (height(v) < height(lStart)) lStart = v;
    for (const v of right) if (height(v) > height(rStart)) rStart = v;

    // Walk each end outwards from the points nearest the split line until nothing is on the wrong
    // side of the bridge. Neither end steps back, so it's linear; the limit only guards rounding.
    let steps = 0;
    const limit = 2 * (left.length + right.length) + 10;
    const step = () => { if (++steps > limit) throw new Error('Bridge walk lost'); };

    let l = lStart, r = rStart, moved = true;
    while (moved) {
      moved = false;
      while (this.orient(l, r, leftPrev.get(l)) < 0) { l = leftPrev.get(l); moved = true; step(); }
      while (this.orient(l, r, rightNext.get(r)) < 0) { r = rightNext.get(r); moved = true; step(); }
    }
    const lLow = l, rLow = r;

    l = lStart; r = rStart; moved = true;
    while (moved) {
      moved = false;
      while (this.orient(r, l, leftNext.get(l)) < 0) { l = leftNext.get(l); moved = true; step(); }
      while (this.orient(r, l, rightPrev.get(r)) < 0) { r = rightPrev.get(r); moved = true; step(); }
    }
    const lHigh = l, rHigh = r;

    // The sides of the hulls facing each other, from the low bridge to the high one: left's bottom
    // counterclockwise, right's top clockwise
    let leftChain = walk(left, lLow, lHigh, 1);
    let rightChain = walk(right, rLow, rHigh, -1);
    // Both bridges at one point: the rest of that hull lies in the gap, so the zip goes around it
    if (lLow === lHigh && left.length > 1) leftChain = [lLow, ...walk(left, leftNext.get(lLow), lLow, 1)];
    if (rLow === rHigh && right.length > 1) rightChain = [rLow, ...walk(right, rightPrev.get(rLow), rLow, -1)];

    // Zip: the base edge l-r climbs from the low bridge to the high one, each step a triangle with
    // the next point of one chain. It must turn counterclockwise, and its corner at the base's other
    // end must stay out of that hull's own corner there; one of the two always fits.
    let i = 0, j = 0;
    while (i < leftChain.length - 1 || j < rightChain.length - 1) {
      const l = leftChain[i], r = rightChain[j];
      const l2 = i + 1 < leftChain.length ? leftChain[i + 1] : -1;
      const r2 = j + 1 < rightChain.length ? rightChain[j + 1] : -1;
      let takeLeft = l2 >= 0 && this.orient(r, l2, l) > 0
        && !this.anglesOverlap(r, l2, l, rightNext.get(r), rightPrev.get(r));
      const takeRight = r2 >= 0 && this.orient(r, r2, l) > 0
        && !this.anglesOverlap(l, r, r2, leftNext.get(l), leftPrev.get(l));
      // Both fit: like Guibas and Stolfi, the triangle whose circumcircle leaves the other out
      if (takeLeft && takeRight) takeLeft = this.inCircle(r, l2, l, r2) <= 0;
      let a, b, c;
      if (takeLeft) { [a, b, c] = [r, l2, l]; i++; }
      else if (takeRight) { [a, b, c] = [r, r2, l]; j++; }
      else throw new Error(`Zip stuck at base ${l}-${r}`);
      this.addTriangle(a, b, c);
      this.legalize([[a, b], [b, c], [c, a]]);
    }

    // Joined hull: right's bottom, the high bridge, left's top, the low bridge
    return [...walk(right, rLow, rHigh, 1), ...walk(left, lHigh, lLow, 1)];
  }

  circumcentre(a, b, c) {
    // Measured from a, which keeps the numbers small
    const x = this.x, y = this.y;
    const bx = x[b] - x[a], by = y[b] - y[a], cx = x[c] - x[a], cy = y[c] - y[a];
    const b2 = bx * bx + by * by, c2 = cx * cx + cy * cy, d = 2 * (bx * cy - by * cx);
    return [x[a] + (cy * b2 - by * c2) / d, y[a] + (bx * c2 - cx * b2) / d];
  }

  voronoi(stump, box) {
    // The dual of the triangulation, one edge for each of its edges, as [x1, y1, x2, y2] segments.
    // Across an inner edge, the circumcentres of its two triangles are joined. A hull edge's dual
    // is a ray to infinity outwards from its triangle's circumcentre, along the edge's outward
    // normal, through the edge's midpoint, since circumcentres lie on every edge's perpendicular
    // bisector. It is cut at least stump past the hull edge, or past the circumcentre if that is
    // already outside, and with box, [x0, y0, x1, y1] around the points, runs on to its edge. The
    // midpoint is inside the box, so that is measured from there.
    const n = this.n, x = this.x, y = this.y, edges = [], rays = [];
    for (const [key, c] of this.apex) {
      const a = Math.floor(key / n), b = key - a * n;
      const d = this.getApex(b, a);
      if (d >= 0) {
        if (a < b) edges.push([...this.circumcentre(a, b, c), ...this.circumcentre(b, a, d)]);
        continue;
      }
      const [ux, uy] = this.circumcentre(a, b, c);
      // The triangle is on the left of a -> b, the outside on the right
      const length = Math.hypot(x[b] - x[a], y[b] - y[a]);
      const nx = (y[b] - y[a]) / length, ny = -(x[b] - x[a]) / length;
      const mx = (x[a] + x[b]) / 2, my = (y[a] + y[b]) / 2;
      let beyond = Math.max((ux - mx) * nx + (uy - my) * ny, 0) + stump;
      if (box) beyond = Math.max(beyond, exitDistance(mx, my, nx, ny, box));
      rays.push([ux, uy, mx + beyond * nx, my + beyond * ny]);
    }
    return {edges, rays};
  }
}

function exitDistance(px, py, ux, uy, [x0, y0, x1, y1]) {
  // How far from px, py, inside the box, a ray along the unit vector ux, uy leaves it
  let t = Infinity;
  if (ux > 0) t = Math.min(t, (x1 - px) / ux); else if (ux < 0) t = Math.min(t, (x0 - px) / ux);
  if (uy > 0) t = Math.min(t, (y1 - py) / uy); else if (uy < 0) t = Math.min(t, (y0 - py) / uy);
  return t;
}

function walk(hull, start, stop, step) {
  // Vertices of a counterclockwise hull from start to stop inclusive, stepping +1 or -1
  let i = hull.indexOf(start);
  const chain = [start];
  while (chain[chain.length - 1] !== stop) {
    i = (i + step + hull.length) % hull.length;
    chain.push(hull[i]);
    if (chain.length > hull.length + 1) throw new Error('Hull walk lost');
  }
  return chain;
}
