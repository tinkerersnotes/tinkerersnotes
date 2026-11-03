#!/usr/bin/env python3
"""Incremental Delaunay triangulation: points inserted one by one into a big virtual triangle.

The virtual triangle is equilateral, its incircle the circumcircle of the points' bounding box (times
--scale). Each point is located by walking from a virtual corner across the edges it lies behind,
splits its triangle into three, and every edge the new triangles share with neighbours is checked
against the empty circle rule and flipped if needed, in a wave outwards. At the end the virtual
corners and their triangles are removed and the outer boundary is walked to collect the hull.

Any finite virtual triangle can be too small: a thin triangle on a hull edge has a circumcircle
reaching arbitrarily far out, and if it reaches a virtual corner, the triangle loses to ones with
that corner and goes with it. --scale inf puts the corners at infinity instead: corner k is
centre + λ u_k as λ grows without bound, every test involving a corner is a polynomial in λ, and
its sign is the sign of the highest nonzero coefficient. Then the result is exactly Delaunay with
the convex hull as its boundary.

The picture compares the result with the convex hull from divide_and_conquer.py: what the triangles
leave of it uncovered is filled red. Its third panel is the dual, the Voronoi diagram.

    python incremental.py [N] [--seed SEED] [--scale SCALE|inf] [--save picture.png]
"""
import argparse

import numpy as np

from divide_and_conquer import Triangulation, cross, draw_dual, inside_circle, orient


# Polynomials in λ as lists of coefficients from the constant up
def poly_mul(p, q):
    product = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            product[i + j] += a * b
    return product


def poly_add(p, q, sign=1):
    n = max(len(p), len(q))
    p, q = p + [0.0] * (n - len(p)), q + [0.0] * (n - len(q))
    return [a + sign * b for a, b in zip(p, q)]


def poly_sub(p, q):
    return poly_add(p, q, -1)


def leading(p):
    """The highest nonzero coefficient, which decides the sign as λ grows without bound."""
    return next((c for c in reversed(p) if c), 0.0)


class IncrementalTriangulation:
    # Same edge store, flips and Voronoi diagram as the divide and conquer version
    remove_triangle = Triangulation.remove_triangle
    legalize = Triangulation.legalize
    triangles = Triangulation.triangles
    illegal_edges = Triangulation.illegal_edges
    circumcentre = Triangulation.circumcentre
    voronoi = Triangulation.voronoi

    def __init__(self, points, scale=1.0):
        points = np.asarray(points, dtype=float)
        n = self.n = len(points)
        self.at_infinity = np.isinf(scale)
        self.apex = {}
        self.edge_from = {}  # Vertex -> a vertex it has an edge to, where walks can start
        self.flips = self.steps = 0
        self.points = [tuple(p) for p in points]
        self.virtual, self.with_virtual = (), []
        if n <= 3:
            # Already finished: a point, a segment or a triangle, made counterclockwise
            ids = list(range(n))
            if n == 3 and orient(*self.points) < 0:
                ids = [0, 2, 1]
            if n == 3:
                self.add_triangle(*ids)
            self.hull = ids
            return

        # The circumcircle of the bounding box is the incircle of the virtual triangle, whose corners
        # are then twice the radius away from the centre
        lo, hi = points.min(axis=0), points.max(axis=0)
        centre = (lo + hi) / 2
        directions = [(np.cos(a), np.sin(a)) for a in np.radians([90, 210, 330])]
        self.virtual = (n, n + 1, n + 2)
        if self.at_infinity:
            # Corners have no place: tests take them as centre + λ u, their points stay unused
            self.centre = tuple(centre)
            self.direction = dict(zip(self.virtual, directions))
            self.points += [(np.nan, np.nan)] * 3
        else:
            radius = scale * np.linalg.norm(hi - lo) / 2
            self.box_circle = centre, radius
            self.points += [tuple(centre + 2 * radius * np.array(u)) for u in directions]
        self.add_triangle(*self.virtual)

        for v in range(n):
            self.insert(v)

        self.with_virtual = self.triangles()
        for triangle in self.with_virtual:
            if max(triangle) >= n:
                self.remove_triangle(*triangle)
        self.hull = self.boundary()

    def add_triangle(self, a, b, c):
        Triangulation.add_triangle(self, a, b, c)
        # Every triangle a split or a flip removes is replaced by triangles at the same corners,
        # so this keeps every vertex's entry an existing edge
        self.edge_from[a], self.edge_from[b], self.edge_from[c] = b, c, a

    def lifted(self, v):
        """Vertex v as p + λ u: a virtual corner at infinity has its direction u, a point u = 0."""
        if v in self.virtual:
            return self.centre, self.direction[v]
        return self.points[v], (0.0, 0.0)

    def side(self, a, b, c):
        """How far c lies to the left of the line a -> b, as twice the triangle's signed area. With
        corners at infinity, it is the coefficients of λ², λ and 1, compared as tuples, which orders
        them as λ grows without bound."""
        P = self.points
        if not self.at_infinity or max(a, b, c) < self.n:
            return 0.0, 0.0, orient(P[a], P[b], P[c])
        (pa, ua), (pb, ub), (pc, uc) = map(self.lifted, (a, b, c))
        # (b - a) x (c - a) with b - a = p + λ u and c - a = q + λ w
        p, u = np.subtract(pb, pa), np.subtract(ub, ua)
        q, w = np.subtract(pc, pa), np.subtract(uc, ua)
        return cross(u, w), cross(p, w) + cross(u, q), cross(p, q)

    def inside(self, a, b, c, d):
        """True when d is clearly inside the circle through the counterclockwise triangle a, b, c.
        With a corner at infinity among them, the circle test is a polynomial in λ."""
        P = self.points
        if not self.at_infinity or max(a, b, c, d) < self.n:
            return inside_circle(P[a], P[b], P[c], P[d])
        # Measured from a corner at infinity, the λ⁴ and λ³ terms cancel on paper but leave rounding
        # noise, whose sign would decide. So measure from a point: swapping two of the four flips the
        # circle test's sign, and then the leading term is one geometric quantity, not a difference.
        sign = 1
        if d >= self.n:
            corners = [a, b, c]
            i = next(i for i, v in enumerate(corners) if v < self.n)  # One of the four is a point
            corners[i], d = d, corners[i]
            a, b, c = corners
            sign = -1
        # The rows of the circle test, x, y and x² + y² of each corner relative to d, as polynomials
        pd, ud = self.lifted(d)
        rows = []
        for v in (a, b, c):
            pv, uv = self.lifted(v)
            p, u = np.subtract(pv, pd), np.subtract(uv, ud)
            rows.append(([p[0], u[0]], [p[1], u[1]], [p @ p, 2 * (p @ u), u @ u]))
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = rows
        det = poly_add(poly_sub(poly_mul(x1, poly_sub(poly_mul(y2, z3), poly_mul(z2, y3))),
                                poly_mul(y1, poly_sub(poly_mul(x2, z3), poly_mul(z2, x3)))),
                       poly_mul(z1, poly_sub(poly_mul(x2, y3), poly_mul(y2, x3))))
        return sign * leading(det) > 0

    def locate(self, v):
        """The counterclockwise triangle containing point v, walking from the first virtual corner.
        While v lies behind an edge of the current triangle, cross the edge it lies furthest behind.
        In a Delaunay triangulation such a walk never goes round in circles."""
        a = self.virtual[0]
        b = self.edge_from[a]
        c = self.apex[a, b]
        while True:
            x, y, _ = min(((a, b, c), (b, c, a), (c, a, b)), key=lambda e: self.side(e[0], e[1], v))
            if self.side(x, y, v) >= (0, 0, 0):
                return a, b, c
            a, b = y, x
            c = self.apex[a, b]
            self.steps += 1

    def insert(self, v):
        """Split the triangle containing point v into three around it, then flip outwards."""
        a, b, c = self.locate(v)
        self.remove_triangle(a, b, c)
        for x, y in ((a, b), (b, c), (c, a)):
            self.add_triangle(x, y, v)
        self.legalize([(a, b), (b, c), (c, a)])

    def boundary(self):
        """The outer boundary, counterclockwise: edges with a triangle on their left only. Walked
        from the leftmost point, which is on it."""
        P = self.points
        out = {}
        for a, b in self.apex:
            if (b, a) not in self.apex:
                out.setdefault(a, []).append(b)
        # A vertex the boundary passes twice, or a point no triangle kept
        self.pinched = sorted(v for v, ends in out.items() if len(ends) > 1)
        self.lost = sorted(set(range(self.n)) - {v for edge in self.apex for v in edge})
        if not out:
            return []  # Every triangle had a virtual corner
        start = min(out, key=lambda v: P[v])
        # Each boundary edge is walked once. Every vertex has as many boundary edges in as out, so
        # the walk can only end back at the start, through a pinched vertex twice if need be.
        hull = [start]
        while True:
            b = out[hull[-1]].pop()
            if b == start:
                return hull
            hull.append(b)

    def is_convex(self):
        P = self.points
        h = self.hull
        return all(orient(P[h[i - 2]], P[h[i - 1]], P[h[i]]) > 0 for i in range(len(h)))


def virtual_size(scale):
    return 'virtual corners at infinity' if np.isinf(scale) else f'virtual triangle scale {scale:g}'


def draw(t, title, save=None):
    import matplotlib
    if save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    P = np.array(t.points)
    fig, (insertion, result, dual) = plt.subplots(1, 3, figsize=(21, 7))
    for ax in (insertion, result, dual):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    def edges(triangles):
        return {tuple(sorted(e)) for tri in triangles for e in zip(tri, tri[1:] + tri[:1])}

    # Insertion: everything before the virtual triangle goes, the box and its circle
    if t.virtual:
        real = [e for e in edges(t.with_virtual) if max(e) < t.n]
        to_virtual = [e for e in edges(t.with_virtual) if min(e) < t.n <= max(e)]
        lo, hi = P[:t.n].min(axis=0), P[:t.n].max(axis=0)
        if t.at_infinity:
            # An edge to a corner at infinity is a ray: all of them to one corner run parallel
            length = 0.4 * np.linalg.norm(hi - lo)
            segments = [(P[v], P[v] + length * np.array(t.direction[k])) for v, k in to_virtual]
            label = 'Rays to virtual corners at infinity'
        else:
            segments = [P[list(e)] for e in to_virtual]
            label = 'Edges to virtual corners'
        insertion.add_collection(LineCollection(segments, colors='tab:orange', lw=0.6))
        insertion.add_collection(LineCollection([P[list(e)] for e in real], colors='0.55', lw=0.6))
        # A line collection's legend entry can take another colour: an empty line stands in for it
        insertion.plot([], [], color='tab:orange', lw=0.6, label=label)
        if not t.at_infinity:
            corners = P[list(t.virtual) + [t.virtual[0]]]
            insertion.plot(*corners.T, color='tab:orange', lw=1.5, label='Virtual triangle')
            (cx, cy), radius = t.box_circle
            insertion.add_patch(plt.Circle((cx, cy), radius, fill=False, ls='--', color='tab:purple',
                                           label='Its incircle'))
        insertion.autoscale_view()
        insertion.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ls=':', color='tab:purple',
                                          label='Bounding box'))
        insertion.plot(*P[:t.n].T, 'o', color='black', ms=2)
        insertion.legend(loc='upper right', fontsize=8)
    insertion.set_title(f'Insertion: {t.flips} flips, {t.steps} walk steps')

    # Result, against the convex hull from divide_and_conquer.py: what the triangles leave of it
    # uncovered shows red. Comparing triangles one by one would not do: points on one circle have
    # many Delaunay triangulations.
    convex_hull = Triangulation(P[:t.n]).hull
    expected = max(2 * t.n - len(convex_hull) - 2, 0)
    triangles = t.triangles()
    if len(triangles) < expected:
        result.fill(*P[convex_hull].T, color='tab:red', alpha=0.35, lw=0, label='Convex hull not covered')
    result.add_collection(PolyCollection([P[list(tri)] for tri in triangles], facecolors='white',
                                         edgecolors='0.55', lw=0.8))
    hull = P[t.hull + t.hull[:1]]
    result.plot(*hull.T, color='black', lw=1.5,
                label='Outer boundary' + ('' if t.is_convex() else ', not convex'))
    result.plot(*P[:t.n].T, 'o', color='tab:blue', ms=4)
    result.legend(loc='upper right', fontsize=8)
    result.set_title(f'After removing the virtual triangle: {len(triangles)} of {expected} triangles')

    # Dual of the result. With triangles missing, the rays follow the dented boundary.
    draw_dual(dual, t, P[:t.n], t.hull)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))  # Room for the title above both panels
    if save:
        fig.savefig(save, dpi=120)
        print(f'Saved {save}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('n', nargs='?', type=int, default=27, help='number of points (default 27)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help="the virtual triangle's incircle over the bounding box's circumcircle "
                             "(default 1), inf for corners at infinity")
    parser.add_argument('--save', metavar='PNG', help='save the picture instead of showing it')
    args = parser.parse_args()

    points = np.random.default_rng(args.seed).uniform(size=(args.n, 2))
    t = IncrementalTriangulation(points, args.scale)
    print(f'{args.n} points: {len(t.triangles())} triangles, boundary of {len(t.hull)} '
          f'({"convex" if t.is_convex() else "not convex"}), {t.flips} flips, {t.steps} walk steps, '
          f'{len(t.illegal_edges())} illegal edges left')
    draw(t, f'Incremental triangulation of {args.n} uniform random points (seed {args.seed}, '
            f'{virtual_size(args.scale)})', args.save)


if __name__ == '__main__':
    main()
