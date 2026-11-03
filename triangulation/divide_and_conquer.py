#!/usr/bin/env python3
"""Divide and conquer Delaunay triangulation of random points, drawn with matplotlib.

Divide: a line through the points' centre of mass splits them into a left and a right half,
each triangulated by its own recursive call. Leaves: one point, two points (a segment) or three
(a triangle, its hull in counterclockwise order).

Conquer: the halves' convex hulls are joined by their two bridges, the common tangents. The gap
between the bridges is zipped up with triangles from one bridge to the other, and every edge whose
two triangles break the empty circle rule is flipped (Lawson), which can reach deep into either
half. When no edge is left to flip, the triangulation is Delaunay.

Dual: joining the circumcentres of triangles across each inner edge gives the Voronoi diagram, and
each hull edge gets a ray outwards, drawn to the edge of the view instead of to infinity.

Points are assumed to have no three on a line, which uniformly distributed random points satisfy
with probability 1. Four or more on a circle are fine: any of their triangulations is Delaunay, and the
circle test ignores differences within rounding noise, so such edges never flip back and forth.

    python divide_and_conquer.py [N] [--seed SEED] [--save picture.png]
"""
import argparse

import numpy as np


def cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def orient(p, q, r):
    """Twice the signed area of triangle pqr: positive when p, q, r turn counterclockwise."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def in_circle(a, b, c, d):
    """Positive when d lies inside the circle through the counterclockwise triangle abc."""
    (ax, ay), (bx, by), (cx, cy) = [(p[0] - d[0], p[1] - d[1]) for p in (a, b, c)]
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    return (ax * (by * c2 - b2 * cy)
            - ay * (bx * c2 - b2 * cx)
            + a2 * (bx * cy - by * cx))


def inside_circle(a, b, c, d):
    """True when d lies clearly inside the circle through the counterclockwise triangle abc: the circle
    test beats its rounding noise, a small fraction of the sum of its terms' sizes. A point on the
    circle, like any of many points placed on one circle, is not inside."""
    (ax, ay), (bx, by), (cx, cy) = [(abs(p[0] - d[0]), abs(p[1] - d[1])) for p in (a, b, c)]
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    size = ax * (by * c2 + b2 * cy) + ay * (bx * c2 + b2 * cx) + a2 * (bx * cy + by * cx)
    return in_circle(a, b, c, d) > 1e-12 * size


def inside_angle(o, a, b, x):
    """True when x lies strictly inside the angle at o, less than pi, turning counterclockwise from
    a to b. The angle of a segment end (a == b) has no inside."""
    return orient(o, a, x) > 0 and orient(o, x, b) > 0


def angles_overlap(o, a1, a2, b1, b2):
    """True when the angles at o from a1 to a2 and from b1 to b2, each less than pi, overlap: then
    one of them starts inside the other."""
    return inside_angle(o, a1, a2, b1) or inside_angle(o, b1, b2, a1)


def exit_distance(p, u, box):
    """How far from p, inside box (x0, y0, x1, y1), a ray along the unit vector u leaves it."""
    x0, y0, x1, y1 = box
    t = np.inf
    for v, d, low, high in ((p[0], u[0], x0, x1), (p[1], u[1], y0, y1)):
        if d > 0:
            t = min(t, (high - v) / d)
        elif d < 0:
            t = min(t, (low - v) / d)
    return t


def walk(hull, start, stop, step):
    """Vertices of a counterclockwise hull from start to stop inclusive, stepping +1 (counterclockwise)
    or -1 (clockwise)."""
    i = hull.index(start)
    chain = [start]
    while chain[-1] != stop:
        i = (i + step) % len(hull)
        chain.append(hull[i])
    return chain


class Triangulation:
    def __init__(self, points):
        self.points = [tuple(p) for p in points]
        # Directed edge (a, b) -> c for every counterclockwise triangle a, b, c. The triangle on the
        # other side of the edge, if any, is under (b, a).
        self.apex = {}
        self.flips = 0
        # For drawing: (depth, centroid, direction, extent along the line) of every split,
        # (depth, low bridge, high bridge) of every merge, the leaves' point indices
        self.splits = []
        self.bridges = []
        self.leaves = []
        self.hull = self.triangulate(list(range(len(points))))

    def add_triangle(self, a, b, c):
        self.apex[a, b], self.apex[b, c], self.apex[c, a] = c, a, b

    def remove_triangle(self, a, b, c):
        del self.apex[a, b], self.apex[b, c], self.apex[c, a]

    def triangles(self):
        return sorted({min((a, b, c), (b, c, a), (c, a, b)) for (a, b), c in self.apex.items()})

    def inside(self, a, b, c, d):
        """True when point d is clearly inside the circle through the counterclockwise triangle a, b, c."""
        P = self.points
        return inside_circle(P[a], P[b], P[c], P[d])

    def legalize(self, edges):
        """Flip edges until none of them, nor any edge a flip touches, breaks the empty circle rule.
        Other edges keep their two triangles, so stay as legal as they were."""
        stack = list(edges)
        while stack:
            a, b = stack.pop()
            c, d = self.apex.get((a, b)), self.apex.get((b, a))
            if c is None or d is None or not self.inside(a, b, c, d):
                continue  # A hull edge, an edge flipped away meanwhile, or a legal edge
            # Triangles a, b, c and b, a, d make the convex quadrilateral a, d, b, c: swap the
            # diagonal a-b for c-d and check the four outer edges against their own neighbours
            self.remove_triangle(a, b, c)
            self.remove_triangle(b, a, d)
            self.add_triangle(a, d, c)
            self.add_triangle(d, b, c)
            self.flips += 1
            stack += [(a, d), (d, b), (b, c), (c, a)]

    def illegal_edges(self):
        return [(a, b) for (a, b), c in self.apex.items()
                if (b, a) in self.apex and self.inside(a, b, c, self.apex[b, a])]

    def circumcentre(self, a, b, c):
        """Centre of the circle through points a, b and c, measured from a to keep the numbers small."""
        P = self.points
        (ax, ay), (bx, by), (cx, cy) = P[a], P[b], P[c]
        bx, by, cx, cy = bx - ax, by - ay, cx - ax, cy - ay
        b2, c2, d = bx * bx + by * by, cx * cx + cy * cy, 2 * (bx * cy - by * cx)
        return ax + (cy * b2 - by * c2) / d, ay + (bx * c2 - cx * b2) / d

    def voronoi(self, stump, box=None):
        """The dual of the triangulation, the Voronoi diagram: one edge for each of its edges, as
        ((x1, y1), (x2, y2)). Returns the edges joining the circumcentres of the two triangles on
        either side of an inner edge, and the rays for hull edges.

        A hull edge's dual is a ray to infinity, outwards from its triangle's circumcentre, along the
        edge's outward normal and through its midpoint, since a circumcentre lies on the
        perpendicular bisector of every edge. It is cut at a virtual vertex at least stump past the
        hull edge, or past the circumcentre when that is outside already, as for a triangle obtuse
        opposite the edge. With box, (x0, y0, x1, y1) around the points, it runs on to the box's
        edge. The midpoint is inside the box, so that is measured from there."""
        P = self.points
        edges, rays = [], []
        for (a, b), c in self.apex.items():
            d = self.apex.get((b, a))
            if d is not None:
                if a < b:
                    edges.append((self.circumcentre(a, b, c), self.circumcentre(b, a, d)))
                continue
            centre = np.array(self.circumcentre(a, b, c))
            pa, pb = np.array(P[a]), np.array(P[b])
            # The triangle is on the left of a -> b, the outside on the right
            normal = np.array([pb[1] - pa[1], pa[0] - pb[0]]) / np.linalg.norm(pb - pa)
            middle = (pa + pb) / 2
            beyond = max((centre - middle) @ normal, 0) + stump
            if box is not None:
                beyond = max(beyond, exit_distance(middle, normal, box))
            rays.append((tuple(centre), tuple(middle + beyond * normal)))
        return edges, rays

    def triangulate(self, ids, depth=0):
        """Triangulate the points ids, return their convex hull in counterclockwise order."""
        P = self.points
        if len(ids) <= 3:
            self.leaves.append(ids)
            if len(ids) == 3:
                a, b, c = ids
                if orient(P[a], P[b], P[c]) < 0:
                    b, c = c, b
                self.add_triangle(a, b, c)
                return [a, b, c]
            return list(ids)  # A point or a segment is its own hull

        # The split line goes through the centre of mass along the direction the points spread the
        # least, so it cuts across their widest extent and the halves come out about even
        xy = np.array([P[v] for v in ids])
        centroid = xy.mean(axis=0)
        s = np.linalg.eigh(np.cov(xy.T))[1][:, 0]
        if (s[0], s[1]) < (0, 0):
            s = -s  # Along +x, so the left half is the upper one, as in the merge's picture
        offsets = xy - centroid
        heights = offsets @ np.array([-s[1], s[0]])  # Distance to the left of the line
        along = offsets @ s
        # A point right on the line goes by which way along the line it lies
        on_left = (heights > 0) | ((heights == 0) & (along <= 0))
        self.splits.append((depth, centroid, s, (along.min(), along.max())))

        left = self.triangulate([v for v, l in zip(ids, on_left) if l], depth + 1)
        right = self.triangulate([v for v, l in zip(ids, on_left) if not l], depth + 1)
        return self.merge(left, right, centroid, s, depth)

    def merge(self, left, right, centroid, s, depth):
        """Join the triangulations inside the convex hulls left and right (counterclockwise, on the
        left and right of the line through centroid along s), return the joined convex hull.

        In coordinates with the split line as the x axis, left is above and right below. The low
        bridge l_low -> r_low closes the joined hull at small x, r_high -> l_high at large x, and
        both have every point on their left."""
        P = self.points
        nxt = lambda hull: {v: hull[(i + 1) % len(hull)] for i, v in enumerate(hull)}
        prv = lambda hull: {v: hull[i - 1] for i, v in enumerate(hull)}
        left_next, left_prev, right_next, right_prev = nxt(left), prv(left), nxt(right), prv(right)

        # Start from the points nearest to the split line and walk each end outwards until nothing
        # is on the wrong side of the bridge. Neither end ever steps back, so the walk takes at most
        # one step per hull vertex: linear, not quadratic.
        height = lambda v: cross(s, np.subtract(P[v], centroid))
        l_start, r_start = min(left, key=height), max(right, key=height)

        l, r = l_start, r_start
        moved = True
        while moved:
            moved = False
            while orient(P[l], P[r], P[left_prev[l]]) < 0:
                l, moved = left_prev[l], True
            while orient(P[l], P[r], P[right_next[r]]) < 0:
                r, moved = right_next[r], True
        l_low, r_low = l, r

        l, r = l_start, r_start
        moved = True
        while moved:
            moved = False
            while orient(P[r], P[l], P[left_next[l]]) < 0:
                l, moved = left_next[l], True
            while orient(P[r], P[l], P[right_prev[r]]) < 0:
                r, moved = right_prev[r], True
        l_high, r_high = l, r
        self.bridges.append((depth, (l_low, r_low), (l_high, r_high)))

        # The sides of the hulls facing each other, from the low bridge to the high one: left's
        # bottom counterclockwise, right's top clockwise
        left_chain = walk(left, l_low, l_high, +1)
        right_chain = walk(right, r_low, r_high, -1)
        # Both bridges at one point: the joined hull keeps only that point of this hull, so the rest
        # of it lies in the gap and the zip goes all the way around it (both sides of a segment)
        if l_low == l_high and len(left) > 1:
            left_chain = [l_low] + walk(left, left_next[l_low], l_low, +1)
        if r_low == r_high and len(right) > 1:
            right_chain = [r_low] + walk(right, right_prev[r_low], r_low, -1)

        # Zip: the base edge l-r climbs from the low bridge to the high one, each step a triangle
        # with the next point of one chain. The triangle must turn counterclockwise, which keeps it
        # outside the hull it takes the point from, and its corner at the base's other end must
        # stay out of that hull's own corner there. Every triangulation of the gap has a triangle
        # on the base, made with one of the two next points, so one of them always fits.
        i = j = 0
        while i < len(left_chain) - 1 or j < len(right_chain) - 1:
            l, r = left_chain[i], right_chain[j]
            l2 = left_chain[i + 1] if i + 1 < len(left_chain) else None
            r2 = right_chain[j + 1] if j + 1 < len(right_chain) else None
            take_left = (l2 is not None and orient(P[r], P[l2], P[l]) > 0
                         and not angles_overlap(P[r], P[l2], P[l], P[right_next[r]], P[right_prev[r]]))
            take_right = (r2 is not None and orient(P[r], P[r2], P[l]) > 0
                          and not angles_overlap(P[l], P[r], P[r2], P[left_next[l]], P[left_prev[l]]))
            if take_left and take_right:
                # Both fit: like Guibas and Stolfi, take the triangle whose circumcircle leaves the
                # other candidate out, so fewer flips follow
                take_left = in_circle(P[r], P[l2], P[l], P[r2]) <= 0
            if take_left:
                triangle, i = (r, l2, l), i + 1
            elif take_right:
                triangle, j = (r, r2, l), j + 1
            else:
                raise AssertionError(f'Zip stuck at base {l}-{r}')
            self.add_triangle(*triangle)
            a, b, c = triangle
            self.legalize([(a, b), (b, c), (c, a)])

        # Joined hull: right's bottom, the high bridge, left's top, the low bridge
        return walk(right, r_low, r_high, +1) + walk(left, l_high, l_low, +1)


def draw_dual(ax, t, P, hull):
    """The Voronoi diagram of triangulation t on the axes ax, over the triangulation drawn faint, its
    vertices the circumcentres. Thin triangles have far away circumcentres, so the view stays on the
    points P, the hull's points among them."""
    from matplotlib.collections import LineCollection

    triangles = t.triangles()
    lo, hi = P.min(axis=0), P.max(axis=0)
    margin = 0.15 * (hi - lo).max()
    edges = {tuple(sorted(e)) for tri in triangles for e in zip(tri, tri[1:] + tri[:1])}
    voronoi, rays = t.voronoi(stump=0.06 * (hi - lo).max(), box=(*(lo - margin), *(hi + margin)))
    ax.add_collection(LineCollection([P[list(e)] for e in edges], colors='0.8', lw=0.6))
    ax.add_collection(LineCollection(voronoi, colors='tab:orange', lw=1.2))
    ax.add_collection(LineCollection(rays, colors='tab:orange', lw=1.2, linestyles=':'))
    centres = np.array([t.circumcentre(*tri) for tri in triangles]).reshape(-1, 2)
    ax.plot(*centres.T, '.', color='tab:orange', ms=4, label='Circumcentres')
    ax.plot([], [], color='tab:orange', lw=1.2, label='Voronoi edges')
    ax.plot([], [], color='tab:orange', lw=1.2, ls=':', label='Rays past hull edges')
    if hull:
        ax.plot(*P[hull + hull[:1]].T, color='0.55', lw=1)
    ax.plot(*P.T, 'o', color='black', ms=4)
    ax.set_xlim(lo[0] - margin, hi[0] + margin)
    ax.set_ylim(lo[1] - margin, hi[1] + margin)
    ax.set_title(f'Dual: {len(voronoi)} Voronoi edges, {len(rays)} rays')
    if triangles:
        ax.legend(loc='upper right', fontsize=8)


def draw(t, title, save=None):
    import matplotlib
    if save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    P = np.array(t.points)
    triangles = t.triangles()
    max_depth = max((d for d, *_ in t.splits), default=0)
    depth_color = plt.cm.viridis(np.linspace(0, 0.85, max_depth + 1))

    fig, (divide, conquer, dual) = plt.subplots(1, 3, figsize=(21, 7))
    for ax in (divide, conquer, dual):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # Divide: every split line across the extent of the points it splits, and the leaves
    labelled = set()
    for depth, centroid, s, (lo, hi) in sorted(t.splits, key=lambda split: split[0]):
        a, b = centroid + lo * s, centroid + hi * s
        divide.plot(*zip(a, b), color=depth_color[depth], lw=3 - 2 * depth / max(max_depth, 1),
                    label=None if depth in labelled else f'Split, depth {depth}')
        divide.plot(*centroid, 'x', color=depth_color[depth], ms=6)
        labelled.add(depth)
    for ids in t.leaves:
        corners = P[ids + ids[:1]]
        if len(ids) == 3:
            divide.fill(*corners.T, color='tab:orange', alpha=0.25, lw=0)
        divide.plot(*corners.T, color='tab:orange', lw=1)
    divide.plot(*P.T, 'o', color='black', ms=4)
    divide.set_title(f'Divide: {len(t.splits)} centre of mass splits, {len(t.leaves)} leaves')
    if t.splits:
        divide.legend(loc='upper right', fontsize=8)

    # Conquer: the Delaunay triangulation, the hull and the top merge's split line and bridges
    edges = {tuple(sorted(e)) for tri in triangles for e in zip(tri, tri[1:] + tri[:1])}
    conquer.add_collection(LineCollection([P[list(e)] for e in edges], colors='0.55', lw=0.8))
    hull = P[t.hull + t.hull[:1]]
    conquer.plot(*hull.T, color='black', lw=1.5, label='Convex hull')
    if t.splits:
        depth, centroid, s, (lo, hi) = t.splits[0]
        normal = np.array([-s[1], s[0]])
        left = (P - centroid) @ normal > 0
        conquer.plot(*P[left].T, 'o', color='tab:blue', ms=5, label='Left half')
        conquer.plot(*P[~left].T, 'o', color='tab:green', ms=5, label='Right half')
        conquer.plot(*zip(centroid + lo * s, centroid + hi * s), '--', color=depth_color[0], lw=1.5,
                     label='Top split line')
        _, low, high = t.bridges[-1]
        for k, bridge in enumerate((low, high)):
            conquer.plot(*P[list(bridge)].T, color='tab:red', lw=3, label='Top bridges' if k == 0 else None)
    else:
        conquer.plot(*P.T, 'o', color='black', ms=5)
    conquer.set_title(f'Conquer: {len(triangles)} Delaunay triangles, {t.flips} edge flips')
    conquer.legend(loc='upper right', fontsize=8)

    draw_dual(dual, t, P, t.hull)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))  # Room for the title above the panels
    if save:
        fig.savefig(save, dpi=120)
        print(f'Saved {save}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('n', nargs='?', type=int, default=27, help='number of points (default 27)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('--save', metavar='PNG', help='save the picture instead of showing it')
    args = parser.parse_args()

    points = np.random.default_rng(args.seed).uniform(size=(args.n, 2))
    t = Triangulation(points)
    print(f'{args.n} points: {len(t.triangles())} triangles, hull of {len(t.hull)}, '
          f'{t.flips} flips, {len(t.illegal_edges())} illegal edges left')
    draw(t, f'Divide and conquer triangulation of {args.n} uniform random points (seed {args.seed})',
         args.save)


if __name__ == '__main__':
    main()
