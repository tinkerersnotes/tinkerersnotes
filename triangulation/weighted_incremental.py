#!/usr/bin/env python3
"""Weighted Delaunay (regular) triangulation, built incrementally: the lower convex hull of 3D points.

Each point (x, y) has a height h. The regular triangulation is the lower convex hull of the lifted
points (x, y, h), projected back onto the plane. With h = x² + y² it is the Delaunay triangulation;
a weight w makes h = x² + y² - w. Points above the lower hull are redundant: no triangle uses them.

Points are inserted one by one into the virtual triangle of incremental.py, with its corners at
infinity, lifted onto the paraboloid there, which puts them above every face of the points' lower
hull. A point on or above the plane of the triangle it falls in is redundant from the start, and
stays so: more points only lower the hull. Otherwise it splits that triangle into three, and every
edge opposite it, its link, is checked: when the far point across the edge is below the plane of
the new point's triangle, the edge is flipped. Two triangles become two when they make a convex
quadrilateral. When the quadrilateral has a reflex corner surrounded by just three triangles, the
three become one and the corner's point is redundant from then on (Edelsbrunner and Shah). Any other
reflex corner is left: flips of its other link edges bring it down to three triangles.

convex_hull_3d() glues the lower hull of the heights and the lower hull of their negatives, which is
the upper hull, along the outline of the points seen from above.

    python weighted_incremental.py [N] [--seed SEED] [--save picture.png]
"""
import argparse

import numpy as np

from divide_and_conquer import orient
from incremental import IncrementalTriangulation, leading, poly_add, poly_mul, poly_sub


def below_plane(a, b, c, d):
    """True when the point d = (x, y, h) lies clearly below the plane through a, b and c, whose
    (x, y) turn counterclockwise: the determinant beats its rounding noise, a small fraction of the
    sum of its terms' sizes. With h = x² + y², this is the circle test."""
    (ax, ay, ah), (bx, by, bh), (cx, cy, ch) = [(p[0] - d[0], p[1] - d[1], p[2] - d[2]) for p in (a, b, c)]
    det = ax * (by * ch - bh * cy) - ay * (bx * ch - bh * cx) + ah * (bx * cy - by * cx)
    size = (abs(ax) * (abs(by * ch) + abs(bh * cy)) + abs(ay) * (abs(bx * ch) + abs(bh * cx))
            + abs(ah) * (abs(bx * cy) + abs(by * cx)))
    return det > 1e-12 * size


def poly_det3(rows):
    """Determinant of three rows of polynomials in λ."""
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = rows
    return poly_add(poly_sub(poly_mul(x1, poly_sub(poly_mul(y2, z3), poly_mul(z2, y3))),
                             poly_mul(y1, poly_sub(poly_mul(x2, z3), poly_mul(z2, x3)))),
                    poly_mul(z1, poly_sub(poly_mul(x2, y3), poly_mul(y2, x3))))


class WeightedTriangulation(IncrementalTriangulation):
    """The regular triangulation of points with heights. Redundant points are in self.redundant;
    self.triangles() are the lower hull's faces."""

    def __init__(self, points, heights):
        self.heights = [float(h) for h in heights]
        self.redundant = set()
        self.removals = 0
        # Always with corners at infinity: a finite virtual triangle could spoil the lower hull
        super().__init__(points, scale=np.inf)

    def height(self, v):
        """Vertex v's lifted height, as a polynomial in λ. A corner at infinity, centre + λ u, is
        lifted onto the paraboloid: |centre + λ u|²."""
        if v in self.virtual:
            (cx, cy), (ux, uy) = self.centre, self.direction[v]
            return [cx * cx + cy * cy, 2 * (cx * ux + cy * uy), ux * ux + uy * uy]
        return [self.heights[v]]

    def inside(self, a, b, c, d):
        """True when the lifted point of d lies clearly below the plane through the lifted points
        of the counterclockwise triangle a, b, c: then the edge across from d is not locally regular.
        With a corner at infinity among them, the test is a polynomial in λ, measured from a point,
        not a corner, as in incremental.py, so leading terms don't cancel into rounding noise."""
        P, H = self.points, self.heights
        if max(a, b, c, d) < self.n:
            return below_plane(*[(P[v][0], P[v][1], H[v]) for v in (a, b, c, d)])
        sign = 1
        if d >= self.n:
            corners = [a, b, c]
            i = next(i for i, v in enumerate(corners) if v < self.n)  # One of the four is a point
            corners[i], d = d, corners[i]
            a, b, c = corners
            sign = -1
        pd, ud = self.lifted(d)
        hd = self.height(d)
        rows = []
        for v in (a, b, c):
            pv, uv = self.lifted(v)
            p, u = np.subtract(pv, pd), np.subtract(uv, ud)
            rows.append(([p[0], u[0]], [p[1], u[1]], poly_sub(self.height(v), hd)))
        return sign * leading(poly_det3(rows)) > 0

    def insert(self, v):
        """Split the triangle containing point v into three around it and flip its link, or leave
        v out when it is on or above that triangle's plane."""
        a, b, c = self.locate(v)
        if not self.inside(a, b, c, v):
            self.redundant.add(v)
            return
        self.remove_triangle(a, b, c)
        for x, y in ((a, b), (b, c), (c, a)):
            self.add_triangle(x, y, v)
        self.flip_link(v, [(a, b), (b, c), (c, a)])

    def flip_link(self, p, edges):
        """Flip the link edges of the new point p until all are locally regular."""
        stack = list(edges)
        while stack:
            a, b = stack.pop()
            if self.apex.get((a, b)) != p:
                continue  # Flipped away meanwhile: no longer a link edge of p
            d = self.apex.get((b, a))
            if d is None or not self.inside(a, b, p, d):
                continue  # On the virtual triangle's boundary, or locally regular
            # Triangles a, b, p and b, a, d: the quadrilateral a, d, b, p is convex at d and p
            convex_a = self.side(p, a, d) > (0, 0, 0)
            convex_b = self.side(d, b, p) > (0, 0, 0)
            if convex_a and convex_b:
                # Two to two: swap the diagonal a-b for p-d
                self.remove_triangle(a, b, p)
                self.remove_triangle(b, a, d)
                self.add_triangle(a, d, p)
                self.add_triangle(d, b, p)
                self.flips += 1
                stack += [(a, d), (d, b)]
            elif not convex_a and self.apex.get((a, p)) == d:
                # Three to one: a lies inside triangle b, p, d, surrounded by these three triangles,
                # and below its plane no longer: a is redundant
                self.remove_triangle(a, b, p)
                self.remove_triangle(b, a, d)
                self.remove_triangle(a, p, d)
                self.add_triangle(b, p, d)
                self.redundant.add(a)
                self.removals += 1
                stack.append((d, b))
            elif not convex_b and self.apex.get((b, d)) == p:
                # Three to one, around b
                self.remove_triangle(a, b, p)
                self.remove_triangle(b, a, d)
                self.remove_triangle(b, d, p)
                self.add_triangle(a, d, p)
                self.redundant.add(b)
                self.removals += 1
                stack.append((a, d))
            # Otherwise a reflex corner with more than three triangles: flips of its other link
            # edges take it down to three, and the last of them removes it


def convex_hull_3d(points):
    """The convex hull of 3D points as triangles (a, b, c), counterclockwise seen from outside: the
    lower hull, faces turned to face down, and the upper hull, the lower hull of the heights' negatives.
    They meet along the outline seen from above, the convex hull of the points' (x, y)."""
    points = np.asarray(points, dtype=float)
    xy = points[:, :2]
    lower = WeightedTriangulation(xy, points[:, 2])
    upper = WeightedTriangulation(xy, -points[:, 2])
    faces = [(a, c, b) for a, b, c in lower.triangles()] + upper.triangles()
    return faces, lower, upper


def draw(points, faces, lower, upper, title, save=None):
    import matplotlib
    if save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    P = np.asarray(points, dtype=float)
    fig = plt.figure(figsize=(21, 7))
    panels = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3, projection='3d')]

    for ax, t, name, colour in ((panels[0], lower, 'Lower hull', 'tab:blue'),
                                (panels[1], upper, 'Upper hull', 'tab:orange')):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_collection(PolyCollection([P[list(tri), :2] for tri in t.triangles()], facecolors=colour,
                                         alpha=0.15, edgecolors=colour, lw=0.8))
        used = [v for v in range(len(P)) if v not in t.redundant]
        ax.plot(*P[used, :2].T, 'o', color=colour, ms=4, label='Vertices')
        ax.plot(*P[sorted(t.redundant), :2].T, 'o', mfc='none', color='0.6', ms=4,
                label='Redundant: above this hull')
        ax.plot(*P[t.hull + t.hull[:1], :2].T, color='tab:red', lw=1.5, label='Outline seen from above')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(f'{name} from above: {len(t.triangles())} triangles, {len(used)} vertices, '
                     f'{t.flips} flips, {t.removals} removals')

    ax = panels[2]
    lower_faces = [P[[a, c, b]] for a, b, c in lower.triangles()]
    upper_faces = [P[list(tri)] for tri in upper.triangles()]
    ax.add_collection3d(Poly3DCollection(lower_faces, facecolors='tab:blue', alpha=0.35, edgecolors='0.3', lw=0.5))
    ax.add_collection3d(Poly3DCollection(upper_faces, facecolors='tab:orange', alpha=0.35, edgecolors='0.3', lw=0.5))
    outline = P[lower.hull + lower.hull[:1]]
    ax.plot(*outline.T, color='tab:red', lw=2)
    on_hull = sorted({v for face in faces for v in face})
    inside = sorted(set(range(len(P))) - set(on_hull))
    ax.scatter(*P[on_hull].T, color='black', s=8)
    ax.scatter(*P[inside].T, color='0.6', s=6)
    lo, hi = P.min(axis=0), P.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_title(f'Convex hull: {len(faces)} faces, {len(on_hull)} of {len(P)} points; '
                 f'outline in red')

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        fig.savefig(save, dpi=120)
        print(f'Saved {save}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('n', nargs='?', type=int, default=50, help='number of points (default 50)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('--save', metavar='PNG', help='save the picture instead of showing it')
    args = parser.parse_args()

    points = np.random.default_rng(args.seed).uniform(size=(args.n, 3))
    faces, lower, upper = convex_hull_3d(points)
    print(f'{args.n} points: convex hull of {len(faces)} faces, lower hull {len(lower.triangles())} '
          f'triangles ({len(lower.redundant)} redundant points), upper hull {len(upper.triangles())} '
          f'triangles ({len(upper.redundant)} redundant points)')
    draw(points, faces, lower, upper,
         f'Convex hull of {args.n} uniform random points in a cube (seed {args.seed}) from two weighted '
         f'triangulations', args.save)


if __name__ == '__main__':
    main()
