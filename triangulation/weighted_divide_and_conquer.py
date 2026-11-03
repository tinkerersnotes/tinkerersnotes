#!/usr/bin/env python3
"""Weighted Delaunay (regular) triangulation by divide and conquer: the lower convex hull of 3D points.

The same recursion as divide_and_conquer.py, with heights: the flips test the lifted plane, not the
circle, and a merge may find a point of one half redundant once the other half is there, removing it
by a three to one flip. weighted_incremental.py is the source of truth to check it against.

The merge zips the gap as before, then flips. An illegal edge a-b, between triangles a, b, c and
b, a, d, is flipped two to two when the quadrilateral a, d, b, c is convex. When it isn't, one corner,
say a, lies inside triangle c, b, d, and a is redundant: had a been below the plane of c, b, d, the
lower hull of those four points would be the three triangles around a, and d would be above the plane
of a, b, c, the edge legal. So a goes, its hole is triangulated again, and flipping goes on. With
three triangles around a, that is the three to one flip; with more, the hole's triangles are cut off
ear by ear, the ear with the lowest plane over a first. Measured on 5000-point sets, that needs about 17%
fewer flips afterwards than the highest plane first, the order Devillers' Delaunay deletion uses for a
point below its neighbours' circles; a redundant point is above its neighbours' planes.

A first attempt only did three to one flips and left any other reflex corner for later flips to free,
as an insertion can. It got stuck on 2% to 21% of the hard test sets, the most on heights on a dome,
where every point inside the outline is redundant. self.stuck counts the edges left illegal; with
removal it stays empty.

    python weighted_divide_and_conquer.py [N] [--seed SEED]
"""
import argparse

import numpy as np

from divide_and_conquer import Triangulation, orient
from weighted_incremental import WeightedTriangulation, below_plane


class WeightedDivideAndConquer(Triangulation):
    """The regular triangulation of points with heights, by divide and conquer. Redundant points are
    in self.redundant, and illegal edges no flip could fix in self.stuck."""

    def __init__(self, points, heights):
        self.heights = [float(h) for h in heights]
        self.redundant = set()
        self.removals = 0
        self.stuck = []
        self.deferred = []  # Illegal edges whose redundant corner is still on the zip's open edge
        self.edge_from = {}  # Vertex -> a vertex it has an edge to, where its ring of neighbours starts
        super().__init__(points)
        self.stuck = self.illegal_edges()

    def add_triangle(self, a, b, c):
        Triangulation.add_triangle(self, a, b, c)
        # Every triangle a flip or a removal takes away is replaced by triangles at the same corners,
        # the removed point's apart, so this keeps every remaining vertex's entry an existing edge
        self.edge_from[a], self.edge_from[b], self.edge_from[c] = b, c, a

    def inside(self, a, b, c, d):
        """True when the lifted point of d lies clearly below the plane through those of the
        counterclockwise triangle a, b, c."""
        P, H = self.points, self.heights
        return below_plane(*[(P[v][0], P[v][1], H[v]) for v in (a, b, c, d)])

    def legalize(self, edges):
        """Flip edges until none of them, nor any edge a flip touches, is illegal, or the illegal
        ones left can't be flipped."""
        P = self.points
        stack = list(edges)
        while stack:
            a, b = stack.pop()
            c, d = self.apex.get((a, b)), self.apex.get((b, a))
            if c is None or d is None or not self.inside(a, b, c, d):
                continue
            # Triangles a, b, c and b, a, d: the quadrilateral a, d, b, c is convex at c and d
            convex_a = orient(P[c], P[a], P[d]) > 0
            convex_b = orient(P[d], P[b], P[c]) > 0
            if convex_a and convex_b:
                self.remove_triangle(a, b, c)
                self.remove_triangle(b, a, d)
                self.add_triangle(a, d, c)
                self.add_triangle(d, b, c)
                self.flips += 1
                stack += [(a, d), (d, b), (b, c), (c, a)]
            else:
                # The reflex corner is redundant. On the zip's open edge it has no closed hole yet:
                # the retry after the zip removes it then.
                edges = self.remove_vertex(a if not convex_a else b)
                if edges:
                    stack += edges
                else:
                    self.deferred.append((a, b))

    def remove_vertex(self, v):
        """Remove redundant point v and triangulate its hole again, returning the edges to check, or
        none when v's triangles don't close around it."""
        P, H = self.points, self.heights
        # v's neighbours counterclockwise: each triangle v, x, y leads on to y
        start = self.edge_from[v]
        ring = [start]
        while True:
            nxt = self.apex.get((v, ring[-1]))
            if nxt is None:
                return []
            if nxt == start:
                break
            ring.append(nxt)
        for x, y in zip(ring, ring[1:] + ring[:1]):
            self.remove_triangle(v, x, y)

        def height_over_v(a, b, c):
            # The plane through the lifted a, b and c, at v's place
            (ax, ay), (bx, by), (cx, cy), (vx, vy) = P[a], P[b], P[c], P[v]
            normal = np.cross([bx - ax, by - ay, H[b] - H[a]], [cx - ax, cy - ay, H[c] - H[a]])
            return H[a] - (normal[0] * (vx - ax) + normal[1] * (vy - ay)) / normal[2]

        # The hole is star-shaped around v, so it has ears: corners turning counterclockwise with no
        # other corner of the hole inside. Cut off the one whose plane is lowest over v.
        hole, edges = list(ring), []
        while len(hole) > 3:
            ears = []
            for i, b in enumerate(hole):
                a, c = hole[i - 1], hole[(i + 1) % len(hole)]
                if orient(P[a], P[b], P[c]) <= 0:
                    continue
                if any(orient(P[a], P[b], P[x]) > 0 and orient(P[b], P[c], P[x]) > 0
                       and orient(P[c], P[a], P[x]) > 0 for x in hole if x not in (a, b, c)):
                    continue
                ears.append((height_over_v(a, b, c), i))
            _, i = min(ears)
            a, b, c = hole[i - 1], hole[i], hole[(i + 1) % len(hole)]
            self.add_triangle(a, b, c)
            edges += [(a, b), (b, c), (c, a)]
            hole.pop(i)
        self.add_triangle(*hole)
        edges += list(zip(hole, hole[1:] + hole[:1]))
        self.redundant.add(v)
        self.removals += 1
        return edges

    def merge(self, left, right, centroid, s, depth):
        hull = super().merge(left, right, centroid, s, depth)
        # Every other edge the flips met is legal by now, or was queued again when its triangles
        # changed, so only the deferred ones can still be illegal. With the zip done, their corners'
        # holes are closed; retry them until nothing moves.
        while self.deferred:
            before = (self.flips, self.removals)
            deferred, self.deferred = self.deferred, []
            self.legalize(deferred)
            if (self.flips, self.removals) == before:
                break
        self.deferred = []
        return hull


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('n', nargs='?', type=int, default=50, help='number of points (default 50)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default 1)')
    args = parser.parse_args()

    points = np.random.default_rng(args.seed).uniform(size=(args.n, 3))
    t = WeightedDivideAndConquer(points[:, :2], points[:, 2])
    truth = WeightedTriangulation(points[:, :2], points[:, 2])
    same = {tuple(sorted(x)) for x in t.triangles()} == {tuple(sorted(x)) for x in truth.triangles()}
    print(f'{args.n} points: lower hull of {len(t.triangles())} triangles, {len(t.redundant)} redundant points, '
          f'{t.flips} flips, {t.removals} removals, {len(t.stuck)} illegal edges stuck; '
          f'{"the same as" if same else "differs from"} the incremental version')


if __name__ == '__main__':
    main()
