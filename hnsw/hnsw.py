#!/usr/bin/env python3
"""Approximate nearest neighbour search with Hierarchical Navigable Small World graphs (HNSW).

Follows Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using
Hierarchical Navigable Small World graphs", https://arxiv.org/abs/1603.09320. Algorithm numbers
below are the paper's.

Layers: every element gets a random top layer l = floor(-ln(U) * mL), so each layer holds about
1/M of the elements of the layer below (mL = 1/ln M). Layer 0 holds them all. Each layer is a
proximity graph over its elements: up to M links per element on the upper layers, 2M on layer 0.

Search (alg. 5): start at the entry point on the top layer and walk greedily towards the query,
always to a closer neighbour, until no neighbour is closer (alg. 2 with ef = 1). That element is
the entry point one layer down. On layer 0 the walk keeps a list of the ef closest elements found
so far and expands them nearest first, which backtracks out of local minima.

Insert (alg. 1): search for the new element as if it were a query, down to its own top layer,
then on each of its layers pick its neighbours among the efConstruction closest found (alg. 3:
the M nearest, or alg. 4: a heuristic) and link both ways. A neighbour with too many links is
pruned back by the same selection.

The heuristic (alg. 4) takes candidates nearest first and keeps one only if it is closer to the
new element than to every one already kept. That is the relative neighbourhood graph's rule,
whose graph is a subgraph of the Delaunay graph: links spread out in diverse directions instead
of bunching up on one side, and clusters stay connected to each other.

Distances are squared Euclidean ('l2', which orders the same as Euclidean) or 1 - cos ('cosine',
vectors normalised once when added). Vectors are numpy rows; the graph is plain Python lists.

    python hnsw.py [N] [--dim D] [--queries Q] [--k K] [--seed SEED]
"""
import argparse
import heapq
import math
import random
import time

import numpy as np


class HNSW:
    def __init__(self, dim, M=16, ef_construction=100, metric='l2', heuristic=True,
                 extend_candidates=False, keep_pruned=False, m_max0=None, m_l=None, seed=None):
        if metric not in ('l2', 'cosine'):
            raise ValueError(f'unknown metric {metric!r}')
        self.dim = dim
        self.M = M
        self.m_max = M                                       # links per element on layers above 0
        self.m_max0 = 2 * M if m_max0 is None else m_max0    # links per element on layer 0
        self.m_l = 1 / math.log(M) if m_l is None else m_l   # level normalisation
        self.ef_construction = ef_construction
        self.metric = metric
        self.heuristic = heuristic                  # alg. 4 if true, alg. 3 otherwise
        self.extend_candidates = extend_candidates  # alg. 4 flags
        self.keep_pruned = keep_pruned
        self.rng = random.Random(seed)
        self.vectors = np.empty((16, dim))
        self.count = 0
        self.links = []         # links[i][layer]: ids of element i's neighbours on that layer
        self.entry = None       # entry point, an element on the top layer
        self.top = -1           # top layer
        self.distance_count = 0  # distances computed by searches, inserts' searches included

    def __len__(self):
        return self.count

    def level(self, i):
        """Element i's top layer."""
        return len(self.links[i]) - 1

    def prepare(self, vector):
        v = np.asarray(vector, dtype=float).reshape(self.dim)
        if self.metric == 'cosine':
            v = v / np.linalg.norm(v)
        return v

    def distances(self, q, ids):
        """Distances from vector q to the elements ids, as a list of floats."""
        self.distance_count += len(ids)
        v = self.vectors[ids]
        if self.metric == 'l2':
            v = v - q
            return np.einsum('ij,ij->i', v, v).tolist()
        return (1 - v @ q).tolist()

    def pairwise(self, ids):
        """Distances between the elements ids, as a list of rows."""
        v = self.vectors[ids]
        gram = v @ v.T
        if self.metric == 'l2':
            norms = np.diag(gram)
            return (norms[:, None] + norms[None, :] - 2 * gram).tolist()
        return (1 - gram).tolist()

    def search_layer(self, q, entry, ef, layer, trace=None):
        """Alg. 2: the ef elements closest to q found by a best first walk on one layer.

        entry and the result are lists of (distance, id), the result sorted nearest first. The walk
        expands the nearest unexpanded candidate until it is farther than the farthest of the ef
        found. A trace list, if given, gets the elements expanded in order and those visited.
        """
        visited = {e for _, e in entry}
        candidates = list(entry)                 # min-heap on distance
        heapq.heapify(candidates)
        found = [(-d, e) for d, e in entry]      # max-heap on distance
        heapq.heapify(found)
        expanded = []
        while candidates:
            d, c = heapq.heappop(candidates)
            if d > -found[0][0]:
                break  # every element in found is expanded or closer
            expanded.append(c)
            new = [e for e in self.links[c][layer] if e not in visited]
            if not new:
                continue
            visited.update(new)
            for de, e in zip(self.distances(q, new), new):
                if len(found) < ef or de < -found[0][0]:
                    heapq.heappush(candidates, (de, e))
                    heapq.heappush(found, (-de, e))
                    if len(found) > ef:
                        heapq.heappop(found)
        if trace is not None:
            trace.append({'layer': layer, 'expanded': expanded, 'visited': visited})
        return sorted((-d, e) for d, e in found)

    def select_neighbours(self, q, candidates, m, layer, base=None):
        """Alg. 3 or 4: up to m of the candidates, a list of (distance to q, id), as q's neighbours.

        base is q's own id when q is already in the graph, kept out of the extended candidates.
        """
        candidates = sorted(candidates)
        if not self.heuristic:
            return candidates[:m]
        if self.extend_candidates:
            seen = {e for _, e in candidates}
            seen.add(base)
            extra = []
            for _, e in candidates:
                for a in self.links[e][layer]:
                    if a not in seen:
                        seen.add(a)
                        extra.append(a)
            if extra:
                candidates = sorted(candidates + list(zip(self.distances(q, extra), extra)))
        between = self.pairwise([e for _, e in candidates])
        kept, discarded = [], []   # indices into candidates
        for j, (d, _) in enumerate(candidates):
            if len(kept) == m:
                break
            # Kept if closer to q than to every element kept so far
            if all(d < between[j][k] for k in kept):
                kept.append(j)
            else:
                discarded.append(j)
        if self.keep_pruned:
            kept += discarded[:m - len(kept)]
        return [candidates[j] for j in sorted(kept)]

    def add(self, vector):
        """Alg. 1: insert one vector, returning its id."""
        q = self.prepare(vector)
        i = self.count
        if i == len(self.vectors):
            self.vectors = np.resize(self.vectors, (2 * i, self.dim))
        self.vectors[i] = q
        self.count += 1
        level = int(-math.log(1 - self.rng.random()) * self.m_l)
        self.links.append([[] for _ in range(level + 1)])
        if self.entry is None:
            self.entry, self.top = i, level
            return i

        entry = [(self.distances(q, [self.entry])[0], self.entry)]
        for layer in range(self.top, level, -1):
            entry = self.search_layer(q, entry, 1, layer)[:1]
        for layer in range(min(self.top, level), -1, -1):
            found = self.search_layer(q, entry, self.ef_construction, layer)
            neighbours = self.select_neighbours(q, found, self.M, layer, base=i)
            self.links[i][layer] = [e for _, e in neighbours]
            m_max = self.m_max0 if layer == 0 else self.m_max
            for _, e in neighbours:
                links = self.links[e][layer]
                links.append(i)
                if len(links) > m_max:
                    # Shrink e's links by the same selection, with e as the base
                    v = self.vectors[e]
                    pruned = self.select_neighbours(
                        v, list(zip(self.distances(v, links), links)), m_max, layer, base=e)
                    self.links[e][layer] = [x for _, x in pruned]
            entry = found
        if level > self.top:
            self.entry, self.top = i, level
        return i

    def add_items(self, vectors):
        return [self.add(v) for v in vectors]

    def search(self, query, k=1, ef=None, trace=None):
        """Alg. 5: ids and distances of the approximate k nearest elements, nearest first.

        ef, at least k, is the size of the found list on layer 0: bigger is slower and more
        accurate. Distances are Euclidean for 'l2'. A trace list gets one entry per layer, as in
        search_layer.
        """
        ef = max(k, ef or k)
        q = self.prepare(query)
        entry = [(self.distances(q, [self.entry])[0], self.entry)]
        for layer in range(self.top, 0, -1):
            entry = self.search_layer(q, entry, 1, layer, trace)[:1]
        found = self.search_layer(q, entry, ef, 0, trace)[:k]
        ids = np.array([e for _, e in found], dtype=int)
        dist = np.array([d for d, _ in found])
        if self.metric == 'l2':
            dist = np.sqrt(np.maximum(dist, 0))
        return ids, dist

    def search_many(self, queries, k=1, ef=None):
        """search for each row of queries: arrays of shape (queries, k) of ids and distances."""
        results = [self.search(q, k, ef) for q in queries]
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])

    def layer_elements(self, layer):
        return [i for i in range(self.count) if self.level(i) >= layer]

    def edges(self, layer):
        """Undirected edges of a layer's graph as (i, j) with i < j; a link either way counts."""
        return sorted({(min(i, j), max(i, j))
                       for i in self.layer_elements(layer) for j in self.links[i][layer]})

    def degrees(self, layer):
        return np.array([len(self.links[i][layer]) for i in self.layer_elements(layer)])


def exact_knn(data, queries, k, metric='l2', chunk=256):
    """Brute force: ids and distances of the exact k nearest rows of data for each query."""
    data = np.asarray(data, dtype=float)
    queries = np.asarray(queries, dtype=float)
    if metric == 'cosine':
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    norms = np.einsum('ij,ij->i', data, data)
    ids, dists = [], []
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]
        if metric == 'l2':
            d = norms[None, :] - 2 * q @ data.T + np.einsum('ij,ij->i', q, q)[:, None]
        else:
            d = 1 - q @ data.T
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(d, part, axis=1)
        order = np.argsort(part_d, axis=1)
        ids.append(np.take_along_axis(part, order, axis=1))
        dists.append(np.take_along_axis(part_d, order, axis=1))
    ids, dists = np.vstack(ids), np.vstack(dists)
    if metric == 'l2':
        dists = np.sqrt(np.maximum(dists, 0))
    return ids, dists


def recall(found, truth):
    """Share of the true k nearest neighbours that were found, over all queries."""
    return np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('n', nargs='?', type=int, default=10000, help='number of vectors')
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--queries', type=int, default=200)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = rng.normal(size=(args.n, args.dim))
    queries = rng.normal(size=(args.queries, args.dim))
    truth, _ = exact_knn(data, queries, args.k)

    index = HNSW(args.dim, seed=args.seed)
    start = time.perf_counter()
    index.add_items(data)
    print(f'{args.n} vectors of {args.dim} dimensions: built in {time.perf_counter() - start:.1f} s, '
          f'{index.top + 1} layers')
    for ef in (10, 20, 40, 80, 160):
        index.distance_count = 0
        start = time.perf_counter()
        found, _ = index.search_many(queries, args.k, ef)
        elapsed = time.perf_counter() - start
        print(f'ef {ef:4}: recall@{args.k} {recall(found, truth):.3f}, '
              f'{index.distance_count / args.queries:7.0f} distances and '
              f'{1000 * elapsed / args.queries:.2f} ms per query')


if __name__ == '__main__':
    main()
