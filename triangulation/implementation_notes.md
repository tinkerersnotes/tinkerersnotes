# Delaunay triangulation: implementation notes

Notes on building two Delaunay triangulations in pure Python with numpy, drawn with matplotlib:

- [divide_and_conquer.py](divide_and_conquer.py), from the prompt in [triangulation.md](triangulation.md);
- [incremental.py](incremental.py), points inserted one by one into a big virtual triangle.

Both also build the dual, the Voronoi diagram. [moving_points.html](moving_points.html) runs a JavaScript port
of the divide and conquer version on moving points, every frame, and is published on the GitHub Pages site.

For each: what was built, what went wrong along the way, and how the result was tested. Then how the two
Python versions compare in speed. The notebooks [divide_and_conquer.ipynb](divide_and_conquer.ipynb) and
[incremental.ipynb](incremental.ipynb) run them on the same four point sets.

Both share one edge store. A dictionary maps a directed edge `(a, b)` to the apex `c` of the
counterclockwise triangle `a, b, c`; the triangle across the edge is under `(b, a)`, so a flip is two
removals and two insertions. Both also share the flips, `legalize()`, which call the circle test through
a `self.inside(a, b, c, d)` method so that the incremental version can override it.

# Part 1: divide and conquer

## What was built

- **Divide.** A line through the points' centre of mass, along the direction they spread the least
  (the smallest eigenvector of their covariance), so it cuts across their widest extent. Points on
  its left go to one recursive call, points on its right to the other. The direction is turned to point
  along +x, so the left half is the upper one, as in the prompt's coordinates for the merge.
- **Leaves.** One point, a segment, or a triangle made counterclockwise with one orientation test.
  Each is its own convex hull.
- **Conquer.** Find the two bridges between the halves' hulls, zip the gap between them with
  triangles, and flip every edge that breaks the empty circle rule until none does.
- **Picture.** Left: the split lines by recursion depth and the leaves. Right: the result, the hull,
  the top split line and its two bridges.

## Comments on the observations

**0. Project hull points onto the split line.** It's only needed to choose where the bridge search
starts: the point of each hull nearest the split line. After that, orientation tests do all the work and
nothing needs projecting.

**1. Visible sides give candidates, but not all of them.** Confirmed, in two ways:

- The part of a hull facing the gap can be the *whole* hull. That happens when both bridges end at
  the same vertex; see the second bug below.
- The final Delaunay edges across the split can end at points *inside* a half's hull, not on it. The
  zip only connects hull points; flips reach the rest.

**2. The visible sequence lies between the farthest projections, with edges all one way along the
line.** That holds in the common case but isn't exact. The facing part of a hull runs from one bridge
end to the other, and the bridge ends can lie past the farthest projections. Take a narrow upper half
and a lower half much wider to the left: the left bridge leans outwards and touches the upper hull on
its top side, past its leftmost point. In the extreme case, both bridges end at one vertex and the
facing part is the whole hull, with edges running both ways. The bridge walk finds the facing part
directly, so no sign condition is needed.

**3a. The hulls built by the recursion are convex hulls.** Yes, by induction. Leaves are trivially
convex, and the joined hull is the lower half's back side, then a bridge, the upper half's back side,
and the other bridge.

**3b. Two segments join the hulls.** Yes, exactly two (the upper and lower common tangents). That
relies on the halves being separated by a line, which the split guarantees.

**4. Finding a bridge: up to quadratic?** It's linear. Start from the two points nearest the split line.
Step the upper end while its neighbour is on the wrong side of the bridge, then the lower end likewise,
and repeat until neither moves. Neither end ever steps back, so a bridge costs at most one step per
hull vertex. The other bridge is the same walk in the other direction. In the prompt's circle and
semicircle example, the walk goes from the nearest points straight to the tangent points.
Binary-search versions find a bridge in logarithmic time (Overmars and van Leeuwen), but the zip after
it is linear anyway.

The *naive and incorrect* pairing of the two lowest-x points is indeed wrong, and the walk is the fix.
The Graham scan alternative is correct, and linear if the two chains are merged in order of projection.
But it builds the whole joined hull when only two edges of it are needed.

**5. Bridges belong to the triangulation; add points and flip.** Every triangulation of a point set has
the convex hull as its boundary, so the bridges are in all of them. Delaunay doesn't minimise the sum of
angles, which is π per triangle and the same for every triangulation of the set (always 2n − h − 2
triangles for n points with h on the hull). It maximises the smallest angle.

The flip approach is correct. If every edge is *locally* Delaunay (the circle through one of its
triangles doesn't contain the other triangle's far point), the whole triangulation is Delaunay. So:

- After each zip triangle, its edges are checked.
- After each flip, the four outer edges of the quadrilateral are checked, as the prompt describes.
- Any edge not checked kept both of its triangles, so it stays legal.
- When nothing is left to check, the result is Delaunay.

Flips do reach into the halves and remove old edges. That is the work Guibas and Stolfi do in their
merge by deleting edges as the zip goes, which avoids flips entirely.

## The zip

The gap between the bridges is a polygon bounded by the two facing chains. Its triangles are built
upwards from the low bridge:

- A base edge `l-r` climbs to the high bridge.
- Each step adds the triangle with the next point of either the upper chain (`l2`) or the lower one
  (`r2`).
- If both fit, the choice is Guibas and Stolfi's: the triangle whose circumcircle leaves the other
  candidate out. That keeps the number of flips low.

A candidate fits when:

1. the triangle turns counterclockwise, which keeps it outside the hull the new point comes from, and
2. its corner at the base's *other* end doesn't overlap that hull's own corner there. Two angles under
   π at one vertex overlap exactly when one starts strictly inside the other.

One candidate always fits. A diagonal between two points of the same chain would lie inside that
convex hull, so every diagonal of the gap crosses from one chain to the other. So the base's triangle,
in any triangulation of the gap, uses `l2` or `r2`.

## Bugs found by testing

1. **The "in sight" test was too strict.** The first version required the new point to be past the
   line through the other chain's next edge. Where a chain bends away, that line fails both candidates
   even though a triangle fits, and the zip got stuck. It was replaced by the corner-overlap test
   above, which is exact.
2. **Both bridges at one vertex.** I had assumed the rest of that hull was then on the outside. It's the
   opposite: the joined hull keeps only that vertex of the hull, so the rest lies in the gap, and the zip
   must go all the way around it. A segment poking into the gap, zipped along both its sides, is the
   two-point version of the same case.
3. **Eigenvector sign.** `eigh` returns either sign, so the "left" half sometimes came out below the
   line. That was cosmetic but confusing next to the prompt's picture. The direction now points
   along +x.
4. **Points on one circle.** The circle and semicircle dataset puts 100 points exactly on each curve,
   breaking the "no four on a circle" assumption. For such points the circle test returns rounding
   noise around zero, and an edge could flip back and forth forever. Any triangulation of points on
   one circle is Delaunay, so ties don't matter; the test just has to ignore the noise. The first fix,
   a fixed tolerance of 1e-12 × (extent of the points)⁴, turned out to be wrong; the incremental version
   exposed it (bug 1 there). The test now ignores differences below 1e-12 of the sum of its own terms'
   sizes, the way robust geometry code bounds rounding error. That dataset needs no flips at all.

Still assumed: no three points on a line. Collinear points would make a leaf a degenerate triangle,
and orientation ties in the bridge walk.

## Testing

scipy isn't in the venv, so instead of comparing with `scipy.spatial.Delaunay`, each result is checked
against properties every correct answer must have:

- **Triangle count** 2n − h − 2, with h the hull size.
- **Orientation**: every triangle is counterclockwise.
- **Coverage**: the triangles' total area equals the hull's area. Together with the count and the
  orientation, this rules out overlaps and holes.
- **Hull**: every point lies on the left of (or on) every hull edge, so the hull is convex,
  counterclockwise and contains everything.
- **Delaunay, by brute force**: no point lies inside any triangle's circumcircle, which is O(n × triangles).
  `illegal_edges()` also does the cheap local check that the flips rely on.

The datasets were 3000 random sets with fixed seeds, 3 to 79 points each, in three families:

- uniform in a square;
- long and thin (normal, scaled 5 : 0.2), which makes skewed splits and long bridges;
- in a thin ring near a circle, where most points are on the hulls and the facing chains are long.

The four notebook datasets passed the same checks, including the exact points on one circle. All of it
was rerun after each change to the circle test.

When a merge failed, a small harness wrapped `Triangulation.merge` and printed:

- the two hulls with coordinates;
- the bridges, and whether every point is on the correct side of each;
- the facing chains;
- the orientation tests at the stuck base.

That's how both zip bugs were tracked down within one or two runs each.

The pictures were rendered headless (`--save`, matplotlib's Agg backend) and checked by eye. The
notebook was run end to end with `jupyter nbconvert --execute`.

### Numbers

| Points | Triangles | Hull | Flips |
|---|---|---|---|
| 27 uniform (seed 1) | 41 | 11 | 20 |
| 225 uniform (seed 1) | 432 | 16 | 227 |
| 300 in a donut (seed 1) | 573 | 25 | 303 |
| 100 on a circle + 100 on a semicircle | 282 | 116 | 0 |
| 1000 uniform (seed 1) | 1982 | — | 1191 |

That's roughly 1.3 flips per point in total over all the merges.

# Part 2: incremental

## What was built

As specified:

- **1, 2 or 3 points** are already finished: a point, a segment, or a triangle made counterclockwise.
- **The virtual triangle** is equilateral, its incircle the circumcircle of the points' bounding box.
  Its corners are twice the incircle's radius from the centre. A `--scale` option multiplies that
  radius.
- **Locating a point.** The walk starts at the first virtual corner. While the point lies behind an
  edge of the current triangle (its orientation against the edge is negative), cross the edge it
  lies furthest behind, measured as the orientation, so distance times edge length. In a Delaunay
  triangulation such a walk can't go round in circles, and the triangulation is Delaunay again after
  every insertion. Each vertex keeps one edge it has, where walks can start. A split or a flip only
  replaces triangles with triangles at the same corners, so that edge always exists.
- **Inserting** splits the containing triangle into three around the point, then checks the three outer
  edges against their neighbours. Each flip checks the four edges around it, the same wave as in the
  divide and conquer merge, with the same code.
- **Finishing.** Remove every triangle with a virtual corner, then walk the outer boundary: edges with a
  triangle on their left only, each walked once.
- **Picture.** Left: the triangulation just before the virtual corners go, with the box and the
  incircle. Right: the result. Red fills the part of the true convex hull (from divide and conquer) that
  no triangle covers.

## A finite virtual triangle loses triangles on the hull

At the specified size, 547 of 600 random sets came out with Delaunay triangles missing and a boundary
that is not convex. Every triangle that is kept is correct; some are just missing.

**The mechanism.** The one triangle missing from the 27-point picture, measured:

```
missing triangle (8, 18, 9): hull edge 18-9 length 0.313, apex 8 only 0.0147 from it
  circumradius 0.84 (the points span 1), reaches 1.67 past the hull edge
  virtual corner 28: 0.14 from the circle's centre, INSIDE
  kept instead: (8, 18, 28), (8, 28, 9)
```

1. **A sliver on a hull edge has a huge circumcircle.** For an apex at height `h` over an edge that its
   foot divides into `p` and `q`, the circle reaches about `p·q / h` past the edge (intersecting chords):
   here 0.156 × 0.157 / 0.0147 ≈ 1.67, more than the point set is wide. As `h` goes to 0 it has no
   limit.
2. **Outside the hull, the only points are the virtual corners.** Inside, other points keep a Delaunay
   circle small, since it must be empty. Beyond a hull edge nothing does, so the bulge reaches a corner.
3. **The flip does what it should.** The corner is inside the triangle's circle, so the hull edge
   `18-9` is illegal and flips to `8-28`.
4. **Removing the corners takes both new triangles.** The hull edge is gone and the boundary runs
   `18 → 8 → 9`: a dent at 8, and a red sliver.

So the result is the Delaunay triangulation of the points *plus the corners*, and the corners are close
enough to change it. "Every point is inside the virtual triangle" isn't the condition that matters.
What matters is that no corner is inside any Delaunay circumcircle, and that depends on the flattest
triangle on the hull, which can be arbitrarily flat. When a hull point is nearly in line with both of its
hull neighbours, all its triangles can go, and the point is lost.

**A bigger triangle only makes it rarer.** On 600 random sets of 4 to 119 points (uniform, thin and
ring, 200 each):

| Scale | Sets with triangles missing | Triangles missing | Boundary not convex | Sets with points lost |
|---|---|---|---|---|
| 1 (as specified) | 547 | 6.7% | 536 | 84 (3 lost everything) |
| 3 | 475 | 3.0% | 469 | 17 |
| 10 | 341 | 1.2% | 341 | 1 |
| 100 | 69 | 0.13% | 69 | 0 |
| 1000 | 6 | 0.01% | 6 | 0 |
| 10⁶ | 0 | 0 | 0 | 0 |

A triangle survives roughly while `h > p·q / D`, with `D` the corner's distance from the edge. Scaling
moves `D` but not the slivers, and random points sometimes lie almost exactly in line with a hull edge.
Very large corners also cost floating point precision. Long, thin sets fail most (200 of 200 at scale 1):
their hulls have long, flat edges. The circle and semicircle set is the worst of the notebook ones: its
long tangents, with the arcs hugging them, make a run of slivers, and it loses 27 of 282 triangles.

**A square bounding box doesn't help.** Its circumcircle is the rectangle's times
√2 · max(w, h) / √(w² + h²), between 1 and 1.41, so it is the same as a scale of at most 1.41. Measured on the
same sets: 546 instead of 547 sets with triangles missing, and 12% fewer missing triangles. The thin sets
gain most, since their factor is the full 1.41.

## Corners at infinity

`--scale inf` puts corner `k` at `centre + λ u_k`, with `u_k` its direction, as `λ` grows without bound:

- Every test involving a corner is a polynomial in `λ`: degree 2 for the orientation test, up to 4 for
  the circle test. Its sign is the sign of its highest nonzero coefficient, as for large enough `λ`.
- Tests among real points only are unchanged.
- The orientation test returns its coefficients as a tuple, `(λ², λ, 1)`. Tuples compare the same way as
  the values do for large `λ`, so the walk's "furthest behind" choice still works.

Geometrically, the circle through a hull edge and a corner at infinity becomes the half-plane beyond the
edge. No point is ever in it, so no hull triangle loses to a corner. The result is exactly the Delaunay
triangulation, with the convex hull as its boundary. The picture draws edges to a corner at infinity as
parallel rays.

The default stays scale 1, so both versions can be compared. The notebook draws each set both ways.

## Bugs found by testing

1. **A fixed circle-test tolerance.** The first tolerance, 1e-12 × (extent)⁴, was scaled to the whole
   point set, virtual corners included. On a thin set it grew to 6e-6, larger than a real violation of
   1.8e-6, so a needed flip was skipped. The comparison with divide and conquer caught it: a kept
   triangle that wasn't Delaunay, which should be impossible. One fixed tolerance can't suit both large
   and small triangles. The test is now relative to its own terms, in both versions; divide and conquer
   had the same weakness, hidden.
2. **A boundary walk that never ended.** When the boundary pinched through a vertex twice, following "the
   first boundary edge out" could circle a loop that never returned to the start, appending until the
   process was killed for memory. Now each boundary edge is used once. Every vertex has as many boundary
   edges in as out, so such a walk can only end back at the start.
3. **No triangle left at all.** At scale 1, three sets lost every triangle. The boundary walk now returns an
   empty hull for that.
4. **Comparing triangles one by one.** The first picture filled divide and conquer's triangles that the
   result lacked. For the circle and semicircle that marked 192 triangles that were just a different valid
   choice: points on one circle have many Delaunay triangulations. The picture now fills the uncovered part
   of the convex hull instead, which doesn't depend on the choice.
5. **Cancellation at infinity.** The first version of the polynomial circle test broke the triangulation. When
   the point being tested was itself a corner, everything was measured relative to it. The λ⁴ and λ³ terms
   then cancel exactly on paper but leave rounding noise, and that noise decided the test. Swapping two of
   the four points only flips the test's sign. So a real point is swapped in as the reference, and then
   each leading coefficient is one geometric quantity, not a difference of nearly equal ones.

## Testing

Divide and conquer, already tested, was the reference:

- **The scale table** above compared triangle sets on 600 random sets. It also checked the invariant behind
  the analysis: no kept triangle was ever outside the Delaunay triangulation, and no edge was illegal, at
  any scale.
- **Corners at infinity** gave triangle sets identical to divide and conquer on the 3000 random sets
  of part 1.
- **The notebook sets and 2000 random points** were checked by property at infinity, since the circle and
  semicircle has no unique answer. Checks: the full triangle count, a convex boundary equal to the true
  hull, no lost points, all triangles counterclockwise, and (up to 300 points) the brute force empty
  circle test.

After the runaway boundary walk, test runs went under a memory limit (`ulimit -v`) and a `timeout`, so a
bug would fail instead of taking the machine down.

One visual check needed pixels. The rays in a downscaled preview looked grey; reading the PNG's pixels
showed pure orange. The preview was wrong, not the drawing.

### Numbers

The notebook sets, scale 1 → corners at infinity:

| Points | Triangles | Boundary | Flips | Walk steps |
|---|---|---|---|---|
| 27 uniform (seed 1) | 40 → 41 | 12, dented → 11, convex | 41 → 44 | 109 → 109 |
| 225 uniform (seed 1) | 429 → 432 | 19 → 16 | 603 → 620 | 3063 → 2630 |
| 300 in a donut (seed 1) | 568 → 573 | 30 → 25 | 844 → 846 | 4177 → 3996 |
| 100 on a circle + 100 on a semicircle | 255 → 282 | 143 → 116 | 1659 → 2187 | 3919 → 3472 |

# Performance

Uniform random points, one run each, pure Python on the Linux VM (expect ±10% noise). The incremental
version has corners at infinity. "Sorted" is a variant used only in the benchmark: points inserted along a
snake through a grid of cells, each walk starting at the previous point.

| n | Divide and conquer | Incremental | of which walk | Walk steps per point | Scale 1 | Sorted | Its steps per point |
|---|---|---|---|---|---|---|---|
| 1 000 | 0.08 s | 0.21 s | 58% | 25 | 0.15 s | 0.21 s | 3.4 |
| 2 000 | 0.16 s | 0.48 s | 65% | 33 | 0.35 s | 0.38 s | 3.5 |
| 5 000 | 0.38 s | 1.51 s | 72% | 51 | 1.22 s | 0.93 s | 3.8 |
| 10 000 | 0.78 s | 3.91 s | 79% | 83 | 3.14 s | 1.82 s | 4.1 |
| 20 000 | 1.56 s | 9.47 s | 82% | 106 | 8.29 s | 3.62 s | 4.3 |
| 50 000 | 3.94 s | 37.27 s | 88% | 186 | 31.33 s | 9.12 s | 4.5 |

- **Divide and conquer grows almost linearly**: 2.5 times the points from 20 000 to 50 000 took 2.53 times
  as long. It flips about 1.3 to 1.4 edges per point in total.
- **The incremental walk grows like √n.** Every walk starts at a virtual corner and crosses about √n
  triangles, so the whole run grows like n^1.5: 2.5 times the points took 3.94 times as long, and
  2.5^1.5 = 3.95. At 50 000 points the walk is 88% of the time, and divide and conquer is 9.5 times faster.
- **The flips are not the problem.** Incremental flips about 3 edges per point, more than divide and
  conquer, but constant in n. A new point starts with 3 neighbours and gains one per flip, and ends up
  with about 6 on average, hence about 3 flips. Without the walk, incremental at 20 000 points would take
  about 1.7 s, close to divide and conquer's 1.56 s.
- **Corners at infinity cost 15 to 25%** over scale 1: polynomial circle tests are slower than plain ones.
  Since scale 1 is wrong, that is the price of a correct answer.
- **Spatial order fixes the walk.** Sorted, walks stay around 4 steps per point at any n, the run becomes
  linear, and incremental stays about 2.3 times slower than divide and conquer.

These are pure Python timings: the ratios matter more than the seconds. Compiled, both would be far
faster, and the gap would likely narrow, since each walk step costs a lot in Python.

# Part 3: the dual, the Voronoi diagram

For each triangle, take its circumcentre, and join it to the circumcentres of its up to three neighbours
across its edges. That is the Voronoi diagram: each Delaunay edge has exactly one dual edge, perpendicular
to it, made of the points equally far from its two ends and further from every other point. A hull edge
has only one triangle, so its dual is a ray to infinity. As asked, it is drawn as a short stump to a virtual
vertex outside the hull.

- **`Triangulation.circumcentre(a, b, c)`**, measured from `a` to keep the numbers small.
- **`Triangulation.voronoi(stump)`** returns the joined circumcentres for inner edges and the stumps for
  hull edges, one entry per Delaunay edge. The incremental class borrows both methods.
- **Where a stump goes.** A circumcentre lies on the perpendicular bisector of each of its triangle's
  edges. So the ray for a hull edge starts at the circumcentre, runs along the edge's outward normal and
  passes through the edge's midpoint. The stump ends `stump` past the edge's midpoint, or past the
  circumcentre when that is already outside the hull, which happens for a triangle obtuse opposite the
  hull edge. Either way it ends outside.
- **Picture.** A third panel, "Dual", in both scripts, drawn by `draw_dual()` in divide_and_conquer.py. Thin
  triangles have far away circumcentres, so the view stays on the points with a 15% margin.

Points on one circle share one circumcentre. For the circle and semicircle set, the dual of each shape's
inside collapses to one point, its centre, with the stumps radiating out like spokes. The edges counted
there include many of zero length, between triangles sharing a centre. For the incremental version at scale 1, the
stumps follow the dented boundary: 143 for the circle and semicircle instead of 116 hull edges.

**Testing**, on 1500 random sets in the three families:

- dual edges and stumps together number 3n − h − 3, one per Delaunay edge, with one stump per hull edge;
- each circumcentre is equally far from its triangle's three corners;
- each dual edge is perpendicular to its Delaunay edge;
- at the middle of each dual edge, the nearest points are that edge's two ends, the Voronoi property;
- every stump ends outside the hull.

With corners at infinity, the incremental version gives the same diagram as divide and conquer.

# Part 4: the moving points page

[moving_points.html](moving_points.html) is a single page that opens from disk, published by the Pages
workflow next to the seam carving demos.

- **The triangulation** is the divide and conquer version ported to JavaScript, rebuilt from scratch every
  frame, since divide and conquer has no incremental update. Points are indices into coordinate arrays.
  A directed edge's key is `a·n + b` in a `Map`. The port keeps the relative circle test and the zip's
  corner test, and adds guards: a walk or a flip count that runs away throws. If a frame hits a
  degenerate case, the page skips its triangles instead of freezing, and the status line counts such
  frames.
- **Controls.**
  - A points slider, 3 to 2000. Sliding keeps the moving points and adds or drops some.
  - Play, to pause and resume, and Reset, for new uniform positions, speeds and directions.
  - Show Delaunay and show Voronoi.
  - Own colours, off by default: each point gets a random hue, and each Delaunay edge fades from one end's
    colour to the other's. The Voronoi diagram stays orange.
- **Drift and bounce.** Each point keeps the speed (3% to 12% of the box's shorter side per second) and
  direction it got at the start. A wall reflects its position and turns its velocity around. A frame
  advances at most 0.05 s, so a background tab doesn't make points jump.
- **Repel, up to 50 points.** Every pair pushes apart with acceleration `k / d²`, `k = 0.0003 L³` for the
  box's shorter side `L`, which takes n² work per step; the slider's maximum drops to 50. Distances are
  softened within `0.02 L`, so a near miss can't fling a point away. Friction, `dv/dt = a − 0.5 v`, applied
  as an exact decay over each 1/240 s step, keeps speeds bounded. Walls still bounce. The constants come
  from a simulation of 50 points:
  - Without friction, speeds stayed near the drifting ones, but nothing bounded them in principle.
  - With friction 0.5 per second, points start about as fast as drifting ones. They slow down by about
    10 s and come to rest by about 30 s, about two thirds of them pressed towards the walls, as
    repelling charges in a box are.
  - Stronger friction settles faster but leaves the start sluggish.

**Testing.**

- **The JavaScript port against divide_and_conquer.py**, on 1500 random sets of 1 to 119 points. The
  Python side wrote the points and its results to JSON, and Node ran the code between the page's
  `BEGIN triangulation` and `END triangulation` markers. Triangle sets and hulls were identical, and so
  were the Voronoi edges and stumps: none of 245 617 segments differed after rounding to 1e-6 of the set's
  size.
- **The page in the browser pane:** pause, reset, the slider, the 50-point limit and its release, the
  show checkboxes and own colours. After 20 simulated seconds of repelling, no point was outside the box
  and no coordinate was NaN.
- **Speed.** A triangulation takes 0.2 ms for 150 points and 4.3 ms for 2000, in Node. Drawing 2000
  points takes about 23 ms in grey and 28 ms in own colours in the VM's browser pane, since each colour
  gradient is its own stroke.

Two traps in testing, worth knowing next time:

- **A hidden browser pane gets few animation frames.** Points barely moved and the status lagged. That was
  the pane, not the page: velocities read back right, and frames came at 41 a second once the pane was
  shown.
- **A screenshot can be stale.** One showed checkboxes unchecked that the page reported checked, and two
  clicks placed from it landed on other controls. Reading the state from the page after each click, and
  taking a fresh screenshot, settled it.

# Open threads

- **Commit the checks and the benchmark.** The property checks, the comparison between the two
  versions, and the benchmark live only in a scratch folder.
- **Shorter walks.** Insert in spatial order and walk from the last point, as the benchmark's sorted
  variant does. Combined with randomness (BRIO, biased randomised insertion order), that keeps walks
  short without sorted order's bad cases.
- **Flip counts against n.** How many flips does a merge need: a constant per point, as the numbers
  suggest, or worse on adversarial inputs? Lawson's flips are O(n²) in the worst case, starting from
  an arbitrary triangulation.
- **Guibas–Stolfi merge.** Delete edges instead of flipping, for comparison: same result, no flips.
- **Splits.** The centre of mass doesn't guarantee even halves on skewed point sets. The median along
  the widest direction would. Alternating cuts (Dwyer) are faster in expectation for uniform points.
- **Exact predicates.** Floating point orientation and circle tests can be wrong for nearly
  degenerate inputs; Shewchuk's adaptive predicates make them exact.
- **Collinear points.** Degenerate leaves and ties in the bridge walk.
- **Animation.** Showing the zip and the flips of each merge step by step, or the flip wave of each
  insertion, would make a good talk visual. The moving points page animates the result, not the steps.
- **Faster repulsion.** The page limits repelling to 50 points because every pair interacts. A grid of
  cells, or a Barnes–Hut tree, would make it n log n.
- **Voronoi cells.** The dual is drawn as edges. Filling each point's cell, in its own colour, would need
  the cell polygons, clipped to the box at the hull.
