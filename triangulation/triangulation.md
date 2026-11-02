Let's build something else now.

Create a python script that will implement a divide and conquer triangulation procedure and visualize the result via matplotlib. We start with N uniformly distributed points (for the example N may be 27) in 2d Euclidean space E^2.

Recursion "divide" part in the common case if there are more than 3 points.:
a) Choose a center mass line (any of them). Points to the "left" and to the "right of this line are chosen to their own triangulation procedure call.

Recursion leaf handling (to return triangulation result and its outer hull):
1)  One point is just a point, itself. Its outer hull is hull is one point, itself. Recursion deeper cannot continue. Simple.
2) Two points is just a segment. Its outer hull is one segment, itself. Recursion deeper cannot continue. Simple.
3) Three points is a triangle. It outer hull is one triangle, itself. Recursion deeper cannot continue. Not so simple because outer hull must be in counter clockwise order.

Recursion "conquer" / merge part.
<This will be somewhat research-ish part>
0) Project outer hull points onto the split line.
1) Observation 1: all points on the visible sides of each hull are candidates for adding into triangulation but the list of is not exhaustive in some cases.
2) Observation 2: sequence of points of outer hull between farthest projection points onto the split line is on the visible side if this sequence contains the shortest distance (to the split line) and scalar product of split line `s` and `v_k= p_(k+1) -p_k` points of outer hull segments between farthest points all the same sign, i.e., `s * v_k >= 0` for the left side hull (point order is counter clockwise), `s * v_k <= 0` for points of the visible side of the right side.
3) Observation 3a: outer hulls built by our "divide and conquer" algorithms are convex hulls.
4) Observation 3b: just two segments are needed to connect right side and left side hulls to combined outer hull that is convex hull.
5) Observation 4:
Let's change coordinates. Split line becomes axis 0x, points projected onto the split line get their x coordinate. Distance to line is y coordinate. Let's assume that lowest x coordinate of the right (former right that is now the down side) side is 0. It is naive and incorrect to assume that lowest x left point can be connected to lowest x point of right side.
Let's assume lowest x point of left side is greater than zero. Farthest right side point lowest (=0) point may not have unobstructed (by edge to the adjacent point of right side hull) view on lowest point of left side. In that case iterate right side point on hull backward until unobstructed view of lowest x point of left side is achieved.
We now start looking from unobstructed view point of right side and lowest x point of left side. Try iterating left side point backwards while unobstructed view point of right side is still visible. Try iterating right side point backwards while max backside point is visible. And so forth. That is up to quadratic time of visited hull nodes on both sides. See example below. Or is here faster incremental step algorithm? Angle between any two edges facing inside of the hull must be less than or equal to pi.

Image example: right side is a fine precision (many many many edges) circle  while left side a fine precision semicircle with flat bottom. Convex will he closed by line tangential to two circles (that is circle and round part of upper semicircle).

Or (big OR). Use Graham scan algorithm for convex hull combining right side visible hull points between lowest x to highest x with left side hull points between lowest x and highest y on the back invisible side of left hull. This should (should?) be correct. All we need to find is just one edge bridging two hulls. This may be a hint for early stop.

Left side points between lowest x point to bridging hull edge are added to triangulation connection candidates.

Same logic applies to find the other edge bridging two outer hull.

6) Observation 5: outer hull of triangulation minimizing sum of angles is a convex hull. So, two edges found using observation 4 must be present in triangulation.
What if we can start with bridging edge (containing 2 points from two sides). Keep adding candidate points to triangulation and checking whether edge flip is need. When two triangles share an edge, two circumcircles around of each of these two triangles should not contain leftover point of other triangle. Triangles should swap inner edge with previously not used diagonal edge if  circumcircle of one triangle included 3rd point of other triangle. Up to 4 (that is if they are not on the outer hull) outer edges should be checked against adjacent triangle for need to rebalance if new circumcircles started to include other triangle points.