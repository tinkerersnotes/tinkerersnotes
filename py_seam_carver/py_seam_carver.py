"""Pure Python seam carver, a line-by-line counterpart of cy_seam_carver.CySeamCarver.

Same public interface as the other seam carver modules, no build step needed.
"""
from math import sqrt

from PIL import Image


class PyMeshNode:
    # Fixed attribute set like CyMeshNode's cdef fields, and much less memory
    # per node than a __dict__ for a mesh of width * height nodes
    __slots__ = ('x', 'y', 'cost', 'r', 'g', 'b', 'up', 'down', 'left', 'right', 'prev', 'evaluated_epoch')

    def __init__(self, x, y, pixels):
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.x = x
        self.y = y
        self.prev = None
        self.cost = -1.0
        self.evaluated_epoch = 0  # Last incremental update that evaluated this node
        pix = pixels[x, y]
        self.r = pix[0]
        self.g = pix[1]
        self.b = pix[2]


class PySeamCarver:
    def __init__(self, image):
        self._image = image if image.mode == 'RGB' else image.convert('RGB')
        self._width, self._height = self._image.size
        self._pixels = self._image.load()

        self._mesh = []
        self._create_mesh_nodes()
        self._link_mesh_nodes()

        # Nodes whose children may need a cost update, one list per row
        self._update_heads = [[] for _ in range(self._height)]
        self._has_update_heads = False
        # Incremented once per removed seam, so evaluated_epoch never needs resetting
        self._epoch = 0

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def _node_index(self, x, y):
        return y * self._width + x

    def _create_mesh_nodes(self):
        if self._height <= 0 or self._width <= 0:
            raise ValueError('image must be non-empty')
        for y in range(self._height):
            for x in range(self._width):
                i = self._node_index(x, y)
                assert(len(self._mesh) == i)
                self._mesh.append(PyMeshNode(x, y, self._pixels))
        self._top_left = self._mesh[0]

    def _link_mesh_nodes(self):
        for y in range(self._height):
            for x in range(self._width):
                i = self._node_index(x, y)
                node = self._mesh[i]
                if x > 0:
                    left = self._mesh[self._node_index(x - 1, y)]
                    left.right = node
                    node.left = left
                if x < self._width - 1:
                    right = self._mesh[self._node_index(x + 1, y)]
                    right.left = node
                    node.right = right
                if y > 0:
                    up = self._mesh[self._node_index(x, y - 1)]
                    up.down = node
                    node.up = up
                if y < self._height - 1:
                    down = self._mesh[self._node_index(x, y + 1)]
                    down.up = node
                    node.down = down

    def _distance(self, n1, n2):
        r = n1.r - n2.r
        g = n1.g - n2.g
        b = n1.b - n2.b
        return sqrt(r*r + g*g + b*b)

    def _path_cost(self, parent, to_node):
        return (parent.cost if parent.cost >= 0.0 else 0.0) + self._distance(parent, to_node)

    def _update_cost(self, from_node, to_node):
        cost = self._path_cost(from_node, to_node)

        if to_node.cost < 0.0 or to_node.cost > cost:
            to_node.cost = cost
            to_node.prev = from_node

    def _propagate_low_cost_paths(self):
        node = self._top_left
        while node is not None:
            node.cost = -1.0
            node.prev = None
            node = node.right
        layer_top = self._top_left
        while layer_top.down is not None:
            n = layer_top
            while n is not None:
                n.down.cost = -1.0
                n.down.prev = None
                n = n.right

            layer_top = layer_top.down
            n = layer_top
            while n is not None:
                up = n.up
                up_left = up.left
                up_right = up.right

                self._update_cost(up, n)
                if up_left is not None:
                    self._update_cost(up_left, n)
                if up_right is not None:
                    self._update_cost(up_right, n)
                n = n.right
        return layer_top

    def _update_cost_if_needed(self, to_node):
        if to_node.prev is not None:
            if to_node.up is to_node.prev:
                pass
            elif to_node.up.left is to_node.prev:
                pass
            elif to_node.up.right is to_node.prev:
                pass
            else:
                # Previous lowest cost path node was in removed seam
                to_node.cost = -1.0
                to_node.prev = None
        else:
            to_node.cost = -1.0

        # Parents are checked in the order up, up-left, up-right, the first
        # strictly cheaper one wins
        up = to_node.up
        best_parent = up
        achievable_cost = self._path_cost(up, to_node)
        if up.left is not None:
            cost = self._path_cost(up.left, to_node)
            if cost < achievable_cost:
                achievable_cost = cost
                best_parent = up.left
        if up.right is not None:
            cost = self._path_cost(up.right, to_node)
            if cost < achievable_cost:
                achievable_cost = cost
                best_parent = up.right

        # Compared exactly on purpose: with unchanged parents the recomputed cost is identical,
        # so only real changes propagate. A tolerance let stale costs and paths survive and
        # drift from a full recompute.
        if to_node.cost != achievable_cost:
            to_node.cost = achievable_cost
            to_node.prev = best_parent
            return True

        # Same cost: still take the parent a full recompute picks on a tie. Children only
        # read the cost, so nothing below needs updating.
        to_node.prev = best_parent
        return False

    # A node may be queued twice (as a seam neighbour and after its cost changed),
    # the second visit is cheap since all its children are already evaluated.
    def _enqueue_update(self, node):
        if node is not None:
            layer = self._update_heads[node.y]
            layer.append(node)

    def _evaluate_child(self, child):
        # Rows are processed top to bottom, so all parents of child are final
        # by now and evaluating it again could not change its cost.
        if child is None or child.evaluated_epoch == self._epoch:
            return
        child.evaluated_epoch = self._epoch
        if self._update_cost_if_needed(child):
            self._enqueue_update(child)

    def _update_cost_incrementally(self):
        for depth in range(self._height):
            # Only rows below depth grow here, so iterating this row is safe
            layer = self._update_heads[depth]
            for node in layer:
                if node.down is None:
                    continue
                self._evaluate_child(node.down)
                self._evaluate_child(node.down.left)
                self._evaluate_child(node.down.right)

        node = self._top_left
        while node is not None:
            layer_top = node
            node = node.down
        return layer_top

    def _find_cheapest_seam(self, bottom_left):
        cheapest = bottom_left
        n = bottom_left
        while n is not None:
            if n.cost < cheapest.cost:
                cheapest = n
            n = n.right
        return cheapest

    def _remove_vertical_seam(self, incremental_cost_updates):
        if incremental_cost_updates and self._has_update_heads:
            bottom_left = self._update_cost_incrementally()
        else:
            bottom_left = self._propagate_low_cost_paths()

        cheapest = self._find_cheapest_seam(bottom_left)
        seam_node = cheapest

        for layer in self._update_heads:
            layer.clear()
        self._has_update_heads = False
        # The next incremental update marks evaluated nodes with a new epoch
        self._epoch += 1

        prev_removed_node = None
        while seam_node is not None:
            left = seam_node.left
            right = seam_node.right
            if left is not None:
                left.right = right
            if right is not None:
                right.left = left

            if seam_node is self._top_left:
                # Special node tracking top left corner
                self._top_left = self._top_left.right

            if prev_removed_node is not None:
                if prev_removed_node.up is seam_node:
                    # Seam node is removed vertically
                    down = seam_node.down
                    down_left = down.left
                    down_right = down.right
                    if down_left is not None:
                        down_left.prev = None
                    if down_right is not None:
                        down_right.prev = None
                else:
                    if prev_removed_node.up is seam_node.right:
                        # Left diagonal seam
                        seam_node.down.up = seam_node.right
                        seam_node.right.down = seam_node.down
                    else:
                        # Right diagonal seam
                        seam_node.down.up = seam_node.left
                        seam_node.left.down = seam_node.down
            prev_removed_node = seam_node
            seam_node = seam_node.prev

            # Head of incremental low cost propagation
            if incremental_cost_updates:
                self._enqueue_update(prev_removed_node.left)
                self._enqueue_update(prev_removed_node.right)
                self._has_update_heads = True
        self._width -= 1

    def shrink_horizontally(self, pixels_to_shrink, incremental_cost_updates=True):
        if pixels_to_shrink < 0 or pixels_to_shrink >= self._width:
            raise IndexError('pixels_to_shrink must be in [0, width)')
        for i in range(pixels_to_shrink):
            self._remove_vertical_seam(incremental_cost_updates)

    def get_shrunk_image(self):
        shrunk_image = Image.new(mode="RGB", size=(self._width, self._height))
        shrunk_pixels = shrunk_image.load()

        y = 0
        row = self._top_left
        while row is not None:
            x = 0
            column = row
            while column is not None:
                shrunk_pixels[x, y] = self._pixels[column.x, column.y]
                column = column.right
                x += 1
            row = row.down
            y += 1
        return shrunk_image
