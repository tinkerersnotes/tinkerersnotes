
from PIL import Image
from libc.math cimport sqrt
import cython

cdef class CyMeshNode:
    cdef public int x
    cdef public int y
    cdef public double cost
    cdef public unsigned char r
    cdef public unsigned char g
    cdef public unsigned char b
    cdef public CyMeshNode up
    cdef public CyMeshNode down
    cdef public CyMeshNode left
    cdef public CyMeshNode right
    cdef public CyMeshNode prev

    def __init__(self, x: int, y: int, pixels):
        self.up = None
        self.down = None
        self.left = None
        self.right = None
        self.x = x
        self.y = y
        self.prev = None
        self.cost = -1.0
        pix = pixels[x, y]
        self.r = pix[0]
        self.g = pix[1]
        self.b = pix[2]

cdef class CySeamCarver:
    cdef int _width
    cdef int _height
    cdef object _image
    cdef object _pixels
    cdef list[CyMeshNode] _mesh
    cdef dict[cython.int, set[CyMeshNode]] _seam_update_head
    cdef CyMeshNode _top_left

    def __init__(self, image):
        self._image = image if image.mode == 'RGB' else image.convert('RGB')
        self._width, self._height = self._image.size
        self._pixels = self._image.load()

        self._mesh: list[CyMeshNode] = []
        self._create_mesh_nodes()
        self._link_mesh_nodes()

        self._seam_update_head: dict[cython.int, set[CyMeshNode]] = {}

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def _node_index(self, x: cython.int, y: cython.int) -> cython.int:
        return y * self._width + x

    def _create_mesh_nodes(self):
        x: cython.int
        y: cython.int
        i: cython.int
        if self._height <= 0 or self._width <= 0:
            raise ValueError('image must be non-empty')
        for y in range(self._height):
            for x in range(self._width):
                i = self._node_index(x, y)
                assert(len(self._mesh) == i)
                self._mesh.append(CyMeshNode(x, y, self._pixels))
        self._top_left = self._mesh[0]

    def _link_mesh_nodes(self):
        x: cython.int
        y: cython.int
        i: cython.int
        node: CyMeshNode
        for y in range(self._height):
            for x in range(self._width):
                i = self._node_index(x, y)
                node = self._mesh[i]
                if x > 0:
                    left: CyMeshNode = self._mesh[self._node_index(x - 1, y)]
                    left.right = node
                    node.left = left
                if x < self._width - 1:
                    right: CyMeshNode = self._mesh[self._node_index(x + 1, y)]
                    right.left = node
                    node.right = right
                if y > 0:
                    up: CyMeshNode = self._mesh[self._node_index(x, y - 1)]
                    up.down = node
                    node.up = up
                if y < self._height - 1:
                    down: CyMeshNode = self._mesh[self._node_index(x, y + 1)]
                    down.up = node
                    node.down = down

    def _distance(self, n1: CyMeshNode, n2: CyMeshNode) -> cython.double:
        r: cython.double = n1.r - n2.r
        g: cython.double = n1.g - n2.g
        b: cython.double = n1.b - n2.b
        return sqrt(r*r + g*g + b*b)

    def _update_cost(self, from_node: CyMeshNode, to_node: CyMeshNode):
        cost: cython.double = from_node.cost if from_node.cost >= 0.0 else 0.0
        cost += self._distance(from_node, to_node)

        if to_node.cost < 0.0 or to_node.cost > cost:
            to_node.cost = cost
            to_node.prev = from_node

    def _propagate_low_cost_paths(self):
        layer_top: CyMeshNode
        n: CyMeshNode
        node: CyMeshNode = self._top_left
        while node is not None:
            node.cost = -1.0
            node.prev = None
            node = node.right
        layer_top = self._top_left
        while layer_top.down is not None:
            n: CyMeshNode
            n = layer_top
            while n is not None:
                n.down.cost = -1
                n.down.prev = None
                n = n.right

            layer_top = layer_top.down
            n = layer_top
            while n is not None:
                up: CyMeshNode
                up = n.up
                up_left: CyMeshNode = up.left
                up_right: CyMeshNode = up.right

                self._update_cost(up, n)
                if up_left is not None:
                    self._update_cost(up_left, n)
                if up_right is not None:
                    self._update_cost(up_right, n)
                n = n.right
        return layer_top

    def _update_cost_if_needed(self, to_node: CyMeshNode) -> cython.bool:
        cost: cython.double
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

        parent_nodes: list[CyMeshNode]
        parent_nodes = [to_node.up]
        if to_node.up.left is not None:
            parent_nodes.append(to_node.up.left)
        if to_node.up.right is not None:
            parent_nodes.append(to_node.up.right)

        achievable_cost: cython.double = -1.0
        best_parent: CyMeshNode = None
        for parent_node in parent_nodes:
            cost = (parent_node.cost if parent_node.cost >= 0.0 else 0.0) + self._distance(parent_node, to_node)
            if achievable_cost < 0.0 or cost < achievable_cost:
                achievable_cost = cost
                best_parent = parent_node

        if not (achievable_cost * 0.9999 <= to_node.cost <= achievable_cost * 1.0001):
            to_node.cost = achievable_cost
            to_node.prev = best_parent
            return True

        return False

    def _update_cost_incrementally(self):
        node: CyMeshNode
        layer_top: CyMeshNode
        depth: cython.int

        for depth in range(self._height):
            if depth > 0:
                del self._seam_update_head[depth - 1]
            for node in self._seam_update_head[depth]:
                if node.down is None:
                    continue

                if self._update_cost_if_needed(node.down):
                        self._seam_update_head[depth+1].add(node.down)

                if node.down.left is not None:
                    if self._update_cost_if_needed(node.down.left):
                        self._seam_update_head[depth+1].add(node.down.left)
                if node.down.right is not None:
                    if self._update_cost_if_needed(node.down.right):
                        self._seam_update_head[depth+1].add(node.down.right)

        node = self._top_left
        while node is not None:
            layer_top = node
            node = node.down
        return layer_top

    def _find_cheapest_seam(self, bottom_left: CyMeshNode) -> CyMeshNode:
        cheapest: CyMeshNode = bottom_left
        n: CyMeshNode = bottom_left
        while n is not None:
            if n.cost < cheapest.cost:
                cheapest = n
            n = n.right
        return cheapest

    def _remove_vertical_seam(self, incremental_cost_updates: cython.bool):
        node: CyMeshNode
        layer_top: CyMeshNode
        seam_node: CyMeshNode

        bottom_left: CyMeshNode
        if incremental_cost_updates and len(self._seam_update_head) > 0:
            bottom_left = self._update_cost_incrementally()
            self._seam_update_head = {}
        else:
            bottom_left = self._propagate_low_cost_paths()

        cheapest: CyMeshNode = self._find_cheapest_seam(bottom_left)
        seam_node = cheapest

        self._seam_update_head = {}

        depth: cython.int = self._height - 1
        prev_removed_node: CyMeshNode = None
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
                self._seam_update_head[depth] = set()
                if prev_removed_node.left is not None:
                    self._seam_update_head[depth].add(prev_removed_node.left)
                if prev_removed_node.right is not None:
                    self._seam_update_head[depth].add(prev_removed_node.right)
            depth -= 1
        self._width -= 1

    def shrink_horizontally(self, pixels_to_shrink: cython.int, incremental_cost_updates: cython.bool = True):
        if pixels_to_shrink < 0 or pixels_to_shrink >= self._width:
            raise IndexError('pixels_to_shrink must be in [0, width)')
        for i in range(pixels_to_shrink):
            self._remove_vertical_seam(incremental_cost_updates)

    def get_shrunk_image(self):
        shrunk_image = Image.new(mode="RGB", size=(self._width, self._height))
        shrunk_pixels = shrunk_image.load()

        x: cython.int
        y: cython.int = 0;
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
