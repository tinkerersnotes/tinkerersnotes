
from PIL import Image
from libc.math cimport sqrt
import random
import time
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
    cdef public int width
    cdef public int height
    cdef public object pixels
    cdef public list[CyMeshNode] mesh
    cdef public dict[cython.int, set[CyMeshNode]] seam_update_head
    cdef public CyMeshNode top_left
    
    def __init__(self, image):
        self.width, self.height = image.size
        self.pixels = image.load()
    
        self.mesh: list[CyMeshNode] = []
        self.create_mesh_nodes()
        self.link_mesh_nodes()

        self.seam_update_head: dict[cython.int, set[CyMeshNode]] = {}

    def node_index(self, x: cython.int, y: cython.int) -> cython.int:
        return y * self.width + x

    def create_mesh_nodes(self):
        x: cython.int
        y: cython.int
        i: cython.int
        assert(self.height > 0 and self.width > 0)
        for y in range(self.height):
            for x in range(self.width):
                i = self.node_index(x, y)
                assert(len(self.mesh) == i)
                self.mesh.append(CyMeshNode(x, y, self.pixels))
        self.top_left = self.mesh[0]

    def link_mesh_nodes(self):
        x: cython.int
        y: cython.int
        i: cython.int
        node: CyMeshNode
        for y in range(self.height):
            for x in range(self.width):
                i = self.node_index(x, y)
                node = self.mesh[i]
                if x > 0:
                    left: CyMeshNode = self.mesh[self.node_index(x - 1, y)]
                    left.right = node
                    node.left = left
                if x < self.width - 1:
                    right: CyMeshNode = self.mesh[self.node_index(x + 1, y)]
                    right.left = node
                    node.right = right
                if y > 0:
                    up: CyMeshNode = self.mesh[self.node_index(x, y - 1)]
                    up.down = node
                    node.up = up
                if y < self.height - 1:
                    down: CyMeshNode = self.mesh[self.node_index(x, y + 1)]
                    down.up = node
                    node.down = down

    def distance(self, n1: CyMeshNode, n2: CyMeshNode) -> cython.double: 
        r: cython.double = n1.r - n2.r
        g: cython.double = n1.g - n2.g
        b: cython.double = n1.b - n2.b
        return sqrt(r*r + g*g + b*b)

    def update_cost(self, from_node: CyMeshNode, to_node: CyMeshNode):
        cost: cython.double = from_node.cost if from_node.cost >= 0.0 else 0.0
        cost += self.distance(from_node, to_node)
        
        if to_node.cost < 0.0 or to_node.cost > cost:
            to_node.cost = cost
            to_node.prev = from_node

    def propagate_low_cost_paths(self):
        layer_top: CyMeshNode
        n: CyMeshNode
        node: CyMeshNode = self.top_left
        while node is not None:
            node.cost = -1.0
            node.prev = None
            node = node.right
        layer_top = self.top_left
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
                
                self.update_cost(up, n)
                if up_left is not None:
                    self.update_cost(up_left, n)
                if up_right is not None:
                    self.update_cost(up_right, n)
                n = n.right
        return layer_top

    def update_cost_if_needed(self, from_node: CyMeshNode, to_node: CyMeshNode) -> cython.bool:
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
            cost = (parent_node.cost if parent_node.cost >= 0.0 else 0.0) + self.distance(parent_node, to_node)
            if achievable_cost < 0.0 or cost < achievable_cost:
                achievable_cost = cost
                best_parent = parent_node

        #if to_node.cost != achievable_cost:
        if not (achievable_cost * 0.9999 <= to_node.cost <= achievable_cost * 1.0001):
            #print(f'Changing best parent of node ({to_node.x},{to_node.y}) with cost {to_node.cost} to {achievable_cost} of prev({best_parent.x},{best_parent.y})')
            to_node.cost = achievable_cost
            to_node.prev = best_parent
            return True   
        
        return False

    def update_cost_incrementally(self):
        node: CyMeshNode
        layer_top: CyMeshNode
        depth: cython.int

        for depth in range(self.height):
            if depth > 0:
                del self.seam_update_head[depth - 1]
            for node in self.seam_update_head[depth]:
                if node.down is None:
                    continue

                if self.update_cost_if_needed(node, node.down):
                        self.seam_update_head[depth+1].add(node.down)
                
                if node.down.left is not None:
                    if self.update_cost_if_needed(node, node.down.left):
                        self.seam_update_head[depth+1].add(node.down.left)
                if node.down.right is not None:
                    if self.update_cost_if_needed(node, node.down.right):
                        self.seam_update_head[depth+1].add(node.down.right)
            
        node = self.top_left
        while node is not None:
            layer_top = node
            node = node.down
        return layer_top

    def find_cheapest_seam(self, bottom_left: CyMeshNode) -> CyMeshNode:
        cheapest: CyMeshNode = bottom_left
        n: CyMeshNode = bottom_left
        while n is not None:
            if n.cost < cheapest.cost:
                cheapest = n
            n = n.right
        return cheapest

    def remove_vertical_seam(self, incremental_cost_updates: cython.bool):
        node: CyMeshNode
        layer_top: CyMeshNode
        seam_node: CyMeshNode

        bottom_left: CyMeshNode
        if incremental_cost_updates and len(self.seam_update_head) > 0:
            bottom_left = self.update_cost_incrementally()
            self.seam_update_head = {}
        else:
            bottom_left = self.propagate_low_cost_paths()
        
        cheapest: CyMeshNode = self.find_cheapest_seam(bottom_left)
        seam_node = cheapest

        self.seam_update_head = {}
        
        depth: cython.int = self.height - 1
        prev_removed_node: CyMeshNode = None
        while seam_node is not None:         
            left = seam_node.left
            right = seam_node.right
            if left is not None:
                left.right = right
            if right is not None:
                right.left = left

            if seam_node is self.top_left:
                #Special node tracking top left corner
                self.top_left = self.top_left.right
                
            if prev_removed_node is not None:
                if prev_removed_node.up is seam_node:
                    #Seam node is removed vertically!")
                    down = seam_node.down
                    down_left = down.left
                    down_right = down.right
                    if down_left is not None:
                        down_left.prev = None
                    if down_right is not None:
                        down_right.prev = None
                else:
                    if prev_removed_node.up is seam_node.right:
                        #Left diagonal seam
                        seam_node.down.up = seam_node.right
                        seam_node.right.down = seam_node.down
                    else:
                        #Right diagonal seam
                        seam_node.down.up = seam_node.left
                        seam_node.left.down = seam_node.down
            prev_removed_node = seam_node
            seam_node = seam_node.prev
            
            # Head of incremental low cost propagation
            if incremental_cost_updates:
                self.seam_update_head[depth] = set()
                if prev_removed_node.left is not None:
                    self.seam_update_head[depth].add(prev_removed_node.left)
                if prev_removed_node.right is not None:
                    self.seam_update_head[depth].add(prev_removed_node.right)
            depth -= 1

    def shrink_horizontally(self, pixels_to_shrink: cython.int, incremental_cost_updates: cython.bool):
        for i in range(pixels_to_shrink):
            self.remove_vertical_seam(incremental_cost_updates)

    def get_shrunk_image(self):
        new_width: cython.int = 0
        new_height: cython.int = 0;
        node = self.top_left
        while node is not None:
            new_width += 1
            node = node.right
        node = self.top_left
        while node is not None:
            new_height += 1
            node = node.down
        
        shrunk_image = Image.new(mode="RGBA", size=(new_width, new_height))
        shrunk_pixels = shrunk_image.load()
        
        x: cython.int
        y: cython.int = 0;
        row = self.top_left
        while row is not None:
            x = 0
            column = row
            while column is not None:
                shrunk_pixels[x, y] = self.pixels[column.x, column.y]
                column = column.right
                x += 1
            row = row.down
            y += 1
        return shrunk_image

from IPython.display import display

# Read image
image = Image.open('tundra_1280x958.jpeg')
print('original image:')
display(image)

sc = CySeamCarver(image)

start_time = time.time()
pixels_to_shrink = 600
incremental_cost_updates = True
sc.shrink_horizontally(pixels_to_shrink, incremental_cost_updates)
finish_time = time.time()
print(f"{finish_time - start_time} seconds to shrink {pixels_to_shrink} pixels")
shrunk_image = sc.get_shrunk_image()

shrunk_image.save('tundra_resized_1230x958.png', optimize=True)
print('shrunk image:')
display(shrunk_image)
