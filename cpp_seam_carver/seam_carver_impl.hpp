#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace seam_carving {

// C++ counterpart of CyMeshNode: one pixel of the image linked to its
// neighbours. Removing a seam only relinks pointers, nodes are never moved.
struct MeshNode {
    MeshNode(int x, int y, const std::uint8_t* rgb)
        : x(x), y(y), r(rgb[0]), g(rgb[1]), b(rgb[2]) {}

    MeshNode* up = nullptr;
    MeshNode* down = nullptr;
    MeshNode* left = nullptr;
    MeshNode* right = nullptr;
    MeshNode* prev = nullptr;  // Parent on the cheapest path from the top row
    double cost = -1.0;
    int x;
    int y;  // Never changes, rows are not removed by horizontal shrinking
    std::uint32_t evaluated_epoch = 0;  // Last incremental update that evaluated this node
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
};

// C++ counterpart of CySeamCarver.
class SeamCarver {
public:
    // rgb is a row-major buffer of width * height * 3 bytes.
    SeamCarver(const std::uint8_t* rgb, int width, int height);

    // Nodes point into nodes_, so a copy would point into the wrong buffer.
    SeamCarver(const SeamCarver&) = delete;
    SeamCarver& operator=(const SeamCarver&) = delete;

    void shrink_horizontally(int pixels_to_shrink, bool incremental_cost_updates);

    int width() const { return width_; }
    int height() const { return height_; }

    // Row-major RGB bytes of the current (shrunk) image, width() * height() * 3.
    std::vector<std::uint8_t> shrunk_rgb() const;

private:
    void link_mesh_nodes();
    MeshNode* propagate_low_cost_paths();
    MeshNode* update_cost_incrementally();
    void remove_vertical_seam(bool incremental_cost_updates);
    void enqueue_update(MeshNode* node);

    static double distance(const MeshNode* n1, const MeshNode* n2);
    static void update_cost(MeshNode* from_node, MeshNode* to_node);
    static bool update_cost_if_needed(MeshNode* to_node);
    static MeshNode* find_cheapest_seam(MeshNode* bottom_left);

    int width_;
    int height_;
    // Single contiguous allocation for the whole mesh. Its capacity is
    // reserved once and never exceeded, so MeshNode pointers stay valid.
    std::vector<MeshNode> nodes_;
    MeshNode* top_left_ = nullptr;

    // Counterpart of seam_update_head: nodes whose cost may be stale, per row.
    std::vector<std::vector<MeshNode*>> update_heads_;
    bool has_update_heads_ = false;
    // Incremented once per removed seam, so evaluated_epoch never needs resetting
    std::uint32_t epoch_ = 0;
};

}  // namespace seam_carving
