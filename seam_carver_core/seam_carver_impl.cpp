#include "seam_carver_impl.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace seam_carving {

SeamCarver::SeamCarver(const std::uint8_t* rgb, int width, int height)
    : width_(width), height_(height) {
    if (rgb == nullptr || width <= 0 || height <= 0) {
        throw std::invalid_argument("image must be non-empty");
    }
    const std::size_t node_count = static_cast<std::size_t>(width) * height;
    nodes_.reserve(node_count);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            nodes_.emplace_back(x, y, rgb + 3 * (static_cast<std::size_t>(y) * width + x));
        }
    }
    assert(nodes_.capacity() == node_count);
    link_mesh_nodes();
    top_left_ = &nodes_.front();
    update_heads_.resize(height);
}

void SeamCarver::link_mesh_nodes() {
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            MeshNode* node = &nodes_[static_cast<std::size_t>(y) * width_ + x];
            if (x > 0) {
                MeshNode* left = node - 1;
                left->right = node;
                node->left = left;
            }
            if (y > 0) {
                MeshNode* up = node - width_;
                up->down = node;
                node->up = up;
            }
        }
    }
}

double SeamCarver::distance(const MeshNode* n1, const MeshNode* n2) {
    const double r = double(n1->r) - n2->r;
    const double g = double(n1->g) - n2->g;
    const double b = double(n1->b) - n2->b;
    return std::sqrt(r * r + g * g + b * b);
}

void SeamCarver::update_cost(MeshNode* from_node, MeshNode* to_node) {
    double cost = from_node->cost >= 0.0 ? from_node->cost : 0.0;
    cost += distance(from_node, to_node);

    if (to_node->cost < 0.0 || to_node->cost > cost) {
        to_node->cost = cost;
        to_node->prev = from_node;
    }
}

MeshNode* SeamCarver::propagate_low_cost_paths() {
    for (MeshNode* node = top_left_; node != nullptr; node = node->right) {
        node->cost = -1.0;
        node->prev = nullptr;
    }
    MeshNode* layer_top = top_left_;
    while (layer_top->down != nullptr) {
        for (MeshNode* n = layer_top; n != nullptr; n = n->right) {
            n->down->cost = -1.0;
            n->down->prev = nullptr;
        }

        layer_top = layer_top->down;
        for (MeshNode* n = layer_top; n != nullptr; n = n->right) {
            MeshNode* up = n->up;
            update_cost(up, n);
            if (up->left != nullptr) {
                update_cost(up->left, n);
            }
            if (up->right != nullptr) {
                update_cost(up->right, n);
            }
        }
    }
    return layer_top;
}

bool SeamCarver::update_cost_if_needed(MeshNode* to_node) {
    MeshNode* up = to_node->up;
    if (to_node->prev != nullptr) {
        if (to_node->prev != up && to_node->prev != up->left && to_node->prev != up->right) {
            // Previous lowest cost path node was in removed seam
            to_node->cost = -1.0;
            to_node->prev = nullptr;
        }
    } else {
        to_node->cost = -1.0;
    }

    MeshNode* const parent_nodes[] = {up, up->left, up->right};
    double achievable_cost = -1.0;
    MeshNode* best_parent = nullptr;
    for (MeshNode* parent_node : parent_nodes) {
        if (parent_node == nullptr) {
            continue;
        }
        const double cost = (parent_node->cost >= 0.0 ? parent_node->cost : 0.0)
                            + distance(parent_node, to_node);
        if (achievable_cost < 0.0 || cost < achievable_cost) {
            achievable_cost = cost;
            best_parent = parent_node;
        }
    }

    // Compared exactly on purpose: with unchanged parents the recomputed cost is bit-identical,
    // so only real changes propagate. A tolerance let stale costs and paths survive and drift
    // from a full recompute.
    if (to_node->cost != achievable_cost) {
        to_node->cost = achievable_cost;
        to_node->prev = best_parent;
        return true;
    }
    // Same cost: still take the parent a full recompute picks on a tie. Children only read
    // the cost, so nothing below needs updating.
    to_node->prev = best_parent;
    return false;
}

// A node may be queued twice (as a seam neighbour and after its cost changed),
// the second visit is cheap since all its children are already evaluated.
void SeamCarver::enqueue_update(MeshNode* node) {
    if (node != nullptr) {
        update_heads_[node->y].push_back(node);
    }
}

MeshNode* SeamCarver::update_cost_incrementally() {
    for (int depth = 0; depth < height_; ++depth) {
        // Only update_heads_[depth + 1] grows here, so indexing stays valid.
        const std::vector<MeshNode*>& layer = update_heads_[depth];
        for (std::size_t i = 0; i < layer.size(); ++i) {
            MeshNode* down = layer[i]->down;
            if (down == nullptr) {
                continue;
            }
            for (MeshNode* child : {down, down->left, down->right}) {
                // Rows are processed top to bottom, so all parents of child are final
                // by now and evaluating it again could not change its cost.
                if (child == nullptr || child->evaluated_epoch == epoch_) {
                    continue;
                }
                child->evaluated_epoch = epoch_;
                if (update_cost_if_needed(child)) {
                    enqueue_update(child);
                }
            }
        }
    }

    MeshNode* layer_top = top_left_;
    while (layer_top->down != nullptr) {
        layer_top = layer_top->down;
    }
    return layer_top;
}

MeshNode* SeamCarver::find_cheapest_seam(MeshNode* bottom_left) {
    MeshNode* cheapest = bottom_left;
    for (MeshNode* n = bottom_left; n != nullptr; n = n->right) {
        if (n->cost < cheapest->cost) {
            cheapest = n;
        }
    }
    return cheapest;
}

void SeamCarver::remove_vertical_seam(bool incremental_cost_updates) {
    MeshNode* bottom_left = incremental_cost_updates && has_update_heads_
                                ? update_cost_incrementally()
                                : propagate_low_cost_paths();

    for (std::vector<MeshNode*>& layer : update_heads_) {
        layer.clear();
    }
    has_update_heads_ = false;
    // The next incremental update marks evaluated nodes with a new epoch.
    ++epoch_;

    MeshNode* seam_node = find_cheapest_seam(bottom_left);
    MeshNode* prev_removed_node = nullptr;
    while (seam_node != nullptr) {
        MeshNode* left = seam_node->left;
        MeshNode* right = seam_node->right;
        if (left != nullptr) {
            left->right = right;
        }
        if (right != nullptr) {
            right->left = left;
        }

        if (seam_node == top_left_) {
            // Special node tracking top left corner
            top_left_ = top_left_->right;
        }

        if (prev_removed_node != nullptr) {
            if (prev_removed_node->up == seam_node) {
                // Seam node is removed vertically
                MeshNode* down = seam_node->down;
                if (down->left != nullptr) {
                    down->left->prev = nullptr;
                }
                if (down->right != nullptr) {
                    down->right->prev = nullptr;
                }
            } else if (prev_removed_node->up == seam_node->right) {
                // Left diagonal seam
                seam_node->down->up = seam_node->right;
                seam_node->right->down = seam_node->down;
            } else {
                // Right diagonal seam
                seam_node->down->up = seam_node->left;
                seam_node->left->down = seam_node->down;
            }
        }
        prev_removed_node = seam_node;
        seam_node = seam_node->prev;

        // Head of incremental low cost propagation
        if (incremental_cost_updates) {
            enqueue_update(prev_removed_node->left);
            enqueue_update(prev_removed_node->right);
            has_update_heads_ = true;
        }
    }
    --width_;
}

void SeamCarver::shrink_horizontally(int pixels_to_shrink, bool incremental_cost_updates) {
    if (pixels_to_shrink < 0 || pixels_to_shrink >= width_) {
        throw std::out_of_range("pixels_to_shrink must be in [0, width)");
    }
    for (int i = 0; i < pixels_to_shrink; ++i) {
        remove_vertical_seam(incremental_cost_updates);
    }
}

std::vector<std::uint8_t> SeamCarver::shrunk_rgb() const {
    std::vector<std::uint8_t> rgb;
    rgb.reserve(static_cast<std::size_t>(width_) * height_ * 3);
    for (const MeshNode* row = top_left_; row != nullptr; row = row->down) {
        for (const MeshNode* column = row; column != nullptr; column = column->right) {
            rgb.push_back(column->r);
            rgb.push_back(column->g);
            rgb.push_back(column->b);
        }
    }
    assert(rgb.size() == static_cast<std::size_t>(width_) * height_ * 3);
    return rgb;
}

}  // namespace seam_carving
