#include "mcts/arena_node_store.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace alphazero::mcts {

ArenaNodeStore::ArenaNodeStore(std::size_t capacity)
    : arena_(capacity),
      node_epoch_(capacity, 0U) {
    if (capacity == 0U) {
        throw std::invalid_argument("ArenaNodeStore capacity must be positive");
    }
}

NodeId ArenaNodeStore::allocate() {
    NodeId id = NULL_NODE;
    if (!free_list_.empty()) {
        id = free_list_.back();
        free_list_.pop_back();
    } else {
        if (next_free_ >= arena_.size()) {
            throw std::runtime_error("ArenaNodeStore capacity exceeded");
        }
        id = next_free_++;
    }

    arena_[static_cast<std::size_t>(id)].reset();
    node_epoch_[static_cast<std::size_t>(id)] = epoch_;
    ++live_nodes_;
    return id;
}

MCTSNode& ArenaNodeStore::get(NodeId id) {
    validate_live_node(id);
    return arena_[static_cast<std::size_t>(id)];
}

const MCTSNode& ArenaNodeStore::get(NodeId id) const {
    validate_live_node(id);
    return arena_[static_cast<std::size_t>(id)];
}

void ArenaNodeStore::release_subtree(NodeId root) {
    if (root == NULL_NODE) {
        return;
    }
    if (!is_within_capacity(root)) {
        throw std::out_of_range("ArenaNodeStore release_subtree root is out of range");
    }
    if (!is_live(root)) {
        return;
    }

    std::vector<NodeId> stack;
    stack.push_back(root);

    while (!stack.empty()) {
        const NodeId node_id = stack.back();
        stack.pop_back();

        if (!is_live(node_id)) {
            continue;
        }

        const MCTSNode& node = arena_[static_cast<std::size_t>(node_id)];
        for (const NodeId child : node.children) {
            if (!is_live(child)) {
                continue;
            }

            // ArenaNodeStore currently assumes a tree topology.
            if (arena_[static_cast<std::size_t>(child)].parent != node_id) {
                continue;
            }
            stack.push_back(child);
        }

        release_single_node(node_id);
    }
}

void ArenaNodeStore::reset() {
    next_free_ = 0;
    free_list_.clear();
    live_nodes_ = 0;

    if (epoch_ == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(node_epoch_.begin(), node_epoch_.end(), 0U);
        epoch_ = 1;
    } else {
        ++epoch_;
    }
}

std::size_t ArenaNodeStore::nodes_allocated() const { return live_nodes_; }

std::size_t ArenaNodeStore::memory_used_bytes() const { return live_nodes_ * sizeof(MCTSNode); }

NodeId ArenaNodeStore::reuse_subtree(NodeId old_root, NodeId preserved_child) {
    validate_live_node(old_root);

    if (preserved_child == NULL_NODE) {
        release_subtree(old_root);
        return NULL_NODE;
    }
    if (!is_live(preserved_child)) {
        throw std::invalid_argument("ArenaNodeStore preserved child must be an allocated node");
    }
    if (arena_[static_cast<std::size_t>(preserved_child)].parent != old_root) {
        throw std::invalid_argument("ArenaNodeStore preserved child must be a direct child of old_root");
    }

    const auto children = arena_[static_cast<std::size_t>(old_root)].children;
    for (const NodeId child : children) {
        if (child == NULL_NODE || child == preserved_child) {
            continue;
        }
        release_subtree(child);
    }

    release_single_node(old_root);

    MCTSNode& new_root = get(preserved_child);
    new_root.parent = NULL_NODE;
    new_root.parent_action = -1;
    return preserved_child;
}

std::size_t ArenaNodeStore::capacity() const noexcept { return arena_.size(); }

bool ArenaNodeStore::is_within_capacity(NodeId id) const noexcept {
    return id != NULL_NODE && static_cast<std::size_t>(id) < arena_.size();
}

bool ArenaNodeStore::is_live(NodeId id) const noexcept {
    if (!is_within_capacity(id)) {
        return false;
    }
    return node_epoch_[static_cast<std::size_t>(id)] == epoch_;
}

void ArenaNodeStore::validate_live_node(NodeId id) const {
    if (!is_live(id)) {
        throw std::out_of_range("ArenaNodeStore node id is not allocated: " + std::to_string(id));
    }
}

void ArenaNodeStore::release_single_node(NodeId id) {
    if (!is_live(id)) {
        return;
    }

    arena_[static_cast<std::size_t>(id)].reset();
    node_epoch_[static_cast<std::size_t>(id)] = 0U;
    free_list_.push_back(id);
    --live_nodes_;
}

}  // namespace alphazero::mcts
