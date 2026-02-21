#include "mcts/arena_node_store.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

namespace {

void link_parent_child(
    alphazero::mcts::ArenaNodeStore& store,
    const alphazero::mcts::NodeId parent,
    const std::size_t edge_index,
    const alphazero::mcts::NodeId child,
    const std::int16_t parent_action = 0) {
    auto& parent_node = store.get(parent);
    parent_node.children[edge_index] = child;
    auto& child_node = store.get(child);
    child_node.parent = parent;
    child_node.parent_action = parent_action;
}

}  // namespace

// WHY: Callers size arenas per game; default capacity and initial accounting must match the MCTS spec.
TEST(ArenaNodeStoreTest, UsesExpectedDefaultCapacityAndStartsEmpty) {
    alphazero::mcts::ArenaNodeStore store;

    EXPECT_EQ(store.capacity(), alphazero::mcts::ArenaNodeStore::kDefaultCapacity);
    EXPECT_EQ(store.nodes_allocated(), 0U);
    EXPECT_EQ(store.memory_used_bytes(), 0U);
}

// WHY: The allocator contract underpins every simulation, so ID assignment, capacity checks, and accounting must
// remain deterministic.
TEST(ArenaNodeStoreTest, AllocatesSequentialIdsAndTracksMemory) {
    alphazero::mcts::ArenaNodeStore store(2);

    const alphazero::mcts::NodeId first = store.allocate();
    const alphazero::mcts::NodeId second = store.allocate();

    EXPECT_EQ(first, 0U);
    EXPECT_EQ(second, 1U);
    EXPECT_EQ(store.nodes_allocated(), 2U);
    EXPECT_EQ(store.memory_used_bytes(), 2U * sizeof(alphazero::mcts::MCTSNode));

    EXPECT_THROW(static_cast<void>(store.allocate()), std::runtime_error);
    EXPECT_THROW(static_cast<void>(store.get(alphazero::mcts::NULL_NODE)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(store.get(3U)), std::out_of_range);
}

// WHY: Subtree release is how MCTS frees abandoned lines, so releasing one branch must not affect unrelated nodes and
// should make released IDs reusable.
TEST(ArenaNodeStoreTest, ReleaseSubtreeFreesOnlyDescendantsAndReusesIds) {
    alphazero::mcts::ArenaNodeStore store(8);

    const alphazero::mcts::NodeId root = store.allocate();
    const alphazero::mcts::NodeId left = store.allocate();
    const alphazero::mcts::NodeId right = store.allocate();
    const alphazero::mcts::NodeId left_grandchild = store.allocate();

    link_parent_child(store, root, 0, left, 11);
    link_parent_child(store, root, 1, right, 22);
    link_parent_child(store, left, 0, left_grandchild, 33);

    store.release_subtree(left);

    EXPECT_EQ(store.nodes_allocated(), 2U);
    EXPECT_EQ(store.memory_used_bytes(), 2U * sizeof(alphazero::mcts::MCTSNode));
    EXPECT_NO_THROW(static_cast<void>(store.get(root)));
    EXPECT_NO_THROW(static_cast<void>(store.get(right)));
    EXPECT_THROW(static_cast<void>(store.get(left)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(store.get(left_grandchild)), std::out_of_range);

    const alphazero::mcts::NodeId recycled_a = store.allocate();
    const alphazero::mcts::NodeId recycled_b = store.allocate();
    const std::unordered_set<alphazero::mcts::NodeId> recycled_ids{recycled_a, recycled_b};

    EXPECT_EQ(recycled_ids.size(), 2U);
    EXPECT_TRUE(recycled_ids.contains(left));
    EXPECT_TRUE(recycled_ids.contains(left_grandchild));
    EXPECT_EQ(store.memory_used_bytes(), 4U * sizeof(alphazero::mcts::MCTSNode));
}

// WHY: Tree reuse after move selection must preserve the chosen child subtree and release every sibling branch.
TEST(ArenaNodeStoreTest, ReuseSubtreeKeepsChosenChildAndReleasesSiblings) {
    alphazero::mcts::ArenaNodeStore store(12);

    const alphazero::mcts::NodeId root = store.allocate();
    const alphazero::mcts::NodeId keep = store.allocate();
    const alphazero::mcts::NodeId drop_a = store.allocate();
    const alphazero::mcts::NodeId drop_b = store.allocate();
    const alphazero::mcts::NodeId keep_grandchild = store.allocate();
    const alphazero::mcts::NodeId drop_grandchild = store.allocate();

    link_parent_child(store, root, 0, keep, 1);
    link_parent_child(store, root, 1, drop_a, 2);
    link_parent_child(store, root, 2, drop_b, 3);
    link_parent_child(store, keep, 0, keep_grandchild, 4);
    link_parent_child(store, drop_a, 0, drop_grandchild, 5);

    const alphazero::mcts::NodeId new_root = store.reuse_subtree(root, keep);

    EXPECT_EQ(new_root, keep);
    EXPECT_EQ(store.nodes_allocated(), 2U);
    EXPECT_EQ(store.memory_used_bytes(), 2U * sizeof(alphazero::mcts::MCTSNode));
    EXPECT_THROW(static_cast<void>(store.get(root)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(store.get(drop_a)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(store.get(drop_b)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(store.get(drop_grandchild)), std::out_of_range);

    const auto& new_root_node = store.get(new_root);
    EXPECT_EQ(new_root_node.parent, alphazero::mcts::NULL_NODE);
    EXPECT_EQ(new_root_node.parent_action, -1);
    EXPECT_NO_THROW(static_cast<void>(store.get(keep_grandchild)));
}

// WHY: Reset is invoked between games; it must invalidate old IDs and rewind allocation without scanning the arena.
TEST(ArenaNodeStoreTest, ResetInvalidatesExistingNodesAndRewindsAllocator) {
    alphazero::mcts::ArenaNodeStore store(6);

    const alphazero::mcts::NodeId first = store.allocate();
    const alphazero::mcts::NodeId second = store.allocate();
    EXPECT_EQ(first, 0U);
    EXPECT_EQ(second, 1U);

    store.reset();

    EXPECT_EQ(store.nodes_allocated(), 0U);
    EXPECT_EQ(store.memory_used_bytes(), 0U);
    EXPECT_THROW(static_cast<void>(store.get(first)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(store.get(second)), std::out_of_range);

    const alphazero::mcts::NodeId rewind_first = store.allocate();
    const alphazero::mcts::NodeId rewind_second = store.allocate();
    EXPECT_EQ(rewind_first, 0U);
    EXPECT_EQ(rewind_second, 1U);
}

// WHY: Release paths are called defensively in reuse flows, so null/already-freed roots should behave as no-ops.
TEST(ArenaNodeStoreTest, ReleaseSubtreeTreatsNullOrAlreadyReleasedRootsAsNoOps) {
    alphazero::mcts::ArenaNodeStore store(4);

    const alphazero::mcts::NodeId root = store.allocate();
    store.release_subtree(alphazero::mcts::NULL_NODE);
    EXPECT_EQ(store.nodes_allocated(), 1U);
    EXPECT_EQ(store.memory_used_bytes(), sizeof(alphazero::mcts::MCTSNode));

    store.release_subtree(root);
    EXPECT_EQ(store.nodes_allocated(), 0U);
    EXPECT_EQ(store.memory_used_bytes(), 0U);

    EXPECT_NO_THROW(store.release_subtree(root));
}

// WHY: Defensive release paths should fail fast on impossible IDs to prevent silent corruption during tree cleanup.
TEST(ArenaNodeStoreTest, ReleaseSubtreeRejectsOutOfRangeRootId) {
    alphazero::mcts::ArenaNodeStore store(4);

    EXPECT_THROW(store.release_subtree(99U), std::out_of_range);
}
