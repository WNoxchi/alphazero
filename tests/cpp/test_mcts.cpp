#include "mcts/mcts_node.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

template <typename T, std::size_t N>
[[nodiscard]] bool all_equal(const std::array<T, N>& values, const T expected) {
    for (const T value : values) {
        if (value != expected) {
            return false;
        }
    }
    return true;
}

}  // namespace

// WHY: Capacity constants and NodeId sentinel are shared by the whole MCTS stack and must stay stable.
TEST(MctsNodeStructureTest, ExposesExpectedCompileTimeCapacitiesAndNodeIdContract) {
    EXPECT_TRUE((std::is_same_v<alphazero::mcts::NodeId, std::uint32_t>));
    EXPECT_EQ(alphazero::mcts::NULL_NODE, std::numeric_limits<alphazero::mcts::NodeId>::max());

    EXPECT_EQ(alphazero::mcts::kChessMaxActions, 218);
    EXPECT_EQ(alphazero::mcts::kGoMaxActions, 362);
    EXPECT_EQ(alphazero::mcts::ChessMCTSNode::kMaxActions, alphazero::mcts::kChessMaxActions);
    EXPECT_EQ(alphazero::mcts::GoMCTSNode::kMaxActions, alphazero::mcts::kGoMaxActions);
    EXPECT_EQ(alphazero::mcts::MCTSNode::kMaxActions, alphazero::mcts::kGoMaxActions);
}

// WHY: Freshly allocated nodes must start from a deterministic zeroed state with explicit null-child sentinels.
TEST(MctsNodeStructureTest, DefaultConstructionInitializesAllFieldsForSafeUse) {
    alphazero::mcts::MCTSNode node{};

    EXPECT_TRUE(all_equal(node.visit_count, 0));
    EXPECT_TRUE(all_equal(node.total_value, 0.0F));
    EXPECT_TRUE(all_equal(node.mean_value, 0.0F));
    EXPECT_TRUE(all_equal(node.prior, 0.0F));
    EXPECT_TRUE(all_equal(node.actions, static_cast<std::int16_t>(-1)));
    EXPECT_TRUE(all_equal(node.children, alphazero::mcts::NULL_NODE));
    EXPECT_TRUE(all_equal(node.virtual_loss, 0));

    EXPECT_EQ(node.num_actions, 0);
    EXPECT_EQ(node.total_visits, 0);
    EXPECT_FLOAT_EQ(node.node_value, 0.0F);
    EXPECT_EQ(node.parent, alphazero::mcts::NULL_NODE);
    EXPECT_EQ(node.parent_action, -1);
}

// WHY: Reset is used when recycling nodes, so it must clear any previous simulation state exactly.
TEST(MctsNodeStructureTest, ResetRestoresNodeToDefaultStateAfterMutation) {
    alphazero::mcts::MCTSNode node{};

    node.visit_count[0] = 17;
    node.total_value[0] = 3.5F;
    node.mean_value[0] = 0.2F;
    node.prior[0] = 0.4F;
    node.actions[0] = 9;
    node.num_actions = 1;
    node.total_visits = 17;
    node.node_value = -0.75F;
    node.children[0] = 42;
    node.parent = 7;
    node.parent_action = 9;
    node.virtual_loss[0] = 2;

    node.reset();

    EXPECT_TRUE(all_equal(node.visit_count, 0));
    EXPECT_TRUE(all_equal(node.total_value, 0.0F));
    EXPECT_TRUE(all_equal(node.mean_value, 0.0F));
    EXPECT_TRUE(all_equal(node.prior, 0.0F));
    EXPECT_TRUE(all_equal(node.actions, static_cast<std::int16_t>(-1)));
    EXPECT_TRUE(all_equal(node.children, alphazero::mcts::NULL_NODE));
    EXPECT_TRUE(all_equal(node.virtual_loss, 0));

    EXPECT_EQ(node.num_actions, 0);
    EXPECT_EQ(node.total_visits, 0);
    EXPECT_FLOAT_EQ(node.node_value, 0.0F);
    EXPECT_EQ(node.parent, alphazero::mcts::NULL_NODE);
    EXPECT_EQ(node.parent_action, -1);
}

// WHY: SoA arrays must be contiguous to support vectorized PUCT computations with predictable stride.
TEST(MctsNodeStructureTest, SoAArraysAreContiguousInMemory) {
    alphazero::mcts::MCTSNode node{};

    const std::ptrdiff_t visit_stride =
        reinterpret_cast<const std::byte*>(&node.visit_count[1]) -
        reinterpret_cast<const std::byte*>(&node.visit_count[0]);
    const std::ptrdiff_t total_value_stride =
        reinterpret_cast<const std::byte*>(&node.total_value[1]) -
        reinterpret_cast<const std::byte*>(&node.total_value[0]);
    const std::ptrdiff_t mean_value_stride =
        reinterpret_cast<const std::byte*>(&node.mean_value[1]) -
        reinterpret_cast<const std::byte*>(&node.mean_value[0]);
    const std::ptrdiff_t prior_stride =
        reinterpret_cast<const std::byte*>(&node.prior[1]) -
        reinterpret_cast<const std::byte*>(&node.prior[0]);
    const std::ptrdiff_t action_stride =
        reinterpret_cast<const std::byte*>(&node.actions[1]) -
        reinterpret_cast<const std::byte*>(&node.actions[0]);
    const std::ptrdiff_t child_stride =
        reinterpret_cast<const std::byte*>(&node.children[1]) -
        reinterpret_cast<const std::byte*>(&node.children[0]);
    const std::ptrdiff_t virtual_loss_stride =
        reinterpret_cast<const std::byte*>(&node.virtual_loss[1]) -
        reinterpret_cast<const std::byte*>(&node.virtual_loss[0]);

    EXPECT_EQ(visit_stride, static_cast<std::ptrdiff_t>(sizeof(node.visit_count[0])));
    EXPECT_EQ(total_value_stride, static_cast<std::ptrdiff_t>(sizeof(node.total_value[0])));
    EXPECT_EQ(mean_value_stride, static_cast<std::ptrdiff_t>(sizeof(node.mean_value[0])));
    EXPECT_EQ(prior_stride, static_cast<std::ptrdiff_t>(sizeof(node.prior[0])));
    EXPECT_EQ(action_stride, static_cast<std::ptrdiff_t>(sizeof(node.actions[0])));
    EXPECT_EQ(child_stride, static_cast<std::ptrdiff_t>(sizeof(node.children[0])));
    EXPECT_EQ(virtual_loss_stride, static_cast<std::ptrdiff_t>(sizeof(node.virtual_loss[0])));
}
