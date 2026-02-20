#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace alphazero::mcts {

using NodeId = std::uint32_t;
inline constexpr NodeId NULL_NODE = std::numeric_limits<NodeId>::max();

inline constexpr int kChessMaxActions = 218;
inline constexpr int kGoMaxActions = 362;

template <int MaxActions>
struct MCTSNodeT {
    static_assert(MaxActions > 0, "MCTSNodeT requires a positive action capacity.");

    static constexpr int kMaxActions = MaxActions;

    // --- Edge statistics (SoA for vectorized PUCT) ---
    std::array<std::int32_t, MaxActions> visit_count{};
    std::array<float, MaxActions> total_value{};
    std::array<float, MaxActions> mean_value{};
    std::array<float, MaxActions> prior{};

    // --- Node metadata ---
    std::array<std::int16_t, MaxActions> actions{};
    std::int16_t num_actions = 0;
    std::int32_t total_visits = 0;
    float node_value = 0.0F;

    // --- Tree structure ---
    std::array<NodeId, MaxActions> children{};
    NodeId parent = NULL_NODE;
    std::int16_t parent_action = -1;

    // --- Virtual loss tracking ---
    std::array<std::int32_t, MaxActions> virtual_loss{};

    constexpr MCTSNodeT() noexcept { reset(); }

    constexpr void reset() noexcept {
        visit_count.fill(0);
        total_value.fill(0.0F);
        mean_value.fill(0.0F);
        prior.fill(0.0F);

        actions.fill(-1);
        num_actions = 0;
        total_visits = 0;
        node_value = 0.0F;

        children.fill(NULL_NODE);
        parent = NULL_NODE;
        parent_action = -1;

        virtual_loss.fill(0);
    }
};

using ChessMCTSNode = MCTSNodeT<kChessMaxActions>;
using GoMCTSNode = MCTSNodeT<kGoMaxActions>;

// Default node size supports every currently implemented game.
using MCTSNode = GoMCTSNode;

static_assert(std::is_standard_layout_v<ChessMCTSNode>);
static_assert(std::is_standard_layout_v<GoMCTSNode>);

}  // namespace alphazero::mcts
