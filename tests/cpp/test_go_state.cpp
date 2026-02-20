#include "games/go/go_config.h"
#include "games/go/go_state.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::go::GoPosition;
using alphazero::go::GoState;
using alphazero::go::kActionSpaceSize;
using alphazero::go::kBlack;
using alphazero::go::kBoardArea;
using alphazero::go::kBoardSize;
using alphazero::go::kEmpty;
using alphazero::go::kPassAction;
using alphazero::go::kWhite;

[[nodiscard]] constexpr int I(int row, int col) { return alphazero::go::to_intersection(row, col); }

[[nodiscard]] constexpr int feature_index(int plane, int intersection) {
    return (plane * kBoardArea) + intersection;
}

[[nodiscard]] std::unique_ptr<GoState> apply_action_as_go_state(const GoState& state, int action) {
    std::unique_ptr<alphazero::GameState> next_base = state.apply_action(action);
    auto typed = std::unique_ptr<GoState>(dynamic_cast<GoState*>(next_base.release()));
    if (!typed) {
        throw std::runtime_error("GoState::apply_action returned non-go state");
    }
    return typed;
}

}  // namespace

// WHY: Runtime game config is consumed by MCTS/network pipelines and must match the spec exactly for Go.
TEST(GoStateTest, GoGameConfigMatchesSpecAndCreatesInitialState) {
    const alphazero::go::GoGameConfig config = alphazero::go::go_game_config();
    EXPECT_EQ(config.name, "go");
    EXPECT_EQ(config.board_rows, 19);
    EXPECT_EQ(config.board_cols, 19);
    EXPECT_EQ(config.planes_per_step, 2);
    EXPECT_EQ(config.num_history_steps, 8);
    EXPECT_EQ(config.constant_planes, 1);
    EXPECT_EQ(config.total_input_channels, 17);
    EXPECT_EQ(config.action_space_size, 362);
    EXPECT_FLOAT_EQ(config.dirichlet_alpha, 0.03F);
    EXPECT_EQ(config.max_game_length, 722);
    EXPECT_EQ(config.value_head_type, alphazero::GameConfig::ValueHeadType::SCALAR);
    EXPECT_TRUE(config.supports_symmetry);
    EXPECT_EQ(config.num_symmetries, 8);

    std::unique_ptr<alphazero::GameState> game = config.new_game();
    auto* go_state = dynamic_cast<GoState*>(game.get());
    ASSERT_NE(go_state, nullptr);
    EXPECT_EQ(go_state->current_player(), GoState::kBlackPlayer);
    EXPECT_EQ(go_state->history_size(), 1);
    EXPECT_EQ(go_state->legal_actions().size(), static_cast<std::size_t>(kActionSpaceSize));
}

// WHY: Self-play depends on immutable transitions and legal-action filtering, including robust rejection of illegal moves.
TEST(GoStateTest, ApplyActionTransitionsStateAndRejectsIllegalMoves) {
    const GoState state{};

    std::unique_ptr<GoState> next_state = apply_action_as_go_state(state, I(3, 3));
    ASSERT_NE(next_state, nullptr);
    EXPECT_EQ(alphazero::go::stone_at(next_state->position(), I(3, 3)), kBlack);
    EXPECT_EQ(next_state->position().side_to_move, kWhite);
    EXPECT_EQ(next_state->current_player(), GoState::kWhitePlayer);
    EXPECT_EQ(next_state->position().move_number, 1);
    EXPECT_EQ(next_state->history_size(), 2);
    EXPECT_EQ(next_state->legal_actions().size(), static_cast<std::size_t>(kActionSpaceSize - 1));
    EXPECT_NE(next_state->hash(), state.hash());

    std::unique_ptr<alphazero::GameState> cloned_base = next_state->clone();
    auto cloned = std::unique_ptr<GoState>(dynamic_cast<GoState*>(cloned_base.release()));
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->hash(), next_state->hash());
    EXPECT_EQ(cloned->to_string(), next_state->to_string());

    EXPECT_THROW(state.apply_action(-1), std::invalid_argument);
    EXPECT_THROW(next_state->apply_action(I(3, 3)), std::invalid_argument);
}

// WHY: Correct terminal/outcome semantics ensure training labels are accurate when games end by pass or move cap.
TEST(GoStateTest, TerminalConditionsUsePassesAndMoveCapAndScoreFromTrompTaylor) {
    GoPosition pass_terminal{};
    pass_terminal.komi = 0.0F;
    pass_terminal.consecutive_passes = 2;
    alphazero::go::set_stone(&pass_terminal, I(0, 0), kBlack);

    const GoState pass_terminal_state(pass_terminal);
    EXPECT_TRUE(pass_terminal_state.is_terminal());
    EXPECT_FLOAT_EQ(pass_terminal_state.outcome(GoState::kBlackPlayer), 1.0F);
    EXPECT_FLOAT_EQ(pass_terminal_state.outcome(GoState::kWhitePlayer), -1.0F);
    EXPECT_THROW(pass_terminal_state.apply_action(kPassAction), std::invalid_argument);

    GoPosition max_length_terminal{};
    max_length_terminal.komi = 0.0F;
    max_length_terminal.move_number = GoState::kMaxGameLength;
    alphazero::go::set_stone(&max_length_terminal, I(0, 0), kWhite);
    const GoState max_length_state(max_length_terminal);
    EXPECT_TRUE(max_length_state.is_terminal());
    EXPECT_FLOAT_EQ(max_length_state.outcome(GoState::kBlackPlayer), -1.0F);
    EXPECT_FLOAT_EQ(max_length_state.outcome(GoState::kWhitePlayer), 1.0F);

    const GoState non_terminal{};
    EXPECT_FALSE(non_terminal.is_terminal());
    EXPECT_FLOAT_EQ(non_terminal.outcome(GoState::kBlackPlayer), 0.0F);
    EXPECT_THROW(
        {
            (void)non_terminal.outcome(2);
        },
        std::invalid_argument);
}

// WHY: Go requires T=8 internal history for NN input; this guards copy-on-write ancestry and zero-fill semantics.
TEST(GoStateTest, HistoryRetainsEightStatesAndEncodeUsesCurrentPerspective) {
    const GoState initial{};
    std::vector<float> initial_encoded(GoState::kTotalInputChannels * kBoardArea, -1.0F);
    initial.encode(initial_encoded.data());

    for (int history_index = 1; history_index < GoState::kHistorySteps; ++history_index) {
        const int base_plane = history_index * GoState::kPlanesPerStep;
        for (int intersection = 0; intersection < kBoardArea; ++intersection) {
            EXPECT_FLOAT_EQ(initial_encoded[feature_index(base_plane, intersection)], 0.0F);
            EXPECT_FLOAT_EQ(initial_encoded[feature_index(base_plane + 1, intersection)], 0.0F);
        }
    }
    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        EXPECT_FLOAT_EQ(initial_encoded[feature_index(GoState::kTotalInputChannels - 1, intersection)], 1.0F);
    }

    std::vector<int> actions = {
        I(0, 0),
        I(0, 1),
        I(1, 0),
        I(1, 1),
        I(2, 0),
        I(2, 1),
        I(3, 0),
        I(3, 1),
        I(4, 0),
    };

    std::unique_ptr<GoState> state = std::make_unique<GoState>();
    for (int action : actions) {
        state = apply_action_as_go_state(*state, action);
    }
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->history_size(), GoState::kHistorySteps);
    EXPECT_EQ(state->history_position(0).move_number, 9);
    EXPECT_EQ(state->history_position(GoState::kHistorySteps - 1).move_number, 2);

    const GoPosition& oldest_visible = state->history_position(GoState::kHistorySteps - 1);
    EXPECT_EQ(alphazero::go::stone_at(oldest_visible, actions[0]), kBlack);
    EXPECT_EQ(alphazero::go::stone_at(oldest_visible, actions[1]), kWhite);
    EXPECT_EQ(alphazero::go::stone_at(oldest_visible, actions[2]), kEmpty);
    EXPECT_THROW(
        {
            const GoPosition& ignored = state->history_position(GoState::kHistorySteps);
            (void)ignored;
        },
        std::out_of_range);

    std::vector<float> encoded(GoState::kTotalInputChannels * kBoardArea, 0.0F);
    state->encode(encoded.data());

    const int newest_plane_base = 0;
    EXPECT_FLOAT_EQ(encoded[feature_index(newest_plane_base + 0, actions[1])], 1.0F);
    EXPECT_FLOAT_EQ(encoded[feature_index(newest_plane_base + 0, actions[0])], 0.0F);
    EXPECT_FLOAT_EQ(encoded[feature_index(newest_plane_base + 1, actions[0])], 1.0F);

    const int oldest_plane_base = (GoState::kHistorySteps - 1) * GoState::kPlanesPerStep;
    EXPECT_FLOAT_EQ(encoded[feature_index(oldest_plane_base + 0, actions[1])], 1.0F);
    EXPECT_FLOAT_EQ(encoded[feature_index(oldest_plane_base + 1, actions[0])], 1.0F);
    EXPECT_FLOAT_EQ(encoded[feature_index(oldest_plane_base + 1, actions[2])], 0.0F);

    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        EXPECT_FLOAT_EQ(encoded[feature_index(GoState::kTotalInputChannels - 1, intersection)], 0.0F);
    }
}
