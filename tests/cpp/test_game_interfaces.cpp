#include "games/game_config.h"
#include "games/game_state.h"

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {

class MinimalGameState final : public alphazero::GameState {
public:
    explicit MinimalGameState(int player = 0, int last_action = -1) : player_(player), last_action_(last_action) {}

    [[nodiscard]] std::unique_ptr<alphazero::GameState> apply_action(int action) const override {
        return std::make_unique<MinimalGameState>((player_ + 1) % 2, action);
    }

    [[nodiscard]] std::vector<int> legal_actions() const override { return {0, 1, 2}; }

    [[nodiscard]] bool is_terminal() const override { return false; }

    [[nodiscard]] float outcome(int /*player*/) const override { return 0.0F; }

    [[nodiscard]] int current_player() const override { return player_; }

    void encode(float* buffer) const override {
        if (buffer != nullptr) {
            buffer[0] = static_cast<float>(player_);
            buffer[1] = static_cast<float>(last_action_);
        }
    }

    [[nodiscard]] std::unique_ptr<alphazero::GameState> clone() const override {
        return std::make_unique<MinimalGameState>(*this);
    }

    [[nodiscard]] std::uint64_t hash() const override {
        return static_cast<std::uint64_t>(player_ * 31 + (last_action_ + 1));
    }

    [[nodiscard]] std::string to_string() const override {
        return "MinimalGameState(player=" + std::to_string(player_) +
               ", last_action=" + std::to_string(last_action_) + ")";
    }

private:
    int player_;
    int last_action_;
};

class MinimalGameConfig final : public alphazero::GameConfig {
public:
    MinimalGameConfig() {
        name = "minimal";
        board_rows = 1;
        board_cols = 1;
        planes_per_step = 1;
        num_history_steps = 1;
        constant_planes = 0;
        total_input_channels = 1;
        action_space_size = 3;
        dirichlet_alpha = 0.3F;
        max_game_length = 8;
        value_head_type = ValueHeadType::SCALAR;
        supports_symmetry = false;
        num_symmetries = 1;
    }

    [[nodiscard]] std::unique_ptr<alphazero::GameState> new_game() const override {
        return std::make_unique<MinimalGameState>();
    }
};

}  // namespace

// WHY: These checks lock the abstract interface so downstream code can rely on the exact API surface.
TEST(GameInterfaceContractTest, GameStateMethodSignaturesMatchSpec) {
    using ApplyActionSig = std::unique_ptr<alphazero::GameState> (alphazero::GameState::*)(int) const;
    using LegalActionsSig = std::vector<int> (alphazero::GameState::*)() const;
    using IsTerminalSig = bool (alphazero::GameState::*)() const;
    using OutcomeSig = float (alphazero::GameState::*)(int) const;
    using CurrentPlayerSig = int (alphazero::GameState::*)() const;
    using EncodeSig = void (alphazero::GameState::*)(float*) const;
    using CloneSig = std::unique_ptr<alphazero::GameState> (alphazero::GameState::*)() const;
    using HashSig = std::uint64_t (alphazero::GameState::*)() const;
    using ToStringSig = std::string (alphazero::GameState::*)() const;

    EXPECT_TRUE(std::is_abstract_v<alphazero::GameState>);
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::apply_action), ApplyActionSig>));
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::legal_actions), LegalActionsSig>));
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::is_terminal), IsTerminalSig>));
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::outcome), OutcomeSig>));
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::current_player), CurrentPlayerSig>));
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::encode), EncodeSig>));
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::clone), CloneSig>));
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::hash), HashSig>));
    EXPECT_TRUE((std::is_same_v<decltype(&alphazero::GameState::to_string), ToStringSig>));
}

// WHY: get_symmetries() is used by the training pipeline; identity must be available even before game-specific overrides.
TEST(GameInterfaceContractTest, DefaultSymmetryIsIdentityTransform) {
    MinimalGameConfig config;
    auto transforms = config.get_symmetries();

    ASSERT_EQ(transforms.size(), 1U);
    ASSERT_NE(transforms.front(), nullptr);

    std::array<float, 6> board{1.0F, -2.0F, 3.5F, 4.0F, 0.0F, 8.0F};
    const std::array<float, 6> board_before = board;
    transforms.front()->transform_board(board.data(), 1, 2, 3);
    EXPECT_EQ(board, board_before);

    std::array<float, 4> policy{0.1F, 0.2F, 0.3F, 0.4F};
    const std::array<float, 4> policy_before = policy;
    transforms.front()->transform_policy(policy.data(), static_cast<int>(policy.size()));
    EXPECT_EQ(policy, policy_before);
}

// WHY: This ensures the config factory can produce a valid GameState implementation that obeys the interface.
TEST(GameInterfaceContractTest, ConfigFactoryProducesCloneableGameState) {
    MinimalGameConfig config;
    std::unique_ptr<alphazero::GameState> state = config.new_game();
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->current_player(), 0);

    std::unique_ptr<alphazero::GameState> next_state = state->apply_action(2);
    ASSERT_NE(next_state, nullptr);
    EXPECT_EQ(next_state->current_player(), 1);

    std::unique_ptr<alphazero::GameState> cloned = next_state->clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_EQ(cloned->hash(), next_state->hash());
    EXPECT_EQ(cloned->to_string(), next_state->to_string());
}
