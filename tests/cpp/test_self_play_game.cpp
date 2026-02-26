#include "selfplay/self_play_game.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::GameConfig;
using alphazero::GameState;
using alphazero::mcts::EvaluationResult;
using alphazero::selfplay::GameTerminationReason;
using alphazero::selfplay::ReplayBuffer;
using alphazero::selfplay::ReplayPosition;
using alphazero::selfplay::SelfPlayGame;
using alphazero::selfplay::SelfPlayGameConfig;
using alphazero::selfplay::SelfPlayGameResult;

struct ToyStateSpec {
    int current_player = 0;
    bool terminal = false;
    std::array<float, 2> terminal_outcome{0.0F, 0.0F};
    std::vector<int> legal_actions;
    std::unordered_map<int, int> transitions;
};

struct ToyGameModel {
    int action_space_size = 1;
    int root_state = 0;
    std::vector<ToyStateSpec> states;
};

class ToyGameState final : public GameState {
public:
    ToyGameState(std::shared_ptr<const ToyGameModel> model, int state_id)
        : model_(std::move(model)),
          state_id_(state_id) {
        if (model_ == nullptr) {
            throw std::invalid_argument("ToyGameState model must be non-null");
        }
        if (state_id_ < 0 || state_id_ >= static_cast<int>(model_->states.size())) {
            throw std::invalid_argument("ToyGameState state id is out of range");
        }
    }

    [[nodiscard]] int state_id() const { return state_id_; }

    [[nodiscard]] std::unique_ptr<GameState> apply_action(int action) const override {
        const ToyStateSpec& spec = model_->states[static_cast<std::size_t>(state_id_)];
        if (spec.terminal) {
            throw std::invalid_argument("ToyGameState cannot transition from terminal state");
        }

        const auto it = spec.transitions.find(action);
        if (it == spec.transitions.end()) {
            throw std::invalid_argument("ToyGameState action is not legal in this state");
        }
        return std::make_unique<ToyGameState>(model_, it->second);
    }

    [[nodiscard]] std::vector<int> legal_actions() const override {
        const ToyStateSpec& spec = model_->states[static_cast<std::size_t>(state_id_)];
        return spec.terminal ? std::vector<int>{} : spec.legal_actions;
    }

    [[nodiscard]] bool is_terminal() const override {
        return model_->states[static_cast<std::size_t>(state_id_)].terminal;
    }

    [[nodiscard]] float outcome(int player) const override {
        if (!is_terminal()) {
            throw std::logic_error("ToyGameState outcome is only valid for terminal states");
        }
        if (player != 0 && player != 1) {
            throw std::invalid_argument("ToyGameState player must be 0 or 1");
        }
        return model_->states[static_cast<std::size_t>(state_id_)].terminal_outcome[static_cast<std::size_t>(player)];
    }

    [[nodiscard]] int current_player() const override {
        return model_->states[static_cast<std::size_t>(state_id_)].current_player;
    }

    void encode(float* buffer) const override {
        if (buffer == nullptr) {
            throw std::invalid_argument("ToyGameState encode buffer must be non-null");
        }
        buffer[0] = static_cast<float>(state_id_);
    }

    [[nodiscard]] std::unique_ptr<GameState> clone() const override {
        return std::make_unique<ToyGameState>(*this);
    }

    [[nodiscard]] std::uint64_t hash() const override {
        return static_cast<std::uint64_t>(state_id_ + 1);
    }

    [[nodiscard]] std::string to_string() const override {
        return "ToyState(" + std::to_string(state_id_) + ")";
    }

private:
    std::shared_ptr<const ToyGameModel> model_;
    int state_id_ = 0;
};

class ToyGameConfig final : public GameConfig {
public:
    ToyGameConfig(std::shared_ptr<const ToyGameModel> model, int max_game_length)
        : model_(std::move(model)) {
        if (model_ == nullptr) {
            throw std::invalid_argument("ToyGameConfig model must be non-null");
        }

        name = "toy";
        board_rows = 1;
        board_cols = 1;
        planes_per_step = 1;
        num_history_steps = 1;
        constant_planes = 0;
        total_input_channels = 1;
        action_space_size = model_->action_space_size;
        dirichlet_alpha = 0.3F;
        this->max_game_length = max_game_length;
        value_head_type = ValueHeadType::SCALAR;
        supports_symmetry = false;
        num_symmetries = 1;
    }

    [[nodiscard]] std::unique_ptr<GameState> new_game() const override {
        return std::make_unique<ToyGameState>(model_, model_->root_state);
    }

private:
    std::shared_ptr<const ToyGameModel> model_;
};

[[nodiscard]] std::shared_ptr<ToyGameModel> make_model(
    int action_space_size,
    std::vector<ToyStateSpec> states,
    int root_state = 0) {
    auto model = std::make_shared<ToyGameModel>();
    model->action_space_size = action_space_size;
    model->root_state = root_state;
    model->states = std::move(states);
    return model;
}

[[nodiscard]] SelfPlayGame::EvaluateFn make_evaluator(std::unordered_map<int, EvaluationResult> table) {
    return [table = std::move(table)](const GameState& state) -> EvaluationResult {
        const auto* toy_state = dynamic_cast<const ToyGameState*>(&state);
        if (toy_state == nullptr) {
            throw std::invalid_argument("Toy evaluator received a non-toy state");
        }

        const auto it = table.find(toy_state->state_id());
        if (it == table.end()) {
            throw std::invalid_argument("Toy evaluator has no entry for the requested state");
        }
        return it->second;
    };
}

[[nodiscard]] std::vector<ReplayPosition> sorted_replay_positions(const ReplayBuffer& replay_buffer) {
    const std::vector<ReplayPosition> sampled = replay_buffer.sample(replay_buffer.size());
    std::vector<ReplayPosition> sorted = sampled;
    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const ReplayPosition& left, const ReplayPosition& right) { return left.move_number < right.move_number; });
    return sorted;
}

[[nodiscard]] SelfPlayGameConfig default_test_selfplay_config() {
    SelfPlayGameConfig config{};
    config.simulations_per_move = 16U;
    config.mcts_threads = 2U;
    config.node_arena_capacity = 128U;
    config.enable_dirichlet_noise = false;
    config.temperature = 0.0F;
    config.temperature_moves = 0;
    config.enable_resignation = false;
    config.resign_threshold = -0.9F;
    config.resign_disable_fraction = 0.0F;
    config.random_seed = 17U;
    return config;
}

// WHY: Playout-cap configuration is validated at construction so misconfigured training runs fail fast before worker
// threads start.
TEST(SelfPlayGameTest, RejectsZeroReducedSimulations) {
    const auto model = make_model(
        1,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/8);
    ReplayBuffer replay_buffer(4U, 41U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = 0.0F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig config = default_test_selfplay_config();
    config.enable_playout_cap = true;
    config.reduced_simulations = 0U;

    EXPECT_THROW((void)SelfPlayGame(game_config, replay_buffer, evaluator, config), std::invalid_argument);
}

// WHY: The reduced simulation budget must be a true reduction; larger values would invert intended playout-cap
// semantics and break future weighting logic.
TEST(SelfPlayGameTest, RejectsReducedSimulationsAboveFullBudget) {
    const auto model = make_model(
        1,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/8);
    ReplayBuffer replay_buffer(4U, 43U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = 0.0F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig config = default_test_selfplay_config();
    config.enable_playout_cap = true;
    config.reduced_simulations = config.simulations_per_move + 1U;

    EXPECT_THROW((void)SelfPlayGame(game_config, replay_buffer, evaluator, config), std::invalid_argument);
}

// WHY: Probability knobs are user-configured from YAML and must reject out-of-range values to prevent undefined
// stochastic behavior when playout caps are enabled.
TEST(SelfPlayGameTest, RejectsPlayoutProbabilityOutsideUnitInterval) {
    const auto model = make_model(
        1,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/8);
    ReplayBuffer replay_buffer(4U, 47U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = 0.0F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig negative_prob = default_test_selfplay_config();
    negative_prob.enable_playout_cap = true;
    negative_prob.full_playout_probability = -0.01F;
    EXPECT_THROW((void)SelfPlayGame(game_config, replay_buffer, evaluator, negative_prob), std::invalid_argument);

    SelfPlayGameConfig over_one_prob = default_test_selfplay_config();
    over_one_prob.enable_playout_cap = true;
    over_one_prob.full_playout_probability = 1.01F;
    EXPECT_THROW((void)SelfPlayGame(game_config, replay_buffer, evaluator, over_one_prob), std::invalid_argument);
}

// WHY: Dirichlet-randomization bounds come from YAML and must be validated up front to avoid launching games with
// invalid per-game noise schedules.
TEST(SelfPlayGameTest, RejectsInvalidDirichletRandomizationBounds) {
    const auto model = make_model(
        1,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/8);
    ReplayBuffer replay_buffer(4U, 53U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = 0.0F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig bad_order = default_test_selfplay_config();
    bad_order.randomize_dirichlet_epsilon = true;
    bad_order.dirichlet_epsilon_min = 0.4F;
    bad_order.dirichlet_epsilon_max = 0.2F;
    EXPECT_THROW((void)SelfPlayGame(game_config, replay_buffer, evaluator, bad_order), std::invalid_argument);

    SelfPlayGameConfig out_of_range = default_test_selfplay_config();
    out_of_range.randomize_dirichlet_epsilon = true;
    out_of_range.dirichlet_epsilon_min = -0.01F;
    out_of_range.dirichlet_epsilon_max = 0.3F;
    EXPECT_THROW((void)SelfPlayGame(game_config, replay_buffer, evaluator, out_of_range), std::invalid_argument);
}

// WHY: When playout-cap mode always selects the reduced simulation budget, replay samples must carry a reduced
// training weight so loss scaling matches search effort for those positions.
TEST(SelfPlayGameTest, PlayoutCapUsesReducedWeightWhenFullBudgetIsNeverSelected) {
    const auto model = make_model(
        1,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/8);
    ReplayBuffer replay_buffer(8U, 71U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = 0.0F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig config = default_test_selfplay_config();
    config.enable_playout_cap = true;
    config.simulations_per_move = 16U;
    config.reduced_simulations = 4U;
    config.full_playout_probability = 0.0F;

    SelfPlayGame game(game_config, replay_buffer, evaluator, config);
    const SelfPlayGameResult result = game.play(/*game_id=*/70U);

    EXPECT_EQ(result.termination_reason, GameTerminationReason::kNatural);
    ASSERT_EQ(replay_buffer.size(), 1U);
    const std::vector<ReplayPosition> replay = sorted_replay_positions(replay_buffer);
    ASSERT_EQ(replay.size(), 1U);
    EXPECT_FLOAT_EQ(replay[0].training_weight, 0.25F);
}

// WHY: The playout-cap feature must leave training weights unchanged when the full simulation budget is chosen every
// move so enabling the feature with probability 1.0 is behaviorally equivalent to baseline self-play.
TEST(SelfPlayGameTest, PlayoutCapKeepsUnitWeightWhenFullBudgetAlwaysSelected) {
    const auto model = make_model(
        1,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/8);
    ReplayBuffer replay_buffer(8U, 73U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = 0.0F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig config = default_test_selfplay_config();
    config.enable_playout_cap = true;
    config.simulations_per_move = 16U;
    config.reduced_simulations = 4U;
    config.full_playout_probability = 1.0F;

    SelfPlayGame game(game_config, replay_buffer, evaluator, config);
    const SelfPlayGameResult result = game.play(/*game_id=*/72U);

    EXPECT_EQ(result.termination_reason, GameTerminationReason::kNatural);
    ASSERT_EQ(replay_buffer.size(), 1U);
    const std::vector<ReplayPosition> replay = sorted_replay_positions(replay_buffer);
    ASSERT_EQ(replay.size(), 1U);
    EXPECT_FLOAT_EQ(replay[0].training_weight, 1.0F);
}

// WHY: A complete self-play game should emit one replay sample per played move with outcomes labeled from the player
// to move at each sampled state, and should report subtree reuse when advancing the root.
TEST(SelfPlayGameTest, NaturalTerminationBackfillsPerspectiveTargetsAndTracksReuse) {
    const auto model = make_model(
        2,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0, 1},
                .transitions = {{0, 1}, {1, 2}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 3}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {-1.0F, 1.0F},
                .legal_actions = {},
                .transitions = {},
            },
            ToyStateSpec{
                .current_player = 0,
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/64);
    ReplayBuffer replay_buffer(16U, 1234U);

    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {3.0F, -3.0F}, .value = 0.2F, .policy_is_logits = true}},
        {1, EvaluationResult{.policy = {2.0F, -2.0F}, .value = -0.7F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig config = default_test_selfplay_config();
    SelfPlayGame game(game_config, replay_buffer, evaluator, config);
    const SelfPlayGameResult result = game.play(/*game_id=*/77U);

    EXPECT_EQ(result.termination_reason, GameTerminationReason::kNatural);
    EXPECT_EQ(result.move_count, 2U);
    EXPECT_EQ(result.action_history, (std::vector<int>{0, 0}));
    EXPECT_EQ(result.replay_positions_written, 2U);
    EXPECT_GE(result.reused_subtree_count, 1U);
    EXPECT_FLOAT_EQ(result.outcome_player0, 1.0F);
    EXPECT_FLOAT_EQ(result.outcome_player1, -1.0F);

    ASSERT_EQ(replay_buffer.size(), 2U);
    const std::vector<ReplayPosition> replay = sorted_replay_positions(replay_buffer);
    ASSERT_EQ(replay.size(), 2U);

    EXPECT_EQ(replay[0].move_number, 0U);
    EXPECT_FLOAT_EQ(replay[0].encoded_state[0], 0.0F);
    EXPECT_FLOAT_EQ(replay[0].value, 1.0F);
    EXPECT_FLOAT_EQ(replay[0].training_weight, 1.0F);
    EXPECT_EQ(replay[0].value_wdl, (std::array<float, 3>{1.0F, 0.0F, 0.0F}));
    EXPECT_EQ(replay[0].policy_size, 2U);
    EXPECT_FLOAT_EQ(replay[0].policy[0], 1.0F);
    EXPECT_FLOAT_EQ(replay[0].policy[1], 0.0F);

    EXPECT_EQ(replay[1].move_number, 1U);
    EXPECT_FLOAT_EQ(replay[1].encoded_state[0], 1.0F);
    EXPECT_FLOAT_EQ(replay[1].value, -1.0F);
    EXPECT_FLOAT_EQ(replay[1].training_weight, 1.0F);
    EXPECT_EQ(replay[1].value_wdl, (std::array<float, 3>{0.0F, 0.0F, 1.0F}));
    EXPECT_EQ(replay[1].policy_size, 2U);
    EXPECT_FLOAT_EQ(replay[1].policy[0], 1.0F);
    EXPECT_FLOAT_EQ(replay[1].policy[1], 0.0F);
}

// WHY: When resignation is enabled and thresholds are crossed, the game should terminate immediately without adding
// replay samples for unplayed moves, and outcomes must reflect resignation from the side to move.
TEST(SelfPlayGameTest, ResignationTerminatesGameWhenEnabledAndNotDisabled) {
    const auto model = make_model(
        2,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {-1.0F, 1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/16);
    ReplayBuffer replay_buffer(8U, 42U);

    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {1.0F, -1.0F}, .value = -1.0F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig config = default_test_selfplay_config();
    config.enable_resignation = true;
    config.resign_threshold = -0.5F;
    config.resign_disable_fraction = 0.0F;
    config.simulations_per_move = 8U;

    SelfPlayGame game(game_config, replay_buffer, evaluator, config);
    const SelfPlayGameResult result = game.play(/*game_id=*/11U);

    EXPECT_EQ(result.termination_reason, GameTerminationReason::kResignation);
    EXPECT_FALSE(result.resignation_was_disabled);
    EXPECT_TRUE(result.resignation_would_have_triggered);
    EXPECT_EQ(result.move_count, 0U);
    EXPECT_TRUE(result.action_history.empty());
    EXPECT_EQ(result.replay_positions_written, 0U);
    EXPECT_FLOAT_EQ(result.outcome_player0, -1.0F);
    EXPECT_FLOAT_EQ(result.outcome_player1, 1.0F);
    EXPECT_EQ(replay_buffer.size(), 0U);
}

// WHY: The per-game resignation-disable fraction must allow games to continue through positions that otherwise would
// trigger resignation so calibration games still produce data.
TEST(SelfPlayGameTest, ResignationDisableFractionForcesPlaythrough) {
    const auto model = make_model(
        2,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {-1.0F, 1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/16);
    ReplayBuffer replay_buffer(8U, 99U);

    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {1.0F, -1.0F}, .value = -1.0F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig config = default_test_selfplay_config();
    config.enable_resignation = true;
    config.resign_threshold = -0.5F;
    config.resign_disable_fraction = 1.0F;
    config.simulations_per_move = 8U;

    SelfPlayGame game(game_config, replay_buffer, evaluator, config);
    const SelfPlayGameResult result = game.play(/*game_id=*/12U);

    EXPECT_EQ(result.termination_reason, GameTerminationReason::kNatural);
    EXPECT_TRUE(result.resignation_was_disabled);
    EXPECT_TRUE(result.resignation_would_have_triggered);
    EXPECT_EQ(result.move_count, 1U);
    EXPECT_EQ(result.action_history, (std::vector<int>{0}));
    EXPECT_EQ(result.replay_positions_written, 1U);
    EXPECT_FLOAT_EQ(result.outcome_player0, -1.0F);
    EXPECT_FLOAT_EQ(result.outcome_player1, 1.0F);

    ASSERT_EQ(replay_buffer.size(), 1U);
    const std::vector<ReplayPosition> replay = sorted_replay_positions(replay_buffer);
    ASSERT_EQ(replay.size(), 1U);
    EXPECT_FLOAT_EQ(replay[0].value, -1.0F);
    EXPECT_FLOAT_EQ(replay[0].training_weight, 1.0F);
    EXPECT_EQ(replay[0].value_wdl, (std::array<float, 3>{0.0F, 0.0F, 1.0F}));
}

// WHY: If the configured max game length is reached before natural terminal conditions, self-play must adjudicate the
// game and backfill replay values consistently.
TEST(SelfPlayGameTest, MaxLengthAdjudicationStopsNonTerminalGameAndBackfillsDrawTargets) {
    const auto model = make_model(
        2,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 1}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0},
                .transitions = {{0, 2}},
            },
            ToyStateSpec{
                .current_player = 0,
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model, /*max_game_length=*/1);
    ReplayBuffer replay_buffer(8U, 5U);

    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {2.0F, -2.0F}, .value = 0.1F, .policy_is_logits = true}},
        {1, EvaluationResult{.policy = {2.0F, -2.0F}, .value = 0.1F, .policy_is_logits = true}},
    });

    SelfPlayGameConfig config = default_test_selfplay_config();
    config.simulations_per_move = 4U;

    SelfPlayGame game(game_config, replay_buffer, evaluator, config);
    const SelfPlayGameResult result = game.play(/*game_id=*/13U);

    EXPECT_EQ(result.termination_reason, GameTerminationReason::kMaxLengthAdjudication);
    EXPECT_EQ(result.move_count, 1U);
    EXPECT_EQ(result.action_history, (std::vector<int>{0}));
    EXPECT_EQ(result.replay_positions_written, 1U);
    EXPECT_FLOAT_EQ(result.outcome_player0, 0.0F);
    EXPECT_FLOAT_EQ(result.outcome_player1, 0.0F);

    ASSERT_EQ(replay_buffer.size(), 1U);
    const std::vector<ReplayPosition> replay = sorted_replay_positions(replay_buffer);
    ASSERT_EQ(replay.size(), 1U);
    EXPECT_FLOAT_EQ(replay[0].value, 0.0F);
    EXPECT_FLOAT_EQ(replay[0].training_weight, 1.0F);
    EXPECT_EQ(replay[0].value_wdl, (std::array<float, 3>{0.0F, 1.0F, 0.0F}));
}

}  // namespace
