#include "selfplay/self_play_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
using alphazero::selfplay::SelfPlayManager;
using alphazero::selfplay::SelfPlayManagerConfig;
using alphazero::selfplay::SelfPlayMetricsSnapshot;

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
    explicit ToyGameConfig(std::shared_ptr<const ToyGameModel> model)
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
        max_game_length = 32;
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

[[nodiscard]] SelfPlayManager::EvaluateFn make_evaluator(
    std::unordered_map<int, EvaluationResult> table,
    std::chrono::milliseconds delay = std::chrono::milliseconds(0)) {
    return [table = std::move(table), delay](const GameState& state) -> EvaluationResult {
        const auto* toy_state = dynamic_cast<const ToyGameState*>(&state);
        if (toy_state == nullptr) {
            throw std::invalid_argument("Toy evaluator received a non-toy state");
        }
        if (delay.count() > 0) {
            std::this_thread::sleep_for(delay);
        }

        const auto it = table.find(toy_state->state_id());
        if (it == table.end()) {
            throw std::invalid_argument("Toy evaluator has no entry for the requested state");
        }
        return it->second;
    };
}

[[nodiscard]] SelfPlayManagerConfig default_manager_config(
    std::size_t concurrent_games,
    std::size_t max_games_per_slot,
    std::size_t simulations_per_move,
    std::size_t threads_per_game) {
    SelfPlayManagerConfig config{};
    config.concurrent_games = concurrent_games;
    config.max_games_per_slot = max_games_per_slot;
    config.initial_game_id = 1000U;
    config.random_seed = 17U;

    config.game_config.simulations_per_move = simulations_per_move;
    config.game_config.mcts_threads = threads_per_game;
    config.game_config.node_arena_capacity = 128U;
    config.game_config.enable_dirichlet_noise = false;
    config.game_config.temperature = 0.0F;
    config.game_config.temperature_moves = 0;
    config.game_config.enable_resignation = false;
    config.game_config.resign_disable_fraction = 0.0F;
    config.game_config.random_seed = 99U;
    return config;
}

[[nodiscard]] bool wait_until(
    const std::function<bool()>& condition,
    const std::chrono::milliseconds timeout,
    const std::chrono::milliseconds poll_interval = std::chrono::milliseconds(2)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(poll_interval);
    }
    return condition();
}

// WHY: The manager's primary contract is to keep M slots busy and recycle each slot into a new game after completion,
// continuously feeding replay data.
TEST(SelfPlayManagerTest, RunsConfiguredSlotsAndRecyclesFinishedGames) {
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
                .terminal_outcome = {-1.0F, 1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model);
    ReplayBuffer replay_buffer(256U, 1234U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = 0.4F, .policy_is_logits = true}},
    });

    const std::size_t slots = 3U;
    const std::size_t games_per_slot = 3U;
    const std::size_t simulations_per_move = 4U;
    SelfPlayManagerConfig config = default_manager_config(slots, games_per_slot, simulations_per_move, 2U);

    SelfPlayManager manager(game_config, replay_buffer, evaluator, config);
    manager.start();

    const std::size_t expected_games = slots * games_per_slot;
    const bool completed = wait_until(
        [&manager, expected_games] {
            const SelfPlayMetricsSnapshot snapshot = manager.metrics();
            return snapshot.games_completed >= expected_games && !manager.is_running();
        },
        std::chrono::seconds(2));
    manager.stop();

    ASSERT_TRUE(completed);
    const SelfPlayMetricsSnapshot snapshot = manager.metrics();
    EXPECT_EQ(snapshot.configured_slots, slots);
    EXPECT_EQ(snapshot.threads_per_game, 2U);
    EXPECT_EQ(snapshot.games_completed, expected_games);
    EXPECT_EQ(snapshot.replay_positions_written, expected_games);
    EXPECT_EQ(snapshot.total_moves, expected_games);
    EXPECT_EQ(snapshot.total_simulations, expected_games * simulations_per_move);
    EXPECT_EQ(snapshot.natural_terminations, expected_games);
    EXPECT_EQ(snapshot.resignation_terminations, 0U);
    EXPECT_EQ(snapshot.max_length_adjudications, 0U);
    EXPECT_TRUE(snapshot.has_latest_game);
    EXPECT_EQ(snapshot.latest_game_length, 1U);
    EXPECT_EQ(snapshot.active_slots, 0U);
    EXPECT_GT(snapshot.games_per_hour, 0.0);
    EXPECT_GT(snapshot.moves_per_second, 0.0);
    EXPECT_GT(snapshot.avg_simulations_per_second, 0.0);
    EXPECT_EQ(replay_buffer.size(), expected_games);
}

// WHY: Resignation outcomes directly affect replay targets and must be observable in aggregate metrics so thresholds can
// be tuned from production telemetry.
TEST(SelfPlayManagerTest, TracksResignationTerminationsAndSimulationThroughput) {
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
    ToyGameConfig game_config(model);
    ReplayBuffer replay_buffer(16U, 7U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {1.0F, -1.0F}, .value = -1.0F, .policy_is_logits = true}},
    });

    SelfPlayManagerConfig config = default_manager_config(/*concurrent_games=*/1U, /*max_games_per_slot=*/1U, 8U, 2U);
    config.game_config.enable_resignation = true;
    config.game_config.resign_threshold = -0.5F;
    config.game_config.resign_disable_fraction = 0.0F;

    SelfPlayManager manager(game_config, replay_buffer, evaluator, config);
    manager.start();
    const bool completed =
        wait_until([&manager] { return manager.metrics().games_completed >= 1U; }, std::chrono::seconds(1));
    manager.stop();

    ASSERT_TRUE(completed);
    const SelfPlayMetricsSnapshot snapshot = manager.metrics();
    EXPECT_EQ(snapshot.games_completed, 1U);
    EXPECT_EQ(snapshot.resignation_terminations, 1U);
    EXPECT_EQ(snapshot.natural_terminations, 0U);
    EXPECT_EQ(snapshot.total_moves, 0U);
    EXPECT_EQ(snapshot.replay_positions_written, 0U);
    EXPECT_EQ(snapshot.total_simulations, 8U);
    EXPECT_TRUE(snapshot.has_latest_game);
    EXPECT_TRUE(snapshot.latest_game_resigned);
    EXPECT_FALSE(snapshot.latest_resignation_disabled);
    EXPECT_EQ(replay_buffer.size(), 0U);
}

// WHY: Disabled-resignation calibration games must report false positives when a position that would have resigned
// eventually wins, otherwise resign-threshold tuning cannot be done safely.
TEST(SelfPlayManagerTest, CountsDisabledResignationFalsePositives) {
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
    ToyGameConfig game_config(model);
    ReplayBuffer replay_buffer(16U, 29U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = -1.0F, .policy_is_logits = true}},
        {1, EvaluationResult{.policy = {0.0F}, .value = 1.0F, .policy_is_logits = true}},
    });

    SelfPlayManagerConfig config = default_manager_config(/*concurrent_games=*/1U, /*max_games_per_slot=*/1U, 1U, 1U);
    config.game_config.enable_resignation = true;
    config.game_config.resign_threshold = -0.5F;
    config.game_config.resign_disable_fraction = 1.0F;

    SelfPlayManager manager(game_config, replay_buffer, evaluator, config);
    manager.start();
    const bool completed =
        wait_until([&manager] { return manager.metrics().games_completed >= 1U; }, std::chrono::seconds(1));
    manager.stop();

    ASSERT_TRUE(completed);
    const SelfPlayMetricsSnapshot snapshot = manager.metrics();
    EXPECT_EQ(snapshot.games_completed, 1U);
    EXPECT_EQ(snapshot.resignation_terminations, 0U);
    EXPECT_EQ(snapshot.natural_terminations, 1U);
    EXPECT_EQ(snapshot.resignation_disabled_games, 1U);
    EXPECT_EQ(snapshot.resignation_false_positive_games, 1U);
    EXPECT_TRUE(snapshot.latest_resignation_disabled);
    EXPECT_TRUE(snapshot.latest_resignation_false_positive);
    EXPECT_EQ(snapshot.latest_game_length, 2U);
    EXPECT_EQ(snapshot.total_simulations, 2U);
    EXPECT_EQ(replay_buffer.size(), 2U);
}

// WHY: Per-game Dirichlet epsilon randomization must override the base epsilon so workers can run controlled
// low-noise/high-noise mixes from a single config.
TEST(SelfPlayManagerTest, RandomizedDirichletEpsilonOverridesPerGameConfig) {
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
                .terminal = true,
                .terminal_outcome = {1.0F, -1.0F},
                .legal_actions = {},
                .transitions = {},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = true,
                .terminal_outcome = {-1.0F, 1.0F},
                .legal_actions = {},
                .transitions = {},
            },
        });
    ToyGameConfig game_config(model);
    ReplayBuffer replay_buffer(64U, 37U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F, 0.0F}, .value = 0.0F, .policy_is_logits = true}},
    });

    SelfPlayManagerConfig config = default_manager_config(/*concurrent_games=*/1U, /*max_games_per_slot=*/24U, 1U, 1U);
    config.game_config.enable_dirichlet_noise = true;
    config.game_config.dirichlet_epsilon = 1.0F;
    config.game_config.randomize_dirichlet_epsilon = true;
    config.game_config.dirichlet_epsilon_min = 0.0F;
    config.game_config.dirichlet_epsilon_max = 0.0F;
    config.game_config.temperature = 0.0F;
    config.game_config.temperature_moves = 0;

    std::mutex callback_mutex;
    std::vector<int> first_actions;
    auto completion_callback =
        [&callback_mutex, &first_actions](const std::size_t /*slot_index*/, const alphazero::selfplay::SelfPlayGameResult& result) {
            std::lock_guard lock(callback_mutex);
            if (!result.action_history.empty()) {
                first_actions.push_back(result.action_history.front());
            }
        };

    SelfPlayManager manager(game_config, replay_buffer, evaluator, config, completion_callback);
    manager.start();
    const bool completed =
        wait_until([&manager] { return manager.metrics().games_completed >= 24U; }, std::chrono::seconds(2));
    manager.stop();

    ASSERT_TRUE(completed);
    ASSERT_EQ(first_actions.size(), 24U);
    EXPECT_TRUE(std::all_of(first_actions.begin(), first_actions.end(), [](const int action) { return action == 0; }));
}

// WHY: Invalid randomized Dirichlet bounds should fail at manager construction before any worker threads launch.
TEST(SelfPlayManagerTest, RejectsInvalidRandomizedDirichletBounds) {
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
    ToyGameConfig game_config(model);
    ReplayBuffer replay_buffer(16U, 39U);
    const auto evaluator = make_evaluator({
        {0, EvaluationResult{.policy = {0.0F}, .value = 0.0F, .policy_is_logits = true}},
    });

    SelfPlayManagerConfig config = default_manager_config(/*concurrent_games=*/1U, /*max_games_per_slot=*/1U, 1U, 1U);
    config.game_config.randomize_dirichlet_epsilon = true;
    config.game_config.dirichlet_epsilon_min = 0.4F;
    config.game_config.dirichlet_epsilon_max = 0.3F;

    EXPECT_THROW((void)SelfPlayManager(game_config, replay_buffer, evaluator, config), std::invalid_argument);
}

// WHY: All configured slots should become active at startup; this is the root-parallelism requirement that drives
// batched inference utilization in the larger pipeline.
TEST(SelfPlayManagerTest, ActivatesAllConfiguredSlots) {
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
    ToyGameConfig game_config(model);
    ReplayBuffer replay_buffer(32U, 31U);
    const auto evaluator = make_evaluator(
        {
            {0, EvaluationResult{.policy = {0.0F}, .value = 0.0F, .policy_is_logits = true}},
        },
        std::chrono::milliseconds(80));

    SelfPlayManagerConfig config = default_manager_config(/*concurrent_games=*/4U, /*max_games_per_slot=*/1U, 1U, 1U);

    SelfPlayManager manager(game_config, replay_buffer, evaluator, config);
    manager.start();

    const bool saw_all_slots = wait_until(
        [&manager, &config] { return manager.metrics().active_slots == config.concurrent_games; },
        std::chrono::milliseconds(500));
    const bool completed =
        wait_until([&manager] { return manager.metrics().games_completed >= 4U; }, std::chrono::seconds(2));
    manager.stop();

    ASSERT_TRUE(completed);
    EXPECT_TRUE(saw_all_slots);

    const SelfPlayMetricsSnapshot snapshot = manager.metrics();
    EXPECT_EQ(snapshot.games_completed, 4U);
    EXPECT_EQ(snapshot.active_slots, 0U);
}

}  // namespace
