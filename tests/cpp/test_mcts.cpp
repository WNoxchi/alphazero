#include "mcts/mcts_search.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "games/chess/chess_config.h"
#include "games/go/go_config.h"
#include "mcts/arena_node_store.h"
#include "mcts/mcts_node.h"

#include <gtest/gtest.h>

namespace {

using alphazero::GameState;
using alphazero::mcts::ArenaNodeStore;
using alphazero::mcts::EvaluationResult;
using alphazero::mcts::MCTSNode;
using alphazero::mcts::MctsSearch;
using alphazero::mcts::NodeId;
using alphazero::mcts::RuntimeMctsSearch;
using alphazero::mcts::SearchConfig;

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
            throw std::invalid_argument("ToyGameState state id out of range");
        }
    }

    [[nodiscard]] int state_id() const { return state_id_; }

    [[nodiscard]] std::unique_ptr<GameState> apply_action(int action) const override {
        const ToyStateSpec& spec = model_->states[static_cast<std::size_t>(state_id_)];
        if (spec.terminal) {
            throw std::invalid_argument("ToyGameState cannot apply actions from a terminal state");
        }

        const auto transition_it = spec.transitions.find(action);
        if (transition_it == spec.transitions.end()) {
            throw std::invalid_argument("ToyGameState action is not legal in this state");
        }

        return std::make_unique<ToyGameState>(model_, transition_it->second);
    }

    [[nodiscard]] std::vector<int> legal_actions() const override {
        const ToyStateSpec& spec = model_->states[static_cast<std::size_t>(state_id_)];
        if (spec.terminal) {
            return {};
        }
        return spec.legal_actions;
    }

    [[nodiscard]] bool is_terminal() const override {
        return model_->states[static_cast<std::size_t>(state_id_)].terminal;
    }

    [[nodiscard]] float outcome(int player) const override {
        if (!is_terminal()) {
            throw std::logic_error("ToyGameState outcome is only valid for terminal states");
        }
        if (player < 0 || player > 1) {
            throw std::invalid_argument("ToyGameState player index must be 0 or 1");
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

class ToyGameConfig final : public alphazero::GameConfig {
public:
    explicit ToyGameConfig(std::shared_ptr<const ToyGameModel> model) : model_(std::move(model)) {
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
        max_game_length = 512;
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
    model->states = std::move(states);
    model->root_state = root_state;
    return model;
}

[[nodiscard]] alphazero::mcts::EvaluateFn make_evaluator(std::unordered_map<int, EvaluationResult> table) {
    return [table = std::move(table)](const GameState& state) -> EvaluationResult {
        const auto* toy_state = dynamic_cast<const ToyGameState*>(&state);
        if (toy_state == nullptr) {
            throw std::invalid_argument("Toy evaluator received a non-toy state");
        }

        const auto it = table.find(toy_state->state_id());
        if (it == table.end()) {
            throw std::invalid_argument("Toy evaluator has no entry for requested state");
        }
        return it->second;
    };
}

template <typename T, std::size_t N>
[[nodiscard]] bool all_equal(const std::array<T, N>& values, const T expected) {
    for (const T value : values) {
        if (value != expected) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] int find_action_slot(const MCTSNode& node, int action) {
    for (int i = 0; i < node.num_actions; ++i) {
        if (node.actions[static_cast<std::size_t>(i)] == action) {
            return i;
        }
    }
    return -1;
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

// WHY: Runtime dispatch must choose chess-sized nodes for chess configs and go-sized nodes for go configs.
TEST(MctsNodeStructureTest, RuntimeSearchDispatchesToGameSpecificNodeCapacity) {
    const alphazero::chess::ChessGameConfig chess_config{};
    const alphazero::go::GoGameConfig go_config{};

    const RuntimeMctsSearch chess_search(chess_config, SearchConfig{}, /*node_arena_capacity=*/64U);
    const RuntimeMctsSearch go_search(go_config, SearchConfig{}, /*node_arena_capacity=*/64U);

    EXPECT_EQ(chess_search.node_capacity_actions(), alphazero::mcts::kChessMaxActions);
    EXPECT_EQ(go_search.node_capacity_actions(), alphazero::mcts::kGoMaxActions);
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

// WHY: PUCT search must learn to prefer stronger continuations from backup values, not just raw priors.
TEST(MctsSearchTest, PuctVisitDistributionShiftsTowardBetterActionDespiteWorsePrior) {
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

    ToyGameConfig config(model);
    ArenaNodeStore store(256);
    SearchConfig search_config{};
    search_config.enable_dirichlet_noise = false;
    MctsSearch search(store, config, search_config);
    search.set_root_state(config.new_game());

    const auto evaluator = make_evaluator({
        {0,
         EvaluationResult{
             .policy = {0.1F, 2.0F},
             .value = 0.0F,
             .policy_is_logits = true,
         }},
    });

    search.run_simulations(64, evaluator);

    const auto strong_edge = search.root_edge_stats(0);
    const auto weak_edge = search.root_edge_stats(1);
    ASSERT_TRUE(strong_edge.has_value());
    ASSERT_TRUE(weak_edge.has_value());

    EXPECT_GT(strong_edge->visit_count, weak_edge->visit_count);
    EXPECT_GT(strong_edge->mean_value, weak_edge->mean_value);
    EXPECT_EQ(search.select_action(40), 0);
}

// WHY: Backup must negate values per ply so parent and child Q estimates have opposite signs when players alternate.
TEST(MctsSearchTest, BackupNegatesValueAcrossAlternatingPlayers) {
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

    ToyGameConfig config(model);
    ArenaNodeStore store(256);
    MctsSearch search(store, config);
    search.set_root_state(config.new_game());

    const auto evaluator = make_evaluator({
        {0,
         EvaluationResult{
             .policy = {1.0F},
             .value = 0.0F,
             .policy_is_logits = false,
         }},
        {1,
         EvaluationResult{
             .policy = {1.0F},
             .value = 0.0F,
             .policy_is_logits = false,
         }},
    });

    search.run_simulations(2, evaluator);

    const auto root_edge = search.root_edge_stats(0);
    ASSERT_TRUE(root_edge.has_value());
    ASSERT_NE(root_edge->child, alphazero::mcts::NULL_NODE);

    const NodeId child_id = root_edge->child;
    const MCTSNode& child_node = store.get(child_id);
    const int child_slot = find_action_slot(child_node, 0);
    ASSERT_GE(child_slot, 0);

    EXPECT_EQ(root_edge->visit_count, 2);
    EXPECT_FLOAT_EQ(root_edge->total_value, 1.0F);
    EXPECT_FLOAT_EQ(root_edge->mean_value, 0.5F);

    EXPECT_EQ(child_node.visit_count[static_cast<std::size_t>(child_slot)], 1);
    EXPECT_FLOAT_EQ(child_node.total_value[static_cast<std::size_t>(child_slot)], -1.0F);
    EXPECT_FLOAT_EQ(child_node.mean_value[static_cast<std::size_t>(child_slot)], -1.0F);
}

// WHY: Virtual loss must be visible while a simulation is in-flight and fully reverted before real backup is applied.
TEST(MctsSearchTest, VirtualLossIsAppliedDuringSelectionAndRevertedDuringBackup) {
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
                .legal_actions = {},
                .transitions = {},
            },
        });

    ToyGameConfig config(model);
    ArenaNodeStore store(256);
    SearchConfig search_config{};
    search_config.enable_dirichlet_noise = false;
    MctsSearch search(store, config, search_config);
    search.set_root_state(config.new_game());

    std::promise<void> entered_leaf_eval;
    std::shared_future<void> entered_leaf_future = entered_leaf_eval.get_future().share();
    std::promise<void> release_leaf_eval;
    std::shared_future<void> release_leaf_future = release_leaf_eval.get_future().share();
    std::atomic<bool> entered_signaled{false};

    const alphazero::mcts::EvaluateFn evaluator =
        [&](const GameState& state) -> EvaluationResult {
        const auto* toy_state = dynamic_cast<const ToyGameState*>(&state);
        if (toy_state == nullptr) {
            throw std::invalid_argument("Expected toy state in evaluator");
        }

        if (toy_state->state_id() == 0) {
            return EvaluationResult{
                .policy = {1.0F},
                .value = 0.0F,
                .policy_is_logits = false,
            };
        }

        if (!entered_signaled.exchange(true)) {
            entered_leaf_eval.set_value();
        }
        release_leaf_future.wait();

        return EvaluationResult{
            .policy = {1.0F},
            .value = 0.4F,
            .policy_is_logits = false,
        };
    };

    std::thread worker([&search, &evaluator] { search.run_simulation(evaluator); });

    entered_leaf_future.wait();

    const auto inflight = search.root_edge_stats(0);
    ASSERT_TRUE(inflight.has_value());
    EXPECT_EQ(inflight->visit_count, 1);
    EXPECT_EQ(inflight->virtual_loss, 1);
    EXPECT_EQ(inflight->total_value, -1.0F);
    EXPECT_EQ(inflight->mean_value, -1.0F);

    release_leaf_eval.set_value();
    worker.join();

    const auto settled = search.root_edge_stats(0);
    ASSERT_TRUE(settled.has_value());
    EXPECT_EQ(settled->visit_count, 1);
    EXPECT_EQ(settled->virtual_loss, 0);
    EXPECT_FLOAT_EQ(settled->total_value, -0.4F);
    EXPECT_FLOAT_EQ(settled->mean_value, -0.4F);
}

// WHY: FPU must match the Leela-style reduction formula used by selection for unvisited actions.
TEST(MctsSearchTest, ComputesLeelaStyleFpuReductionFromVisitedPriorMass) {
    MCTSNode node{};
    node.num_actions = 3;
    node.node_value = 0.2F;
    node.prior[0] = 0.5F;
    node.prior[1] = 0.3F;
    node.prior[2] = 0.2F;
    node.visit_count[0] = 10;
    node.visit_count[1] = 0;
    node.visit_count[2] = 5;

    const float fpu = alphazero::mcts::compute_fpu_value(node, 0.25F);
    const float expected = 0.2F - (0.25F * std::sqrt(0.7F));
    EXPECT_NEAR(fpu, expected, 1.0e-6F);
}

// WHY: Root FPU should not counteract root exploration when Dirichlet noise is active, while in-tree FPU
// reduction should still bias unvisited children toward the parent value estimate.
TEST(MctsSearchTest, UsesRootFpuOverrideOnlyAtRootAndKeepsInTreeReduction) {
    {
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
                    .legal_actions = {},
                    .transitions = {},
                },
                ToyStateSpec{
                    .current_player = 1,
                    .terminal = false,
                    .terminal_outcome = {0.0F, 0.0F},
                    .legal_actions = {},
                    .transitions = {},
                },
            });

        ToyGameConfig config(model);
        ArenaNodeStore store(256);
        SearchConfig search_config{};
        search_config.c_puct = 1.0F;
        search_config.c_fpu = 0.2F;
        search_config.c_fpu_root = 0.0F;
        search_config.enable_dirichlet_noise = false;

        MctsSearch search(store, config, search_config);
        search.set_root_state(config.new_game());

        const auto evaluator = make_evaluator({
            {0,
             EvaluationResult{
                 .policy = {0.8F, 0.2F},
                 .value = 0.0F,
                 .policy_is_logits = false,
             }},
            {1,
             EvaluationResult{
                 .policy = {0.0F, 0.0F},
                 .value = 0.3F,
                 .policy_is_logits = false,
             }},
            {2,
             EvaluationResult{
                 .policy = {0.0F, 0.0F},
                 .value = 0.3F,
                 .policy_is_logits = false,
             }},
        });

        search.run_simulations(2, evaluator);

        const auto edge0 = search.root_edge_stats(0);
        const auto edge1 = search.root_edge_stats(1);
        ASSERT_TRUE(edge0.has_value());
        ASSERT_TRUE(edge1.has_value());
        EXPECT_EQ(edge0->visit_count, 1);
        EXPECT_EQ(edge1->visit_count, 1);
    }

    {
        const auto model = make_model(
            3,
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
                    .legal_actions = {1, 2},
                    .transitions = {{1, 2}, {2, 3}},
                },
                ToyStateSpec{
                    .current_player = 0,
                    .terminal = false,
                    .terminal_outcome = {0.0F, 0.0F},
                    .legal_actions = {},
                    .transitions = {},
                },
                ToyStateSpec{
                    .current_player = 0,
                    .terminal = false,
                    .terminal_outcome = {0.0F, 0.0F},
                    .legal_actions = {},
                    .transitions = {},
                },
            });

        ToyGameConfig config(model);
        ArenaNodeStore store(256);
        SearchConfig search_config{};
        search_config.c_puct = 1.0F;
        search_config.c_fpu = 0.2F;
        search_config.c_fpu_root = 0.0F;
        search_config.enable_dirichlet_noise = false;

        MctsSearch search(store, config, search_config);
        search.set_root_state(config.new_game());

        const auto evaluator = make_evaluator({
            {0,
             EvaluationResult{
                 .policy = {1.0F, 0.0F, 0.0F},
                 .value = 0.0F,
                 .policy_is_logits = false,
             }},
            {1,
             EvaluationResult{
                 .policy = {0.0F, 0.8F, 0.2F},
                 .value = 0.0F,
                 .policy_is_logits = false,
             }},
            {2,
             EvaluationResult{
                 .policy = {0.0F, 0.0F, 0.0F},
                 .value = 0.3F,
                 .policy_is_logits = false,
             }},
            {3,
             EvaluationResult{
                 .policy = {0.0F, 0.0F, 0.0F},
                 .value = 0.3F,
                 .policy_is_logits = false,
             }},
        });

        search.run_simulations(3, evaluator);

        const auto root_edge = search.root_edge_stats(0);
        ASSERT_TRUE(root_edge.has_value());
        ASSERT_NE(root_edge->child, alphazero::mcts::NULL_NODE);

        const MCTSNode& child_node = store.get(root_edge->child);
        const int action1_slot = find_action_slot(child_node, 1);
        const int action2_slot = find_action_slot(child_node, 2);
        ASSERT_GE(action1_slot, 0);
        ASSERT_GE(action2_slot, 0);

        EXPECT_EQ(child_node.visit_count[static_cast<std::size_t>(action1_slot)], 2);
        EXPECT_EQ(child_node.visit_count[static_cast<std::size_t>(action2_slot)], 0);
    }
}

// WHY: Dirichlet exploration noise must be injected at the root only; child priors should stay equal to NN outputs.
TEST(MctsSearchTest, AppliesDirichletNoiseOnlyAtRoot) {
    const auto model = make_model(
        4,
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
                .legal_actions = {2, 3},
                .transitions = {{2, 3}, {3, 4}},
            },
            ToyStateSpec{
                .current_player = 1,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {2, 3},
                .transitions = {{2, 5}, {3, 6}},
            },
            ToyStateSpec{.current_player = 0, .terminal = true, .terminal_outcome = {0.0F, 0.0F}},
            ToyStateSpec{.current_player = 0, .terminal = true, .terminal_outcome = {0.0F, 0.0F}},
            ToyStateSpec{.current_player = 0, .terminal = true, .terminal_outcome = {0.0F, 0.0F}},
            ToyStateSpec{.current_player = 0, .terminal = true, .terminal_outcome = {0.0F, 0.0F}},
        });

    ToyGameConfig config(model);
    ArenaNodeStore store(256);

    SearchConfig search_config{};
    search_config.enable_dirichlet_noise = true;
    search_config.dirichlet_epsilon = 0.25F;
    search_config.random_seed = 7U;

    MctsSearch search(store, config, search_config);
    search.set_root_state(config.new_game());

    const auto evaluator = make_evaluator({
        {0,
         EvaluationResult{
             .policy = {1.5F, 0.5F, -5.0F, -5.0F},
             .value = 0.0F,
             .policy_is_logits = true,
         }},
        {1,
         EvaluationResult{
             .policy = {-5.0F, -5.0F, 2.0F, 0.0F},
             .value = 0.0F,
             .policy_is_logits = true,
         }},
        {2,
         EvaluationResult{
             .policy = {-5.0F, -5.0F, 2.0F, 0.0F},
             .value = 0.0F,
             .policy_is_logits = true,
         }},
    });

    search.run_simulation(evaluator);

    const auto root_edge_0 = search.root_edge_stats(0);
    const auto root_edge_1 = search.root_edge_stats(1);
    ASSERT_TRUE(root_edge_0.has_value());
    ASSERT_TRUE(root_edge_1.has_value());

    const float raw_root_0 = std::exp(1.0F) / (std::exp(1.0F) + 1.0F);
    const float raw_root_1 = 1.0F / (std::exp(1.0F) + 1.0F);

    EXPECT_NEAR(root_edge_0->prior + root_edge_1->prior, 1.0F, 1.0e-5F);
    EXPECT_TRUE(std::fabs(root_edge_0->prior - raw_root_0) > 1.0e-4F ||
                std::fabs(root_edge_1->prior - raw_root_1) > 1.0e-4F);

    const NodeId expanded_child =
        root_edge_0->child != alphazero::mcts::NULL_NODE ? root_edge_0->child : root_edge_1->child;
    ASSERT_NE(expanded_child, alphazero::mcts::NULL_NODE);

    const MCTSNode& child_node = store.get(expanded_child);
    const int action2_slot = find_action_slot(child_node, 2);
    const int action3_slot = find_action_slot(child_node, 3);
    ASSERT_GE(action2_slot, 0);
    ASSERT_GE(action3_slot, 0);

    const float raw_child_2 = std::exp(2.0F) / (std::exp(2.0F) + 1.0F);
    const float raw_child_3 = 1.0F / (std::exp(2.0F) + 1.0F);
    EXPECT_NEAR(child_node.prior[static_cast<std::size_t>(action2_slot)], raw_child_2, 1.0e-6F);
    EXPECT_NEAR(child_node.prior[static_cast<std::size_t>(action3_slot)], raw_child_3, 1.0e-6F);
}

// WHY: Temperature controls training target generation and move sampling; both early stochastic and late greedy regimes
// must match the spec.
TEST(MctsSearchTest, ProducesCorrectTemperaturePolicyAcrossEarlyAndLateMoveRegimes) {
    const auto model = make_model(
        3,
        {
            ToyStateSpec{
                .current_player = 0,
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {0, 1, 2},
                .transitions = {{0, 1}, {1, 2}, {2, 3}},
            },
            ToyStateSpec{.current_player = 1, .terminal = true, .terminal_outcome = {0.0F, 0.0F}},
            ToyStateSpec{.current_player = 1, .terminal = true, .terminal_outcome = {0.0F, 0.0F}},
            ToyStateSpec{.current_player = 1, .terminal = true, .terminal_outcome = {0.0F, 0.0F}},
        });

    ToyGameConfig config(model);
    ArenaNodeStore store(256);
    SearchConfig search_config{};
    search_config.temperature = 1.0F;
    search_config.temperature_moves = 30;

    MctsSearch search(store, config, search_config);
    search.set_root_state(config.new_game());

    const auto evaluator = make_evaluator({
        {0,
         EvaluationResult{
             .policy = {1.0F, 1.0F, 1.0F},
             .value = 0.0F,
             .policy_is_logits = false,
         }},
    });
    search.run_simulation(evaluator);

    MCTSNode& root_node = store.get(search.root_id());
    root_node.visit_count[0] = 10;
    root_node.visit_count[1] = 5;
    root_node.visit_count[2] = 0;
    root_node.total_visits = 15;

    const std::vector<float> early_policy = search.root_policy_target(10);
    EXPECT_NEAR(early_policy[0], 10.0F / 15.0F, 1.0e-6F);
    EXPECT_NEAR(early_policy[1], 5.0F / 15.0F, 1.0e-6F);
    EXPECT_NEAR(early_policy[2], 0.0F, 1.0e-6F);

    const std::vector<float> late_policy = search.root_policy_target(40);
    EXPECT_FLOAT_EQ(late_policy[0], 1.0F);
    EXPECT_FLOAT_EQ(late_policy[1], 0.0F);
    EXPECT_FLOAT_EQ(late_policy[2], 0.0F);
    EXPECT_EQ(search.select_action(40), 0);
}

// WHY: Tree reuse is required for throughput; promoting a selected child to root must keep its accumulated search stats.
TEST(MctsSearchTest, TreeReusePromotesChildToRootAndPreservesItsStatistics) {
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
                .terminal = false,
                .terminal_outcome = {0.0F, 0.0F},
                .legal_actions = {1},
                .transitions = {{1, 4}},
            },
            ToyStateSpec{.current_player = 0, .terminal = true, .terminal_outcome = {1.0F, -1.0F}},
            ToyStateSpec{.current_player = 0, .terminal = true, .terminal_outcome = {-1.0F, 1.0F}},
        });

    ToyGameConfig config(model);
    ArenaNodeStore store(512);
    SearchConfig search_config{};
    search_config.enable_dirichlet_noise = false;

    MctsSearch search(store, config, search_config);
    search.set_root_state(config.new_game());

    const auto evaluator = make_evaluator({
        {0,
         EvaluationResult{
             .policy = {1.0F, 1.0F},
             .value = 0.0F,
             .policy_is_logits = false,
         }},
        {1,
         EvaluationResult{
             .policy = {1.0F, 0.0F},
             .value = 0.2F,
             .policy_is_logits = false,
         }},
        {2,
         EvaluationResult{
             .policy = {0.0F, 1.0F},
             .value = -0.2F,
             .policy_is_logits = false,
         }},
    });

    search.run_simulations(24, evaluator);

    const auto edge0 = search.root_edge_stats(0);
    const auto edge1 = search.root_edge_stats(1);
    ASSERT_TRUE(edge0.has_value());
    ASSERT_TRUE(edge1.has_value());

    const int chosen_action = edge0->child != alphazero::mcts::NULL_NODE ? 0 : 1;
    const NodeId old_root = search.root_id();
    const NodeId child_before = chosen_action == 0 ? edge0->child : edge1->child;
    ASSERT_NE(child_before, alphazero::mcts::NULL_NODE);

    const MCTSNode child_snapshot = store.get(child_before);
    const std::size_t nodes_before = store.nodes_allocated();

    search.advance_root(chosen_action);

    EXPECT_EQ(search.root_id(), child_before);
    EXPECT_LT(store.nodes_allocated(), nodes_before);

    const MCTSNode& new_root = store.get(search.root_id());
    EXPECT_EQ(new_root.parent, alphazero::mcts::NULL_NODE);
    EXPECT_EQ(new_root.parent_action, -1);
    EXPECT_EQ(new_root.num_actions, child_snapshot.num_actions);
    EXPECT_EQ(new_root.total_visits, child_snapshot.total_visits);

    EXPECT_THROW(static_cast<void>(store.get(old_root)), std::out_of_range);
}

// WHY: Node mutexes should be materialized only when a node is first locked; root allocation/reset should not
// pre-populate mutex entries.
TEST(MctsSearchTest, RootAllocationKeepsNodeMutexMapEmptyUntilLockIsNeeded) {
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
            },
        });

    ToyGameConfig config(model);
    ArenaNodeStore store(64);
    SearchConfig search_config{};
    search_config.enable_dirichlet_noise = false;

    MctsSearch search(store, config, search_config);
    search.set_root_state(config.new_game());

    EXPECT_EQ(search.cached_node_mutex_count(), 0U);

    const std::vector<float> policy_before_expansion = search.root_policy_target(/*move_number=*/0);
    EXPECT_EQ(policy_before_expansion.size(), static_cast<std::size_t>(config.action_space_size));
    EXPECT_EQ(search.cached_node_mutex_count(), 1U);

    search.set_root_state(config.new_game());
    EXPECT_EQ(search.cached_node_mutex_count(), 0U);
}

// WHY: The search core is used with tree parallelism; under contention, visit accounting and virtual-loss cleanup must
// remain consistent.
TEST(MctsSearchTest, ConcurrentSimulationsAccumulateVisitsWithoutLeakingVirtualLoss) {
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

    ToyGameConfig config(model);
    ArenaNodeStore store(1024);
    SearchConfig search_config{};
    search_config.enable_dirichlet_noise = false;

    MctsSearch search(store, config, search_config);
    search.set_root_state(config.new_game());

    const auto evaluator = make_evaluator({
        {0,
         EvaluationResult{
             .policy = {1.0F},
             .value = 0.0F,
             .policy_is_logits = false,
         }},
    });

    constexpr int kThreads = 8;
    constexpr int kSimulationsPerThread = 50;

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&search, &evaluator] {
            for (int i = 0; i < kSimulationsPerThread; ++i) {
                search.run_simulation(evaluator);
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    const auto root_edge = search.root_edge_stats(0);
    ASSERT_TRUE(root_edge.has_value());
    EXPECT_EQ(root_edge->visit_count, kThreads * kSimulationsPerThread);
    EXPECT_EQ(root_edge->virtual_loss, 0);
    EXPECT_EQ(root_edge->visit_count, store.get(search.root_id()).total_visits);
}
