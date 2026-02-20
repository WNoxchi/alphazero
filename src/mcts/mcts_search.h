#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "games/game_config.h"
#include "games/game_state.h"
#include "mcts/mcts_node.h"
#include "mcts/node_store.h"

namespace alphazero::mcts {

struct SearchConfig {
    std::size_t simulations_per_move = 800;
    float c_puct = 2.5F;
    float c_fpu = 0.25F;

    bool enable_dirichlet_noise = false;
    float dirichlet_epsilon = 0.25F;
    float dirichlet_alpha_override = 0.0F;

    float temperature = 1.0F;
    int temperature_moves = 30;

    bool enable_resignation = true;
    float resign_threshold = -0.9F;

    std::uint64_t random_seed = 0xC0FFEE1234567890ULL;
};

struct EvaluationResult {
    // Policy over the full action space. Interpretation depends on `policy_is_logits`.
    std::vector<float> policy;
    // Scalar value in [-1, 1] from current-player perspective.
    float value = 0.0F;
    // true: `policy` is logits and must be softmaxed over legal actions.
    // false: `policy` is non-negative probabilities and must be renormalized over legal actions.
    bool policy_is_logits = true;
};

struct EdgeStats {
    int action = -1;
    std::int32_t visit_count = 0;
    float total_value = 0.0F;
    float mean_value = 0.0F;
    float prior = 0.0F;
    std::int32_t virtual_loss = 0;
    NodeId child = NULL_NODE;
};

using EvaluateFn = std::function<EvaluationResult(const GameState&)>;

[[nodiscard]] float compute_fpu_value(const MCTSNode& node, float c_fpu);

class MctsSearch {
public:
    MctsSearch(NodeStore& node_store, const GameConfig& game_config, SearchConfig config = {});

    void set_root_state(std::unique_ptr<GameState> root_state);

    [[nodiscard]] bool has_root() const;
    [[nodiscard]] NodeId root_id() const;
    [[nodiscard]] const GameState& root_state() const;

    void run_simulations(std::size_t simulation_count, const EvaluateFn& evaluator);
    void run_simulations(const EvaluateFn& evaluator);
    void run_simulation(const EvaluateFn& evaluator);

    [[nodiscard]] std::vector<float> root_policy_target(int move_number) const;
    [[nodiscard]] int select_action(int move_number);

    void advance_root(int action);
    [[nodiscard]] bool should_resign() const;

    void apply_dirichlet_noise_to_root();
    [[nodiscard]] std::optional<EdgeStats> root_edge_stats(int action) const;

private:
    struct TraversedEdge {
        NodeId node_id = NULL_NODE;
        int action_slot = -1;
    };

    struct LeafTarget {
        NodeId existing_node = NULL_NODE;
        NodeId parent_node = NULL_NODE;
        int parent_slot = -1;
        int parent_action = -1;
    };

    [[nodiscard]] std::shared_ptr<std::mutex> node_mutex(NodeId node_id) const;
    [[nodiscard]] NodeId allocate_node();

    void reset_store_for_new_root();
    void clear_node_mutexes();

    void ensure_root_expanded(const EvaluateFn& evaluator);
    void maybe_apply_root_dirichlet_noise();

    [[nodiscard]] std::vector<float> masked_policy(
        const EvaluationResult& eval_result,
        const std::vector<int>& legal_actions) const;

    void initialize_node(
        MCTSNode* node,
        const GameState& state,
        const EvaluationResult& eval_result,
        NodeId parent,
        int parent_action) const;

    [[nodiscard]] int select_action_slot(const MCTSNode& node) const;
    [[nodiscard]] static int find_action_slot(const MCTSNode& node, int action);

    static void apply_virtual_loss(MCTSNode* node, int action_slot);
    static void revert_virtual_loss(MCTSNode* node, int action_slot);
    static void apply_backup(MCTSNode* node, int action_slot, float value);

    [[nodiscard]] float dirichlet_alpha() const;
    [[nodiscard]] std::vector<float> sample_dirichlet(int size, float alpha);
    [[nodiscard]] int argmax_visit_slot(const MCTSNode& node) const;
    [[nodiscard]] float temperature_for_move(int move_number) const;

    NodeStore& node_store_;
    const GameConfig& game_config_;
    SearchConfig config_;

    std::unique_ptr<GameState> root_state_;
    NodeId root_id_ = NULL_NODE;

    mutable std::shared_mutex root_mutex_;
    mutable std::mutex store_mutex_;
    mutable std::mutex node_mutex_map_mutex_;
    mutable std::unordered_map<NodeId, std::shared_ptr<std::mutex>> node_mutexes_;

    mutable std::mutex rng_mutex_;
    std::mt19937_64 rng_;

    std::mutex root_expand_mutex_;
    std::mutex root_noise_mutex_;
    std::atomic<bool> root_expanded_{false};
    std::atomic<bool> root_noise_applied_{false};
};

}  // namespace alphazero::mcts
