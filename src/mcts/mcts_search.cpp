#include "mcts/mcts_search.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mcts/arena_node_store.h"

namespace alphazero::mcts {
namespace {

constexpr float kEpsilon = 1.0e-7F;

[[nodiscard]] bool nearly_equal(float left, float right) {
    return std::fabs(left - right) <= kEpsilon;
}

}  // namespace

template <typename NodeType>
float compute_fpu_value(const NodeType& node, float c_fpu) {
    float visited_prior_sum = 0.0F;
    for (int i = 0; i < node.num_actions; ++i) {
        if (node.visit_count[static_cast<std::size_t>(i)] > 0) {
            visited_prior_sum += std::max(0.0F, node.prior[static_cast<std::size_t>(i)]);
        }
    }

    return node.node_value - (c_fpu * std::sqrt(std::max(0.0F, visited_prior_sum)));
}

template <typename NodeType>
MctsSearchT<NodeType>::MctsSearchT(NodeStoreT<NodeType>& node_store, const GameConfig& game_config, SearchConfig config)
    : node_store_(node_store),
      game_config_(game_config),
      config_(config),
      rng_(config.random_seed) {
    if (game_config_.action_space_size <= 0) {
        throw std::invalid_argument("MctsSearch requires a positive action space size");
    }
    if (!(config_.c_puct > 0.0F) || !std::isfinite(config_.c_puct)) {
        throw std::invalid_argument("MctsSearch c_puct must be finite and > 0");
    }
    if (!(config_.c_fpu >= 0.0F) || !std::isfinite(config_.c_fpu)) {
        throw std::invalid_argument("MctsSearch c_fpu must be finite and >= 0");
    }
    if (!std::isfinite(config_.temperature) || config_.temperature < 0.0F) {
        throw std::invalid_argument("MctsSearch temperature must be finite and >= 0");
    }
    if (!std::isfinite(config_.dirichlet_epsilon) || config_.dirichlet_epsilon < 0.0F ||
        config_.dirichlet_epsilon > 1.0F) {
        throw std::invalid_argument("MctsSearch dirichlet_epsilon must be finite and in [0, 1]");
    }
    if (config_.enable_dirichlet_noise && !(dirichlet_alpha() > 0.0F)) {
        throw std::invalid_argument("MctsSearch requires positive Dirichlet alpha when root noise is enabled");
    }
}

template <typename NodeType>
void MctsSearchT<NodeType>::set_root_state(std::unique_ptr<GameState> root_state) {
    if (root_state == nullptr) {
        throw std::invalid_argument("MctsSearch root state must be non-null");
    }

    std::unique_lock root_lock(root_mutex_);

    reset_store_for_new_root();
    root_state_ = std::move(root_state);
    root_id_ = allocate_node();
    root_expanded_ = false;
    root_noise_applied_ = false;
}

template <typename NodeType>
bool MctsSearchT<NodeType>::has_root() const {
    std::shared_lock root_lock(root_mutex_);
    return root_id_ != NULL_NODE && root_state_ != nullptr;
}

template <typename NodeType>
NodeId MctsSearchT<NodeType>::root_id() const {
    std::shared_lock root_lock(root_mutex_);
    return root_id_;
}

template <typename NodeType>
const GameState& MctsSearchT<NodeType>::root_state() const {
    std::shared_lock root_lock(root_mutex_);
    if (root_state_ == nullptr) {
        throw std::logic_error("MctsSearch root state is not initialized");
    }
    return *root_state_;
}

template <typename NodeType>
void MctsSearchT<NodeType>::run_simulations(std::size_t simulation_count, const EvaluateFn& evaluator) {
    for (std::size_t i = 0; i < simulation_count; ++i) {
        run_simulation(evaluator);
    }
}

template <typename NodeType>
void MctsSearchT<NodeType>::run_simulations(const EvaluateFn& evaluator) {
    run_simulations(config_.simulations_per_move, evaluator);
}

template <typename NodeType>
void MctsSearchT<NodeType>::run_simulation(const EvaluateFn& evaluator) {
    if (!evaluator) {
        throw std::invalid_argument("MctsSearch evaluator callback must be set");
    }

    ensure_root_expanded(evaluator);

    NodeId root_id = NULL_NODE;
    std::unique_ptr<GameState> state;
    {
        std::shared_lock root_lock(root_mutex_);
        if (root_id_ == NULL_NODE || root_state_ == nullptr) {
            throw std::logic_error("MctsSearch root state is not initialized");
        }
        root_id = root_id_;
        state = root_state_->clone();
    }

    if (state->is_terminal()) {
        return;
    }

    std::vector<TraversedEdge> path;
    LeafTarget leaf_target{};
    NodeId current = root_id;

    while (true) {
        if (state->is_terminal()) {
            leaf_target.existing_node = current;
            break;
        }

        const std::shared_ptr<std::mutex> current_mutex = node_mutex(current);
        int selected_slot = -1;
        int selected_action = -1;
        NodeId child_id = NULL_NODE;

        {
            std::scoped_lock node_lock(*current_mutex);
            NodeType& node = node_store_.get(current);
            if (node.num_actions <= 0) {
                leaf_target.existing_node = current;
                break;
            }

            selected_slot = select_action_slot(node);
            selected_action = node.actions[static_cast<std::size_t>(selected_slot)];
            apply_virtual_loss(&node, selected_slot);
            child_id = node.children[static_cast<std::size_t>(selected_slot)];
        }

        path.push_back(TraversedEdge{
            .node_id = current,
            .action_slot = selected_slot,
        });

        state = state->apply_action(selected_action);
        if (child_id == NULL_NODE) {
            leaf_target.parent_node = current;
            leaf_target.parent_slot = selected_slot;
            leaf_target.parent_action = selected_action;
            break;
        }

        current = child_id;
    }

    const bool terminal_leaf = state->is_terminal();
    EvaluationResult eval_result;
    float leaf_value = 0.0F;

    if (terminal_leaf) {
        leaf_value = state->outcome(state->current_player());
    } else {
        eval_result = evaluator(*state);
        leaf_value = eval_result.value;
        if (!std::isfinite(leaf_value)) {
            throw std::invalid_argument("MctsSearch evaluator returned non-finite value");
        }
    }

    if (leaf_target.existing_node != NULL_NODE) {
        const std::shared_ptr<std::mutex> leaf_mutex = node_mutex(leaf_target.existing_node);
        std::scoped_lock leaf_lock(*leaf_mutex);

        NodeType& leaf_node = node_store_.get(leaf_target.existing_node);
        if (terminal_leaf) {
            leaf_node.node_value = leaf_value;
        } else if (leaf_node.num_actions == 0) {
            initialize_node(&leaf_node, *state, eval_result, leaf_node.parent, leaf_node.parent_action);
        }
    } else if (leaf_target.parent_node != NULL_NODE) {
        NodeId child_id = NULL_NODE;
        {
            const std::shared_ptr<std::mutex> parent_mutex = node_mutex(leaf_target.parent_node);
            std::scoped_lock parent_lock(*parent_mutex);

            NodeType& parent_node = node_store_.get(leaf_target.parent_node);
            if (leaf_target.parent_slot < 0 || leaf_target.parent_slot >= parent_node.num_actions) {
                throw std::logic_error("MctsSearch parent slot out of range during expansion");
            }
            child_id = parent_node.children[static_cast<std::size_t>(leaf_target.parent_slot)];
        }

        if (child_id == NULL_NODE) {
            const NodeId candidate_id = allocate_node();
            {
                const std::shared_ptr<std::mutex> candidate_mutex = node_mutex(candidate_id);
                std::scoped_lock candidate_lock(*candidate_mutex);

                NodeType& candidate = node_store_.get(candidate_id);
                if (terminal_leaf) {
                    candidate.reset();
                    candidate.node_value = leaf_value;
                    candidate.parent = leaf_target.parent_node;
                    candidate.parent_action = static_cast<std::int16_t>(leaf_target.parent_action);
                } else {
                    initialize_node(&candidate, *state, eval_result, leaf_target.parent_node, leaf_target.parent_action);
                }
            }

            const std::shared_ptr<std::mutex> parent_mutex = node_mutex(leaf_target.parent_node);
            std::scoped_lock parent_lock(*parent_mutex);

            NodeType& parent_node = node_store_.get(leaf_target.parent_node);
            NodeId& child_slot_ref = parent_node.children[static_cast<std::size_t>(leaf_target.parent_slot)];
            if (child_slot_ref == NULL_NODE) {
                child_slot_ref = candidate_id;
                child_id = candidate_id;
            } else {
                child_id = child_slot_ref;
                std::scoped_lock store_lock(store_mutex_);
                node_store_.release_subtree(candidate_id);
            }
        }
    }

    float backed_up_value = leaf_value;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        const std::shared_ptr<std::mutex> parent_mutex = node_mutex(it->node_id);
        std::scoped_lock parent_lock(*parent_mutex);

        NodeType& parent_node = node_store_.get(it->node_id);
        revert_virtual_loss(&parent_node, it->action_slot);
        backed_up_value = -backed_up_value;
        apply_backup(&parent_node, it->action_slot, backed_up_value);
    }
}

template <typename NodeType>
std::vector<float> MctsSearchT<NodeType>::root_policy_target(const int move_number) const {
    NodeId root_id = NULL_NODE;
    {
        std::shared_lock root_lock(root_mutex_);
        if (root_id_ == NULL_NODE) {
            throw std::logic_error("MctsSearch root node is not initialized");
        }
        root_id = root_id_;
    }

    std::vector<float> policy(static_cast<std::size_t>(game_config_.action_space_size), 0.0F);
    const std::shared_ptr<std::mutex> root_node_mutex = node_mutex(root_id);
    std::scoped_lock root_node_lock(*root_node_mutex);

    const NodeType& root_node = node_store_.get(root_id);
    if (root_node.num_actions <= 0) {
        return policy;
    }

    const float temperature = temperature_for_move(move_number);
    const int best_slot = argmax_visit_slot(root_node);

    if (temperature <= kEpsilon) {
        const int best_action = root_node.actions[static_cast<std::size_t>(best_slot)];
        policy[static_cast<std::size_t>(best_action)] = 1.0F;
        return policy;
    }

    std::vector<float> weights(static_cast<std::size_t>(root_node.num_actions), 0.0F);
    float weight_sum = 0.0F;
    const float exponent = 1.0F / temperature;

    for (int i = 0; i < root_node.num_actions; ++i) {
        const float visit_count = static_cast<float>(std::max(0, root_node.visit_count[static_cast<std::size_t>(i)]));
        const float weight = std::pow(visit_count, exponent);
        weights[static_cast<std::size_t>(i)] = std::isfinite(weight) ? weight : 0.0F;
        weight_sum += weights[static_cast<std::size_t>(i)];
    }

    if (!(weight_sum > 0.0F)) {
        for (int i = 0; i < root_node.num_actions; ++i) {
            weights[static_cast<std::size_t>(i)] = std::max(0.0F, root_node.prior[static_cast<std::size_t>(i)]);
        }
        weight_sum = 0.0F;
        for (const float weight : weights) {
            weight_sum += weight;
        }
    }

    if (!(weight_sum > 0.0F)) {
        const float uniform = 1.0F / static_cast<float>(root_node.num_actions);
        for (int i = 0; i < root_node.num_actions; ++i) {
            const int action = root_node.actions[static_cast<std::size_t>(i)];
            policy[static_cast<std::size_t>(action)] = uniform;
        }
        return policy;
    }

    for (int i = 0; i < root_node.num_actions; ++i) {
        const int action = root_node.actions[static_cast<std::size_t>(i)];
        policy[static_cast<std::size_t>(action)] = weights[static_cast<std::size_t>(i)] / weight_sum;
    }
    return policy;
}

template <typename NodeType>
int MctsSearchT<NodeType>::select_action(const int move_number) {
    NodeId root_id = NULL_NODE;
    {
        std::shared_lock root_lock(root_mutex_);
        if (root_id_ == NULL_NODE) {
            throw std::logic_error("MctsSearch root node is not initialized");
        }
        root_id = root_id_;
    }

    const std::shared_ptr<std::mutex> root_node_mutex = node_mutex(root_id);
    std::scoped_lock root_node_lock(*root_node_mutex);

    const NodeType& root_node = node_store_.get(root_id);
    if (root_node.num_actions <= 0) {
        throw std::logic_error("MctsSearch cannot select action from an unexpanded root");
    }

    const float temperature = temperature_for_move(move_number);
    const int best_slot = argmax_visit_slot(root_node);
    if (temperature <= kEpsilon) {
        return root_node.actions[static_cast<std::size_t>(best_slot)];
    }

    std::vector<double> weights(static_cast<std::size_t>(root_node.num_actions), 0.0);
    double weight_sum = 0.0;
    const double exponent = 1.0 / static_cast<double>(temperature);

    for (int i = 0; i < root_node.num_actions; ++i) {
        const double visit_count = static_cast<double>(std::max(0, root_node.visit_count[static_cast<std::size_t>(i)]));
        const double weight = std::pow(visit_count, exponent);
        weights[static_cast<std::size_t>(i)] = std::isfinite(weight) ? weight : 0.0;
        weight_sum += weights[static_cast<std::size_t>(i)];
    }

    if (!(weight_sum > 0.0)) {
        for (int i = 0; i < root_node.num_actions; ++i) {
            weights[static_cast<std::size_t>(i)] =
                static_cast<double>(std::max(0.0F, root_node.prior[static_cast<std::size_t>(i)]));
        }
        weight_sum = 0.0;
        for (const double weight : weights) {
            weight_sum += weight;
        }
    }

    if (!(weight_sum > 0.0)) {
        return root_node.actions[static_cast<std::size_t>(best_slot)];
    }

    std::size_t sampled_slot = 0;
    {
        std::scoped_lock rng_lock(rng_mutex_);
        std::discrete_distribution<std::size_t> distribution(weights.begin(), weights.end());
        sampled_slot = distribution(rng_);
    }

    if (sampled_slot >= static_cast<std::size_t>(root_node.num_actions)) {
        sampled_slot = static_cast<std::size_t>(best_slot);
    }
    return root_node.actions[sampled_slot];
}

template <typename NodeType>
void MctsSearchT<NodeType>::advance_root(const int action) {
    std::unique_lock root_lock(root_mutex_);
    if (root_id_ == NULL_NODE || root_state_ == nullptr) {
        throw std::logic_error("MctsSearch root state is not initialized");
    }

    const NodeId old_root = root_id_;
    std::unique_ptr<GameState> next_state = root_state_->apply_action(action);

    NodeId selected_child = NULL_NODE;
    std::vector<NodeId> siblings_to_release;
    {
        const std::shared_ptr<std::mutex> old_root_mutex = node_mutex(old_root);
        std::scoped_lock old_root_lock(*old_root_mutex);

        NodeType& old_root_node = node_store_.get(old_root);
        const int action_slot = find_action_slot(old_root_node, action);
        if (action_slot < 0) {
            throw std::invalid_argument("MctsSearch advance_root action was not legal at root");
        }

        selected_child = old_root_node.children[static_cast<std::size_t>(action_slot)];
        for (int i = 0; i < old_root_node.num_actions; ++i) {
            const NodeId child = old_root_node.children[static_cast<std::size_t>(i)];
            if (child != NULL_NODE && child != selected_child) {
                siblings_to_release.push_back(child);
            }
        }
    }

    if (selected_child == NULL_NODE) {
        {
            std::scoped_lock store_lock(store_mutex_);
            node_store_.release_subtree(old_root);
        }
        root_id_ = allocate_node();
        root_state_ = std::move(next_state);
        root_expanded_ = false;
        root_noise_applied_ = false;
        clear_node_mutexes();
        return;
    }

    if (auto* arena_store = dynamic_cast<ArenaNodeStoreT<NodeType>*>(&node_store_); arena_store != nullptr) {
        std::scoped_lock store_lock(store_mutex_);
        root_id_ = arena_store->reuse_subtree(old_root, selected_child);
    } else {
        for (const NodeId sibling : siblings_to_release) {
            std::scoped_lock store_lock(store_mutex_);
            node_store_.release_subtree(sibling);
        }

        const std::shared_ptr<std::mutex> child_mutex = node_mutex(selected_child);
        std::scoped_lock child_lock(*child_mutex);
        NodeType& child_node = node_store_.get(selected_child);
        child_node.parent = NULL_NODE;
        child_node.parent_action = -1;
        root_id_ = selected_child;
    }

    root_state_ = std::move(next_state);
    root_expanded_ = false;
    root_noise_applied_ = false;
    clear_node_mutexes();
}

template <typename NodeType>
bool MctsSearchT<NodeType>::should_resign() const {
    if (!config_.enable_resignation) {
        return false;
    }

    NodeId root_id = NULL_NODE;
    {
        std::shared_lock root_lock(root_mutex_);
        if (root_id_ == NULL_NODE) {
            return false;
        }
        root_id = root_id_;
    }

    const std::shared_ptr<std::mutex> root_node_mutex = node_mutex(root_id);
    std::scoped_lock root_node_lock(*root_node_mutex);

    const NodeType& root_node = node_store_.get(root_id);
    if (root_node.num_actions <= 0) {
        return false;
    }
    if (!(root_node.node_value < config_.resign_threshold)) {
        return false;
    }

    float best_child_q = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < root_node.num_actions; ++i) {
        if (root_node.visit_count[static_cast<std::size_t>(i)] <= 0) {
            continue;
        }
        best_child_q = std::max(best_child_q, root_node.mean_value[static_cast<std::size_t>(i)]);
    }

    if (!std::isfinite(best_child_q)) {
        return false;
    }

    return best_child_q < config_.resign_threshold;
}

template <typename NodeType>
void MctsSearchT<NodeType>::apply_dirichlet_noise_to_root() {
    NodeId root_id = NULL_NODE;
    {
        std::shared_lock root_lock(root_mutex_);
        if (root_id_ == NULL_NODE) {
            throw std::logic_error("MctsSearch root node is not initialized");
        }
        root_id = root_id_;
    }

    const std::shared_ptr<std::mutex> root_node_mutex = node_mutex(root_id);
    std::scoped_lock root_node_lock(*root_node_mutex);

    NodeType& root_node = node_store_.get(root_id);
    if (root_node.num_actions <= 0) {
        root_noise_applied_ = true;
        return;
    }

    const float epsilon = config_.dirichlet_epsilon;
    if (epsilon <= 0.0F) {
        root_noise_applied_ = true;
        return;
    }

    const std::vector<float> noise = sample_dirichlet(root_node.num_actions, dirichlet_alpha());
    float prior_sum = 0.0F;
    for (int i = 0; i < root_node.num_actions; ++i) {
        const float mixed = ((1.0F - epsilon) * root_node.prior[static_cast<std::size_t>(i)]) +
            (epsilon * noise[static_cast<std::size_t>(i)]);
        root_node.prior[static_cast<std::size_t>(i)] = std::max(0.0F, mixed);
        prior_sum += root_node.prior[static_cast<std::size_t>(i)];
    }

    if (prior_sum <= 0.0F) {
        const float uniform = 1.0F / static_cast<float>(root_node.num_actions);
        for (int i = 0; i < root_node.num_actions; ++i) {
            root_node.prior[static_cast<std::size_t>(i)] = uniform;
        }
    } else {
        for (int i = 0; i < root_node.num_actions; ++i) {
            root_node.prior[static_cast<std::size_t>(i)] /= prior_sum;
        }
    }

    root_noise_applied_ = true;
}

template <typename NodeType>
std::optional<EdgeStats> MctsSearchT<NodeType>::root_edge_stats(const int action) const {
    NodeId root_id = NULL_NODE;
    {
        std::shared_lock root_lock(root_mutex_);
        if (root_id_ == NULL_NODE) {
            return std::nullopt;
        }
        root_id = root_id_;
    }

    const std::shared_ptr<std::mutex> root_node_mutex = node_mutex(root_id);
    std::scoped_lock root_node_lock(*root_node_mutex);

    const NodeType& root_node = node_store_.get(root_id);
    const int action_slot = find_action_slot(root_node, action);
    if (action_slot < 0) {
        return std::nullopt;
    }

    return EdgeStats{
        .action = action,
        .visit_count = root_node.visit_count[static_cast<std::size_t>(action_slot)],
        .total_value = root_node.total_value[static_cast<std::size_t>(action_slot)],
        .mean_value = root_node.mean_value[static_cast<std::size_t>(action_slot)],
        .prior = root_node.prior[static_cast<std::size_t>(action_slot)],
        .virtual_loss = root_node.virtual_loss[static_cast<std::size_t>(action_slot)],
        .child = root_node.children[static_cast<std::size_t>(action_slot)],
    };
}

template <typename NodeType>
std::shared_ptr<std::mutex> MctsSearchT<NodeType>::node_mutex(const NodeId node_id) const {
    std::scoped_lock node_mutexes_lock(node_mutex_map_mutex_);
    auto [it, inserted] = node_mutexes_.emplace(node_id, std::make_shared<std::mutex>());
    (void)inserted;
    return it->second;
}

template <typename NodeType>
NodeId MctsSearchT<NodeType>::allocate_node() {
    std::scoped_lock store_lock(store_mutex_);
    const NodeId id = node_store_.allocate();

    std::scoped_lock node_mutexes_lock(node_mutex_map_mutex_);
    node_mutexes_.emplace(id, std::make_shared<std::mutex>());
    return id;
}

template <typename NodeType>
void MctsSearchT<NodeType>::reset_store_for_new_root() {
    std::scoped_lock store_lock(store_mutex_);
    node_store_.reset();
    clear_node_mutexes();
}

template <typename NodeType>
void MctsSearchT<NodeType>::clear_node_mutexes() {
    std::scoped_lock node_mutexes_lock(node_mutex_map_mutex_);
    node_mutexes_.clear();
}

template <typename NodeType>
void MctsSearchT<NodeType>::ensure_root_expanded(const EvaluateFn& evaluator) {
    NodeId root_id = NULL_NODE;
    std::unique_ptr<GameState> root_state;
    {
        std::shared_lock root_lock(root_mutex_);
        if (root_id_ == NULL_NODE || root_state_ == nullptr) {
            throw std::logic_error("MctsSearch root state is not initialized");
        }
        root_id = root_id_;
        root_state = root_state_->clone();
    }

    bool needs_expansion = false;
    {
        const std::shared_ptr<std::mutex> root_node_mutex = node_mutex(root_id);
        std::scoped_lock root_node_lock(*root_node_mutex);

        NodeType& root_node = node_store_.get(root_id);
        if (root_state->is_terminal()) {
            root_node.node_value = root_state->outcome(root_state->current_player());
            root_expanded_ = true;
            root_noise_applied_ = true;
            return;
        }

        needs_expansion = root_node.num_actions == 0;
    }

    if (needs_expansion) {
        std::scoped_lock expand_lock(root_expand_mutex_);

        {
            const std::shared_ptr<std::mutex> root_node_mutex = node_mutex(root_id);
            std::scoped_lock root_node_lock(*root_node_mutex);
            NodeType& root_node = node_store_.get(root_id);
            needs_expansion = root_node.num_actions == 0;
        }

        if (needs_expansion) {
            const EvaluationResult eval_result = evaluator(*root_state);
            if (!std::isfinite(eval_result.value)) {
                throw std::invalid_argument("MctsSearch evaluator returned non-finite root value");
            }

            const std::shared_ptr<std::mutex> root_node_mutex = node_mutex(root_id);
            std::scoped_lock root_node_lock(*root_node_mutex);
            NodeType& root_node = node_store_.get(root_id);
            if (root_node.num_actions == 0) {
                initialize_node(&root_node, *root_state, eval_result, NULL_NODE, -1);
            }
        }
    }

    root_expanded_ = true;
    maybe_apply_root_dirichlet_noise();
}

template <typename NodeType>
void MctsSearchT<NodeType>::maybe_apply_root_dirichlet_noise() {
    if (!config_.enable_dirichlet_noise) {
        root_noise_applied_ = true;
        return;
    }
    if (root_noise_applied_) {
        return;
    }

    std::scoped_lock noise_lock(root_noise_mutex_);
    if (root_noise_applied_) {
        return;
    }
    apply_dirichlet_noise_to_root();
}

template <typename NodeType>
std::vector<float> MctsSearchT<NodeType>::masked_policy(
    const EvaluationResult& eval_result,
    const std::vector<int>& legal_actions) const {
    if (legal_actions.empty()) {
        return {};
    }
    if (eval_result.policy.size() != static_cast<std::size_t>(game_config_.action_space_size)) {
        throw std::invalid_argument("MctsSearch evaluator returned policy with wrong action-space size");
    }

    std::vector<float> masked(static_cast<std::size_t>(legal_actions.size()), 0.0F);

    if (eval_result.policy_is_logits) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const int action : legal_actions) {
            if (action < 0 || action >= game_config_.action_space_size) {
                throw std::invalid_argument("MctsSearch legal action index out of range");
            }
            max_logit = std::max(max_logit, eval_result.policy[static_cast<std::size_t>(action)]);
        }

        float sum = 0.0F;
        for (std::size_t i = 0; i < legal_actions.size(); ++i) {
            const int action = legal_actions[i];
            const float shifted = eval_result.policy[static_cast<std::size_t>(action)] - max_logit;
            const float value = std::exp(shifted);
            masked[i] = std::isfinite(value) ? value : 0.0F;
            sum += masked[i];
        }

        if (!(sum > 0.0F)) {
            const float uniform = 1.0F / static_cast<float>(legal_actions.size());
            std::fill(masked.begin(), masked.end(), uniform);
            return masked;
        }
        for (float& value : masked) {
            value /= sum;
        }
        return masked;
    }

    float sum = 0.0F;
    for (std::size_t i = 0; i < legal_actions.size(); ++i) {
        const int action = legal_actions[i];
        if (action < 0 || action >= game_config_.action_space_size) {
            throw std::invalid_argument("MctsSearch legal action index out of range");
        }

        const float raw = eval_result.policy[static_cast<std::size_t>(action)];
        masked[i] = std::isfinite(raw) ? std::max(0.0F, raw) : 0.0F;
        sum += masked[i];
    }

    if (!(sum > 0.0F)) {
        const float uniform = 1.0F / static_cast<float>(legal_actions.size());
        std::fill(masked.begin(), masked.end(), uniform);
        return masked;
    }

    for (float& value : masked) {
        value /= sum;
    }
    return masked;
}

template <typename NodeType>
void MctsSearchT<NodeType>::initialize_node(
    NodeType* node,
    const GameState& state,
    const EvaluationResult& eval_result,
    const NodeId parent,
    const int parent_action) const {
    if (node == nullptr) {
        throw std::invalid_argument("MctsSearch initialize_node requires a non-null node");
    }
    if (!std::isfinite(eval_result.value)) {
        throw std::invalid_argument("MctsSearch evaluator value must be finite");
    }

    node->reset();
    node->node_value = eval_result.value;
    node->parent = parent;
    node->parent_action = static_cast<std::int16_t>(parent_action);

    const std::vector<int> legal_actions = state.legal_actions();
    if (legal_actions.size() > static_cast<std::size_t>(NodeType::kMaxActions)) {
        throw std::runtime_error("MctsSearch legal action count exceeds MCTS node capacity");
    }

    const std::vector<float> priors = masked_policy(eval_result, legal_actions);
    node->num_actions = static_cast<std::int16_t>(legal_actions.size());

    for (std::size_t i = 0; i < legal_actions.size(); ++i) {
        node->actions[i] = static_cast<std::int16_t>(legal_actions[i]);
        node->prior[i] = priors[i];
    }
}

template <typename NodeType>
int MctsSearchT<NodeType>::select_action_slot(const NodeType& node) const {
    if (node.num_actions <= 0) {
        throw std::logic_error("MctsSearch select_action_slot called on an unexpanded node");
    }

    const float fpu = compute_fpu_value(node, config_.c_fpu);
    // Keep at least one effective visit so first selection is prior-driven instead of all-zero ties.
    const float sqrt_total_visits = std::sqrt(static_cast<float>(std::max(1, node.total_visits)));

    int best_slot = 0;
    float best_score = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < node.num_actions; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const float q_value = node.visit_count[index] > 0 ? node.mean_value[index] : fpu;
        const float u_value =
            config_.c_puct * node.prior[index] * sqrt_total_visits / (1.0F + static_cast<float>(node.visit_count[index]));
        const float score = q_value + u_value;

        if (score > best_score) {
            best_score = score;
            best_slot = i;
            continue;
        }

        if (nearly_equal(score, best_score)) {
            if (node.prior[index] > node.prior[static_cast<std::size_t>(best_slot)]) {
                best_slot = i;
            } else if (nearly_equal(node.prior[index], node.prior[static_cast<std::size_t>(best_slot)]) &&
                       node.actions[index] < node.actions[static_cast<std::size_t>(best_slot)]) {
                best_slot = i;
            }
        }
    }

    return best_slot;
}

template <typename NodeType>
int MctsSearchT<NodeType>::find_action_slot(const NodeType& node, const int action) {
    for (int i = 0; i < node.num_actions; ++i) {
        if (node.actions[static_cast<std::size_t>(i)] == action) {
            return i;
        }
    }
    return -1;
}

template <typename NodeType>
void MctsSearchT<NodeType>::apply_virtual_loss(NodeType* node, const int action_slot) {
    if (node == nullptr) {
        throw std::invalid_argument("MctsSearch apply_virtual_loss requires a non-null node");
    }
    if (action_slot < 0 || action_slot >= node->num_actions) {
        throw std::out_of_range("MctsSearch apply_virtual_loss action slot is out of range");
    }

    const std::size_t index = static_cast<std::size_t>(action_slot);
    ++node->virtual_loss[index];
    ++node->visit_count[index];
    ++node->total_visits;
    node->total_value[index] -= 1.0F;
    node->mean_value[index] = node->total_value[index] / static_cast<float>(node->visit_count[index]);
}

template <typename NodeType>
void MctsSearchT<NodeType>::revert_virtual_loss(NodeType* node, const int action_slot) {
    if (node == nullptr) {
        throw std::invalid_argument("MctsSearch revert_virtual_loss requires a non-null node");
    }
    if (action_slot < 0 || action_slot >= node->num_actions) {
        throw std::out_of_range("MctsSearch revert_virtual_loss action slot is out of range");
    }

    const std::size_t index = static_cast<std::size_t>(action_slot);
    if (node->virtual_loss[index] <= 0 || node->visit_count[index] <= 0 || node->total_visits <= 0) {
        throw std::logic_error("MctsSearch virtual loss state is inconsistent during revert");
    }

    --node->virtual_loss[index];
    --node->visit_count[index];
    --node->total_visits;
    node->total_value[index] += 1.0F;
    if (node->visit_count[index] > 0) {
        node->mean_value[index] = node->total_value[index] / static_cast<float>(node->visit_count[index]);
    } else {
        node->mean_value[index] = 0.0F;
    }
}

template <typename NodeType>
void MctsSearchT<NodeType>::apply_backup(NodeType* node, const int action_slot, const float value) {
    if (node == nullptr) {
        throw std::invalid_argument("MctsSearch apply_backup requires a non-null node");
    }
    if (action_slot < 0 || action_slot >= node->num_actions) {
        throw std::out_of_range("MctsSearch apply_backup action slot is out of range");
    }
    if (!std::isfinite(value)) {
        throw std::invalid_argument("MctsSearch backup value must be finite");
    }

    const std::size_t index = static_cast<std::size_t>(action_slot);
    ++node->visit_count[index];
    ++node->total_visits;
    node->total_value[index] += value;
    node->mean_value[index] = node->total_value[index] / static_cast<float>(node->visit_count[index]);
}

template <typename NodeType>
float MctsSearchT<NodeType>::dirichlet_alpha() const {
    if (config_.dirichlet_alpha_override > 0.0F) {
        return config_.dirichlet_alpha_override;
    }
    return game_config_.dirichlet_alpha;
}

template <typename NodeType>
std::vector<float> MctsSearchT<NodeType>::sample_dirichlet(const int size, const float alpha) {
    if (size <= 0) {
        return {};
    }
    if (!(alpha > 0.0F) || !std::isfinite(alpha)) {
        throw std::invalid_argument("MctsSearch Dirichlet alpha must be finite and > 0");
    }

    std::vector<float> samples(static_cast<std::size_t>(size), 0.0F);
    std::gamma_distribution<float> gamma(alpha, 1.0F);

    float sum = 0.0F;
    {
        std::scoped_lock rng_lock(rng_mutex_);
        for (int i = 0; i < size; ++i) {
            const float sample = gamma(rng_);
            samples[static_cast<std::size_t>(i)] = std::max(0.0F, sample);
            sum += samples[static_cast<std::size_t>(i)];
        }
    }

    if (!(sum > 0.0F)) {
        const float uniform = 1.0F / static_cast<float>(size);
        std::fill(samples.begin(), samples.end(), uniform);
        return samples;
    }

    for (float& sample : samples) {
        sample /= sum;
    }
    return samples;
}

template <typename NodeType>
int MctsSearchT<NodeType>::argmax_visit_slot(const NodeType& node) const {
    if (node.num_actions <= 0) {
        throw std::logic_error("MctsSearch argmax_visit_slot requires an expanded node");
    }

    int best_slot = 0;
    for (int i = 1; i < node.num_actions; ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        const std::size_t best_index = static_cast<std::size_t>(best_slot);

        if (node.visit_count[index] > node.visit_count[best_index]) {
            best_slot = i;
            continue;
        }
        if (node.visit_count[index] == node.visit_count[best_index]) {
            if (node.prior[index] > node.prior[best_index]) {
                best_slot = i;
            } else if (nearly_equal(node.prior[index], node.prior[best_index]) &&
                       node.actions[index] < node.actions[best_index]) {
                best_slot = i;
            }
        }
    }
    return best_slot;
}

template <typename NodeType>
float MctsSearchT<NodeType>::temperature_for_move(const int move_number) const {
    if (config_.temperature <= 0.0F || move_number > config_.temperature_moves) {
        return 0.0F;
    }
    return config_.temperature;
}

RuntimeMctsSearch::ChessContext::ChessContext(
    const GameConfig& game_config,
    const SearchConfig config,
    const std::size_t node_arena_capacity)
    : node_store(node_arena_capacity),
      search(node_store, game_config, config) {}

RuntimeMctsSearch::GoContext::GoContext(
    const GameConfig& game_config,
    const SearchConfig config,
    const std::size_t node_arena_capacity)
    : node_store(node_arena_capacity),
      search(node_store, game_config, config) {}

RuntimeMctsSearch::RuntimeMctsSearch(
    const GameConfig& game_config,
    const SearchConfig config,
    const std::size_t node_arena_capacity)
    : search_variant_(
          choose_node_layout(game_config) == NodeLayout::kChess
              ? SearchVariant(std::in_place_type<ChessContext>, game_config, config, node_arena_capacity)
              : SearchVariant(std::in_place_type<GoContext>, game_config, config, node_arena_capacity)) {}

RuntimeMctsSearch::NodeLayout RuntimeMctsSearch::choose_node_layout(const GameConfig& game_config) {
    if (game_config.name == "chess") {
        return NodeLayout::kChess;
    }
    if (game_config.name == "go") {
        return NodeLayout::kGo;
    }
    if (game_config.action_space_size <= kChessMaxActions) {
        return NodeLayout::kChess;
    }
    return NodeLayout::kGo;
}

void RuntimeMctsSearch::set_root_state(std::unique_ptr<GameState> root_state) {
    with_search([&root_state](auto& search) { search.set_root_state(std::move(root_state)); });
}

bool RuntimeMctsSearch::has_root() const {
    return with_search([](const auto& search) { return search.has_root(); });
}

NodeId RuntimeMctsSearch::root_id() const {
    return with_search([](const auto& search) { return search.root_id(); });
}

const GameState& RuntimeMctsSearch::root_state() const {
    return with_search([](const auto& search) -> const GameState& { return search.root_state(); });
}

void RuntimeMctsSearch::run_simulations(const std::size_t simulation_count, const EvaluateFn& evaluator) {
    with_search([simulation_count, &evaluator](auto& search) { search.run_simulations(simulation_count, evaluator); });
}

void RuntimeMctsSearch::run_simulations(const EvaluateFn& evaluator) {
    with_search([&evaluator](auto& search) { search.run_simulations(evaluator); });
}

void RuntimeMctsSearch::run_simulation(const EvaluateFn& evaluator) {
    with_search([&evaluator](auto& search) { search.run_simulation(evaluator); });
}

std::vector<float> RuntimeMctsSearch::root_policy_target(const int move_number) const {
    return with_search([move_number](const auto& search) { return search.root_policy_target(move_number); });
}

int RuntimeMctsSearch::select_action(const int move_number) {
    return with_search([move_number](auto& search) { return search.select_action(move_number); });
}

void RuntimeMctsSearch::advance_root(const int action) {
    with_search([action](auto& search) { search.advance_root(action); });
}

bool RuntimeMctsSearch::should_resign() const {
    return with_search([](const auto& search) { return search.should_resign(); });
}

void RuntimeMctsSearch::apply_dirichlet_noise_to_root() {
    with_search([](auto& search) { search.apply_dirichlet_noise_to_root(); });
}

std::optional<EdgeStats> RuntimeMctsSearch::root_edge_stats(const int action) const {
    return with_search([action](const auto& search) { return search.root_edge_stats(action); });
}

int RuntimeMctsSearch::node_capacity_actions() const noexcept {
    return std::holds_alternative<ChessContext>(search_variant_) ? ChessMCTSNode::kMaxActions : GoMCTSNode::kMaxActions;
}

template float compute_fpu_value<ChessMCTSNode>(const ChessMCTSNode& node, float c_fpu);
template float compute_fpu_value<GoMCTSNode>(const GoMCTSNode& node, float c_fpu);

template class MctsSearchT<ChessMCTSNode>;
template class MctsSearchT<GoMCTSNode>;

}  // namespace alphazero::mcts
