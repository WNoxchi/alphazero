#include "selfplay/self_play_game.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace alphazero::selfplay {
namespace {

constexpr float kOutcomeEpsilon = 1.0e-6F;

[[nodiscard]] mcts::SearchConfig make_search_config(const SelfPlayGameConfig& config) {
    return mcts::SearchConfig{
        .simulations_per_move = config.simulations_per_move,
        .c_puct = config.c_puct,
        .c_fpu = config.c_fpu,
        .enable_dirichlet_noise = config.enable_dirichlet_noise,
        .dirichlet_epsilon = config.dirichlet_epsilon,
        .dirichlet_alpha_override = config.dirichlet_alpha_override,
        .temperature = config.temperature,
        .temperature_moves = config.temperature_moves,
        .enable_resignation = config.enable_resignation,
        .resign_threshold = config.resign_threshold,
        .random_seed = config.random_seed,
    };
}

}  // namespace

SelfPlayGame::SelfPlayGame(
    const GameConfig& game_config,
    ReplayBuffer& replay_buffer,
    const EvaluateFn& evaluator,
    SelfPlayGameConfig config)
    : game_config_(game_config),
      replay_buffer_(replay_buffer),
      evaluator_(evaluator),
      config_(config),
      search_config_(make_search_config(config_)),
      search_(game_config_, search_config_, config_.node_arena_capacity),
      rng_(config_.random_seed) {
    if (!evaluator_) {
        throw std::invalid_argument("SelfPlayGame requires a non-null evaluator callback");
    }
    if (game_config_.board_rows <= 0 || game_config_.board_cols <= 0 || game_config_.total_input_channels <= 0) {
        throw std::invalid_argument("SelfPlayGame requires a valid game input shape");
    }
    if (game_config_.action_space_size <= 0) {
        throw std::invalid_argument("SelfPlayGame requires a positive action space size");
    }
    if (config_.simulations_per_move == 0U) {
        throw std::invalid_argument("SelfPlayGame simulations_per_move must be greater than zero");
    }
    if (config_.enable_playout_cap) {
        if (config_.reduced_simulations == 0U) {
            throw std::invalid_argument("SelfPlayGame reduced_simulations must be greater than zero");
        }
        if (config_.reduced_simulations > config_.simulations_per_move) {
            throw std::invalid_argument("SelfPlayGame reduced_simulations must not exceed simulations_per_move");
        }
        if (!std::isfinite(config_.full_playout_probability) || config_.full_playout_probability < 0.0F ||
            config_.full_playout_probability > 1.0F) {
            throw std::invalid_argument("SelfPlayGame full_playout_probability must be finite and in [0, 1]");
        }
    }
    if (config_.randomize_dirichlet_epsilon) {
        if (!std::isfinite(config_.dirichlet_epsilon_min) || !std::isfinite(config_.dirichlet_epsilon_max)) {
            throw std::invalid_argument(
                "SelfPlayGame dirichlet epsilon randomization bounds must be finite");
        }
        if (config_.dirichlet_epsilon_min < 0.0F || config_.dirichlet_epsilon_min > 1.0F ||
            config_.dirichlet_epsilon_max < 0.0F || config_.dirichlet_epsilon_max > 1.0F) {
            throw std::invalid_argument(
                "SelfPlayGame dirichlet epsilon randomization bounds must be in [0, 1]");
        }
        if (config_.dirichlet_epsilon_min > config_.dirichlet_epsilon_max) {
            throw std::invalid_argument(
                "SelfPlayGame dirichlet_epsilon_min must not exceed dirichlet_epsilon_max");
        }
    }
    if (config_.mcts_threads == 0U) {
        throw std::invalid_argument("SelfPlayGame mcts_threads must be greater than zero");
    }
    if (config_.node_arena_capacity == 0U) {
        throw std::invalid_argument("SelfPlayGame node_arena_capacity must be greater than zero");
    }
    if (!std::isfinite(config_.resign_disable_fraction) || config_.resign_disable_fraction < 0.0F ||
        config_.resign_disable_fraction > 1.0F) {
        throw std::invalid_argument("SelfPlayGame resign_disable_fraction must be finite and in [0, 1]");
    }

    if (encoded_state_size() > ReplayPosition::kMaxEncodedStateSize) {
        throw std::invalid_argument("SelfPlayGame encoded state exceeds ReplayPosition storage capacity");
    }
    if (static_cast<std::size_t>(game_config_.action_space_size) > ReplayPosition::kMaxPolicySize) {
        throw std::invalid_argument("SelfPlayGame action space exceeds ReplayPosition policy storage capacity");
    }
}

SelfPlayGameResult SelfPlayGame::play(const std::uint32_t game_id) {
    SelfPlayGameResult result{};
    result.game_id = game_id;
    result.action_history.clear();

    std::vector<PendingSample> pending_samples;
    pending_samples.reserve(
        game_config_.max_game_length > 0 ? static_cast<std::size_t>(game_config_.max_game_length) : 256U);

    std::unique_ptr<GameState> initial_state = game_config_.new_game();
    if (initial_state == nullptr) {
        throw std::logic_error("SelfPlayGame game config returned null initial state");
    }
    search_.set_root_state(std::move(initial_state));

    result.resignation_was_disabled = choose_resignation_disabled_for_game();
    int resigned_player = -1;

    while (true) {
        const GameState& root = search_.root_state();
        const bool reached_max_length = game_config_.max_game_length > 0 &&
            result.move_count >= static_cast<std::size_t>(game_config_.max_game_length);

        if (root.is_terminal()) {
            result.termination_reason =
                reached_max_length ? GameTerminationReason::kMaxLengthAdjudication : GameTerminationReason::kNatural;
            break;
        }
        if (reached_max_length) {
            result.termination_reason = GameTerminationReason::kMaxLengthAdjudication;
            break;
        }

        bool use_full_simulations = true;
        std::size_t simulations_this_move = config_.simulations_per_move;
        if (config_.enable_playout_cap) {
            std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
            {
                std::scoped_lock lock(rng_mutex_);
                use_full_simulations = distribution(rng_) < config_.full_playout_probability;
            }
            if (!use_full_simulations) {
                simulations_this_move = config_.reduced_simulations;
            }
        }

        run_simulation_batch(simulations_this_move);
        ++result.simulation_batches_executed;
        result.total_simulations += simulations_this_move;

        if (config_.enable_resignation && search_.should_resign()) {
            result.resignation_would_have_triggered = true;
            result.resignation_candidate_player = search_.root_state().current_player();
            if (!result.resignation_was_disabled) {
                resigned_player = result.resignation_candidate_player;
                result.termination_reason = GameTerminationReason::kResignation;
                break;
            }
        }

        const int move_number_for_temperature = static_cast<int>(result.move_count + 1U);
        std::vector<float> policy = search_.root_policy_target(move_number_for_temperature);
        const int action = search_.select_action(move_number_for_temperature);

        const GameState& pre_move_state = search_.root_state();
        PendingSample sample{};
        sample.encoded_state.resize(encoded_state_size());
        pre_move_state.encode(sample.encoded_state.data());
        sample.policy = std::move(policy);
        sample.training_weight = use_full_simulations
            ? 1.0F
            : static_cast<float>(config_.reduced_simulations) / static_cast<float>(config_.simulations_per_move);
        sample.player = pre_move_state.current_player();
        sample.move_number = static_cast<std::uint16_t>(
            std::min<std::size_t>(result.move_count, std::numeric_limits<std::uint16_t>::max()));
        pending_samples.push_back(std::move(sample));

        if (const std::optional<mcts::EdgeStats> edge = search_.root_edge_stats(action);
            edge.has_value() && edge->child != mcts::NULL_NODE) {
            ++result.reused_subtree_count;
        }

        search_.advance_root(action);
        result.action_history.push_back(action);
        result.move_count = result.action_history.size();
    }

    if (result.termination_reason == GameTerminationReason::kResignation) {
        if (resigned_player != 0 && resigned_player != 1) {
            throw std::logic_error("SelfPlayGame resignation triggered for an invalid player index");
        }
        result.outcome_player0 = resigned_player == 0 ? -1.0F : 1.0F;
        result.outcome_player1 = -result.outcome_player0;
    } else {
        const GameState& final_state = search_.root_state();
        if (final_state.is_terminal()) {
            result.outcome_player0 = final_state.outcome(0);
            result.outcome_player1 = final_state.outcome(1);
        } else {
            // Conservative fallback for max-length adjudication when the state implementation does not self-terminate.
            result.outcome_player0 = 0.0F;
            result.outcome_player1 = 0.0F;
        }
    }

    std::vector<ReplayPosition> replay_positions;
    replay_positions.reserve(pending_samples.size());
    for (const PendingSample& sample : pending_samples) {
        if (sample.player != 0 && sample.player != 1) {
            throw std::logic_error("SelfPlayGame recorded sample with invalid player index");
        }
        const float scalar_value = sample.player == 0 ? result.outcome_player0 : result.outcome_player1;
        replay_positions.push_back(ReplayPosition::make(
            sample.encoded_state,
            sample.policy,
            scalar_value,
            wdl_target(scalar_value),
            game_id,
            sample.move_number,
            sample.training_weight));
    }

    replay_buffer_.add_game(replay_positions);
    result.replay_positions_written = replay_positions.size();
    return result;
}

void SelfPlayGame::run_simulation_batch(const std::size_t simulations) {
    if (simulations == 0U) {
        throw std::invalid_argument("SelfPlayGame simulation batch size must be greater than zero");
    }

    if (config_.mcts_threads <= 1U || simulations <= 1U) {
        search_.run_simulations(simulations, evaluator_);
        return;
    }

    const std::size_t worker_count = std::min(config_.mcts_threads, simulations);
    const std::size_t base = simulations / worker_count;
    const std::size_t remainder = simulations % worker_count;

    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    std::exception_ptr first_error;
    std::mutex error_mutex;

    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t sims_for_worker = base + (worker < remainder ? 1U : 0U);
        workers.emplace_back([this, sims_for_worker, &first_error, &error_mutex] {
            try {
                for (std::size_t i = 0; i < sims_for_worker; ++i) {
                    search_.run_simulation(evaluator_);
                }
            } catch (...) {
                std::scoped_lock lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

bool SelfPlayGame::choose_resignation_disabled_for_game() {
    if (!config_.enable_resignation || config_.resign_disable_fraction <= 0.0F) {
        return false;
    }
    if (config_.resign_disable_fraction >= 1.0F) {
        return true;
    }

    std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
    std::scoped_lock lock(rng_mutex_);
    return distribution(rng_) < config_.resign_disable_fraction;
}

std::array<float, ReplayPosition::kWdlSize> SelfPlayGame::wdl_target(const float scalar_value) const {
    if (scalar_value > kOutcomeEpsilon) {
        return {1.0F, 0.0F, 0.0F};
    }
    if (scalar_value < -kOutcomeEpsilon) {
        return {0.0F, 0.0F, 1.0F};
    }
    return {0.0F, 1.0F, 0.0F};
}

std::size_t SelfPlayGame::encoded_state_size() const noexcept {
    return static_cast<std::size_t>(game_config_.total_input_channels) * static_cast<std::size_t>(game_config_.board_rows) *
        static_cast<std::size_t>(game_config_.board_cols);
}

}  // namespace alphazero::selfplay
