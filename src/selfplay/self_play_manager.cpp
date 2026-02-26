#include "selfplay/self_play_manager.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace alphazero::selfplay {
namespace {

constexpr std::uint64_t kSeedMixAddend = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kSeedMixMul1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t kSeedMixMul2 = 0x94D049BB133111EBULL;

}  // namespace

SelfPlayManager::SelfPlayManager(
    const GameConfig& game_config,
    ReplayBuffer& replay_buffer,
    EvaluateFn evaluator,
    SelfPlayManagerConfig config,
    CompletionCallback completion_callback)
    : game_config_(game_config),
      replay_buffer_(replay_buffer),
      evaluator_(std::move(evaluator)),
      config_(config),
      completion_callback_(std::move(completion_callback)),
      next_game_id_(config_.initial_game_id) {
    if (!evaluator_) {
        throw std::invalid_argument("SelfPlayManager requires a non-null evaluator callback");
    }
    if (config_.concurrent_games == 0U) {
        throw std::invalid_argument("SelfPlayManager concurrent_games must be greater than zero");
    }
    if (config_.game_config.mcts_threads == 0U) {
        throw std::invalid_argument("SelfPlayManager threads-per-game must be greater than zero");
    }
    if (config_.game_config.simulations_per_move == 0U) {
        throw std::invalid_argument("SelfPlayManager simulations-per-move must be greater than zero");
    }
    if (config_.game_config.randomize_dirichlet_epsilon) {
        const float epsilon_min = config_.game_config.dirichlet_epsilon_min;
        const float epsilon_max = config_.game_config.dirichlet_epsilon_max;
        if (!std::isfinite(epsilon_min) || !std::isfinite(epsilon_max)) {
            throw std::invalid_argument(
                "SelfPlayManager dirichlet epsilon randomization bounds must be finite");
        }
        if (epsilon_min < 0.0F || epsilon_min > 1.0F || epsilon_max < 0.0F || epsilon_max > 1.0F) {
            throw std::invalid_argument(
                "SelfPlayManager dirichlet epsilon randomization bounds must be in [0, 1]");
        }
        if (epsilon_min > epsilon_max) {
            throw std::invalid_argument(
                "SelfPlayManager dirichlet_epsilon_min must not exceed dirichlet_epsilon_max");
        }
    }
}

SelfPlayManager::~SelfPlayManager() {
    try {
        stop();
    } catch (...) {
        // Destructors must not throw; any worker error is surfaced via metrics().
    }
}

void SelfPlayManager::start() {
    std::unique_lock lifecycle_lock(lifecycle_mutex_);
    if (running_.load(std::memory_order_acquire)) {
        throw std::logic_error("SelfPlayManager is already running");
    }
    if (!workers_.empty()) {
        throw std::logic_error("SelfPlayManager cannot start with unjoined worker threads");
    }

    {
        std::lock_guard error_lock(worker_error_mutex_);
        worker_failed_ = false;
        first_worker_error_ = nullptr;
    }

    reset_metrics_locked();
    stop_requested_.store(false, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    active_slots_.store(0U, std::memory_order_release);
    next_game_id_.store(config_.initial_game_id, std::memory_order_release);
    start_time_ = std::chrono::steady_clock::now();
    end_time_ = start_time_;
    has_start_time_ = true;

    workers_.reserve(config_.concurrent_games);
    try {
        for (std::size_t slot = 0; slot < config_.concurrent_games; ++slot) {
            SelfPlayGameConfig slot_game_config = config_.game_config;
            slot_game_config.mcts_threads = config_.game_config.mcts_threads;
            slot_game_config.random_seed = mix_seed(config_.random_seed, slot);
            workers_.emplace_back([this, slot, slot_game_config] { worker_loop(slot, slot_game_config); });
        }
    } catch (...) {
        stop_requested_.store(true, std::memory_order_release);
        running_.store(false, std::memory_order_release);
        std::vector<std::thread> partial_workers;
        partial_workers.swap(workers_);
        lifecycle_lock.unlock();
        for (std::thread& worker : partial_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        throw;
    }
}

void SelfPlayManager::stop() {
    std::vector<std::thread> workers_to_join;
    {
        std::lock_guard lifecycle_lock(lifecycle_mutex_);
        stop_requested_.store(true, std::memory_order_release);
        workers_to_join.swap(workers_);
    }

    for (std::thread& worker : workers_to_join) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    running_.store(false, std::memory_order_release);
    if (has_start_time_) {
        end_time_ = std::chrono::steady_clock::now();
    }
}

bool SelfPlayManager::is_running() const noexcept { return running_.load(std::memory_order_acquire); }

SelfPlayMetricsSnapshot SelfPlayManager::metrics() const {
    SelfPlayMetricsSnapshot snapshot{};
    snapshot.configured_slots = config_.concurrent_games;
    snapshot.threads_per_game = config_.game_config.mcts_threads;
    snapshot.active_slots = active_slots_.load(std::memory_order_acquire);

    {
        std::lock_guard lock(metrics_mutex_);
        snapshot.games_completed = games_completed_;
        snapshot.replay_positions_written = replay_positions_written_;
        snapshot.total_moves = total_moves_;
        snapshot.total_simulations = total_simulations_;
        snapshot.natural_terminations = natural_terminations_;
        snapshot.resignation_terminations = resignation_terminations_;
        snapshot.max_length_adjudications = max_length_adjudications_;
        snapshot.resignation_disabled_games = resignation_disabled_games_;
        snapshot.resignation_false_positive_games = resignation_false_positive_games_;

        snapshot.has_latest_game = has_latest_game_;
        snapshot.latest_game_id = latest_game_id_;
        snapshot.latest_slot = latest_slot_;
        snapshot.latest_game_length = latest_game_length_;
        snapshot.latest_outcome_player0 = latest_outcome_player0_;
        snapshot.latest_game_resigned = latest_game_resigned_;
        snapshot.latest_resignation_disabled = latest_resignation_disabled_;
        snapshot.latest_resignation_false_positive = latest_resignation_false_positive_;

        if (games_completed_ > 0U) {
            const double games = static_cast<double>(games_completed_);
            snapshot.average_game_length = cumulative_game_length_ / games;
            snapshot.average_outcome_player0 = cumulative_outcome_player0_ / games;
        }
    }

    {
        std::lock_guard error_lock(worker_error_mutex_);
        snapshot.worker_failed = worker_failed_;
    }

    const double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_time()).count();
    if (elapsed_seconds > 0.0) {
        snapshot.moves_per_second = static_cast<double>(snapshot.total_moves) / elapsed_seconds;
        snapshot.games_per_hour = (static_cast<double>(snapshot.games_completed) / elapsed_seconds) * 3600.0;
        snapshot.avg_simulations_per_second = static_cast<double>(snapshot.total_simulations) / elapsed_seconds;
    }
    return snapshot;
}

std::uint64_t SelfPlayManager::mix_seed(const std::uint64_t base_seed, const std::size_t slot_index) noexcept {
    std::uint64_t seed = base_seed + kSeedMixAddend + static_cast<std::uint64_t>(slot_index);
    seed ^= (seed >> 30U);
    seed *= kSeedMixMul1;
    seed ^= (seed >> 27U);
    seed *= kSeedMixMul2;
    seed ^= (seed >> 31U);
    return seed;
}

void SelfPlayManager::worker_loop(const std::size_t slot_index, SelfPlayGameConfig slot_game_config) {
    active_slots_.fetch_add(1U, std::memory_order_acq_rel);

    try {
        std::mt19937_64 slot_rng(slot_game_config.random_seed);
        std::size_t games_played_in_slot = 0U;

        while (!stop_requested_.load(std::memory_order_acquire)) {
            if (config_.max_games_per_slot > 0U && games_played_in_slot >= config_.max_games_per_slot) {
                break;
            }

            SelfPlayGameConfig game_config = slot_game_config;
            game_config.random_seed = slot_rng();
            if (game_config.randomize_dirichlet_epsilon) {
                std::uniform_real_distribution<float> epsilon_distribution(
                    game_config.dirichlet_epsilon_min,
                    game_config.dirichlet_epsilon_max);
                game_config.dirichlet_epsilon = epsilon_distribution(slot_rng);
            }

            SelfPlayGame game(game_config_, replay_buffer_, evaluator_, game_config);
            const std::uint32_t game_id = next_game_id_.fetch_add(1U, std::memory_order_acq_rel);
            const SelfPlayGameResult result = game.play(game_id);
            record_completed_game(slot_index, result);
            ++games_played_in_slot;
        }
    } catch (...) {
        {
            std::lock_guard error_lock(worker_error_mutex_);
            worker_failed_ = true;
            if (!first_worker_error_) {
                first_worker_error_ = std::current_exception();
            }
        }
        stop_requested_.store(true, std::memory_order_release);
    }

    const std::size_t previous_active = active_slots_.fetch_sub(1U, std::memory_order_acq_rel);
    if (previous_active == 1U) {
        running_.store(false, std::memory_order_release);
        std::lock_guard lifecycle_lock(lifecycle_mutex_);
        if (has_start_time_) {
            end_time_ = std::chrono::steady_clock::now();
        }
    }
}

void SelfPlayManager::record_completed_game(const std::size_t slot_index, const SelfPlayGameResult& result) {
    const bool resignation_false_positive = is_resignation_false_positive(result);

    {
        std::lock_guard lock(metrics_mutex_);
        ++games_completed_;
        replay_positions_written_ += result.replay_positions_written;
        total_moves_ += result.move_count;
        total_simulations_ += result.simulation_batches_executed * config_.game_config.simulations_per_move;
        cumulative_game_length_ += static_cast<double>(result.move_count);
        cumulative_outcome_player0_ += static_cast<double>(result.outcome_player0);

        switch (result.termination_reason) {
        case GameTerminationReason::kNatural:
            ++natural_terminations_;
            break;
        case GameTerminationReason::kResignation:
            ++resignation_terminations_;
            break;
        case GameTerminationReason::kMaxLengthAdjudication:
            ++max_length_adjudications_;
            break;
        }

        if (result.resignation_was_disabled) {
            ++resignation_disabled_games_;
        }
        if (resignation_false_positive) {
            ++resignation_false_positive_games_;
        }

        has_latest_game_ = true;
        latest_game_id_ = result.game_id;
        latest_slot_ = slot_index;
        latest_game_length_ = result.move_count;
        latest_outcome_player0_ = result.outcome_player0;
        latest_game_resigned_ = result.termination_reason == GameTerminationReason::kResignation;
        latest_resignation_disabled_ = result.resignation_was_disabled;
        latest_resignation_false_positive_ = resignation_false_positive;
    }

    if (completion_callback_) {
        completion_callback_(slot_index, result);
    }
}

bool SelfPlayManager::is_resignation_false_positive(const SelfPlayGameResult& result) const noexcept {
    if (!result.resignation_was_disabled || !result.resignation_would_have_triggered) {
        return false;
    }

    if (result.resignation_candidate_player == 0) {
        return result.outcome_player0 > 0.0F;
    }
    if (result.resignation_candidate_player == 1) {
        return result.outcome_player1 > 0.0F;
    }
    return false;
}

void SelfPlayManager::reset_metrics_locked() {
    std::lock_guard lock(metrics_mutex_);
    games_completed_ = 0U;
    replay_positions_written_ = 0U;
    total_moves_ = 0U;
    total_simulations_ = 0U;
    natural_terminations_ = 0U;
    resignation_terminations_ = 0U;
    max_length_adjudications_ = 0U;
    resignation_disabled_games_ = 0U;
    resignation_false_positive_games_ = 0U;
    cumulative_game_length_ = 0.0;
    cumulative_outcome_player0_ = 0.0;

    has_latest_game_ = false;
    latest_game_id_ = 0U;
    latest_slot_ = 0U;
    latest_game_length_ = 0U;
    latest_outcome_player0_ = 0.0F;
    latest_game_resigned_ = false;
    latest_resignation_disabled_ = false;
    latest_resignation_false_positive_ = false;
}

std::chrono::steady_clock::duration SelfPlayManager::elapsed_time() const {
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    if (!has_start_time_) {
        return std::chrono::steady_clock::duration::zero();
    }

    const std::chrono::steady_clock::time_point end =
        running_.load(std::memory_order_acquire) ? std::chrono::steady_clock::now() : end_time_;
    if (end < start_time_) {
        return std::chrono::steady_clock::duration::zero();
    }
    return end - start_time_;
}

}  // namespace alphazero::selfplay
