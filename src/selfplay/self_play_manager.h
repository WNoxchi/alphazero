#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "games/game_config.h"
#include "selfplay/replay_buffer.h"
#include "selfplay/self_play_game.h"

namespace alphazero::selfplay {

struct SelfPlayManagerConfig {
    std::size_t concurrent_games = 32U;
    std::size_t max_games_per_slot = 0U;  // 0 means run until stop() is requested.
    std::uint32_t initial_game_id = 1U;
    std::uint64_t random_seed = 0xA17A3E1B5F2C4D69ULL;
    SelfPlayGameConfig game_config{};
};

struct SelfPlayMetricsSnapshot {
    std::size_t configured_slots = 0U;
    std::size_t threads_per_game = 0U;
    std::size_t active_slots = 0U;

    std::size_t games_completed = 0U;
    std::size_t replay_positions_written = 0U;
    std::size_t total_moves = 0U;
    std::size_t total_simulations = 0U;

    std::size_t natural_terminations = 0U;
    std::size_t resignation_terminations = 0U;
    std::size_t max_length_adjudications = 0U;

    std::size_t resignation_disabled_games = 0U;
    std::size_t resignation_false_positive_games = 0U;

    bool has_latest_game = false;
    std::uint32_t latest_game_id = 0U;
    std::size_t latest_slot = 0U;
    std::size_t latest_game_length = 0U;
    float latest_outcome_player0 = 0.0F;
    bool latest_game_resigned = false;
    bool latest_resignation_disabled = false;
    bool latest_resignation_false_positive = false;

    double average_game_length = 0.0;
    double average_outcome_player0 = 0.0;
    double moves_per_second = 0.0;
    double games_per_hour = 0.0;
    double avg_simulations_per_second = 0.0;

    bool worker_failed = false;
};

class SelfPlayManager {
public:
    using EvaluateFn = SelfPlayGame::EvaluateFn;
    using CompletionCallback = std::function<void(std::size_t slot_index, const SelfPlayGameResult& result)>;

    SelfPlayManager(
        const GameConfig& game_config,
        ReplayBuffer& replay_buffer,
        EvaluateFn evaluator,
        SelfPlayManagerConfig config = {},
        CompletionCallback completion_callback = {});
    ~SelfPlayManager();

    SelfPlayManager(const SelfPlayManager&) = delete;
    SelfPlayManager& operator=(const SelfPlayManager&) = delete;
    SelfPlayManager(SelfPlayManager&&) = delete;
    SelfPlayManager& operator=(SelfPlayManager&&) = delete;

    void start();
    void stop();
    void update_simulations_per_move(std::size_t new_sims);

    [[nodiscard]] bool is_running() const noexcept;
    [[nodiscard]] SelfPlayMetricsSnapshot metrics() const;

private:
    static std::uint64_t mix_seed(std::uint64_t base_seed, std::size_t slot_index) noexcept;
    void worker_loop(std::size_t slot_index, SelfPlayGameConfig slot_game_config);
    void record_completed_game(std::size_t slot_index, const SelfPlayGameResult& result);
    [[nodiscard]] bool is_resignation_false_positive(const SelfPlayGameResult& result) const noexcept;
    void reset_metrics_locked();

    [[nodiscard]] std::chrono::steady_clock::duration elapsed_time() const;

    const GameConfig& game_config_;
    ReplayBuffer& replay_buffer_;
    EvaluateFn evaluator_;
    SelfPlayManagerConfig config_;
    CompletionCallback completion_callback_;

    mutable std::mutex lifecycle_mutex_;
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> running_{false};
    std::atomic<std::size_t> active_slots_{0U};
    std::atomic<std::uint32_t> next_game_id_{1U};
    std::atomic<std::size_t> simulations_per_move_{0U};
    std::chrono::steady_clock::time_point start_time_{};
    std::chrono::steady_clock::time_point end_time_{};
    bool has_start_time_ = false;

    mutable std::mutex metrics_mutex_;
    std::size_t games_completed_ = 0U;
    std::size_t replay_positions_written_ = 0U;
    std::size_t total_moves_ = 0U;
    std::size_t total_simulations_ = 0U;
    std::size_t natural_terminations_ = 0U;
    std::size_t resignation_terminations_ = 0U;
    std::size_t max_length_adjudications_ = 0U;
    std::size_t resignation_disabled_games_ = 0U;
    std::size_t resignation_false_positive_games_ = 0U;
    double cumulative_game_length_ = 0.0;
    double cumulative_outcome_player0_ = 0.0;

    bool has_latest_game_ = false;
    std::uint32_t latest_game_id_ = 0U;
    std::size_t latest_slot_ = 0U;
    std::size_t latest_game_length_ = 0U;
    float latest_outcome_player0_ = 0.0F;
    bool latest_game_resigned_ = false;
    bool latest_resignation_disabled_ = false;
    bool latest_resignation_false_positive_ = false;

    mutable std::mutex worker_error_mutex_;
    bool worker_failed_ = false;
    std::exception_ptr first_worker_error_{};
};

}  // namespace alphazero::selfplay
