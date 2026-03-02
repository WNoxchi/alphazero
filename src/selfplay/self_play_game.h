#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <vector>

#include "games/game_config.h"
#include "mcts/mcts_search.h"
#include "selfplay/replay_buffer.h"

namespace alphazero::selfplay {

enum class GameTerminationReason {
    kNatural = 0,
    kResignation = 1,
    kMaxLengthAdjudication = 2,
};

struct SelfPlayGameConfig {
    std::size_t simulations_per_move = 800U;
    std::size_t mcts_threads = 8U;
    std::size_t node_arena_capacity = mcts::ArenaNodeStore::kDefaultCapacity;
    bool enable_playout_cap = false;
    std::size_t reduced_simulations = 50U;
    float full_playout_probability = 0.25F;

    float c_puct = 2.5F;
    float c_fpu = 0.25F;
    float c_fpu_root = -1.0F;

    bool enable_dirichlet_noise = true;
    float dirichlet_epsilon = 0.25F;
    bool randomize_dirichlet_epsilon = false;
    float dirichlet_epsilon_min = 0.15F;
    float dirichlet_epsilon_max = 0.35F;
    float dirichlet_alpha_override = 0.0F;
    bool dynamic_dirichlet_alpha = false;

    float temperature = 1.0F;
    int temperature_moves = 30;

    bool enable_resignation = true;
    float resign_threshold = -0.9F;
    float resign_disable_fraction = 0.1F;
    bool compute_ownership = false;

    std::uint64_t random_seed = 0xD1CEB00B5EED1234ULL;
};

struct SelfPlayGameResult {
    std::uint32_t game_id = 0U;
    std::size_t move_count = 0U;
    std::size_t replay_positions_written = 0U;
    std::size_t reused_subtree_count = 0U;
    std::size_t simulation_batches_executed = 0U;
    std::size_t total_simulations = 0U;

    GameTerminationReason termination_reason = GameTerminationReason::kNatural;
    bool resignation_was_disabled = false;
    bool resignation_would_have_triggered = false;
    int resignation_candidate_player = -1;

    float outcome_player0 = 0.0F;
    float outcome_player1 = 0.0F;

    std::vector<int> action_history;
};

class SelfPlayGame {
public:
    using EvaluateFn = mcts::EvaluateFn;
    using AddGameFn = std::function<void(const std::vector<ReplayPosition>&)>;

    SelfPlayGame(
        const GameConfig& game_config,
        ReplayBuffer& replay_buffer,
        const EvaluateFn& evaluator,
        SelfPlayGameConfig config = {});

    SelfPlayGame(
        const GameConfig& game_config,
        AddGameFn add_game_fn,
        const EvaluateFn& evaluator,
        SelfPlayGameConfig config = {});

    [[nodiscard]] SelfPlayGameResult play(std::uint32_t game_id);

private:
    struct PendingSample {
        std::vector<float> encoded_state;
        std::vector<float> policy;
        float training_weight = 1.0F;
        int player = 0;
        std::uint16_t move_number = 0U;
    };

    void run_simulation_batch(std::size_t simulations);
    [[nodiscard]] bool choose_resignation_disabled_for_game();
    [[nodiscard]] std::array<float, ReplayPosition::kWdlSize> wdl_target(float scalar_value) const;
    [[nodiscard]] std::size_t encoded_state_size() const noexcept;

    const GameConfig& game_config_;
    AddGameFn add_game_fn_;
    const EvaluateFn& evaluator_;
    SelfPlayGameConfig config_;
    mcts::SearchConfig search_config_{};
    mcts::RuntimeMctsSearch search_;
    std::mt19937_64 rng_;
    std::mutex rng_mutex_;
};

}  // namespace alphazero::selfplay
