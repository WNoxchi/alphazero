#include "games/go/go_config.h"

#include <memory>

#include "games/go/go_state.h"

namespace alphazero::go {

GoGameConfig::GoGameConfig() {
    name = "go";

    board_rows = kBoardSize;
    board_cols = kBoardSize;

    planes_per_step = GoState::kPlanesPerStep;
    num_history_steps = GoState::kHistorySteps;
    constant_planes = GoState::kConstantPlanes;
    total_input_channels = GoState::kTotalInputChannels;

    action_space_size = kActionSpaceSize;

    dirichlet_alpha = 0.03F;
    max_game_length = GoState::kMaxGameLength;

    value_head_type = ValueHeadType::SCALAR;

    supports_symmetry = true;
    num_symmetries = 8;
}

std::unique_ptr<GameState> GoGameConfig::new_game() const { return std::make_unique<GoState>(); }

const GoGameConfig& go_game_config() {
    static const GoGameConfig config{};
    return config;
}

}  // namespace alphazero::go
