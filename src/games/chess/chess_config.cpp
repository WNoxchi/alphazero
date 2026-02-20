#include "games/chess/chess_config.h"

#include <memory>

#include "games/chess/chess_state.h"
#include "games/chess/movegen.h"

namespace alphazero::chess {

ChessGameConfig::ChessGameConfig() {
    name = "chess";

    board_rows = 8;
    board_cols = 8;

    planes_per_step = ChessState::kPlanesPerStep;
    num_history_steps = ChessState::kHistorySteps;
    constant_planes = ChessState::kConstantPlanes;
    total_input_channels = ChessState::kTotalInputChannels;

    action_space_size = kActionSpaceSize;

    dirichlet_alpha = 0.3F;
    max_game_length = ChessState::kMaxGameLength;

    value_head_type = ValueHeadType::WDL;

    supports_symmetry = false;
    num_symmetries = 1;
}

std::unique_ptr<GameState> ChessGameConfig::new_game() const { return std::make_unique<ChessState>(); }

const ChessGameConfig& chess_game_config() {
    static const ChessGameConfig config{};
    return config;
}

}  // namespace alphazero::chess
