#pragma once

#include <memory>

#include "games/game_config.h"

namespace alphazero::chess {

class ChessGameConfig final : public GameConfig {
public:
    ChessGameConfig();
    [[nodiscard]] std::unique_ptr<GameState> new_game() const override;
};

[[nodiscard]] const ChessGameConfig& chess_game_config();

}  // namespace alphazero::chess
