#pragma once

#include <memory>
#include <vector>

#include "games/game_config.h"

namespace alphazero::go {

class GoGameConfig final : public GameConfig {
public:
    GoGameConfig();
    [[nodiscard]] std::unique_ptr<GameState> new_game() const override;
    [[nodiscard]] std::vector<std::unique_ptr<SymmetryTransform>> get_symmetries() const override;
};

[[nodiscard]] const GoGameConfig& go_game_config();

}  // namespace alphazero::go
