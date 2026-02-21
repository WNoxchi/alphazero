#pragma once

#include <memory>
#include <string>
#include <vector>

#include "games/game_state.h"

namespace alphazero {

struct SymmetryTransform {
    virtual ~SymmetryTransform() = default;

    // In-place board tensor transform for shape (channels, rows, cols).
    virtual void transform_board(float* board, int channels, int rows, int cols) const = 0;

    // In-place policy vector transform for length action_space_size.
    virtual void transform_policy(float* policy, int action_space_size) const = 0;
};

class IdentitySymmetryTransform final : public SymmetryTransform {
public:
    void transform_board(float* /*board*/, int /*channels*/, int /*rows*/, int /*cols*/) const override {}
    void transform_policy(float* /*policy*/, int /*action_space_size*/) const override {}
};

struct GameConfig {
    enum class ValueHeadType { SCALAR, WDL };

    virtual ~GameConfig() = default;

    std::string name;  // "chess" or "go"

    // Board geometry
    int board_rows = 0;
    int board_cols = 0;

    // Neural-network input encoding
    int planes_per_step = 0;
    int num_history_steps = 0;
    int constant_planes = 0;
    int total_input_channels = 0;

    // Action space
    int action_space_size = 0;

    // MCTS parameters
    float dirichlet_alpha = 0.0F;
    int max_game_length = 0;

    // Value head type
    ValueHeadType value_head_type = ValueHeadType::SCALAR;

    // Symmetry support
    bool supports_symmetry = false;
    int num_symmetries = 1;

    // Create a new game in the initial state.
    [[nodiscard]] virtual std::unique_ptr<GameState> new_game() const = 0;

    // Returns symmetry transforms for this game. Default is identity only.
    [[nodiscard]] virtual std::vector<std::unique_ptr<SymmetryTransform>> get_symmetries() const {
        std::vector<std::unique_ptr<SymmetryTransform>> transforms;
        transforms.emplace_back(std::make_unique<IdentitySymmetryTransform>());
        return transforms;
    }
};

}  // namespace alphazero
