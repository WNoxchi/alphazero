#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace alphazero {

class GameState {
public:
    virtual ~GameState() = default;

    // Apply an action index in [0, action_space_size), returning the next state.
    [[nodiscard]] virtual std::unique_ptr<GameState> apply_action(int action) const = 0;

    // Return legal action indices for the current state.
    [[nodiscard]] virtual std::vector<int> legal_actions() const = 0;

    // Return whether this position is terminal.
    [[nodiscard]] virtual bool is_terminal() const = 0;

    // Terminal outcome from the perspective of `player`: +1 win, 0 draw, -1 loss.
    [[nodiscard]] virtual float outcome(int player) const = 0;

    // Return player to move (0 or 1).
    [[nodiscard]] virtual int current_player() const = 0;

    // Encode this state into a pre-allocated tensor buffer.
    virtual void encode(float* buffer) const = 0;

    // Deep copy of this state.
    [[nodiscard]] virtual std::unique_ptr<GameState> clone() const = 0;

    // Position hash for transpositions/deduplication.
    [[nodiscard]] virtual std::uint64_t hash() const = 0;

    // Human-readable state representation for logging/debugging.
    [[nodiscard]] virtual std::string to_string() const = 0;
};

}  // namespace alphazero
