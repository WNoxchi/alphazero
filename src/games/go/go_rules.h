#pragma once

#include <vector>

#include "games/go/go_state.h"

namespace alphazero::go {

struct StoneGroup {
    int representative = -1;
    int liberty_count = 0;
    int stone_count = 0;

    [[nodiscard]] bool operator==(const StoneGroup& other) const = default;
};

enum class MoveStatus {
    kLegal = 0,
    kInvalidAction,
    kIntersectionOccupied,
    kKoViolation,
    kSuperkoViolation,
    kSelfCapture,
    kInvalidSideToMove,
};

struct MoveResult {
    MoveStatus status = MoveStatus::kInvalidAction;
    GoPosition position{};
    int action = -1;
    int captured_stones = 0;
    int ko_point = -1;

    [[nodiscard]] bool legal() const { return status == MoveStatus::kLegal; }
};

[[nodiscard]] bool is_valid_action(int action);
[[nodiscard]] bool is_pass_action(int action);
[[nodiscard]] bool passes_end_game(const GoPosition& position);

[[nodiscard]] MoveResult play_pass(const GoPosition& position);
[[nodiscard]] MoveResult play_action(const GoPosition& position, int action);
[[nodiscard]] bool is_legal_action(const GoPosition& position, int action);

// Compute connected stone groups and liberty counts using a union-find pass.
[[nodiscard]] std::vector<StoneGroup> compute_stone_groups(const GoPosition& position);

// Returns liberties for the group that contains `intersection`.
// Returns 0 for invalid intersections or empty points.
[[nodiscard]] int liberties_for_intersection(const GoPosition& position, int intersection);

}  // namespace alphazero::go
