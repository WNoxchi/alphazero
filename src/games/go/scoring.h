#pragma once

#include "games/go/go_state.h"

namespace alphazero::go {

struct TrompTaylorScore {
    int black_points = 0;
    int white_points = 0;
    float komi = kDefaultKomi;
    float final_score = 0.0F;

    [[nodiscard]] int winner() const {
        if (final_score > 0.0F) {
            return kBlack;
        }
        if (final_score < 0.0F) {
            return kWhite;
        }
        return kEmpty;
    }
};

// Tromp-Taylor area scoring:
// - Occupied intersections score for their color.
// - Empty regions score for a color only when they are reachable by that color and not the other.
// - Final score is black_points - white_points - komi.
[[nodiscard]] TrompTaylorScore compute_tromp_taylor_score(const GoPosition& position);

}  // namespace alphazero::go
