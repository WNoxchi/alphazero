#include "games/go/scoring.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace alphazero::go {
namespace {

[[nodiscard]] constexpr bool is_black_stone(std::uint8_t stone) { return stone == kBlack; }

[[nodiscard]] constexpr bool is_white_stone(std::uint8_t stone) { return stone == kWhite; }

template <typename Function>
void for_each_neighbor(int intersection, Function&& fn) {
    if (!is_valid_intersection(intersection)) {
        return;
    }

    const int row = intersection_row(intersection);
    const int col = intersection_col(intersection);

    if (row > 0) {
        fn(to_intersection(row - 1, col));
    }
    if (row + 1 < kBoardSize) {
        fn(to_intersection(row + 1, col));
    }
    if (col > 0) {
        fn(to_intersection(row, col - 1));
    }
    if (col + 1 < kBoardSize) {
        fn(to_intersection(row, col + 1));
    }
}

struct EmptyRegionInfo {
    int size = 0;
    bool reaches_black = false;
    bool reaches_white = false;
    std::vector<int> intersections;
};

[[nodiscard]] EmptyRegionInfo analyze_empty_region(
    const GoPosition& position,
    int start,
    std::array<bool, kBoardArea>* visited) {
    EmptyRegionInfo region;
    if (!is_valid_intersection(start) || visited == nullptr || (*visited)[start]) {
        return region;
    }

    std::vector<int> stack;
    stack.push_back(start);
    (*visited)[start] = true;

    while (!stack.empty()) {
        const int current = stack.back();
        stack.pop_back();
        ++region.size;
        region.intersections.push_back(current);

        for_each_neighbor(current, [&](int neighbor) {
            const std::uint8_t stone = stone_at(position, neighbor);
            if (is_black_stone(stone)) {
                region.reaches_black = true;
                return;
            }
            if (is_white_stone(stone)) {
                region.reaches_white = true;
                return;
            }
            if ((*visited)[neighbor]) {
                return;
            }
            (*visited)[neighbor] = true;
            stack.push_back(neighbor);
        });
    }

    return region;
}

}  // namespace

TrompTaylorScore compute_tromp_taylor_score(const GoPosition& position) {
    TrompTaylorScore score;
    score.komi = position.komi;

    std::array<bool, kBoardArea> visited{};
    visited.fill(false);

    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        const std::uint8_t stone = stone_at(position, intersection);
        if (is_black_stone(stone)) {
            ++score.black_points;
            continue;
        }
        if (is_white_stone(stone)) {
            ++score.white_points;
            continue;
        }
        if (visited[intersection]) {
            continue;
        }

        const EmptyRegionInfo region = analyze_empty_region(position, intersection, &visited);
        if (region.reaches_black && !region.reaches_white) {
            score.black_points += region.size;
            continue;
        }
        if (region.reaches_white && !region.reaches_black) {
            score.white_points += region.size;
        }
    }

    score.final_score = static_cast<float>(score.black_points - score.white_points) - score.komi;
    return score;
}

void compute_tromp_taylor_ownership(const GoPosition& position, float* out_ownership) {
    if (out_ownership == nullptr) {
        throw std::invalid_argument("compute_tromp_taylor_ownership requires a non-null output buffer");
    }

    std::fill_n(out_ownership, kBoardArea, 0.0F);
    std::array<bool, kBoardArea> visited{};
    visited.fill(false);

    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        const std::uint8_t stone = stone_at(position, intersection);
        if (is_black_stone(stone)) {
            out_ownership[intersection] = 1.0F;
            continue;
        }
        if (is_white_stone(stone)) {
            out_ownership[intersection] = -1.0F;
            continue;
        }
        if (visited[intersection]) {
            continue;
        }

        const EmptyRegionInfo region = analyze_empty_region(position, intersection, &visited);
        if (region.reaches_black == region.reaches_white) {
            continue;
        }

        const float ownership_value = region.reaches_black ? 1.0F : -1.0F;
        for (const int empty_intersection : region.intersections) {
            out_ownership[empty_intersection] = ownership_value;
        }
    }
}

}  // namespace alphazero::go
