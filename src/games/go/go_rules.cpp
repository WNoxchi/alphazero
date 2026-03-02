#include "games/go/go_rules.h"

#include <array>
#include <cstdint>
#include <vector>

namespace alphazero::go {
namespace {

struct BoardAnalysis {
    std::array<int, kBoardArea> parent{};
    std::array<int, kBoardArea> rank{};
    std::array<int, kBoardArea> root_for_intersection{};
    std::array<int, kBoardArea> liberty_count_for_root{};
    std::array<int, kBoardArea> stone_count_for_root{};
    std::array<bool, kBoardArea> root_used{};
    std::array<std::vector<int>, kBoardArea> stones_for_root{};
    std::vector<StoneGroup> groups{};
};

[[nodiscard]] constexpr bool is_stone(std::uint8_t stone) {
    return stone == kBlack || stone == kWhite;
}

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

int find_root(BoardAnalysis* analysis, int intersection) {
    if (analysis == nullptr || !is_valid_intersection(intersection) || analysis->parent[intersection] < 0) {
        return -1;
    }

    int node = intersection;
    while (analysis->parent[node] != node) {
        node = analysis->parent[node];
    }

    int current = intersection;
    while (analysis->parent[current] != current) {
        const int next = analysis->parent[current];
        analysis->parent[current] = node;
        current = next;
    }
    return node;
}

void union_sets(BoardAnalysis* analysis, int first, int second) {
    if (analysis == nullptr) {
        return;
    }

    int root_first = find_root(analysis, first);
    int root_second = find_root(analysis, second);
    if (root_first < 0 || root_second < 0 || root_first == root_second) {
        return;
    }

    if (analysis->rank[root_first] < analysis->rank[root_second]) {
        std::swap(root_first, root_second);
    }

    analysis->parent[root_second] = root_first;
    if (analysis->rank[root_first] == analysis->rank[root_second]) {
        ++analysis->rank[root_first];
    }
}

[[nodiscard]] BoardAnalysis analyze_board(const GoPosition& position) {
    BoardAnalysis analysis;
    analysis.parent.fill(-1);
    analysis.rank.fill(0);
    analysis.root_for_intersection.fill(-1);
    analysis.liberty_count_for_root.fill(0);
    analysis.stone_count_for_root.fill(0);
    analysis.root_used.fill(false);

    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        if (is_stone(stone_at(position, intersection))) {
            analysis.parent[intersection] = intersection;
        }
    }

    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const int intersection = to_intersection(row, col);
            const std::uint8_t color = stone_at(position, intersection);
            if (!is_stone(color)) {
                continue;
            }

            if (col + 1 < kBoardSize) {
                const int right = to_intersection(row, col + 1);
                if (stone_at(position, right) == color) {
                    union_sets(&analysis, intersection, right);
                }
            }

            if (row + 1 < kBoardSize) {
                const int down = to_intersection(row + 1, col);
                if (stone_at(position, down) == color) {
                    union_sets(&analysis, intersection, down);
                }
            }
        }
    }

    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        if (!is_stone(stone_at(position, intersection))) {
            continue;
        }

        const int root = find_root(&analysis, intersection);
        if (root < 0) {
            continue;
        }
        analysis.root_for_intersection[intersection] = root;
        analysis.root_used[root] = true;
        ++analysis.stone_count_for_root[root];
        analysis.stones_for_root[root].push_back(intersection);
    }

    std::array<int, kBoardArea> liberty_seen_for_root{};
    liberty_seen_for_root.fill(-1);

    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        const std::uint8_t color = stone_at(position, intersection);
        if (!is_stone(color)) {
            continue;
        }

        const int root = analysis.root_for_intersection[intersection];
        if (root < 0) {
            continue;
        }

        for_each_neighbor(intersection, [&](int neighbor) {
            if (stone_at(position, neighbor) != kEmpty) {
                return;
            }
            if (liberty_seen_for_root[neighbor] == root) {
                return;
            }
            liberty_seen_for_root[neighbor] = root;
            ++analysis.liberty_count_for_root[root];
        });
    }

    analysis.groups.reserve(kBoardArea);
    for (int root = 0; root < kBoardArea; ++root) {
        if (!analysis.root_used[root]) {
            continue;
        }
        analysis.groups.push_back(StoneGroup{
            .representative = root,
            .liberty_count = analysis.liberty_count_for_root[root],
            .stone_count = analysis.stone_count_for_root[root],
        });
    }
    return analysis;
}

[[nodiscard]] MoveResult illegal_result(const GoPosition& position, int action, MoveStatus status) {
    return MoveResult{
        .status = status,
        .position = position,
        .action = action,
        .captured_stones = 0,
        .ko_point = position.ko_point,
    };
}

}  // namespace

bool is_valid_action(int action) { return action >= 0 && action < kActionSpaceSize; }

bool is_pass_action(int action) { return action == kPassAction; }

bool passes_end_game(const GoPosition& position) { return position.consecutive_passes >= 2; }

MoveResult play_pass(const GoPosition& position) {
    if (position.move_number < kMinPassMove) {
        return illegal_result(position, kPassAction, MoveStatus::kPassTooEarly);
    }
    if (!is_valid_color(position.side_to_move)) {
        return illegal_result(position, kPassAction, MoveStatus::kInvalidSideToMove);
    }

    GoPosition next = position;
    next.position_history.insert(zobrist_board_hash(position));
    next.side_to_move = opponent_color(position.side_to_move);
    next.ko_point = -1;
    next.move_number = position.move_number + 1;
    next.consecutive_passes = position.consecutive_passes + 1;

    return MoveResult{
        .status = MoveStatus::kLegal,
        .position = std::move(next),
        .action = kPassAction,
        .captured_stones = 0,
        .ko_point = -1,
    };
}

MoveResult play_action(const GoPosition& position, int action) {
    if (is_pass_action(action)) {
        return play_pass(position);
    }
    if (!is_valid_action(action) || !is_valid_intersection(action)) {
        return illegal_result(position, action, MoveStatus::kInvalidAction);
    }
    if (!is_valid_color(position.side_to_move)) {
        return illegal_result(position, action, MoveStatus::kInvalidSideToMove);
    }
    if (stone_at(position, action) != kEmpty) {
        return illegal_result(position, action, MoveStatus::kIntersectionOccupied);
    }
    if (position.ko_point == action) {
        return illegal_result(position, action, MoveStatus::kKoViolation);
    }

    const int side_to_move = position.side_to_move;
    const int opponent = opponent_color(side_to_move);

    GoPosition next = position;
    next.position_history.insert(zobrist_board_hash(position));
    next.ko_point = -1;
    next.consecutive_passes = 0;
    set_stone(&next, action, static_cast<std::uint8_t>(side_to_move));

    BoardAnalysis analysis_after_placement = analyze_board(next);
    std::array<bool, kBoardArea> captured_roots{};
    int captured_stones = 0;
    int captured_intersection = -1;

    for_each_neighbor(action, [&](int neighbor) {
        if (!is_valid_intersection(neighbor) || stone_at(next, neighbor) != static_cast<std::uint8_t>(opponent)) {
            return;
        }

        const int root = analysis_after_placement.root_for_intersection[neighbor];
        if (root < 0 || captured_roots[root] || analysis_after_placement.liberty_count_for_root[root] > 0) {
            return;
        }

        captured_roots[root] = true;
        for (int stone : analysis_after_placement.stones_for_root[root]) {
            if (captured_stones == 0) {
                captured_intersection = stone;
            } else {
                captured_intersection = -1;
            }
            ++captured_stones;
            set_stone(&next, stone, kEmpty);
        }
    });

    BoardAnalysis analysis_after_capture =
        captured_stones > 0 ? analyze_board(next) : std::move(analysis_after_placement);

    const int own_root = analysis_after_capture.root_for_intersection[action];
    const int own_liberties =
        own_root >= 0 ? analysis_after_capture.liberty_count_for_root[own_root] : 0;
    if (own_liberties <= 0) {
        return illegal_result(position, action, MoveStatus::kSelfCapture);
    }

    const std::uint64_t next_hash = zobrist_board_hash(next);
    if (next.position_history.contains(next_hash)) {
        return illegal_result(position, action, MoveStatus::kSuperkoViolation);
    }

    if (captured_stones == 1 && own_root >= 0 &&
        analysis_after_capture.stone_count_for_root[own_root] == 1 &&
        analysis_after_capture.liberty_count_for_root[own_root] == 1) {
        next.ko_point = captured_intersection;
    }

    next.position_history.insert(next_hash);
    next.side_to_move = opponent;
    next.move_number = position.move_number + 1;

    return MoveResult{
        .status = MoveStatus::kLegal,
        .position = std::move(next),
        .action = action,
        .captured_stones = captured_stones,
        .ko_point = next.ko_point,
    };
}

bool is_legal_action(const GoPosition& position, int action) { return play_action(position, action).legal(); }

std::vector<StoneGroup> compute_stone_groups(const GoPosition& position) {
    return analyze_board(position).groups;
}

int liberties_for_intersection(const GoPosition& position, int intersection) {
    if (!is_valid_intersection(intersection)) {
        return 0;
    }

    const std::uint8_t stone = stone_at(position, intersection);
    if (!is_stone(stone)) {
        return 0;
    }

    const BoardAnalysis analysis = analyze_board(position);
    const int root = analysis.root_for_intersection[intersection];
    if (root < 0) {
        return 0;
    }
    return analysis.liberty_count_for_root[root];
}

}  // namespace alphazero::go
