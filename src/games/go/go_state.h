#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>

namespace alphazero::go {

constexpr int kBoardSize = 19;
constexpr int kBoardArea = kBoardSize * kBoardSize;
constexpr int kPassAction = kBoardArea;
constexpr int kActionSpaceSize = kBoardArea + 1;
constexpr int kMaxGameLength = kBoardArea * 2;
constexpr float kDefaultKomi = 7.5F;

enum StoneColor : std::uint8_t {
    kEmpty = 0U,
    kBlack = 1U,
    kWhite = 2U,
};

struct GoPosition {
    // Board state: 0=empty, 1=black, 2=white.
    std::array<std::array<std::uint8_t, kBoardSize>, kBoardSize> board{};

    // Side to move: 1=black, 2=white.
    int side_to_move = kBlack;

    // Ko point encoded as a flat intersection index [0, 360], or -1 when none.
    int ko_point = -1;

    // Komi compensation for white.
    float komi = kDefaultKomi;

    // Number of moves played so far (including passes).
    int move_number = 0;

    // Consecutive passes; game terminates at 2.
    int consecutive_passes = 0;

    // Positional superko history (board-only hashes).
    std::unordered_set<std::uint64_t> position_history{};
};

[[nodiscard]] constexpr bool is_valid_color(int color) { return color == kBlack || color == kWhite; }

[[nodiscard]] constexpr bool is_valid_intersection(int row, int col) {
    return row >= 0 && row < kBoardSize && col >= 0 && col < kBoardSize;
}

[[nodiscard]] constexpr bool is_valid_intersection(int intersection) {
    return intersection >= 0 && intersection < kBoardArea;
}

[[nodiscard]] constexpr int to_intersection(int row, int col) {
    return is_valid_intersection(row, col) ? (row * kBoardSize) + col : -1;
}

[[nodiscard]] constexpr int intersection_row(int intersection) {
    return is_valid_intersection(intersection) ? (intersection / kBoardSize) : -1;
}

[[nodiscard]] constexpr int intersection_col(int intersection) {
    return is_valid_intersection(intersection) ? (intersection % kBoardSize) : -1;
}

[[nodiscard]] constexpr int opponent_color(int color) {
    return color == kBlack ? kWhite : (color == kWhite ? kBlack : kEmpty);
}

[[nodiscard]] std::uint8_t stone_at(const GoPosition& position, int row, int col);
[[nodiscard]] std::uint8_t stone_at(const GoPosition& position, int intersection);

void set_stone(GoPosition* position, int row, int col, std::uint8_t stone);
void set_stone(GoPosition* position, int intersection, std::uint8_t stone);

// Deterministic Zobrist keys and hash update helpers.
[[nodiscard]] std::uint64_t zobrist_stone_key(int intersection, int color);
[[nodiscard]] std::uint64_t zobrist_side_to_move_key(int side_to_move);
[[nodiscard]] std::uint64_t zobrist_ko_point_key(int ko_point);

// Incremental board-hash update for stone placements/captures (XOR toggle).
[[nodiscard]] std::uint64_t zobrist_update_for_stone(std::uint64_t board_hash, int intersection, int color);
[[nodiscard]] std::uint64_t zobrist_update_for_stone(std::uint64_t board_hash, int row, int col, int color);

// Board-only hash (for positional superko).
[[nodiscard]] std::uint64_t zobrist_board_hash(const GoPosition& position);

// Full position hash (board + side-to-move + ko point).
[[nodiscard]] std::uint64_t zobrist_hash(const GoPosition& position);

[[nodiscard]] std::string board_to_string(const GoPosition& position);

}  // namespace alphazero::go
