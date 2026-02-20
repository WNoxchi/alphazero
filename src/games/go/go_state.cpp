#include "games/go/go_state.h"

#include <array>
#include <cstdint>
#include <sstream>

namespace alphazero::go {
namespace {

struct ZobristTable {
    // [0]=black, [1]=white.
    std::array<std::array<std::uint64_t, kBoardArea>, 2> stone{};
    // side_to_move[color], valid indices are kBlack and kWhite.
    std::array<std::uint64_t, 3> side_to_move{};
    // ko_point[0..360], ko_point[361] is "no ko point".
    std::array<std::uint64_t, kBoardArea + 1> ko_point{};
};

[[nodiscard]] constexpr bool is_valid_stone_value(std::uint8_t stone) {
    return stone == kEmpty || stone == kBlack || stone == kWhite;
}

[[nodiscard]] std::uint64_t splitmix64(std::uint64_t* state) {
    std::uint64_t x = (*state += 0x9E3779B97F4A7C15ULL);
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

[[nodiscard]] const ZobristTable& zobrist_table() {
    static const ZobristTable kTable = [] {
        ZobristTable table;
        std::uint64_t seed = 0xC001C0DE1234ABCDULL;

        for (int color = 0; color < 2; ++color) {
            for (int intersection = 0; intersection < kBoardArea; ++intersection) {
                table.stone[color][intersection] = splitmix64(&seed);
            }
        }

        table.side_to_move[kBlack] = splitmix64(&seed);
        table.side_to_move[kWhite] = splitmix64(&seed);

        for (std::uint64_t& key : table.ko_point) {
            key = splitmix64(&seed);
        }

        return table;
    }();
    return kTable;
}

}  // namespace

std::uint8_t stone_at(const GoPosition& position, int row, int col) {
    if (!is_valid_intersection(row, col)) {
        return kEmpty;
    }
    return position.board[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
}

std::uint8_t stone_at(const GoPosition& position, int intersection) {
    if (!is_valid_intersection(intersection)) {
        return kEmpty;
    }
    return stone_at(position, intersection_row(intersection), intersection_col(intersection));
}

void set_stone(GoPosition* position, int row, int col, std::uint8_t stone) {
    if (position == nullptr || !is_valid_intersection(row, col) || !is_valid_stone_value(stone)) {
        return;
    }

    position->board[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = stone;
}

void set_stone(GoPosition* position, int intersection, std::uint8_t stone) {
    if (!is_valid_intersection(intersection)) {
        return;
    }
    set_stone(position, intersection_row(intersection), intersection_col(intersection), stone);
}

std::uint64_t zobrist_stone_key(int intersection, int color) {
    if (!is_valid_intersection(intersection) || !is_valid_color(color)) {
        return 0ULL;
    }

    const auto& table = zobrist_table();
    const int stone_index = color == kBlack ? 0 : 1;
    return table.stone[static_cast<std::size_t>(stone_index)][static_cast<std::size_t>(intersection)];
}

std::uint64_t zobrist_side_to_move_key(int side_to_move) {
    if (!is_valid_color(side_to_move)) {
        return 0ULL;
    }

    const auto& table = zobrist_table();
    return table.side_to_move[static_cast<std::size_t>(side_to_move)];
}

std::uint64_t zobrist_ko_point_key(int ko_point) {
    const auto& table = zobrist_table();
    const int index = is_valid_intersection(ko_point) ? ko_point : kBoardArea;
    return table.ko_point[static_cast<std::size_t>(index)];
}

std::uint64_t zobrist_update_for_stone(std::uint64_t board_hash, int intersection, int color) {
    return board_hash ^ zobrist_stone_key(intersection, color);
}

std::uint64_t zobrist_update_for_stone(std::uint64_t board_hash, int row, int col, int color) {
    return zobrist_update_for_stone(board_hash, to_intersection(row, col), color);
}

std::uint64_t zobrist_board_hash(const GoPosition& position) {
    std::uint64_t hash = 0ULL;
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const std::uint8_t stone = stone_at(position, row, col);
            if (!is_valid_color(stone)) {
                continue;
            }

            const int intersection = to_intersection(row, col);
            hash ^= zobrist_stone_key(intersection, stone);
        }
    }
    return hash;
}

std::uint64_t zobrist_hash(const GoPosition& position) {
    std::uint64_t hash = zobrist_board_hash(position);
    hash ^= zobrist_side_to_move_key(position.side_to_move);
    hash ^= zobrist_ko_point_key(position.ko_point);
    return hash;
}

std::string board_to_string(const GoPosition& position) {
    std::ostringstream out;
    for (int row = kBoardSize - 1; row >= 0; --row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const std::uint8_t stone = stone_at(position, row, col);
            const char symbol = stone == kBlack ? 'X' : (stone == kWhite ? 'O' : '.');
            out << symbol;
        }
        if (row > 0) {
            out << '\n';
        }
    }
    return out.str();
}

}  // namespace alphazero::go
