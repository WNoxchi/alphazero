#include "games/chess/bitboard.h"

#include <array>
#include <bit>
#include <cstdint>
#include <sstream>

namespace alphazero::chess {
namespace {

enum Direction : int {
    kNorth = 0,
    kNorthEast = 1,
    kEast = 2,
    kSouthEast = 3,
    kSouth = 4,
    kSouthWest = 5,
    kWest = 6,
    kNorthWest = 7,
    kDirectionCount = 8,
};

struct ZobristTable {
    std::array<std::array<std::array<std::uint64_t, kBoardSquares>, kPieceTypeCount>, 2> piece{};
    std::array<std::uint64_t, 16> castling{};
    std::array<std::uint64_t, kBoardSquares + 1> en_passant{};
    std::uint64_t side_to_move = 0;
};

[[nodiscard]] constexpr int square_file(int square) { return square & 7; }
[[nodiscard]] constexpr int square_rank(int square) { return square >> 3; }
[[nodiscard]] constexpr int to_square(int file, int rank) { return rank * 8 + file; }

[[nodiscard]] const std::array<int, kDirectionCount>& direction_file_delta() {
    static constexpr std::array<int, kDirectionCount> kDelta = {0, 1, 1, 1, 0, -1, -1, -1};
    return kDelta;
}

[[nodiscard]] const std::array<int, kDirectionCount>& direction_rank_delta() {
    static constexpr std::array<int, kDirectionCount> kDelta = {1, 1, 0, -1, -1, -1, 0, 1};
    return kDelta;
}

[[nodiscard]] const std::array<int, kDirectionCount>& direction_square_delta() {
    static constexpr std::array<int, kDirectionCount> kDelta = {8, 9, 1, -7, -8, -9, -1, 7};
    return kDelta;
}

[[nodiscard]] const std::array<std::array<Bitboard, kDirectionCount>, kBoardSquares>& ray_table() {
    static const auto kRays = [] {
        std::array<std::array<Bitboard, kDirectionCount>, kBoardSquares> rays{};
        const auto& file_delta = direction_file_delta();
        const auto& rank_delta = direction_rank_delta();
        for (int square = 0; square < kBoardSquares; ++square) {
            const int from_file = square_file(square);
            const int from_rank = square_rank(square);
            for (int dir = 0; dir < kDirectionCount; ++dir) {
                int file = from_file + file_delta[dir];
                int rank = from_rank + rank_delta[dir];
                Bitboard attacks = 0ULL;
                while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
                    attacks |= square_bit(to_square(file, rank));
                    file += file_delta[dir];
                    rank += rank_delta[dir];
                }
                rays[square][dir] = attacks;
            }
        }
        return rays;
    }();
    return kRays;
}

[[nodiscard]] const std::array<Bitboard, kBoardSquares>& knight_table() {
    static const auto kKnightAttacks = [] {
        constexpr std::array<int, 8> kFileOffset = {1, 2, 2, 1, -1, -2, -2, -1};
        constexpr std::array<int, 8> kRankOffset = {2, 1, -1, -2, -2, -1, 1, 2};
        std::array<Bitboard, kBoardSquares> table{};
        for (int square = 0; square < kBoardSquares; ++square) {
            const int file = square_file(square);
            const int rank = square_rank(square);
            Bitboard attacks = 0ULL;
            for (std::size_t i = 0; i < kFileOffset.size(); ++i) {
                const int next_file = file + kFileOffset[i];
                const int next_rank = rank + kRankOffset[i];
                if (next_file >= 0 && next_file < 8 && next_rank >= 0 && next_rank < 8) {
                    attacks |= square_bit(to_square(next_file, next_rank));
                }
            }
            table[square] = attacks;
        }
        return table;
    }();
    return kKnightAttacks;
}

[[nodiscard]] const std::array<Bitboard, kBoardSquares>& king_table() {
    static const auto kKingAttacks = [] {
        constexpr std::array<int, 8> kFileOffset = {0, 1, 1, 1, 0, -1, -1, -1};
        constexpr std::array<int, 8> kRankOffset = {1, 1, 0, -1, -1, -1, 0, 1};
        std::array<Bitboard, kBoardSquares> table{};
        for (int square = 0; square < kBoardSquares; ++square) {
            const int file = square_file(square);
            const int rank = square_rank(square);
            Bitboard attacks = 0ULL;
            for (std::size_t i = 0; i < kFileOffset.size(); ++i) {
                const int next_file = file + kFileOffset[i];
                const int next_rank = rank + kRankOffset[i];
                if (next_file >= 0 && next_file < 8 && next_rank >= 0 && next_rank < 8) {
                    attacks |= square_bit(to_square(next_file, next_rank));
                }
            }
            table[square] = attacks;
        }
        return table;
    }();
    return kKingAttacks;
}

[[nodiscard]] Bitboard directional_slider_attacks(int square, Bitboard occupancy, Direction direction) {
    const auto& rays = ray_table();
    const Bitboard ray = rays[square][direction];
    const Bitboard blockers = ray & occupancy;
    if (blockers == 0ULL) {
        return ray;
    }

    const int delta = direction_square_delta()[direction];
    const int blocker_square = delta > 0 ? bit_scan_forward(blockers) : bit_scan_reverse(blockers);
    if (blocker_square < 0) {
        return ray;
    }
    // Remove squares beyond the first blocker while keeping the blocker square itself.
    return ray ^ rays[blocker_square][direction];
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
        std::uint64_t seed = 0xA17ECAFE1234BEEFULL;

        for (int color = 0; color < 2; ++color) {
            for (int piece = 0; piece < kPieceTypeCount; ++piece) {
                for (int square = 0; square < kBoardSquares; ++square) {
                    table.piece[color][piece][square] = splitmix64(&seed);
                }
            }
        }
        for (std::uint64_t& key : table.castling) {
            key = splitmix64(&seed);
        }
        for (std::uint64_t& key : table.en_passant) {
            key = splitmix64(&seed);
        }
        table.side_to_move = splitmix64(&seed);
        return table;
    }();
    return kTable;
}

}  // namespace

int popcount(Bitboard bits) { return std::popcount(bits); }

int bit_scan_forward(Bitboard bits) {
    if (bits == 0ULL) {
        return -1;
    }
    return static_cast<int>(std::countr_zero(bits));
}

int bit_scan_reverse(Bitboard bits) {
    if (bits == 0ULL) {
        return -1;
    }
    return static_cast<int>((kBoardSquares - 1) - std::countl_zero(bits));
}

int pop_lsb(Bitboard* bits) {
    if (bits == nullptr || *bits == 0ULL) {
        return -1;
    }
    const int square = bit_scan_forward(*bits);
    *bits &= (*bits - 1ULL);
    return square;
}

Bitboard knight_attacks(int square) {
    if (!is_valid_square(square)) {
        return 0ULL;
    }
    return knight_table()[square];
}

Bitboard king_attacks(int square) {
    if (!is_valid_square(square)) {
        return 0ULL;
    }
    return king_table()[square];
}

Bitboard bishop_attacks(int square, Bitboard occupancy) {
    if (!is_valid_square(square)) {
        return 0ULL;
    }
    return directional_slider_attacks(square, occupancy, kNorthEast) |
           directional_slider_attacks(square, occupancy, kSouthEast) |
           directional_slider_attacks(square, occupancy, kSouthWest) |
           directional_slider_attacks(square, occupancy, kNorthWest);
}

Bitboard rook_attacks(int square, Bitboard occupancy) {
    if (!is_valid_square(square)) {
        return 0ULL;
    }
    return directional_slider_attacks(square, occupancy, kNorth) |
           directional_slider_attacks(square, occupancy, kEast) |
           directional_slider_attacks(square, occupancy, kSouth) |
           directional_slider_attacks(square, occupancy, kWest);
}

Bitboard queen_attacks(int square, Bitboard occupancy) {
    if (!is_valid_square(square)) {
        return 0ULL;
    }
    return bishop_attacks(square, occupancy) | rook_attacks(square, occupancy);
}

Bitboard pawn_attacks(int color, Bitboard pawns) {
    if (color == kWhite) {
        return shift_north_east(pawns) | shift_north_west(pawns);
    }
    return shift_south_east(pawns) | shift_south_west(pawns);
}

Bitboard occupied_by(const ChessPosition& position, int color) {
    if (color != kWhite && color != kBlack) {
        return 0ULL;
    }

    Bitboard bits = 0ULL;
    for (int piece = 0; piece < kPieceTypeCount; ++piece) {
        bits |= position.pieces[color][piece];
    }
    return bits;
}

Bitboard occupied(const ChessPosition& position) {
    return occupied_by(position, kWhite) | occupied_by(position, kBlack);
}

std::uint64_t zobrist_hash(const ChessPosition& position) {
    const auto& table = zobrist_table();
    std::uint64_t hash = 0ULL;

    for (int color = 0; color < 2; ++color) {
        for (int piece = 0; piece < kPieceTypeCount; ++piece) {
            Bitboard pieces = position.pieces[color][piece];
            while (pieces != 0ULL) {
                const int square = pop_lsb(&pieces);
                if (square >= 0) {
                    hash ^= table.piece[color][piece][square];
                }
            }
        }
    }

    if (position.side_to_move == kBlack) {
        hash ^= table.side_to_move;
    }
    hash ^= table.castling[position.castling & 0x0FU];

    const int en_passant_index =
        is_valid_square(position.en_passant_square) ? position.en_passant_square : kBoardSquares;
    hash ^= table.en_passant[en_passant_index];

    return hash;
}

std::string bitboard_to_string(Bitboard bits) {
    std::ostringstream out;
    for (int rank = 7; rank >= 0; --rank) {
        for (int file = 0; file < 8; ++file) {
            const int square = to_square(file, rank);
            out << ((bits & square_bit(square)) != 0ULL ? '1' : '.');
        }
        if (rank > 0) {
            out << '\n';
        }
    }
    return out.str();
}

}  // namespace alphazero::chess
