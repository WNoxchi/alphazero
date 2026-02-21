#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace alphazero::chess {

using Bitboard = std::uint64_t;

constexpr int kBoardSquares = 64;

// Square indexing: A1=0, B1=1, ..., H8=63.
constexpr Bitboard kFileAMask = 0x0101010101010101ULL;
constexpr Bitboard kFileHMask = 0x8080808080808080ULL;
constexpr Bitboard kNotFileAMask = 0xFEFEFEFEFEFEFEFEULL;
constexpr Bitboard kNotFileHMask = 0x7F7F7F7F7F7F7F7FULL;

enum Color : int {
    kWhite = 0,
    kBlack = 1,
};

enum PieceType : int {
    kPawn = 0,
    kKnight = 1,
    kBishop = 2,
    kRook = 3,
    kQueen = 4,
    kKing = 5,
    kPieceTypeCount = 6,
};

enum CastlingRight : std::uint8_t {
    kWhiteKingSide = 1U << 0U,
    kWhiteQueenSide = 1U << 1U,
    kBlackKingSide = 1U << 2U,
    kBlackQueenSide = 1U << 3U,
};

struct ChessPosition {
    // 12 bitboards: 6 piece types x 2 colors.
    std::uint64_t pieces[2][kPieceTypeCount]{};

    // Side to move: 0=white, 1=black.
    int side_to_move = kWhite;

    // Castling rights (4 bits).
    std::uint8_t castling = 0;

    // En passant target square (-1 when unavailable).
    int en_passant_square = -1;

    // Halfmove clock for 50-move rule.
    int halfmove_clock = 0;

    // Fullmove number, starting at 1.
    int fullmove_number = 1;

    // Count of times this position has repeated in the game history.
    int repetition_count = 1;
};

[[nodiscard]] constexpr bool is_valid_square(int square) { return square >= 0 && square < kBoardSquares; }

[[nodiscard]] constexpr Bitboard square_bit(int square) {
    return is_valid_square(square) ? (1ULL << static_cast<unsigned>(square)) : 0ULL;
}

[[nodiscard]] int popcount(Bitboard bits);
[[nodiscard]] int bit_scan_forward(Bitboard bits);
[[nodiscard]] int bit_scan_reverse(Bitboard bits);

// Removes and returns the index of the least-significant set bit, or -1 if empty.
int pop_lsb(Bitboard* bits);

[[nodiscard]] constexpr Bitboard shift_north(Bitboard bits) { return bits << 8U; }
[[nodiscard]] constexpr Bitboard shift_south(Bitboard bits) { return bits >> 8U; }
[[nodiscard]] constexpr Bitboard shift_east(Bitboard bits) { return (bits & kNotFileHMask) << 1U; }
[[nodiscard]] constexpr Bitboard shift_west(Bitboard bits) { return (bits & kNotFileAMask) >> 1U; }
[[nodiscard]] constexpr Bitboard shift_north_east(Bitboard bits) { return (bits & kNotFileHMask) << 9U; }
[[nodiscard]] constexpr Bitboard shift_north_west(Bitboard bits) { return (bits & kNotFileAMask) << 7U; }
[[nodiscard]] constexpr Bitboard shift_south_east(Bitboard bits) { return (bits & kNotFileHMask) >> 7U; }
[[nodiscard]] constexpr Bitboard shift_south_west(Bitboard bits) { return (bits & kNotFileAMask) >> 9U; }

[[nodiscard]] Bitboard knight_attacks(int square);
[[nodiscard]] Bitboard king_attacks(int square);
[[nodiscard]] Bitboard bishop_attacks(int square, Bitboard occupancy);
[[nodiscard]] Bitboard rook_attacks(int square, Bitboard occupancy);
[[nodiscard]] Bitboard queen_attacks(int square, Bitboard occupancy);

// Aggregated pawn attack masks for all pawns in `pawns`.
[[nodiscard]] Bitboard pawn_attacks(int color, Bitboard pawns);

[[nodiscard]] Bitboard occupied_by(const ChessPosition& position, int color);
[[nodiscard]] Bitboard occupied(const ChessPosition& position);

// Deterministic 64-bit Zobrist hash for repetition detection.
[[nodiscard]] std::uint64_t zobrist_hash(const ChessPosition& position);

[[nodiscard]] std::string bitboard_to_string(Bitboard bits);

}  // namespace alphazero::chess
