#include "games/chess/bitboard.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::chess::Bitboard;
using alphazero::chess::ChessPosition;
using alphazero::chess::kBlack;
using alphazero::chess::kBoardSquares;
using alphazero::chess::kKing;
using alphazero::chess::kPawn;
using alphazero::chess::kQueen;
using alphazero::chess::kRook;
using alphazero::chess::kWhite;
using alphazero::chess::square_bit;

[[nodiscard]] constexpr int square(int file, int rank) { return rank * 8 + file; }

[[nodiscard]] Bitboard naive_knight_attacks(int source_square) {
    constexpr std::array<int, 8> kFileOffset = {1, 2, 2, 1, -1, -2, -2, -1};
    constexpr std::array<int, 8> kRankOffset = {2, 1, -1, -2, -2, -1, 1, 2};
    const int source_file = source_square & 7;
    const int source_rank = source_square >> 3;

    Bitboard attacks = 0ULL;
    for (std::size_t i = 0; i < kFileOffset.size(); ++i) {
        const int file = source_file + kFileOffset[i];
        const int rank = source_rank + kRankOffset[i];
        if (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
            attacks |= square_bit(square(file, rank));
        }
    }
    return attacks;
}

[[nodiscard]] Bitboard naive_king_attacks(int source_square) {
    constexpr std::array<int, 8> kFileOffset = {0, 1, 1, 1, 0, -1, -1, -1};
    constexpr std::array<int, 8> kRankOffset = {1, 1, 0, -1, -1, -1, 0, 1};
    const int source_file = source_square & 7;
    const int source_rank = source_square >> 3;

    Bitboard attacks = 0ULL;
    for (std::size_t i = 0; i < kFileOffset.size(); ++i) {
        const int file = source_file + kFileOffset[i];
        const int rank = source_rank + kRankOffset[i];
        if (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
            attacks |= square_bit(square(file, rank));
        }
    }
    return attacks;
}

[[nodiscard]] Bitboard naive_slider_attacks(int source_square, int file_step, int rank_step, Bitboard occupancy) {
    int file = (source_square & 7) + file_step;
    int rank = (source_square >> 3) + rank_step;

    Bitboard attacks = 0ULL;
    while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
        const int destination = square(file, rank);
        attacks |= square_bit(destination);
        if ((occupancy & square_bit(destination)) != 0ULL) {
            break;
        }
        file += file_step;
        rank += rank_step;
    }
    return attacks;
}

[[nodiscard]] Bitboard naive_bishop_attacks(int source_square, Bitboard occupancy) {
    return naive_slider_attacks(source_square, 1, 1, occupancy) |
           naive_slider_attacks(source_square, 1, -1, occupancy) |
           naive_slider_attacks(source_square, -1, -1, occupancy) |
           naive_slider_attacks(source_square, -1, 1, occupancy);
}

[[nodiscard]] Bitboard naive_rook_attacks(int source_square, Bitboard occupancy) {
    return naive_slider_attacks(source_square, 0, 1, occupancy) |
           naive_slider_attacks(source_square, 1, 0, occupancy) |
           naive_slider_attacks(source_square, 0, -1, occupancy) |
           naive_slider_attacks(source_square, -1, 0, occupancy);
}

[[nodiscard]] Bitboard deterministic_random_mask(std::uint64_t* seed) {
    // WHY: Deterministic occupancy patterns make this test stable across machines/runs.
    std::uint64_t x = (*seed += 0x9E3779B97F4A7C15ULL);
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

}  // namespace

// WHY: These primitives are used everywhere in move generation and must be exact.
TEST(ChessBitboardTest, PopcountAndBitScansBehaveAsExpected) {
    Bitboard bits = square_bit(square(0, 0)) | square_bit(square(7, 0)) | square_bit(square(2, 5)) |
                    square_bit(square(7, 7));
    EXPECT_EQ(alphazero::chess::popcount(bits), 4);
    EXPECT_EQ(alphazero::chess::bit_scan_forward(bits), square(0, 0));
    EXPECT_EQ(alphazero::chess::bit_scan_reverse(bits), square(7, 7));

    Bitboard pop_bits = bits;
    std::vector<int> popped;
    while (pop_bits != 0ULL) {
        popped.push_back(alphazero::chess::pop_lsb(&pop_bits));
    }
    ASSERT_EQ(popped.size(), 4U);
    EXPECT_EQ(popped[0], square(0, 0));
    EXPECT_EQ(popped[1], square(7, 0));
    EXPECT_EQ(popped[2], square(2, 5));
    EXPECT_EQ(popped[3], square(7, 7));
    EXPECT_EQ(alphazero::chess::pop_lsb(&pop_bits), -1);
    EXPECT_EQ(alphazero::chess::bit_scan_forward(0ULL), -1);
    EXPECT_EQ(alphazero::chess::bit_scan_reverse(0ULL), -1);
}

// WHY: File masks in shifts prevent wraparound bugs that corrupt move generation.
TEST(ChessBitboardTest, ShiftOperationsRespectBoardEdges) {
    EXPECT_EQ(alphazero::chess::shift_west(square_bit(square(0, 0))), 0ULL);
    EXPECT_EQ(alphazero::chess::shift_east(square_bit(square(7, 0))), 0ULL);
    EXPECT_EQ(alphazero::chess::shift_north(square_bit(square(0, 7))), 0ULL);
    EXPECT_EQ(alphazero::chess::shift_south(square_bit(square(0, 0))), 0ULL);

    EXPECT_EQ(alphazero::chess::shift_north(square_bit(square(3, 3))), square_bit(square(3, 4)));
    EXPECT_EQ(alphazero::chess::shift_south(square_bit(square(3, 3))), square_bit(square(3, 2)));
    EXPECT_EQ(alphazero::chess::shift_east(square_bit(square(3, 3))), square_bit(square(4, 3)));
    EXPECT_EQ(alphazero::chess::shift_west(square_bit(square(3, 3))), square_bit(square(2, 3)));
    EXPECT_EQ(alphazero::chess::shift_north_east(square_bit(square(3, 3))), square_bit(square(4, 4)));
    EXPECT_EQ(alphazero::chess::shift_north_west(square_bit(square(3, 3))), square_bit(square(2, 4)));
    EXPECT_EQ(alphazero::chess::shift_south_east(square_bit(square(3, 3))), square_bit(square(4, 2)));
    EXPECT_EQ(alphazero::chess::shift_south_west(square_bit(square(3, 3))), square_bit(square(2, 2)));
}

// WHY: Knight/king tables are precomputed once and reused for every node expansion.
TEST(ChessBitboardTest, KnightAndKingAttackTablesMatchNaiveGeneration) {
    for (int source_square = 0; source_square < kBoardSquares; ++source_square) {
        EXPECT_EQ(alphazero::chess::knight_attacks(source_square), naive_knight_attacks(source_square))
            << "square=" << source_square;
        EXPECT_EQ(alphazero::chess::king_attacks(source_square), naive_king_attacks(source_square))
            << "square=" << source_square;
    }
}

// WHY: Sliding-piece attacks are the core of legal move generation and perft correctness.
TEST(ChessBitboardTest, SlidingPieceAttacksMatchNaiveRayTracing) {
    std::vector<Bitboard> occupancies = {
        0ULL,
        0x0000001818000000ULL,  // center cluster.
        0x8100000000000081ULL,  // corner pieces.
        0x00FF00000000FF00ULL,  // ranks 2 and 7.
    };

    std::uint64_t seed = 0x1234FEDCBA987654ULL;
    for (int i = 0; i < 16; ++i) {
        occupancies.push_back(deterministic_random_mask(&seed));
    }

    for (int source_square = 0; source_square < kBoardSquares; ++source_square) {
        for (Bitboard occupancy : occupancies) {
            const Bitboard expected_bishop = naive_bishop_attacks(source_square, occupancy);
            const Bitboard expected_rook = naive_rook_attacks(source_square, occupancy);
            const Bitboard expected_queen = expected_bishop | expected_rook;

            EXPECT_EQ(alphazero::chess::bishop_attacks(source_square, occupancy), expected_bishop)
                << "square=" << source_square << " occupancy=" << occupancy;
            EXPECT_EQ(alphazero::chess::rook_attacks(source_square, occupancy), expected_rook)
                << "square=" << source_square << " occupancy=" << occupancy;
            EXPECT_EQ(alphazero::chess::queen_attacks(source_square, occupancy), expected_queen)
                << "square=" << source_square << " occupancy=" << occupancy;
        }
    }
}

// WHY: Pawn attack masks are direction-sensitive and frequently used for king-safety checks.
TEST(ChessBitboardTest, PawnAttackMasksAreDirectionalAndDoNotWrapFiles) {
    const Bitboard white_pawns = square_bit(square(0, 1)) | square_bit(square(3, 3)) | square_bit(square(7, 1));
    const Bitboard black_pawns = square_bit(square(0, 6)) | square_bit(square(3, 4)) | square_bit(square(7, 6));

    const Bitboard expected_white =
        square_bit(square(1, 2)) | square_bit(square(2, 4)) | square_bit(square(4, 4)) | square_bit(square(6, 2));
    const Bitboard expected_black =
        square_bit(square(1, 5)) | square_bit(square(2, 3)) | square_bit(square(4, 3)) | square_bit(square(6, 5));

    EXPECT_EQ(alphazero::chess::pawn_attacks(kWhite, white_pawns), expected_white);
    EXPECT_EQ(alphazero::chess::pawn_attacks(kBlack, black_pawns), expected_black);
}

// WHY: Correct occupancy aggregation is used by attack generation, pins/check detection, and hashing.
TEST(ChessBitboardTest, OccupancyHelpersAggregatePiecesByColor) {
    ChessPosition position{};
    position.pieces[kWhite][kKing] = square_bit(square(4, 0));
    position.pieces[kWhite][kPawn] = square_bit(square(3, 1));
    position.pieces[kBlack][kKing] = square_bit(square(4, 7));
    position.pieces[kBlack][kQueen] = square_bit(square(3, 6));

    const Bitboard white_expected = square_bit(square(4, 0)) | square_bit(square(3, 1));
    const Bitboard black_expected = square_bit(square(4, 7)) | square_bit(square(3, 6));
    EXPECT_EQ(alphazero::chess::occupied_by(position, kWhite), white_expected);
    EXPECT_EQ(alphazero::chess::occupied_by(position, kBlack), black_expected);
    EXPECT_EQ(alphazero::chess::occupied(position), white_expected | black_expected);
}

// WHY: Repetition detection requires deterministic hashes that change when position state changes.
TEST(ChessBitboardTest, ZobristHashIsStableAndSensitiveToPositionState) {
    ChessPosition baseline{};
    baseline.pieces[kWhite][kKing] = square_bit(square(4, 0));
    baseline.pieces[kWhite][kRook] = square_bit(square(7, 0));
    baseline.pieces[kWhite][kPawn] = square_bit(square(4, 1));
    baseline.pieces[kBlack][kKing] = square_bit(square(4, 7));
    baseline.pieces[kBlack][kRook] = square_bit(square(0, 7));
    baseline.pieces[kBlack][kPawn] = square_bit(square(3, 6));
    baseline.side_to_move = kWhite;
    baseline.castling = static_cast<std::uint8_t>(alphazero::chess::kWhiteKingSide | alphazero::chess::kBlackQueenSide);
    baseline.en_passant_square = square(4, 5);

    const std::uint64_t base_hash = alphazero::chess::zobrist_hash(baseline);
    const std::uint64_t same_hash = alphazero::chess::zobrist_hash(baseline);
    EXPECT_EQ(base_hash, same_hash);

    ChessPosition side_changed = baseline;
    side_changed.side_to_move = kBlack;
    EXPECT_NE(alphazero::chess::zobrist_hash(side_changed), base_hash);

    ChessPosition castling_changed = baseline;
    castling_changed.castling = static_cast<std::uint8_t>(alphazero::chess::kWhiteQueenSide);
    EXPECT_NE(alphazero::chess::zobrist_hash(castling_changed), base_hash);

    ChessPosition ep_changed = baseline;
    ep_changed.en_passant_square = -1;
    EXPECT_NE(alphazero::chess::zobrist_hash(ep_changed), base_hash);

    ChessPosition piece_changed = baseline;
    piece_changed.pieces[kWhite][kPawn] ^= square_bit(square(4, 1));
    piece_changed.pieces[kWhite][kPawn] ^= square_bit(square(4, 2));
    EXPECT_NE(alphazero::chess::zobrist_hash(piece_changed), base_hash);
}
