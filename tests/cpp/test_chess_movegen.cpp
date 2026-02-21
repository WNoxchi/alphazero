#include "games/chess/movegen.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::chess::ChessPosition;
using alphazero::chess::Move;
using alphazero::chess::kActionPlanesPerSquare;
using alphazero::chess::kActionSpaceSize;
using alphazero::chess::kBishop;
using alphazero::chess::kBlack;
using alphazero::chess::kBoardSquares;
using alphazero::chess::kKnight;
using alphazero::chess::kPawn;
using alphazero::chess::kQueen;
using alphazero::chess::kRook;
using alphazero::chess::kWhite;
using alphazero::chess::square_bit;

[[nodiscard]] constexpr int square(int file, int rank) { return rank * 8 + file; }

[[nodiscard]] int algebraic_to_square(const std::string& square_name) {
    if (square_name.size() != 2) {
        return -1;
    }
    const int file = square_name[0] - 'a';
    const int rank = square_name[1] - '1';
    if (file < 0 || file >= 8 || rank < 0 || rank >= 8) {
        return -1;
    }
    return square(file, rank);
}

[[nodiscard]] int piece_type_from_fen_char(char symbol) {
    switch (std::tolower(static_cast<unsigned char>(symbol))) {
        case 'p':
            return kPawn;
        case 'n':
            return kKnight;
        case 'b':
            return kBishop;
        case 'r':
            return kRook;
        case 'q':
            return kQueen;
        case 'k':
            return alphazero::chess::kKing;
        default:
            return -1;
    }
}

[[nodiscard]] ChessPosition position_from_fen(const std::string& fen) {
    ChessPosition position{};

    std::istringstream stream(fen);
    std::string placement;
    std::string side_to_move;
    std::string castling;
    std::string en_passant;
    std::string halfmove;
    std::string fullmove;

    stream >> placement >> side_to_move >> castling >> en_passant >> halfmove >> fullmove;

    int rank = 7;
    int file = 0;
    for (char symbol : placement) {
        if (symbol == '/') {
            --rank;
            file = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(symbol)) != 0) {
            file += symbol - '0';
            continue;
        }

        const int color = std::isupper(static_cast<unsigned char>(symbol)) != 0 ? kWhite : kBlack;
        const int piece = piece_type_from_fen_char(symbol);
        if (piece >= 0 && rank >= 0 && rank < 8 && file >= 0 && file < 8) {
            position.pieces[color][piece] |= square_bit(square(file, rank));
        }
        ++file;
    }

    position.side_to_move = (side_to_move == "b") ? kBlack : kWhite;

    position.castling = 0;
    if (castling.find('K') != std::string::npos) {
        position.castling = static_cast<std::uint8_t>(position.castling | alphazero::chess::kWhiteKingSide);
    }
    if (castling.find('Q') != std::string::npos) {
        position.castling = static_cast<std::uint8_t>(position.castling | alphazero::chess::kWhiteQueenSide);
    }
    if (castling.find('k') != std::string::npos) {
        position.castling = static_cast<std::uint8_t>(position.castling | alphazero::chess::kBlackKingSide);
    }
    if (castling.find('q') != std::string::npos) {
        position.castling = static_cast<std::uint8_t>(position.castling | alphazero::chess::kBlackQueenSide);
    }

    if (en_passant != "-") {
        position.en_passant_square = algebraic_to_square(en_passant);
    } else {
        position.en_passant_square = -1;
    }

    position.halfmove_clock = halfmove.empty() ? 0 : std::stoi(halfmove);
    position.fullmove_number = fullmove.empty() ? 1 : std::stoi(fullmove);
    position.repetition_count = 1;

    return position;
}

[[nodiscard]] std::uint64_t perft(const ChessPosition& position, int depth) {
    if (depth == 0) {
        return 1ULL;
    }

    const std::vector<Move> legal_moves = alphazero::chess::generate_legal_moves(position);
    if (depth == 1) {
        return legal_moves.size();
    }

    std::uint64_t nodes = 0;
    for (const Move& move : legal_moves) {
        nodes += perft(alphazero::chess::apply_move(position, move), depth - 1);
    }
    return nodes;
}

[[nodiscard]] bool move_present(const std::vector<Move>& moves, const Move& candidate) {
    for (const Move& move : moves) {
        if (move == candidate) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool is_knight_delta(const Move& move) {
    const int file_delta = std::abs((move.to & 7) - (move.from & 7));
    const int rank_delta = std::abs((move.to >> 3) - (move.from >> 3));
    return (file_delta == 1 && rank_delta == 2) || (file_delta == 2 && rank_delta == 1);
}

}  // namespace

// WHY: Action-index translation must be exact because NN policy logits and MCTS use only flat action IDs.
TEST(ChessMovegenTest, ActionIndexRoundTripCoversAllMoveFamilies) {
    std::vector<std::string> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",      // baseline openings.
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                          // castling.
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",                            // en passant.
        "5n1r/6P1/8/8/8/8/8/4K2k w - - 0 1",                            // promotions.
        "4k3/8/8/8/8/8/1p6/4K3 b - - 0 1",                              // black-to-move mapping.
    };

    int queen_family_moves = 0;
    int knight_family_moves = 0;
    int underpromotion_moves = 0;
    int castling_moves = 0;
    int en_passant_moves = 0;
    int black_perspective_moves = 0;

    for (const std::string& fen : fens) {
        const ChessPosition position = position_from_fen(fen);
        const std::vector<Move> legal_moves = alphazero::chess::generate_legal_moves(position);
        ASSERT_FALSE(legal_moves.empty()) << "fen=" << fen;

        for (const Move& move : legal_moves) {
            const int action_index = alphazero::chess::semantic_move_to_action_index(move, position.side_to_move);
            ASSERT_GE(action_index, 0) << "fen=" << fen << " from=" << move.from << " to=" << move.to;
            ASSERT_LT(action_index, kActionSpaceSize);

            const std::optional<Move> decoded = alphazero::chess::action_index_to_semantic_move(position, action_index);
            ASSERT_TRUE(decoded.has_value()) << "fen=" << fen << " action=" << action_index;
            EXPECT_EQ(decoded.value(), move) << "fen=" << fen << " action=" << action_index;

            if (position.side_to_move == kBlack) {
                ++black_perspective_moves;
            }
            if (move.is_en_passant) {
                ++en_passant_moves;
            }
            if (move.is_castling) {
                ++castling_moves;
            }
            if (move.promotion == kKnight || move.promotion == kBishop || move.promotion == kRook) {
                ++underpromotion_moves;
            } else if (is_knight_delta(move)) {
                ++knight_family_moves;
            } else {
                ++queen_family_moves;
            }
        }
    }

    EXPECT_GT(queen_family_moves, 0);
    EXPECT_GT(knight_family_moves, 0);
    EXPECT_GT(underpromotion_moves, 0);
    EXPECT_GT(castling_moves, 0);
    EXPECT_GT(en_passant_moves, 0);
    EXPECT_GT(black_perspective_moves, 0);
}

// WHY: Black-to-move mirroring must align with white perspective so policy planes are canonical.
TEST(ChessMovegenTest, BlackPerspectiveUsesCanonicalMirroring) {
    const Move black_underpromotion = {
        .from = square(1, 1),
        .to = square(1, 0),
        .piece = kPawn,
        .promotion = kKnight,
    };
    const Move mirrored_white_underpromotion = {
        .from = square(6, 6),
        .to = square(6, 7),
        .piece = kPawn,
        .promotion = kKnight,
    };

    const int black_index = alphazero::chess::semantic_move_to_action_index(black_underpromotion, kBlack);
    const int white_index = alphazero::chess::semantic_move_to_action_index(mirrored_white_underpromotion, kWhite);

    ASSERT_GE(black_index, 0);
    ASSERT_GE(white_index, 0);
    EXPECT_EQ(black_index, white_index);
}

// WHY: Known perft references catch subtle legality bugs across castling, checks, en passant, and promotions.
TEST(ChessMovegenTest, PerftMatchesReferencePositions) {
    struct PerftCase {
        std::string name;
        std::string fen;
        std::vector<std::uint64_t> expected_by_depth;
        int default_max_depth = 0;
    };

    const bool run_exhaustive_kiwipete =
        std::getenv("ALPHAZERO_EXHAUSTIVE_PERFT") != nullptr;

    const std::vector<PerftCase> test_cases = {
        {
            .name = "initial",
            .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            .expected_by_depth = {20ULL, 400ULL, 8902ULL, 197281ULL, 4865609ULL, 119060324ULL},
            .default_max_depth = 6,
        },
        {
            .name = "kiwipete",
            .fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            .expected_by_depth = {48ULL, 2039ULL, 97862ULL, 4085603ULL, 193690690ULL, 8031647685ULL},
            .default_max_depth = run_exhaustive_kiwipete ? 6 : 5,
        },
        {
            .name = "endgame_rook_pawn",
            .fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            .expected_by_depth = {14ULL, 191ULL, 2812ULL, 43238ULL, 674624ULL, 11030083ULL},
            .default_max_depth = 6,
        },
        {
            .name = "endgame_rook_pawn_mirror",
            .fen = "8/1p1p4/8/K1P3r1/R5pk/4P3/5P2/8 b - - 0 1",
            .expected_by_depth = {14ULL, 191ULL, 2812ULL, 43238ULL, 674624ULL, 11030083ULL},
            .default_max_depth = 6,
        },
    };

    for (const PerftCase& test_case : test_cases) {
        const ChessPosition position = position_from_fen(test_case.fen);
        ASSERT_GT(test_case.default_max_depth, 0) << "case=" << test_case.name;
        ASSERT_LE(test_case.default_max_depth, static_cast<int>(test_case.expected_by_depth.size()))
            << "case=" << test_case.name;

        for (int depth = 1; depth <= test_case.default_max_depth; ++depth) {
            EXPECT_EQ(perft(position, depth), test_case.expected_by_depth[depth - 1])
                << "case=" << test_case.name << " depth=" << depth;
        }
    }
}

// WHY: En passant and castling have extra king-safety constraints that pseudo-legal generation alone misses.
TEST(ChessMovegenTest, KingSafetyFilteringRejectsIllegalSpecialMoves) {
    const ChessPosition castling_through_check = position_from_fen("5r2/8/8/8/8/8/8/4K2R w K - 0 1");
    const std::vector<Move> castling_moves = alphazero::chess::generate_legal_moves(castling_through_check);
    EXPECT_FALSE(move_present(castling_moves, Move{.from = square(4, 0), .to = square(6, 0), .piece = alphazero::chess::kKing, .is_castling = true}));

    const ChessPosition ep_pin = position_from_fen("4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    const Move ep_move = {.from = square(4, 4), .to = square(3, 5), .piece = kPawn, .is_en_passant = true};
    const std::vector<Move> pseudo_moves = alphazero::chess::generate_pseudo_legal_moves(ep_pin);
    const std::vector<Move> legal_moves = alphazero::chess::generate_legal_moves(ep_pin);

    EXPECT_TRUE(move_present(pseudo_moves, ep_move));
    EXPECT_FALSE(move_present(legal_moves, ep_move));
}

// WHY: Decoding must reject illegal action IDs even when they syntactically map to on-board coordinates.
TEST(ChessMovegenTest, ActionDecodingRejectsIllegalMovesInPosition) {
    const ChessPosition start = position_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // a1 -> a2 (queen-plane north 1) is blocked by own pawn in the initial position.
    const int blocked_rook_push = (square(0, 0) * kActionPlanesPerSquare) + 0;
    ASSERT_LT(blocked_rook_push, kActionSpaceSize);
    EXPECT_FALSE(alphazero::chess::action_index_to_semantic_move(start, blocked_rook_push).has_value());

    // Out of range action IDs are invalid by definition.
    EXPECT_FALSE(alphazero::chess::action_index_to_semantic_move(start, kActionSpaceSize).has_value());
}
