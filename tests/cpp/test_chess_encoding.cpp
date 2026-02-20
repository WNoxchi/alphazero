#include "games/chess/chess_state.h"
#include "games/chess/movegen.h"

#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::chess::ChessPosition;
using alphazero::chess::ChessState;
using alphazero::chess::kBlack;
using alphazero::chess::kBlackKingSide;
using alphazero::chess::kBlackQueenSide;
using alphazero::chess::kBoardSquares;
using alphazero::chess::kWhite;
using alphazero::chess::kWhiteKingSide;
using alphazero::chess::kWhiteQueenSide;
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
            return alphazero::chess::kPawn;
        case 'n':
            return alphazero::chess::kKnight;
        case 'b':
            return alphazero::chess::kBishop;
        case 'r':
            return alphazero::chess::kRook;
        case 'q':
            return alphazero::chess::kQueen;
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

    position.side_to_move = side_to_move == "b" ? kBlack : kWhite;
    position.castling = 0;
    if (castling.find('K') != std::string::npos) {
        position.castling = static_cast<std::uint8_t>(position.castling | kWhiteKingSide);
    }
    if (castling.find('Q') != std::string::npos) {
        position.castling = static_cast<std::uint8_t>(position.castling | kWhiteQueenSide);
    }
    if (castling.find('k') != std::string::npos) {
        position.castling = static_cast<std::uint8_t>(position.castling | kBlackKingSide);
    }
    if (castling.find('q') != std::string::npos) {
        position.castling = static_cast<std::uint8_t>(position.castling | kBlackQueenSide);
    }

    position.en_passant_square = en_passant == "-" ? -1 : algebraic_to_square(en_passant);
    position.halfmove_clock = halfmove.empty() ? 0 : std::stoi(halfmove);
    position.fullmove_number = fullmove.empty() ? 1 : std::stoi(fullmove);
    position.repetition_count = 1;
    return position;
}

[[nodiscard]] int orient_square_for_side(int board_square, int side_to_move) {
    return side_to_move == kBlack ? (kBoardSquares - 1) - board_square : board_square;
}

[[nodiscard]] float encoded_value(const std::vector<float>& encoded, int channel, int board_square) {
    return encoded[(channel * kBoardSquares) + board_square];
}

void expect_plane_constant(const std::vector<float>& encoded, int channel, float expected) {
    for (int board_square = 0; board_square < kBoardSquares; ++board_square) {
        EXPECT_FLOAT_EQ(encoded_value(encoded, channel, board_square), expected) << "channel=" << channel;
    }
}

void expect_only_squares_set(
    const std::vector<float>& encoded,
    int channel,
    const std::vector<int>& expected_squares) {
    std::vector<bool> expected(kBoardSquares, false);
    for (int board_square : expected_squares) {
        if (board_square >= 0 && board_square < kBoardSquares) {
            expected[board_square] = true;
        }
    }

    for (int board_square = 0; board_square < kBoardSquares; ++board_square) {
        const float expected_value = expected[board_square] ? 1.0F : 0.0F;
        EXPECT_FLOAT_EQ(encoded_value(encoded, channel, board_square), expected_value)
            << "channel=" << channel << " square=" << board_square;
    }
}

[[nodiscard]] std::vector<float> encode_state(const ChessState& state) {
    std::vector<float> encoded(ChessState::kTotalInputChannels * kBoardSquares, -1.0F);
    state.encode(encoded.data());
    return encoded;
}

[[nodiscard]] std::unique_ptr<ChessState> apply_move_by_squares(const ChessState& state, int from, int to) {
    for (int action : state.legal_actions()) {
        const std::optional<alphazero::chess::Move> move =
            alphazero::chess::action_index_to_semantic_move(state.position(), action);
        if (move.has_value() && move->from == from && move->to == to) {
            std::unique_ptr<alphazero::GameState> next_state = state.apply_action(action);
            auto typed = std::unique_ptr<ChessState>(dynamic_cast<ChessState*>(next_state.release()));
            if (!typed) {
                throw std::runtime_error("Expected ChessState when applying a chess action");
            }
            return typed;
        }
    }

    throw std::runtime_error("Requested move was not legal");
}

}  // namespace

// WHY: This validates the exact 8x8x119 layout contract and initial-position semantics that NN inference depends on.
TEST(ChessEncodingTest, InitialPositionEncodesSpecDefinedPlanesAndZeroFilledHistory) {
    const ChessState state{};
    const std::vector<float> encoded = encode_state(state);

    ASSERT_EQ(encoded.size(), static_cast<std::size_t>(ChessState::kTotalInputChannels * kBoardSquares));

    std::vector<int> white_pawn_squares;
    std::vector<int> black_pawn_squares;
    for (int file = 0; file < 8; ++file) {
        white_pawn_squares.push_back(square(file, 1));
        black_pawn_squares.push_back(square(file, 6));
    }

    expect_only_squares_set(encoded, 0, white_pawn_squares);
    expect_only_squares_set(encoded, 6, black_pawn_squares);
    expect_only_squares_set(encoded, 5, {square(4, 0)});
    expect_only_squares_set(encoded, 11, {square(4, 7)});
    expect_plane_constant(encoded, 12, 1.0F);
    expect_plane_constant(encoded, 13, 0.0F);

    for (int channel = ChessState::kPlanesPerStep; channel < ChessState::kHistorySteps * ChessState::kPlanesPerStep;
         ++channel) {
        expect_plane_constant(encoded, channel, 0.0F);
    }

    const int constant_offset = ChessState::kHistorySteps * ChessState::kPlanesPerStep;
    expect_plane_constant(encoded, constant_offset + 0, 1.0F);
    expect_plane_constant(encoded, constant_offset + 1, 0.0F);
    expect_plane_constant(encoded, constant_offset + 2, 1.0F);
    expect_plane_constant(encoded, constant_offset + 3, 1.0F);
    expect_plane_constant(encoded, constant_offset + 4, 1.0F);
    expect_plane_constant(encoded, constant_offset + 5, 1.0F);
    expect_plane_constant(encoded, constant_offset + 6, 0.0F);
}

// WHY: Black-to-move canonicalization must be stable so policy/action orientation and input features stay aligned.
TEST(ChessEncodingTest, BlackToMoveUsesCanonicalMirroringAndPerspectiveRelativeConstants) {
    const ChessState state(position_from_fen("4k3/8/8/8/8/8/1p6/4K3 b Kq - 25 10"));
    const std::vector<float> encoded = encode_state(state);

    expect_only_squares_set(encoded, 0, {orient_square_for_side(square(1, 1), kBlack)});
    expect_only_squares_set(encoded, 5, {orient_square_for_side(square(4, 7), kBlack)});
    expect_only_squares_set(encoded, 11, {orient_square_for_side(square(4, 0), kBlack)});

    const int constant_offset = ChessState::kHistorySteps * ChessState::kPlanesPerStep;
    expect_plane_constant(encoded, constant_offset + 0, 0.0F);
    expect_plane_constant(encoded, constant_offset + 2, 0.0F);
    expect_plane_constant(encoded, constant_offset + 3, 1.0F);
    expect_plane_constant(encoded, constant_offset + 4, 1.0F);
    expect_plane_constant(encoded, constant_offset + 5, 0.0F);

    EXPECT_NEAR(encoded_value(encoded, constant_offset + 1, 0), 19.0F / 512.0F, 1e-6F);
    EXPECT_NEAR(encoded_value(encoded, constant_offset + 6, 0), 0.25F, 1e-6F);
}

// WHY: Temporal ordering across T=8 history steps must be correct or the network will learn from scrambled context.
TEST(ChessEncodingTest, HistoryPlanesStoreNewestToOldestPositions) {
    std::unique_ptr<ChessState> state = std::make_unique<ChessState>();
    state = apply_move_by_squares(*state, square(4, 1), square(4, 3));  // e2e4
    state = apply_move_by_squares(*state, square(4, 6), square(4, 4));  // e7e5

    const std::vector<float> encoded = encode_state(*state);

    const int latest = 0 * ChessState::kPlanesPerStep;
    const int previous = 1 * ChessState::kPlanesPerStep;
    const int oldest = 2 * ChessState::kPlanesPerStep;
    const int unused = 3 * ChessState::kPlanesPerStep;

    EXPECT_FLOAT_EQ(encoded_value(encoded, latest + 0, square(4, 3)), 1.0F);
    EXPECT_FLOAT_EQ(encoded_value(encoded, latest + 6, square(4, 4)), 1.0F);

    EXPECT_FLOAT_EQ(encoded_value(encoded, previous + 0, square(4, 3)), 1.0F);
    EXPECT_FLOAT_EQ(encoded_value(encoded, previous + 6, square(4, 6)), 1.0F);

    EXPECT_FLOAT_EQ(encoded_value(encoded, oldest + 0, square(4, 1)), 1.0F);
    EXPECT_FLOAT_EQ(encoded_value(encoded, oldest + 6, square(4, 6)), 1.0F);

    expect_plane_constant(encoded, unused + 0, 0.0F);
    expect_plane_constant(encoded, unused + 6, 0.0F);
}

// WHY: The repetition feature planes encode draw-pressure information and must distinguish first and repeated states.
TEST(ChessEncodingTest, RepetitionPlanesDifferentiateSingleAndMultipleOccurrences) {
    std::unique_ptr<ChessState> state = std::make_unique<ChessState>();
    state = apply_move_by_squares(*state, square(6, 0), square(5, 2));  // Ng1-f3
    state = apply_move_by_squares(*state, square(6, 7), square(5, 5));  // Ng8-f6
    state = apply_move_by_squares(*state, square(5, 2), square(6, 0));  // Nf3-g1
    state = apply_move_by_squares(*state, square(5, 5), square(6, 7));  // Nf6-g8

    const std::vector<float> encoded = encode_state(*state);

    const int latest = 0 * ChessState::kPlanesPerStep;
    const int first_occurrence = 4 * ChessState::kPlanesPerStep;
    const int never_reached = 5 * ChessState::kPlanesPerStep;

    expect_plane_constant(encoded, latest + 12, 0.0F);
    expect_plane_constant(encoded, latest + 13, 1.0F);
    expect_plane_constant(encoded, first_occurrence + 12, 1.0F);
    expect_plane_constant(encoded, first_occurrence + 13, 0.0F);
    expect_plane_constant(encoded, never_reached + 12, 0.0F);
    expect_plane_constant(encoded, never_reached + 13, 0.0F);
}
