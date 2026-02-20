#include "games/chess/chess_config.h"
#include "games/chess/chess_state.h"
#include "games/chess/movegen.h"

#include <cctype>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::chess::ChessPosition;
using alphazero::chess::ChessState;
using alphazero::chess::kActionSpaceSize;
using alphazero::chess::kBlack;
using alphazero::chess::kBishop;
using alphazero::chess::kKing;
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
            return kKing;
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

    position.en_passant_square = en_passant == "-" ? -1 : algebraic_to_square(en_passant);
    position.halfmove_clock = halfmove.empty() ? 0 : std::stoi(halfmove);
    position.fullmove_number = fullmove.empty() ? 1 : std::stoi(fullmove);
    position.repetition_count = 1;
    return position;
}

[[nodiscard]] std::unique_ptr<ChessState> apply_move_by_squares(const ChessState& state, int from, int to) {
    for (int action : state.legal_actions()) {
        const std::optional<alphazero::chess::Move> move =
            alphazero::chess::action_index_to_semantic_move(state.position(), action);
        if (move.has_value() && move->from == from && move->to == to) {
            std::unique_ptr<alphazero::GameState> next_state = state.apply_action(action);
            auto typed = std::unique_ptr<ChessState>(dynamic_cast<ChessState*>(next_state.release()));
            if (!typed) {
                throw std::runtime_error("ChessState::apply_action returned non-chess state");
            }
            return typed;
        }
    }

    throw std::runtime_error("Requested move is not legal in this position");
}

}  // namespace

// WHY: The runtime config is the single source of truth for dimensions and must exactly match the spec before
// downstream MCTS/network work relies on it.
TEST(ChessStateTest, ChessGameConfigMatchesSpecAndCreatesInitialState) {
    const alphazero::chess::ChessGameConfig config = alphazero::chess::chess_game_config();
    EXPECT_EQ(config.name, "chess");
    EXPECT_EQ(config.board_rows, 8);
    EXPECT_EQ(config.board_cols, 8);
    EXPECT_EQ(config.planes_per_step, 14);
    EXPECT_EQ(config.num_history_steps, 8);
    EXPECT_EQ(config.constant_planes, 7);
    EXPECT_EQ(config.total_input_channels, 119);
    EXPECT_EQ(config.action_space_size, 4672);
    EXPECT_FLOAT_EQ(config.dirichlet_alpha, 0.3F);
    EXPECT_EQ(config.max_game_length, 512);
    EXPECT_EQ(config.value_head_type, alphazero::GameConfig::ValueHeadType::WDL);
    EXPECT_FALSE(config.supports_symmetry);
    EXPECT_EQ(config.num_symmetries, 1);

    std::unique_ptr<alphazero::GameState> game = config.new_game();
    auto* chess_state = dynamic_cast<ChessState*>(game.get());
    ASSERT_NE(chess_state, nullptr);
    EXPECT_EQ(chess_state->current_player(), kWhite);
    EXPECT_EQ(chess_state->history_size(), 1);
    EXPECT_EQ(chess_state->legal_actions().size(), 20U);
}

// WHY: Action application drives self-play; this checks immutable transition behavior and clone/hash stability.
TEST(ChessStateTest, ApplyActionTransitionsStateAndPreservesCloneIdentity) {
    const ChessState state{};

    int e2e4_action = -1;
    for (int action : state.legal_actions()) {
        const std::optional<alphazero::chess::Move> move =
            alphazero::chess::action_index_to_semantic_move(state.position(), action);
        if (move.has_value() && move->from == square(4, 1) && move->to == square(4, 3)) {
            e2e4_action = action;
            break;
        }
    }

    ASSERT_GE(e2e4_action, 0);
    std::unique_ptr<alphazero::GameState> next_base = state.apply_action(e2e4_action);
    auto next_state = std::unique_ptr<ChessState>(dynamic_cast<ChessState*>(next_base.release()));
    ASSERT_NE(next_state, nullptr);

    EXPECT_EQ(next_state->current_player(), kBlack);
    EXPECT_EQ(next_state->history_size(), 2);
    EXPECT_NE(next_state->hash(), state.hash());
    EXPECT_EQ(next_state->history_position(1).side_to_move, kWhite);

    std::unique_ptr<alphazero::GameState> cloned_base = next_state->clone();
    auto cloned_state = std::unique_ptr<ChessState>(dynamic_cast<ChessState*>(cloned_base.release()));
    ASSERT_NE(cloned_state, nullptr);
    EXPECT_EQ(cloned_state->hash(), next_state->hash());
    EXPECT_EQ(cloned_state->to_string(), next_state->to_string());

    EXPECT_THROW(state.apply_action(kActionSpaceSize), std::invalid_argument);
}

// WHY: Checkmate and stalemate outcomes are fundamental game-end semantics and must map to correct +/-1/0 labels.
TEST(ChessStateTest, TerminalOutcomesDistinguishCheckmateAndStalemate) {
    const ChessState checkmate_state(position_from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));
    EXPECT_TRUE(checkmate_state.is_terminal());
    EXPECT_FLOAT_EQ(checkmate_state.outcome(kBlack), -1.0F);
    EXPECT_FLOAT_EQ(checkmate_state.outcome(kWhite), 1.0F);

    const ChessState stalemate_state(position_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"));
    EXPECT_TRUE(stalemate_state.is_terminal());
    EXPECT_FLOAT_EQ(stalemate_state.outcome(kWhite), 0.0F);
    EXPECT_FLOAT_EQ(stalemate_state.outcome(kBlack), 0.0F);
}

// WHY: Draw rules are separate terminal paths and protect training targets from silently drifting in late-game states.
TEST(ChessStateTest, DrawRulesCoverFiftyMoveRepetitionInsufficientMaterialAndMaxLength) {
    const ChessState fifty_move_state(position_from_fen("7k/8/8/8/8/8/8/K7 w - - 100 50"));
    EXPECT_TRUE(fifty_move_state.is_terminal());
    EXPECT_FLOAT_EQ(fifty_move_state.outcome(kWhite), 0.0F);

    ChessPosition repeated = position_from_fen("7k/8/8/8/8/8/8/K7 w - - 0 1");
    repeated.repetition_count = 3;
    const ChessState repeated_state(repeated);
    EXPECT_TRUE(repeated_state.is_terminal());
    EXPECT_FLOAT_EQ(repeated_state.outcome(kBlack), 0.0F);

    const ChessState insufficient_material_state(position_from_fen("7k/8/8/8/8/8/8/KB6 w - - 0 1"));
    EXPECT_TRUE(insufficient_material_state.is_terminal());
    EXPECT_FLOAT_EQ(insufficient_material_state.outcome(kWhite), 0.0F);

    const ChessState max_length_state(position_from_fen("7k/8/8/8/8/8/8/K7 w - - 0 257"));
    EXPECT_TRUE(max_length_state.is_terminal());
    EXPECT_FLOAT_EQ(max_length_state.outcome(kWhite), 0.0F);
}

// WHY: The spec requires T=8 internal history; this verifies ring-buffer behavior under repeated state transitions.
TEST(ChessStateTest, HistoryBufferRetainsMostRecentEightPositions) {
    std::unique_ptr<ChessState> state = std::make_unique<ChessState>();

    for (int i = 0; i < 4; ++i) {
        state = apply_move_by_squares(*state, square(6, 0), square(5, 2));  // Ng1-f3
        state = apply_move_by_squares(*state, square(6, 7), square(5, 5));  // Ng8-f6
        state = apply_move_by_squares(*state, square(5, 2), square(6, 0));  // Nf3-g1
        state = apply_move_by_squares(*state, square(5, 5), square(6, 7));  // Nf6-g8
    }

    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->history_size(), ChessState::kHistorySteps);
    EXPECT_EQ(state->history_position(0).side_to_move, state->current_player());
    EXPECT_EQ(state->history_position(ChessState::kHistorySteps - 1).side_to_move, kBlack);
    EXPECT_THROW(
        {
            const ChessPosition& ignored = state->history_position(ChessState::kHistorySteps);
            (void)ignored;
        },
        std::out_of_range);
}
