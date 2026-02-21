#include "games/chess/chess_state.h"
#include "games/chess/movegen.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::chess::ChessState;
using alphazero::chess::Move;
using alphazero::chess::action_index_to_semantic_move;
using alphazero::chess::kActionSpaceSize;

[[nodiscard]] constexpr int square(int file, int rank) { return rank * 8 + file; }

[[nodiscard]] int find_action(const ChessState& state, int from, int to, int promotion = -1) {
    for (int action : state.legal_actions()) {
        const std::optional<Move> move = action_index_to_semantic_move(state.position(), action);
        if (!move.has_value()) {
            continue;
        }
        if (move->from == from && move->to == to && (promotion < 0 || move->promotion == promotion)) {
            return action;
        }
    }
    throw std::runtime_error("Requested move is not legal in the current state");
}

[[nodiscard]] std::unique_ptr<ChessState> apply_action(const ChessState& state, int action) {
    std::unique_ptr<alphazero::GameState> next_base = state.apply_action(action);
    auto typed = std::unique_ptr<ChessState>(dynamic_cast<ChessState*>(next_base.release()));
    if (!typed) {
        throw std::runtime_error("Expected ChessState when applying a chess action");
    }
    return typed;
}

}  // namespace

// WHY: FEN is the debugging and fixture interchange format; round-trip identity prevents silent corruption.
TEST(ChessSerializationTest, FenRoundTripPreservesCanonicalStateText) {
    const std::vector<std::string> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/3k4/8/3Pp3/8/4K3/8 b - d3 12 57",
        "4k3/8/8/8/8/8/8/4K3 b - - 83 119",
    };

    for (const std::string& fen : fens) {
        const ChessState state = ChessState::from_fen(fen);
        EXPECT_EQ(state.to_fen(), fen);
    }

    const ChessState initial_state{};
    EXPECT_EQ(initial_state.to_fen(), fens.front());
}

// WHY: Import should fail loudly on malformed FEN so tests/logging cannot proceed with invalid board states.
TEST(ChessSerializationTest, FenParsingRejectsMalformedInputs) {
    const std::vector<std::string> invalid_fens = {
        "8/8/8/8/8/8/8/8 w - - 0",
        "8/8/8/8/8/8/8/7X w - - 0 1",
        "9/8/8/8/8/8/8/K6k w - - 0 1",
        "8/8/8/8/8/8/8/K6k x - - 0 1",
        "8/8/8/8/8/8/8/K6k w KK - 0 1",
        "8/8/8/8/8/8/8/K6k w - e4 0 1",
    };

    for (const std::string& fen : invalid_fens) {
        EXPECT_THROW(
            {
                const ChessState ignored = ChessState::from_fen(fen);
                (void)ignored;
            },
            std::invalid_argument)
            << "fen=" << fen;
    }
}

// WHY: PGN logs are the portable replay artifact; SAN/move-number correctness is required for external tooling.
TEST(ChessSerializationTest, PgnExportEmitsValidTagsAndSanMovetext) {
    std::unique_ptr<ChessState> state = std::make_unique<ChessState>();
    std::vector<int> action_history;

    const int f2f3 = find_action(*state, square(5, 1), square(5, 2));
    action_history.push_back(f2f3);
    state = apply_action(*state, f2f3);

    const int e7e5 = find_action(*state, square(4, 6), square(4, 4));
    action_history.push_back(e7e5);
    state = apply_action(*state, e7e5);

    const int g2g4 = find_action(*state, square(6, 1), square(6, 3));
    action_history.push_back(g2g4);
    state = apply_action(*state, g2g4);

    const int qd8h4 = find_action(*state, square(3, 7), square(7, 3));
    action_history.push_back(qd8h4);

    const std::string pgn = ChessState::actions_to_pgn(action_history, "0-1");
    EXPECT_NE(pgn.find("[Event \"AlphaZero Self-Play\"]"), std::string::npos);
    EXPECT_NE(pgn.find("[Result \"0-1\"]"), std::string::npos);
    EXPECT_NE(pgn.find("1. f3 e5 2. g4 Qh4# 0-1"), std::string::npos);
}

// WHY: Non-initial positions must produce SetUp/FEN tags and proper black-to-move numbering (`n...`) for replay.
TEST(ChessSerializationTest, PgnExportSupportsCustomStartingFenAndBlackMoveNumbering) {
    const std::string start_fen = "4k3/8/8/8/8/8/8/4K3 b - - 0 1";
    const ChessState state = ChessState::from_fen(start_fen);
    const int ke8e7 = find_action(state, square(4, 7), square(4, 6));

    const std::string pgn = ChessState::actions_to_pgn({ke8e7}, "*", start_fen);
    EXPECT_NE(pgn.find("[SetUp \"1\"]"), std::string::npos);
    EXPECT_NE(pgn.find("[FEN \"" + start_fen + "\"]"), std::string::npos);
    EXPECT_NE(pgn.find("1... Ke7 *"), std::string::npos);
}

// WHY: SAN disambiguation is required for PGN parser compatibility when two same-type pieces share a destination.
TEST(ChessSerializationTest, PgnExportDisambiguatesAmbiguousPieceMoves) {
    const std::string start_fen = "7k/8/8/8/8/8/8/KN3N2 w - - 0 1";
    const ChessState state = ChessState::from_fen(start_fen);
    const int nb1d2 = find_action(state, square(1, 0), square(3, 1));

    const std::string pgn = ChessState::actions_to_pgn({nb1d2}, "*", start_fen);
    EXPECT_NE(pgn.find("1. Nbd2 *"), std::string::npos);
}

// WHY: Export-time validation protects pipeline logs from malformed outcomes or illegal action traces.
TEST(ChessSerializationTest, PgnExportRejectsInvalidResultOrIllegalActions) {
    EXPECT_THROW(
        {
            const std::string ignored = ChessState::actions_to_pgn({}, "win");
            (void)ignored;
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            const std::string ignored = ChessState::actions_to_pgn({kActionSpaceSize}, "*");
            (void)ignored;
        },
        std::invalid_argument);
}
