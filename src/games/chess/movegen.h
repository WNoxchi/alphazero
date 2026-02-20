#pragma once

#include <optional>
#include <vector>

#include "games/chess/bitboard.h"

namespace alphazero::chess {

constexpr int kActionPlanesPerSquare = 73;
constexpr int kActionSpaceSize = kBoardSquares * kActionPlanesPerSquare;

struct Move {
    int from = -1;
    int to = -1;
    int piece = -1;
    int promotion = -1;  // -1 when not a promotion, otherwise PieceType.
    bool is_en_passant = false;
    bool is_castling = false;

    [[nodiscard]] bool operator==(const Move& other) const = default;
};

[[nodiscard]] std::vector<Move> generate_pseudo_legal_moves(const ChessPosition& position);
[[nodiscard]] std::vector<Move> generate_legal_moves(const ChessPosition& position);
[[nodiscard]] std::vector<int> legal_action_indices(const ChessPosition& position);

[[nodiscard]] ChessPosition apply_move(const ChessPosition& position, const Move& move);

[[nodiscard]] bool is_square_attacked(const ChessPosition& position, int square, int attacker_color);
[[nodiscard]] bool is_in_check(const ChessPosition& position, int color);

// Convert a semantic move into AlphaZero's 8x8x73 action encoding.
// Returns -1 for invalid/unrepresentable moves.
[[nodiscard]] int semantic_move_to_action_index(const Move& move, int side_to_move);

// Decode an action index into a legal semantic move in `position`.
// Returns nullopt when the action is illegal for the position.
[[nodiscard]] std::optional<Move> action_index_to_semantic_move(const ChessPosition& position, int action_index);

}  // namespace alphazero::chess
