#include "games/chess/movegen.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace alphazero::chess {
namespace {

constexpr int kBoardSize = 8;

struct Delta {
    int file = 0;
    int rank = 0;
};

constexpr std::array<Delta, 8> kQueenMoveDirections = {
    Delta{0, 1},   // N
    Delta{1, 1},   // NE
    Delta{1, 0},   // E
    Delta{1, -1},  // SE
    Delta{0, -1},  // S
    Delta{-1, -1}, // SW
    Delta{-1, 0},  // W
    Delta{-1, 1},  // NW
};

constexpr std::array<Delta, 8> kKnightMoveOffsets = {
    Delta{-1, 2},
    Delta{1, 2},
    Delta{-2, 1},
    Delta{2, 1},
    Delta{-2, -1},
    Delta{2, -1},
    Delta{-1, -2},
    Delta{1, -2},
};

constexpr std::array<Delta, 3> kUnderpromotionDirections = {
    Delta{0, 1},
    Delta{-1, 1},
    Delta{1, 1},
};

[[nodiscard]] constexpr int square_file(int square) { return square & 7; }
[[nodiscard]] constexpr int square_rank(int square) { return square >> 3; }
[[nodiscard]] constexpr int make_square(int file, int rank) { return rank * kBoardSize + file; }

[[nodiscard]] constexpr int opponent_of(int color) { return color == kWhite ? kBlack : kWhite; }

[[nodiscard]] int piece_on_square(const ChessPosition& position, int color, int square) {
    if ((color != kWhite && color != kBlack) || !is_valid_square(square)) {
        return -1;
    }

    const Bitboard mask = square_bit(square);
    for (int piece = 0; piece < kPieceTypeCount; ++piece) {
        if ((position.pieces[color][piece] & mask) != 0ULL) {
            return piece;
        }
    }
    return -1;
}

[[nodiscard]] int king_square(const ChessPosition& position, int color) {
    if (color != kWhite && color != kBlack) {
        return -1;
    }
    return bit_scan_forward(position.pieces[color][kKing]);
}

void add_non_pawn_moves(const ChessPosition& position, int color, int piece, Bitboard pieces, std::vector<Move>* moves) {
    if (moves == nullptr) {
        return;
    }

    const Bitboard own_occupancy = occupied_by(position, color);
    const Bitboard all_occupancy = occupied(position);

    while (pieces != 0ULL) {
        const int from = pop_lsb(&pieces);
        Bitboard attacks = 0ULL;
        switch (piece) {
            case kKnight:
                attacks = knight_attacks(from);
                break;
            case kBishop:
                attacks = bishop_attacks(from, all_occupancy);
                break;
            case kRook:
                attacks = rook_attacks(from, all_occupancy);
                break;
            case kQueen:
                attacks = queen_attacks(from, all_occupancy);
                break;
            default:
                attacks = 0ULL;
                break;
        }

        attacks &= ~own_occupancy;
        while (attacks != 0ULL) {
            const int to = pop_lsb(&attacks);
            moves->push_back(Move{.from = from, .to = to, .piece = piece});
        }
    }
}

void add_king_moves(const ChessPosition& position, int color, std::vector<Move>* moves) {
    if (moves == nullptr) {
        return;
    }

    const int from = king_square(position, color);
    if (!is_valid_square(from)) {
        return;
    }

    Bitboard attacks = king_attacks(from) & ~occupied_by(position, color);
    while (attacks != 0ULL) {
        const int to = pop_lsb(&attacks);
        moves->push_back(Move{.from = from, .to = to, .piece = kKing});
    }
}

void add_pawn_pushes(const ChessPosition& position, int color, int from, std::vector<Move>* moves) {
    if (moves == nullptr || !is_valid_square(from)) {
        return;
    }

    const Bitboard all_occupancy = occupied(position);
    const int forward = color == kWhite ? 8 : -8;
    const int start_rank = color == kWhite ? 1 : 6;
    const int promotion_rank = color == kWhite ? 6 : 1;
    const int to = from + forward;

    if (!is_valid_square(to) || (all_occupancy & square_bit(to)) != 0ULL) {
        return;
    }

    if (square_rank(from) == promotion_rank) {
        moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .promotion = kQueen});
        moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .promotion = kKnight});
        moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .promotion = kBishop});
        moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .promotion = kRook});
        return;
    }

    moves->push_back(Move{.from = from, .to = to, .piece = kPawn});

    if (square_rank(from) != start_rank) {
        return;
    }

    const int double_push_to = from + (forward * 2);
    if (is_valid_square(double_push_to) && (all_occupancy & square_bit(double_push_to)) == 0ULL) {
        moves->push_back(Move{.from = from, .to = double_push_to, .piece = kPawn});
    }
}

void add_pawn_captures(const ChessPosition& position, int color, int from, std::vector<Move>* moves) {
    if (moves == nullptr || !is_valid_square(from)) {
        return;
    }

    const Bitboard opponent_occupancy = occupied_by(position, opponent_of(color));
    const int rank = square_rank(from);
    const int file = square_file(from);
    const int promotion_rank = color == kWhite ? 6 : 1;

    constexpr std::array<int, 2> kWhiteCaptureDelta = {7, 9};
    constexpr std::array<int, 2> kBlackCaptureDelta = {-9, -7};
    const auto& deltas = color == kWhite ? kWhiteCaptureDelta : kBlackCaptureDelta;

    for (int delta : deltas) {
        const int to = from + delta;
        if (!is_valid_square(to)) {
            continue;
        }

        const int file_delta = square_file(to) - file;
        if (std::abs(file_delta) != 1) {
            continue;
        }

        const Bitboard to_mask = square_bit(to);
        const bool is_en_passant_capture = (position.en_passant_square == to);
        const bool is_standard_capture = (opponent_occupancy & to_mask) != 0ULL;

        if (!is_en_passant_capture && !is_standard_capture) {
            continue;
        }

        if (is_en_passant_capture) {
            moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .is_en_passant = true});
            continue;
        }

        if (rank == promotion_rank) {
            moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .promotion = kQueen});
            moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .promotion = kKnight});
            moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .promotion = kBishop});
            moves->push_back(Move{.from = from, .to = to, .piece = kPawn, .promotion = kRook});
            continue;
        }

        moves->push_back(Move{.from = from, .to = to, .piece = kPawn});
    }
}

void add_pawn_moves(const ChessPosition& position, int color, std::vector<Move>* moves) {
    if (moves == nullptr) {
        return;
    }

    Bitboard pawns = position.pieces[color][kPawn];
    while (pawns != 0ULL) {
        const int from = pop_lsb(&pawns);
        add_pawn_pushes(position, color, from, moves);
        add_pawn_captures(position, color, from, moves);
    }
}

void add_castling_moves(const ChessPosition& position, int color, std::vector<Move>* moves) {
    if (moves == nullptr) {
        return;
    }

    const int opponent = opponent_of(color);
    const Bitboard all_occupancy = occupied(position);

    if (color == kWhite) {
        constexpr int kKingStart = 4;
        constexpr int kKingSideRookStart = 7;
        constexpr int kQueenSideRookStart = 0;

        if ((position.castling & kWhiteKingSide) != 0U &&
            (position.pieces[kWhite][kKing] & square_bit(kKingStart)) != 0ULL &&
            (position.pieces[kWhite][kRook] & square_bit(kKingSideRookStart)) != 0ULL &&
            (all_occupancy & (square_bit(5) | square_bit(6))) == 0ULL &&
            !is_square_attacked(position, 4, opponent) && !is_square_attacked(position, 5, opponent) &&
            !is_square_attacked(position, 6, opponent)) {
            moves->push_back(Move{.from = 4, .to = 6, .piece = kKing, .is_castling = true});
        }

        if ((position.castling & kWhiteQueenSide) != 0U &&
            (position.pieces[kWhite][kKing] & square_bit(kKingStart)) != 0ULL &&
            (position.pieces[kWhite][kRook] & square_bit(kQueenSideRookStart)) != 0ULL &&
            (all_occupancy & (square_bit(1) | square_bit(2) | square_bit(3))) == 0ULL &&
            !is_square_attacked(position, 4, opponent) && !is_square_attacked(position, 3, opponent) &&
            !is_square_attacked(position, 2, opponent)) {
            moves->push_back(Move{.from = 4, .to = 2, .piece = kKing, .is_castling = true});
        }
        return;
    }

    constexpr int kKingStart = 60;
    constexpr int kKingSideRookStart = 63;
    constexpr int kQueenSideRookStart = 56;

    if ((position.castling & kBlackKingSide) != 0U &&
        (position.pieces[kBlack][kKing] & square_bit(kKingStart)) != 0ULL &&
        (position.pieces[kBlack][kRook] & square_bit(kKingSideRookStart)) != 0ULL &&
        (all_occupancy & (square_bit(61) | square_bit(62))) == 0ULL &&
        !is_square_attacked(position, 60, opponent) && !is_square_attacked(position, 61, opponent) &&
        !is_square_attacked(position, 62, opponent)) {
        moves->push_back(Move{.from = 60, .to = 62, .piece = kKing, .is_castling = true});
    }

    if ((position.castling & kBlackQueenSide) != 0U &&
        (position.pieces[kBlack][kKing] & square_bit(kKingStart)) != 0ULL &&
        (position.pieces[kBlack][kRook] & square_bit(kQueenSideRookStart)) != 0ULL &&
        (all_occupancy & (square_bit(57) | square_bit(58) | square_bit(59))) == 0ULL &&
        !is_square_attacked(position, 60, opponent) && !is_square_attacked(position, 59, opponent) &&
        !is_square_attacked(position, 58, opponent)) {
        moves->push_back(Move{.from = 60, .to = 58, .piece = kKing, .is_castling = true});
    }
}

[[nodiscard]] int orient_square_for_player(int square, int side_to_move) {
    if (!is_valid_square(square)) {
        return -1;
    }
    if (side_to_move == kBlack) {
        return (kBoardSquares - 1) - square;
    }
    return square;
}

[[nodiscard]] int unorient_square_for_player(int oriented_square, int side_to_move) {
    return orient_square_for_player(oriented_square, side_to_move);
}

[[nodiscard]] bool delta_is_knight_move(int file_delta, int rank_delta, int* offset_index) {
    for (int i = 0; i < static_cast<int>(kKnightMoveOffsets.size()); ++i) {
        if (kKnightMoveOffsets[i].file == file_delta && kKnightMoveOffsets[i].rank == rank_delta) {
            if (offset_index != nullptr) {
                *offset_index = i;
            }
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool delta_is_queen_move(int file_delta, int rank_delta, int* direction_index, int* distance) {
    for (int dir = 0; dir < static_cast<int>(kQueenMoveDirections.size()); ++dir) {
        const int direction_file = kQueenMoveDirections[dir].file;
        const int direction_rank = kQueenMoveDirections[dir].rank;

        for (int step = 1; step <= 7; ++step) {
            if (direction_file * step == file_delta && direction_rank * step == rank_delta) {
                if (direction_index != nullptr) {
                    *direction_index = dir;
                }
                if (distance != nullptr) {
                    *distance = step;
                }
                return true;
            }
        }
    }
    return false;
}

[[nodiscard]] int underpromotion_piece_to_index(int promotion_piece) {
    switch (promotion_piece) {
        case kKnight:
            return 0;
        case kBishop:
            return 1;
        case kRook:
            return 2;
        default:
            return -1;
    }
}

[[nodiscard]] int underpromotion_direction_to_index(int file_delta, int rank_delta) {
    for (int i = 0; i < static_cast<int>(kUnderpromotionDirections.size()); ++i) {
        if (kUnderpromotionDirections[i].file == file_delta && kUnderpromotionDirections[i].rank == rank_delta) {
            return i;
        }
    }
    return -1;
}

[[nodiscard]] bool move_equivalent(const Move& left, const Move& right) {
    return left.from == right.from && left.to == right.to && left.piece == right.piece &&
           left.promotion == right.promotion && left.is_en_passant == right.is_en_passant &&
           left.is_castling == right.is_castling;
}

}  // namespace

std::vector<Move> generate_pseudo_legal_moves(const ChessPosition& position) {
    const int side_to_move = position.side_to_move;
    if (side_to_move != kWhite && side_to_move != kBlack) {
        return {};
    }

    std::vector<Move> moves;
    moves.reserve(256);

    add_pawn_moves(position, side_to_move, &moves);
    add_non_pawn_moves(position, side_to_move, kKnight, position.pieces[side_to_move][kKnight], &moves);
    add_non_pawn_moves(position, side_to_move, kBishop, position.pieces[side_to_move][kBishop], &moves);
    add_non_pawn_moves(position, side_to_move, kRook, position.pieces[side_to_move][kRook], &moves);
    add_non_pawn_moves(position, side_to_move, kQueen, position.pieces[side_to_move][kQueen], &moves);
    add_king_moves(position, side_to_move, &moves);
    add_castling_moves(position, side_to_move, &moves);

    return moves;
}

std::vector<Move> generate_legal_moves(const ChessPosition& position) {
    const int side_to_move = position.side_to_move;
    if (side_to_move != kWhite && side_to_move != kBlack) {
        return {};
    }

    const std::vector<Move> pseudo_moves = generate_pseudo_legal_moves(position);
    std::vector<Move> legal_moves;
    legal_moves.reserve(pseudo_moves.size());

    for (const Move& move : pseudo_moves) {
        const ChessPosition next_position = apply_move(position, move);
        if (!is_in_check(next_position, side_to_move)) {
            legal_moves.push_back(move);
        }
    }

    return legal_moves;
}

std::vector<int> legal_action_indices(const ChessPosition& position) {
    const std::vector<Move> legal_moves = generate_legal_moves(position);
    std::vector<int> actions;
    actions.reserve(legal_moves.size());

    for (const Move& move : legal_moves) {
        const int action_index = semantic_move_to_action_index(move, position.side_to_move);
        if (action_index >= 0) {
            actions.push_back(action_index);
        }
    }
    return actions;
}

ChessPosition apply_move(const ChessPosition& position, const Move& move) {
    ChessPosition next = position;

    const int side_to_move = position.side_to_move;
    if ((side_to_move != kWhite && side_to_move != kBlack) || !is_valid_square(move.from) || !is_valid_square(move.to)) {
        return next;
    }

    int moving_piece = move.piece;
    if (moving_piece < 0 || moving_piece >= kPieceTypeCount ||
        (position.pieces[side_to_move][moving_piece] & square_bit(move.from)) == 0ULL) {
        moving_piece = piece_on_square(position, side_to_move, move.from);
        if (moving_piece < 0 || moving_piece >= kPieceTypeCount) {
            return next;
        }
    }

    const int opponent = opponent_of(side_to_move);

    const Bitboard from_mask = square_bit(move.from);
    const Bitboard to_mask = square_bit(move.to);

    next.pieces[side_to_move][moving_piece] &= ~from_mask;

    bool is_capture = false;
    if (move.is_en_passant) {
        const int captured_square = move.to + (side_to_move == kWhite ? -8 : 8);
        if (is_valid_square(captured_square)) {
            const Bitboard captured_mask = square_bit(captured_square);
            if ((next.pieces[opponent][kPawn] & captured_mask) != 0ULL) {
                next.pieces[opponent][kPawn] &= ~captured_mask;
                is_capture = true;
            }
        }
    } else {
        for (int piece = 0; piece < kPieceTypeCount; ++piece) {
            if ((next.pieces[opponent][piece] & to_mask) != 0ULL) {
                next.pieces[opponent][piece] &= ~to_mask;
                is_capture = true;
                break;
            }
        }
    }

    if (moving_piece == kKing && move.is_castling) {
        if (side_to_move == kWhite) {
            if (move.to == 6) {
                next.pieces[kWhite][kRook] &= ~square_bit(7);
                next.pieces[kWhite][kRook] |= square_bit(5);
            } else if (move.to == 2) {
                next.pieces[kWhite][kRook] &= ~square_bit(0);
                next.pieces[kWhite][kRook] |= square_bit(3);
            }
        } else {
            if (move.to == 62) {
                next.pieces[kBlack][kRook] &= ~square_bit(63);
                next.pieces[kBlack][kRook] |= square_bit(61);
            } else if (move.to == 58) {
                next.pieces[kBlack][kRook] &= ~square_bit(56);
                next.pieces[kBlack][kRook] |= square_bit(59);
            }
        }
    }

    int placed_piece = moving_piece;
    if (moving_piece == kPawn && move.promotion >= 0 && move.promotion < kPieceTypeCount) {
        placed_piece = move.promotion;
    }
    next.pieces[side_to_move][placed_piece] |= to_mask;

    if (moving_piece == kKing) {
        if (side_to_move == kWhite) {
            next.castling = static_cast<std::uint8_t>(next.castling & ~(kWhiteKingSide | kWhiteQueenSide));
        } else {
            next.castling = static_cast<std::uint8_t>(next.castling & ~(kBlackKingSide | kBlackQueenSide));
        }
    }

    if (moving_piece == kRook) {
        if (side_to_move == kWhite) {
            if (move.from == 0) {
                next.castling = static_cast<std::uint8_t>(next.castling & ~kWhiteQueenSide);
            }
            if (move.from == 7) {
                next.castling = static_cast<std::uint8_t>(next.castling & ~kWhiteKingSide);
            }
        } else {
            if (move.from == 56) {
                next.castling = static_cast<std::uint8_t>(next.castling & ~kBlackQueenSide);
            }
            if (move.from == 63) {
                next.castling = static_cast<std::uint8_t>(next.castling & ~kBlackKingSide);
            }
        }
    }

    if (!move.is_en_passant) {
        if (opponent == kWhite) {
            if (move.to == 0) {
                next.castling = static_cast<std::uint8_t>(next.castling & ~kWhiteQueenSide);
            }
            if (move.to == 7) {
                next.castling = static_cast<std::uint8_t>(next.castling & ~kWhiteKingSide);
            }
        } else {
            if (move.to == 56) {
                next.castling = static_cast<std::uint8_t>(next.castling & ~kBlackQueenSide);
            }
            if (move.to == 63) {
                next.castling = static_cast<std::uint8_t>(next.castling & ~kBlackKingSide);
            }
        }
    }

    next.en_passant_square = -1;
    if (moving_piece == kPawn && std::abs(move.to - move.from) == 16) {
        next.en_passant_square = move.from + (side_to_move == kWhite ? 8 : -8);
    }

    next.halfmove_clock = (moving_piece == kPawn || is_capture) ? 0 : (position.halfmove_clock + 1);
    next.fullmove_number = position.fullmove_number + (side_to_move == kBlack ? 1 : 0);

    next.side_to_move = opponent;
    next.repetition_count = 1;

    return next;
}

bool is_square_attacked(const ChessPosition& position, int square, int attacker_color) {
    if (!is_valid_square(square) || (attacker_color != kWhite && attacker_color != kBlack)) {
        return false;
    }

    const Bitboard target_mask = square_bit(square);
    const Bitboard all_occupancy = occupied(position);

    if ((pawn_attacks(attacker_color, position.pieces[attacker_color][kPawn]) & target_mask) != 0ULL) {
        return true;
    }

    if ((knight_attacks(square) & position.pieces[attacker_color][kKnight]) != 0ULL) {
        return true;
    }

    if ((king_attacks(square) & position.pieces[attacker_color][kKing]) != 0ULL) {
        return true;
    }

    const Bitboard bishop_like = position.pieces[attacker_color][kBishop] | position.pieces[attacker_color][kQueen];
    if ((bishop_attacks(square, all_occupancy) & bishop_like) != 0ULL) {
        return true;
    }

    const Bitboard rook_like = position.pieces[attacker_color][kRook] | position.pieces[attacker_color][kQueen];
    if ((rook_attacks(square, all_occupancy) & rook_like) != 0ULL) {
        return true;
    }

    return false;
}

bool is_in_check(const ChessPosition& position, int color) {
    if (color != kWhite && color != kBlack) {
        return false;
    }

    const int square = king_square(position, color);
    if (!is_valid_square(square)) {
        return false;
    }

    return is_square_attacked(position, square, opponent_of(color));
}

int semantic_move_to_action_index(const Move& move, int side_to_move) {
    if ((side_to_move != kWhite && side_to_move != kBlack) || !is_valid_square(move.from) || !is_valid_square(move.to)) {
        return -1;
    }

    const int from = orient_square_for_player(move.from, side_to_move);
    const int to = orient_square_for_player(move.to, side_to_move);
    if (!is_valid_square(from) || !is_valid_square(to)) {
        return -1;
    }

    const int file_delta = square_file(to) - square_file(from);
    const int rank_delta = square_rank(to) - square_rank(from);

    int move_type_index = -1;
    const int underpromotion_piece_index = underpromotion_piece_to_index(move.promotion);
    if (underpromotion_piece_index >= 0) {
        const int underpromotion_direction_index = underpromotion_direction_to_index(file_delta, rank_delta);
        if (underpromotion_direction_index < 0) {
            return -1;
        }
        move_type_index = 64 + (underpromotion_piece_index * 3) + underpromotion_direction_index;
    } else {
        int knight_index = -1;
        if (delta_is_knight_move(file_delta, rank_delta, &knight_index)) {
            move_type_index = 56 + knight_index;
        } else {
            int direction_index = -1;
            int distance = -1;
            if (!delta_is_queen_move(file_delta, rank_delta, &direction_index, &distance)) {
                return -1;
            }
            move_type_index = (direction_index * 7) + (distance - 1);
        }
    }

    if (move_type_index < 0 || move_type_index >= kActionPlanesPerSquare) {
        return -1;
    }

    return from * kActionPlanesPerSquare + move_type_index;
}

std::optional<Move> action_index_to_semantic_move(const ChessPosition& position, int action_index) {
    if (action_index < 0 || action_index >= kActionSpaceSize) {
        return std::nullopt;
    }

    const int side_to_move = position.side_to_move;
    if (side_to_move != kWhite && side_to_move != kBlack) {
        return std::nullopt;
    }

    const int from_oriented = action_index / kActionPlanesPerSquare;
    const int move_type_index = action_index % kActionPlanesPerSquare;

    const int from_file = square_file(from_oriented);
    const int from_rank = square_rank(from_oriented);

    int to_file = -1;
    int to_rank = -1;
    int promotion_piece = -1;

    if (move_type_index < 56) {
        const int direction = move_type_index / 7;
        const int distance = (move_type_index % 7) + 1;
        to_file = from_file + (kQueenMoveDirections[direction].file * distance);
        to_rank = from_rank + (kQueenMoveDirections[direction].rank * distance);
    } else if (move_type_index < 64) {
        const int knight_offset = move_type_index - 56;
        to_file = from_file + kKnightMoveOffsets[knight_offset].file;
        to_rank = from_rank + kKnightMoveOffsets[knight_offset].rank;
    } else {
        const int underpromotion_offset = move_type_index - 64;
        const int piece_index = underpromotion_offset / 3;
        const int direction_index = underpromotion_offset % 3;

        if (piece_index == 0) {
            promotion_piece = kKnight;
        } else if (piece_index == 1) {
            promotion_piece = kBishop;
        } else {
            promotion_piece = kRook;
        }

        to_file = from_file + kUnderpromotionDirections[direction_index].file;
        to_rank = from_rank + kUnderpromotionDirections[direction_index].rank;
    }

    if (to_file < 0 || to_file >= kBoardSize || to_rank < 0 || to_rank >= kBoardSize) {
        return std::nullopt;
    }

    const int to_oriented = make_square(to_file, to_rank);
    const int from = unorient_square_for_player(from_oriented, side_to_move);
    const int to = unorient_square_for_player(to_oriented, side_to_move);

    if (!is_valid_square(from) || !is_valid_square(to)) {
        return std::nullopt;
    }

    const int moving_piece = piece_on_square(position, side_to_move, from);
    if (moving_piece < 0) {
        return std::nullopt;
    }

    Move candidate{.from = from, .to = to, .piece = moving_piece};

    if (promotion_piece >= 0) {
        if (moving_piece != kPawn) {
            return std::nullopt;
        }
        candidate.promotion = promotion_piece;
    }

    if (promotion_piece < 0 && moving_piece == kPawn) {
        const int promotion_rank = side_to_move == kWhite ? 7 : 0;
        if (square_rank(to) == promotion_rank) {
            candidate.promotion = kQueen;
        }
    }

    if (moving_piece == kKing && square_rank(from) == square_rank(to) && std::abs(square_file(to) - square_file(from)) == 2) {
        candidate.is_castling = true;
    }

    if (moving_piece == kPawn && position.en_passant_square == to &&
        (occupied(position) & square_bit(to)) == 0ULL && std::abs(square_file(to) - square_file(from)) == 1) {
        candidate.is_en_passant = true;
    }

    const std::vector<Move> legal_moves = generate_legal_moves(position);
    for (const Move& legal_move : legal_moves) {
        if (move_equivalent(legal_move, candidate)) {
            return legal_move;
        }
    }

    return std::nullopt;
}

}  // namespace alphazero::chess
