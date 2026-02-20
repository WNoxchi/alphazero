#include "games/chess/chess_state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "games/chess/movegen.h"

namespace alphazero::chess {
namespace {

[[nodiscard]] constexpr int opponent_of(int color) { return color == kWhite ? kBlack : kWhite; }

[[nodiscard]] constexpr int square_file(int square) { return square & 7; }
[[nodiscard]] constexpr int square_rank(int square) { return square >> 3; }

[[nodiscard]] constexpr bool is_light_square(int square) {
    return ((square_file(square) + square_rank(square)) & 1) == 0;
}

[[nodiscard]] ChessPosition initial_chess_position() {
    ChessPosition position{};
    position.pieces[kWhite][kPawn] = 0x000000000000FF00ULL;
    position.pieces[kWhite][kKnight] = 0x0000000000000042ULL;
    position.pieces[kWhite][kBishop] = 0x0000000000000024ULL;
    position.pieces[kWhite][kRook] = 0x0000000000000081ULL;
    position.pieces[kWhite][kQueen] = 0x0000000000000008ULL;
    position.pieces[kWhite][kKing] = 0x0000000000000010ULL;

    position.pieces[kBlack][kPawn] = 0x00FF000000000000ULL;
    position.pieces[kBlack][kKnight] = 0x4200000000000000ULL;
    position.pieces[kBlack][kBishop] = 0x2400000000000000ULL;
    position.pieces[kBlack][kRook] = 0x8100000000000000ULL;
    position.pieces[kBlack][kQueen] = 0x0800000000000000ULL;
    position.pieces[kBlack][kKing] = 0x1000000000000000ULL;

    position.side_to_move = kWhite;
    position.castling = static_cast<std::uint8_t>(kWhiteKingSide | kWhiteQueenSide | kBlackKingSide | kBlackQueenSide);
    position.en_passant_square = -1;
    position.halfmove_clock = 0;
    position.fullmove_number = 1;
    position.repetition_count = 1;
    return position;
}

[[nodiscard]] int inferred_ply_count(const ChessPosition& position) {
    const int fullmove_base = std::max(0, position.fullmove_number - 1);
    const int side_offset = position.side_to_move == kBlack ? 1 : 0;
    return (fullmove_base * 2) + side_offset;
}

[[nodiscard]] char piece_symbol_at(const ChessPosition& position, int square) {
    if (!is_valid_square(square)) {
        return '.';
    }

    const Bitboard mask = square_bit(square);
    constexpr std::array<char, kPieceTypeCount> kWhiteSymbols = {'P', 'N', 'B', 'R', 'Q', 'K'};
    constexpr std::array<char, kPieceTypeCount> kBlackSymbols = {'p', 'n', 'b', 'r', 'q', 'k'};

    for (int piece = 0; piece < kPieceTypeCount; ++piece) {
        if ((position.pieces[kWhite][piece] & mask) != 0ULL) {
            return kWhiteSymbols[piece];
        }
        if ((position.pieces[kBlack][piece] & mask) != 0ULL) {
            return kBlackSymbols[piece];
        }
    }

    return '.';
}

[[nodiscard]] std::string castling_to_string(std::uint8_t castling) {
    std::string rights;
    if ((castling & kWhiteKingSide) != 0U) {
        rights.push_back('K');
    }
    if ((castling & kWhiteQueenSide) != 0U) {
        rights.push_back('Q');
    }
    if ((castling & kBlackKingSide) != 0U) {
        rights.push_back('k');
    }
    if ((castling & kBlackQueenSide) != 0U) {
        rights.push_back('q');
    }
    if (rights.empty()) {
        rights = "-";
    }
    return rights;
}

[[nodiscard]] std::string square_to_algebraic(int square) {
    if (!is_valid_square(square)) {
        return "-";
    }

    std::string result(2, ' ');
    result[0] = static_cast<char>('a' + square_file(square));
    result[1] = static_cast<char>('1' + square_rank(square));
    return result;
}

void fill_plane(float* buffer, int channel, float value) {
    if (buffer == nullptr) {
        return;
    }

    const int channel_offset = channel * kBoardSquares;
    for (int square = 0; square < kBoardSquares; ++square) {
        buffer[channel_offset + square] = value;
    }
}

}  // namespace

ChessState::ChessState() : ChessState(initial_chess_position()) {}

ChessState::ChessState(const ChessPosition& position) : position_(position), history_size_(1), ply_count_(inferred_ply_count(position)) {
    if (position_.side_to_move != kWhite && position_.side_to_move != kBlack) {
        throw std::invalid_argument("ChessState requires side_to_move to be white or black");
    }

    position_.repetition_count = std::max(1, position_.repetition_count);
    history_[0] = position_;
    repetition_table_[zobrist_hash(position_)] = position_.repetition_count;
}

ChessState::ChessState(
    ChessPosition position,
    std::array<ChessPosition, kHistorySteps> history,
    int history_size,
    std::unordered_map<std::uint64_t, int> repetition_table,
    int ply_count)
    : position_(std::move(position)),
      history_(std::move(history)),
      history_size_(std::clamp(history_size, 1, kHistorySteps)),
      repetition_table_(std::move(repetition_table)),
      ply_count_(std::max(0, ply_count)) {}

std::unique_ptr<GameState> ChessState::apply_action(int action) const {
    const std::optional<Move> move = action_index_to_semantic_move(position_, action);
    if (!move.has_value()) {
        throw std::invalid_argument("Cannot apply illegal chess action index: " + std::to_string(action));
    }

    ChessPosition next_position = apply_move(position_, move.value());
    std::unordered_map<std::uint64_t, int> next_repetition_table = repetition_table_;
    const std::uint64_t next_hash = zobrist_hash(next_position);
    const int next_repetition_count = ++next_repetition_table[next_hash];
    next_position.repetition_count = next_repetition_count;

    std::array<ChessPosition, kHistorySteps> next_history{};
    next_history[0] = next_position;

    const int next_history_size = std::min(kHistorySteps, history_size_ + 1);
    for (int i = 1; i < next_history_size; ++i) {
        next_history[i] = history_[i - 1];
    }

    return std::unique_ptr<GameState>(new ChessState(
        next_position,
        next_history,
        next_history_size,
        std::move(next_repetition_table),
        ply_count_ + 1));
}

std::vector<int> ChessState::legal_actions() const { return legal_action_indices(position_); }

bool ChessState::is_terminal() const {
    const std::vector<Move> legal_moves = generate_legal_moves(position_);
    if (legal_moves.empty()) {
        return true;
    }

    return position_.halfmove_clock >= 100 || position_.repetition_count >= 3 || ply_count_ >= kMaxGameLength ||
           is_insufficient_material();
}

float ChessState::outcome(int player) const {
    if (player != kWhite && player != kBlack) {
        throw std::invalid_argument("Outcome player must be 0 (white) or 1 (black)");
    }

    const std::vector<Move> legal_moves = generate_legal_moves(position_);
    if (legal_moves.empty()) {
        if (!is_in_check(position_, position_.side_to_move)) {
            return 0.0F;  // Stalemate.
        }

        const int winner = opponent_of(position_.side_to_move);
        return winner == player ? 1.0F : -1.0F;
    }

    if (position_.halfmove_clock >= 100 || position_.repetition_count >= 3 || ply_count_ >= kMaxGameLength ||
        is_insufficient_material()) {
        return 0.0F;
    }

    return 0.0F;
}

int ChessState::current_player() const { return position_.side_to_move; }

void ChessState::encode(float* buffer) const {
    if (buffer == nullptr) {
        throw std::invalid_argument("ChessState::encode requires a non-null output buffer");
    }

    std::fill_n(buffer, kTotalInputChannels * kBoardSquares, 0.0F);

    const int perspective_color = position_.side_to_move;
    for (int history_index = 0; history_index < history_size_; ++history_index) {
        encode_position_planes(history_[history_index], perspective_color, history_index, buffer);
    }

    const int constant_offset = kHistorySteps * kPlanesPerStep;
    fill_plane(buffer, constant_offset + 0, perspective_color == kWhite ? 1.0F : 0.0F);
    fill_plane(buffer, constant_offset + 1, normalized_move_count());

    const bool perspective_is_white = perspective_color == kWhite;
    const bool p1_kingside =
        perspective_is_white ? (position_.castling & kWhiteKingSide) != 0U : (position_.castling & kBlackKingSide) != 0U;
    const bool p1_queenside = perspective_is_white ? (position_.castling & kWhiteQueenSide) != 0U
                                                   : (position_.castling & kBlackQueenSide) != 0U;
    const bool p2_kingside =
        perspective_is_white ? (position_.castling & kBlackKingSide) != 0U : (position_.castling & kWhiteKingSide) != 0U;
    const bool p2_queenside = perspective_is_white ? (position_.castling & kBlackQueenSide) != 0U
                                                   : (position_.castling & kWhiteQueenSide) != 0U;

    fill_plane(buffer, constant_offset + 2, p1_kingside ? 1.0F : 0.0F);
    fill_plane(buffer, constant_offset + 3, p1_queenside ? 1.0F : 0.0F);
    fill_plane(buffer, constant_offset + 4, p2_kingside ? 1.0F : 0.0F);
    fill_plane(buffer, constant_offset + 5, p2_queenside ? 1.0F : 0.0F);
    fill_plane(buffer, constant_offset + 6, normalized_no_progress_count());
}

std::unique_ptr<GameState> ChessState::clone() const { return std::make_unique<ChessState>(*this); }

std::uint64_t ChessState::hash() const { return zobrist_hash(position_); }

std::string ChessState::to_string() const {
    std::ostringstream stream;
    for (int rank = 7; rank >= 0; --rank) {
        stream << (rank + 1) << " ";
        for (int file = 0; file < 8; ++file) {
            const int square = (rank * 8) + file;
            stream << piece_symbol_at(position_, square);
            if (file != 7) {
                stream << ' ';
            }
        }
        stream << '\n';
    }
    stream << "  a b c d e f g h\n";
    stream << "side_to_move: " << (position_.side_to_move == kWhite ? "white" : "black")
           << ", castling: " << castling_to_string(position_.castling)
           << ", en_passant: " << square_to_algebraic(position_.en_passant_square)
           << ", halfmove: " << position_.halfmove_clock
           << ", fullmove: " << position_.fullmove_number
           << ", repetition: " << position_.repetition_count
           << ", ply: " << ply_count_;
    return stream.str();
}

const ChessPosition& ChessState::history_position(int steps_ago) const {
    if (steps_ago < 0 || steps_ago >= history_size_) {
        throw std::out_of_range("Requested chess history step is not available");
    }
    return history_[steps_ago];
}

bool ChessState::is_insufficient_material() const {
    const int white_pawns = popcount(position_.pieces[kWhite][kPawn]);
    const int white_rooks = popcount(position_.pieces[kWhite][kRook]);
    const int white_queens = popcount(position_.pieces[kWhite][kQueen]);
    const int black_pawns = popcount(position_.pieces[kBlack][kPawn]);
    const int black_rooks = popcount(position_.pieces[kBlack][kRook]);
    const int black_queens = popcount(position_.pieces[kBlack][kQueen]);

    if (white_pawns != 0 || white_rooks != 0 || white_queens != 0 || black_pawns != 0 || black_rooks != 0 ||
        black_queens != 0) {
        return false;
    }

    const int white_knights = popcount(position_.pieces[kWhite][kKnight]);
    const int white_bishops = popcount(position_.pieces[kWhite][kBishop]);
    const int black_knights = popcount(position_.pieces[kBlack][kKnight]);
    const int black_bishops = popcount(position_.pieces[kBlack][kBishop]);

    const int white_minor = white_knights + white_bishops;
    const int black_minor = black_knights + black_bishops;
    const int total_minor = white_minor + black_minor;

    if (total_minor <= 1) {
        return true;  // K vs K, K+B vs K, K+N vs K.
    }

    if (white_minor == 0 && black_minor == 2 && black_knights == 2) {
        return true;  // K+NN vs K.
    }
    if (black_minor == 0 && white_minor == 2 && white_knights == 2) {
        return true;  // K vs K+NN.
    }

    if (white_minor == 1 && black_minor == 1) {
        if (white_knights == 1 && black_knights == 1) {
            return true;  // K+N vs K+N.
        }

        if (white_bishops == 1 && black_bishops == 1) {
            const int white_bishop_square = bit_scan_forward(position_.pieces[kWhite][kBishop]);
            const int black_bishop_square = bit_scan_forward(position_.pieces[kBlack][kBishop]);
            return white_bishop_square >= 0 && black_bishop_square >= 0 &&
                   is_light_square(white_bishop_square) == is_light_square(black_bishop_square);
        }
    }

    return false;
}

int ChessState::orient_square_for_side(int square, int side_to_move) {
    if (!is_valid_square(square)) {
        return -1;
    }
    if (side_to_move == kBlack) {
        return (kBoardSquares - 1) - square;
    }
    return square;
}

float ChessState::normalized_move_count() const {
    return std::min(1.0F, static_cast<float>(ply_count_) / static_cast<float>(kMaxGameLength));
}

float ChessState::normalized_no_progress_count() const {
    constexpr float kHalfmoveDrawThreshold = 100.0F;
    return std::min(1.0F, static_cast<float>(position_.halfmove_clock) / kHalfmoveDrawThreshold);
}

void ChessState::encode_position_planes(
    const ChessPosition& encoded_position,
    int perspective_color,
    int history_index,
    float* buffer) {
    if (buffer == nullptr || history_index < 0 || history_index >= kHistorySteps) {
        return;
    }

    const int plane_base = history_index * kPlanesPerStep;
    const int opponent_color = opponent_of(perspective_color);

    for (int piece = 0; piece < kPieceTypeCount; ++piece) {
        Bitboard p1_bits = encoded_position.pieces[perspective_color][piece];
        while (p1_bits != 0ULL) {
            const int square = pop_lsb(&p1_bits);
            const int oriented_square = orient_square_for_side(square, perspective_color);
            if (oriented_square >= 0) {
                buffer[((plane_base + piece) * kBoardSquares) + oriented_square] = 1.0F;
            }
        }

        Bitboard p2_bits = encoded_position.pieces[opponent_color][piece];
        while (p2_bits != 0ULL) {
            const int square = pop_lsb(&p2_bits);
            const int oriented_square = orient_square_for_side(square, perspective_color);
            if (oriented_square >= 0) {
                buffer[((plane_base + 6 + piece) * kBoardSquares) + oriented_square] = 1.0F;
            }
        }
    }

    const float seen_once = encoded_position.repetition_count <= 1 ? 1.0F : 0.0F;
    const float seen_multiple = encoded_position.repetition_count >= 2 ? 1.0F : 0.0F;
    fill_plane(buffer, plane_base + 12, seen_once);
    fill_plane(buffer, plane_base + 13, seen_multiple);
}

}  // namespace alphazero::chess
