#include "games/go/go_state.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "games/go/go_rules.h"
#include "games/go/scoring.h"

namespace alphazero::go {
namespace {

struct ZobristTable {
    // [0]=black, [1]=white.
    std::array<std::array<std::uint64_t, kBoardArea>, 2> stone{};
    // side_to_move[color], valid indices are kBlack and kWhite.
    std::array<std::uint64_t, 3> side_to_move{};
    // ko_point[0..360], ko_point[361] is "no ko point".
    std::array<std::uint64_t, kBoardArea + 1> ko_point{};
};

[[nodiscard]] constexpr bool is_valid_stone_value(std::uint8_t stone) {
    return stone == kEmpty || stone == kBlack || stone == kWhite;
}

[[nodiscard]] const char* move_status_to_string(MoveStatus status) {
    switch (status) {
        case MoveStatus::kLegal:
            return "legal";
        case MoveStatus::kInvalidAction:
            return "invalid_action";
        case MoveStatus::kIntersectionOccupied:
            return "intersection_occupied";
        case MoveStatus::kKoViolation:
            return "ko_violation";
        case MoveStatus::kSuperkoViolation:
            return "superko_violation";
        case MoveStatus::kSelfCapture:
            return "self_capture";
        case MoveStatus::kInvalidSideToMove:
            return "invalid_side_to_move";
    }
    return "unknown";
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
        std::uint64_t seed = 0xC001C0DE1234ABCDULL;

        for (int color = 0; color < 2; ++color) {
            for (int intersection = 0; intersection < kBoardArea; ++intersection) {
                table.stone[color][intersection] = splitmix64(&seed);
            }
        }

        table.side_to_move[kBlack] = splitmix64(&seed);
        table.side_to_move[kWhite] = splitmix64(&seed);

        for (std::uint64_t& key : table.ko_point) {
            key = splitmix64(&seed);
        }

        return table;
    }();
    return kTable;
}

}  // namespace

std::uint8_t stone_at(const GoPosition& position, int row, int col) {
    if (!is_valid_intersection(row, col)) {
        return kEmpty;
    }
    return position.board[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
}

std::uint8_t stone_at(const GoPosition& position, int intersection) {
    if (!is_valid_intersection(intersection)) {
        return kEmpty;
    }
    return stone_at(position, intersection_row(intersection), intersection_col(intersection));
}

void set_stone(GoPosition* position, int row, int col, std::uint8_t stone) {
    if (position == nullptr || !is_valid_intersection(row, col) || !is_valid_stone_value(stone)) {
        return;
    }

    position->board[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = stone;
}

void set_stone(GoPosition* position, int intersection, std::uint8_t stone) {
    if (!is_valid_intersection(intersection)) {
        return;
    }
    set_stone(position, intersection_row(intersection), intersection_col(intersection), stone);
}

std::uint64_t zobrist_stone_key(int intersection, int color) {
    if (!is_valid_intersection(intersection) || !is_valid_color(color)) {
        return 0ULL;
    }

    const auto& table = zobrist_table();
    const int stone_index = color == kBlack ? 0 : 1;
    return table.stone[static_cast<std::size_t>(stone_index)][static_cast<std::size_t>(intersection)];
}

std::uint64_t zobrist_side_to_move_key(int side_to_move) {
    if (!is_valid_color(side_to_move)) {
        return 0ULL;
    }

    const auto& table = zobrist_table();
    return table.side_to_move[static_cast<std::size_t>(side_to_move)];
}

std::uint64_t zobrist_ko_point_key(int ko_point) {
    const auto& table = zobrist_table();
    const int index = is_valid_intersection(ko_point) ? ko_point : kBoardArea;
    return table.ko_point[static_cast<std::size_t>(index)];
}

std::uint64_t zobrist_update_for_stone(std::uint64_t board_hash, int intersection, int color) {
    return board_hash ^ zobrist_stone_key(intersection, color);
}

std::uint64_t zobrist_update_for_stone(std::uint64_t board_hash, int row, int col, int color) {
    return zobrist_update_for_stone(board_hash, to_intersection(row, col), color);
}

std::uint64_t zobrist_board_hash(const GoPosition& position) {
    std::uint64_t hash = 0ULL;
    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const std::uint8_t stone = stone_at(position, row, col);
            if (!is_valid_color(stone)) {
                continue;
            }

            const int intersection = to_intersection(row, col);
            hash ^= zobrist_stone_key(intersection, stone);
        }
    }
    return hash;
}

std::uint64_t zobrist_hash(const GoPosition& position) {
    std::uint64_t hash = zobrist_board_hash(position);
    hash ^= zobrist_side_to_move_key(position.side_to_move);
    hash ^= zobrist_ko_point_key(position.ko_point);
    return hash;
}

std::string board_to_string(const GoPosition& position) {
    std::ostringstream out;
    for (int row = kBoardSize - 1; row >= 0; --row) {
        for (int col = 0; col < kBoardSize; ++col) {
            const std::uint8_t stone = stone_at(position, row, col);
            const char symbol = stone == kBlack ? 'X' : (stone == kWhite ? 'O' : '.');
            out << symbol;
        }
        if (row > 0) {
            out << '\n';
        }
    }
    return out.str();
}

GoState::GoState() = default;

GoState::GoState(const GoPosition& position) : position_(position) {}

GoState::GoState(GoPosition position, std::shared_ptr<const GoState> parent)
    : position_(std::move(position)),
      parent_(std::move(parent)) {}

std::unique_ptr<GameState> GoState::apply_action(int action) const {
    if (is_terminal()) {
        throw std::invalid_argument("Cannot apply actions to a terminal Go state");
    }

    const MoveResult result = play_action(position_, action);
    if (!result.legal()) {
        throw std::invalid_argument(
            "Cannot apply illegal Go action index " + std::to_string(action) + " (" +
            move_status_to_string(result.status) + ")");
    }

    auto parent = std::make_shared<GoState>(*this);
    return std::unique_ptr<GameState>(new GoState(result.position, std::move(parent)));
}

std::vector<int> GoState::legal_actions() const {
    std::vector<int> actions;
    actions.reserve(kActionSpaceSize);

    for (int action = 0; action < kActionSpaceSize; ++action) {
        if (is_legal_action(position_, action)) {
            actions.push_back(action);
        }
    }

    return actions;
}

bool GoState::is_terminal() const {
    return position_.consecutive_passes >= 2 || position_.move_number >= kMaxGameLength;
}

float GoState::outcome(int player) const {
    const int player_color = player_index_to_color(player);
    if (!is_valid_color(player_color)) {
        throw std::invalid_argument("Outcome player must be 0 (black) or 1 (white)");
    }

    if (!is_terminal()) {
        return 0.0F;
    }

    const TrompTaylorScore score = compute_tromp_taylor_score(position_);
    const int winner = score.winner();
    if (winner == kEmpty) {
        return 0.0F;
    }
    return winner == player_color ? 1.0F : -1.0F;
}

int GoState::current_player() const {
    const int player_index = color_to_player_index(position_.side_to_move);
    if (player_index < 0) {
        throw std::logic_error("GoState side_to_move must be black or white");
    }
    return player_index;
}

void GoState::encode(float* buffer) const {
    if (buffer == nullptr) {
        throw std::invalid_argument("GoState::encode requires a non-null output buffer");
    }

    std::fill_n(buffer, kTotalInputChannels * kBoardArea, 0.0F);

    const int perspective_color = position_.side_to_move;
    const GoState* history_cursor = this;
    for (int history_index = 0; history_index < kHistorySteps && history_cursor != nullptr; ++history_index) {
        encode_position_planes(history_cursor->position_, perspective_color, history_index, buffer);
        history_cursor = history_cursor->parent_.get();
    }

    fill_plane(buffer, kHistorySteps * kPlanesPerStep, perspective_color == kBlack ? 1.0F : 0.0F);
}

std::unique_ptr<GameState> GoState::clone() const { return std::make_unique<GoState>(*this); }

std::uint64_t GoState::hash() const { return zobrist_hash(position_); }

std::string GoState::to_string() const {
    std::ostringstream out;
    out << board_to_string(position_) << '\n';
    out << "side_to_move: ";
    if (position_.side_to_move == kBlack) {
        out << "black";
    } else if (position_.side_to_move == kWhite) {
        out << "white";
    } else {
        out << "invalid(" << position_.side_to_move << ")";
    }
    out << ", ko_point: " << position_.ko_point
        << ", move_number: " << position_.move_number
        << ", consecutive_passes: " << position_.consecutive_passes
        << ", komi: " << position_.komi
        << ", history_size: " << history_size();
    return out.str();
}

int GoState::history_size() const {
    int size = 1;
    const GoState* history_cursor = this;
    while (size < kHistorySteps && history_cursor->parent_ != nullptr) {
        ++size;
        history_cursor = history_cursor->parent_.get();
    }
    return size;
}

const GoPosition& GoState::history_position(int steps_ago) const {
    if (steps_ago < 0 || steps_ago >= kHistorySteps) {
        throw std::out_of_range("Requested Go history step is outside the T=8 history window");
    }

    const GoState* history_cursor = this;
    int traversed = 0;
    while (traversed < steps_ago && history_cursor->parent_ != nullptr) {
        history_cursor = history_cursor->parent_.get();
        ++traversed;
    }

    if (traversed != steps_ago) {
        throw std::out_of_range("Requested Go history step is not available");
    }

    return history_cursor->position_;
}

void GoState::fill_plane(float* buffer, int plane_index, float value) {
    if (buffer == nullptr || plane_index < 0 || plane_index >= kTotalInputChannels) {
        return;
    }
    std::fill_n(buffer + (plane_index * kBoardArea), kBoardArea, value);
}

void GoState::encode_position_planes(
    const GoPosition& encoded_position,
    int perspective_color,
    int history_index,
    float* buffer) {
    if (buffer == nullptr || history_index < 0 || history_index >= kHistorySteps || !is_valid_color(perspective_color)) {
        return;
    }

    const int opponent = opponent_color(perspective_color);
    const int plane_base = history_index * kPlanesPerStep;

    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        const std::uint8_t stone = stone_at(encoded_position, intersection);
        if (stone == perspective_color) {
            buffer[((plane_base + 0) * kBoardArea) + intersection] = 1.0F;
        } else if (stone == opponent) {
            buffer[((plane_base + 1) * kBoardArea) + intersection] = 1.0F;
        }
    }
}

}  // namespace alphazero::go
