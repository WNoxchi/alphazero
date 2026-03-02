#include "games/go/go_state.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <stdexcept>
#include <utility>
#include <vector>

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
        case MoveStatus::kPassTooEarly:
            return "pass_too_early";
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

struct SgfProperty {
    std::string identifier;
    std::vector<std::string> values;
};

using SgfNode = std::vector<SgfProperty>;

[[nodiscard]] constexpr bool is_upper_ascii_letter(char symbol) {
    return symbol >= 'A' && symbol <= 'Z';
}

[[nodiscard]] constexpr bool is_sgf_coord_symbol(char symbol) {
    return symbol >= 'a' && symbol < static_cast<char>('a' + kBoardSize);
}

[[nodiscard]] bool parse_integer_token(const std::string& token, int* value) {
    if (value == nullptr || token.empty()) {
        return false;
    }

    std::size_t parsed_length = 0;
    int parsed_value = 0;
    try {
        parsed_value = std::stoi(token, &parsed_length);
    } catch (const std::exception&) {
        return false;
    }

    if (parsed_length != token.size()) {
        return false;
    }
    *value = parsed_value;
    return true;
}

[[nodiscard]] bool parse_float_token(const std::string& token, float* value) {
    if (value == nullptr || token.empty()) {
        return false;
    }

    std::size_t parsed_length = 0;
    float parsed_value = 0.0F;
    try {
        parsed_value = std::stof(token, &parsed_length);
    } catch (const std::exception&) {
        return false;
    }

    if (parsed_length != token.size() || !std::isfinite(parsed_value)) {
        return false;
    }
    *value = parsed_value;
    return true;
}

[[nodiscard]] std::string format_float_for_sgf(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("SGF serialization requires finite floating-point values");
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    std::string text = stream.str();
    const std::size_t dot = text.find('.');
    if (dot != std::string::npos) {
        while (!text.empty() && text.back() == '0') {
            text.pop_back();
        }
        if (!text.empty() && text.back() == '.') {
            text.pop_back();
        }
    }

    if (text.empty() || text == "-0") {
        return "0";
    }
    return text;
}

[[nodiscard]] std::string escape_sgf_value(std::string_view raw) {
    std::string escaped;
    escaped.reserve(raw.size());

    for (char symbol : raw) {
        if (symbol == '\\' || symbol == ']') {
            escaped.push_back('\\');
            escaped.push_back(symbol);
            continue;
        }
        if (symbol == '\r' || symbol == '\n') {
            escaped.push_back(' ');
            continue;
        }
        escaped.push_back(symbol);
    }
    return escaped;
}

[[nodiscard]] int sgf_coordinate_to_action(const std::string& coordinate, bool allow_pass) {
    if (coordinate.empty() || coordinate == "tt") {
        return allow_pass ? kPassAction : -1;
    }

    if (coordinate.size() != 2 || !is_sgf_coord_symbol(coordinate[0]) || !is_sgf_coord_symbol(coordinate[1])) {
        return -1;
    }

    const int col = coordinate[0] - 'a';
    const int sgf_row = coordinate[1] - 'a';
    const int row = (kBoardSize - 1) - sgf_row;
    return to_intersection(row, col);
}

[[nodiscard]] std::string action_to_sgf_coordinate(int action) {
    if (action == kPassAction) {
        return "";
    }
    if (!is_valid_intersection(action)) {
        throw std::invalid_argument("SGF serialization encountered an invalid Go action index");
    }

    const int row = intersection_row(action);
    const int col = intersection_col(action);
    const int sgf_row = (kBoardSize - 1) - row;

    std::string coordinate(2, 'a');
    coordinate[0] = static_cast<char>('a' + col);
    coordinate[1] = static_cast<char>('a' + sgf_row);
    return coordinate;
}

[[nodiscard]] int infer_action_from_transition(const GoPosition& previous, const GoPosition& next, int mover_color) {
    const int opponent = opponent_color(mover_color);

    int placed_stone = -1;
    int placed_count = 0;
    int changed_count = 0;
    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        const std::uint8_t before = stone_at(previous, intersection);
        const std::uint8_t after = stone_at(next, intersection);
        if (before == after) {
            continue;
        }

        ++changed_count;
        if (before == kEmpty && after == mover_color) {
            placed_stone = intersection;
            ++placed_count;
            continue;
        }
        if (before == opponent && after == kEmpty) {
            continue;
        }

        throw std::logic_error("GoState history contains an invalid board transition for SGF export");
    }

    if (placed_count == 0) {
        if (changed_count == 0) {
            return kPassAction;
        }
        throw std::logic_error("GoState history changed board state without placing a stone");
    }
    if (placed_count != 1) {
        throw std::logic_error("GoState history placed multiple stones in a single transition");
    }
    return placed_stone;
}

class SgfParser {
public:
    explicit SgfParser(std::string_view source) : source_(source) {}

    [[nodiscard]] std::vector<SgfNode> parse() {
        skip_whitespace();
        expect('(', "SGF must start with '('");

        std::vector<SgfNode> nodes;
        while (true) {
            skip_whitespace();
            if (at_end()) {
                throw std::invalid_argument("SGF is missing a closing ')'");
            }

            const char token = peek();
            if (token == ';') {
                nodes.push_back(parse_node());
                continue;
            }
            if (token == '(') {
                throw std::invalid_argument("SGF variations are not supported");
            }
            if (token == ')') {
                ++index_;
                break;
            }

            throw std::invalid_argument("SGF contains an unexpected token while parsing the game tree");
        }

        skip_whitespace();
        if (!at_end()) {
            throw std::invalid_argument("SGF contains trailing content after the game tree");
        }
        if (nodes.empty()) {
            throw std::invalid_argument("SGF game tree must contain at least one node");
        }
        return nodes;
    }

private:
    [[nodiscard]] SgfNode parse_node() {
        expect(';', "SGF node must begin with ';'");

        SgfNode node;
        while (true) {
            skip_whitespace();
            if (at_end()) {
                throw std::invalid_argument("SGF ended while parsing a node");
            }

            const char token = peek();
            if (token == ';' || token == ')' || token == '(') {
                break;
            }
            node.push_back(parse_property());
        }
        return node;
    }

    [[nodiscard]] SgfProperty parse_property() {
        std::string identifier;
        while (!at_end() && is_upper_ascii_letter(peek())) {
            identifier.push_back(peek());
            ++index_;
        }

        if (identifier.empty()) {
            throw std::invalid_argument("SGF property identifier must use uppercase letters");
        }

        skip_whitespace();
        std::vector<std::string> values;
        while (!at_end() && peek() == '[') {
            values.push_back(parse_property_value());
            skip_whitespace();
        }

        if (values.empty()) {
            throw std::invalid_argument("SGF property '" + identifier + "' must include at least one value");
        }

        return SgfProperty{
            .identifier = std::move(identifier),
            .values = std::move(values),
        };
    }

    [[nodiscard]] std::string parse_property_value() {
        expect('[', "SGF property value must begin with '['");

        std::string value;
        while (!at_end()) {
            const char symbol = peek();
            ++index_;

            if (symbol == ']') {
                return value;
            }

            if (symbol == '\\') {
                if (at_end()) {
                    throw std::invalid_argument("SGF property value ends with an incomplete escape sequence");
                }

                const char escaped = peek();
                ++index_;
                if (escaped == '\r') {
                    if (!at_end() && peek() == '\n') {
                        ++index_;
                    }
                    continue;
                }
                if (escaped == '\n') {
                    continue;
                }
                value.push_back(escaped);
                continue;
            }

            value.push_back(symbol);
        }

        throw std::invalid_argument("SGF property value is missing a closing ']'");
    }

    void skip_whitespace() {
        while (!at_end()) {
            const unsigned char symbol = static_cast<unsigned char>(peek());
            if (std::isspace(symbol) == 0) {
                break;
            }
            ++index_;
        }
    }

    void expect(char symbol, const char* error) {
        if (at_end() || peek() != symbol) {
            throw std::invalid_argument(error);
        }
        ++index_;
    }

    [[nodiscard]] bool at_end() const { return index_ >= source_.size(); }
    [[nodiscard]] char peek() const { return source_[index_]; }

    std::string_view source_;
    std::size_t index_ = 0;
};

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

GoState GoState::from_sgf(const std::string& sgf) {
    const std::vector<SgfNode> nodes = SgfParser(sgf).parse();

    GoPosition starting_position{};
    bool has_black_setup = false;
    bool has_white_setup = false;
    bool side_to_move_explicit = false;

    const SgfNode& root = nodes.front();
    for (const SgfProperty& property : root) {
        if (property.identifier == "GM") {
            if (property.values.size() != 1 || property.values.front() != "1") {
                throw std::invalid_argument("SGF root property GM must be exactly '1' for Go");
            }
            continue;
        }

        if (property.identifier == "SZ") {
            int parsed_size = 0;
            if (property.values.size() != 1 || !parse_integer_token(property.values.front(), &parsed_size) ||
                parsed_size != kBoardSize) {
                throw std::invalid_argument("SGF root property SZ must be exactly 19");
            }
            continue;
        }

        if (property.identifier == "KM") {
            float parsed_komi = 0.0F;
            if (property.values.size() != 1 || !parse_float_token(property.values.front(), &parsed_komi)) {
                throw std::invalid_argument("SGF root property KM must be a finite floating-point value");
            }
            starting_position.komi = parsed_komi;
            continue;
        }

        if (property.identifier == "PL") {
            if (property.values.size() != 1) {
                throw std::invalid_argument("SGF root property PL must contain exactly one value");
            }
            const std::string& side = property.values.front();
            if (side == "B") {
                starting_position.side_to_move = kBlack;
            } else if (side == "W") {
                starting_position.side_to_move = kWhite;
            } else {
                throw std::invalid_argument("SGF root property PL must be 'B' or 'W'");
            }
            side_to_move_explicit = true;
            continue;
        }

        if (property.identifier == "AB" || property.identifier == "AW") {
            const int color = property.identifier == "AB" ? kBlack : kWhite;
            if (color == kBlack) {
                has_black_setup = true;
            } else {
                has_white_setup = true;
            }

            for (const std::string& coordinate : property.values) {
                const int action = sgf_coordinate_to_action(coordinate, /*allow_pass=*/false);
                if (!is_valid_intersection(action)) {
                    throw std::invalid_argument("SGF setup properties must use valid board coordinates");
                }

                const std::uint8_t existing_stone = stone_at(starting_position, action);
                if (existing_stone != kEmpty && existing_stone != static_cast<std::uint8_t>(color)) {
                    throw std::invalid_argument("SGF setup properties cannot assign both colors to one point");
                }
                set_stone(&starting_position, action, static_cast<std::uint8_t>(color));
            }
            continue;
        }
    }

    if (!side_to_move_explicit && has_black_setup && !has_white_setup) {
        // Standard handicap SGF convention: white plays first after black setup stones.
        starting_position.side_to_move = kWhite;
    }

    GoState state(starting_position);
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        const SgfNode& node = nodes[node_index];

        int node_move_color = kEmpty;
        int node_move_action = -1;
        for (const SgfProperty& property : node) {
            if (property.identifier != "B" && property.identifier != "W") {
                continue;
            }

            if (node_move_color != kEmpty) {
                throw std::invalid_argument("SGF node contains multiple move properties");
            }
            if (property.values.size() != 1) {
                throw std::invalid_argument("SGF move properties must contain exactly one value");
            }

            node_move_color = property.identifier == "B" ? kBlack : kWhite;
            node_move_action = sgf_coordinate_to_action(property.values.front(), /*allow_pass=*/true);
            if (node_move_action < 0 || node_move_action >= kActionSpaceSize) {
                throw std::invalid_argument("SGF move coordinate is invalid for a 19x19 board");
            }
        }

        if (node_move_color == kEmpty) {
            continue;
        }

        if (state.position_.side_to_move != node_move_color) {
            throw std::invalid_argument(
                "SGF move color does not match side-to-move at node " + std::to_string(node_index));
        }

        std::unique_ptr<GameState> next_base;
        try {
            next_base = state.apply_action(node_move_action);
        } catch (const std::exception& error) {
            throw std::invalid_argument(
                "SGF contains an illegal move at node " + std::to_string(node_index) + ": " + error.what());
        }

        auto* typed = dynamic_cast<GoState*>(next_base.get());
        if (typed == nullptr) {
            throw std::logic_error("GoState::from_sgf expected GoState transition result");
        }
        state = *typed;
    }

    return state;
}

std::string GoState::to_sgf(const std::string& result) const {
    std::vector<const GoState*> lineage;
    lineage.reserve(256);

    const GoState* cursor = this;
    while (cursor != nullptr) {
        lineage.push_back(cursor);
        cursor = cursor->parent_.get();
    }
    std::reverse(lineage.begin(), lineage.end());

    if (lineage.empty()) {
        throw std::logic_error("GoState::to_sgf cannot serialize an empty lineage");
    }

    const GoPosition& root_position = lineage.front()->position_;
    if (!is_valid_color(root_position.side_to_move)) {
        throw std::logic_error("GoState::to_sgf cannot serialize a root position with invalid side_to_move");
    }

    std::ostringstream stream;
    stream << "(;";
    stream << "GM[1]";
    stream << "FF[4]";
    stream << "CA[UTF-8]";
    stream << "AP[AlphaZero]";
    stream << "SZ[" << kBoardSize << "]";
    stream << "KM[" << escape_sgf_value(format_float_for_sgf(root_position.komi)) << "]";
    stream << "PB[AlphaZero]";
    stream << "PW[AlphaZero]";
    stream << "RE[" << escape_sgf_value(result.empty() ? "?" : result) << "]";

    if (root_position.side_to_move == kWhite) {
        stream << "PL[W]";
    }

    bool wrote_black_setup = false;
    bool wrote_white_setup = false;
    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        const std::uint8_t stone = stone_at(root_position, intersection);
        if (stone == kBlack) {
            if (!wrote_black_setup) {
                stream << "AB";
                wrote_black_setup = true;
            }
            stream << "[" << action_to_sgf_coordinate(intersection) << "]";
        } else if (stone == kWhite) {
            if (!wrote_white_setup) {
                stream << "AW";
                wrote_white_setup = true;
            }
            stream << "[" << action_to_sgf_coordinate(intersection) << "]";
        } else if (stone != kEmpty) {
            throw std::logic_error("GoState::to_sgf encountered an invalid stone value in root setup");
        }
    }

    for (std::size_t i = 1; i < lineage.size(); ++i) {
        const GoPosition& previous = lineage[i - 1]->position_;
        const GoPosition& next = lineage[i]->position_;
        const int mover = previous.side_to_move;
        if (!is_valid_color(mover) || next.side_to_move != opponent_color(mover)) {
            throw std::logic_error("GoState::to_sgf encountered inconsistent side-to-move history");
        }
        if (next.move_number != previous.move_number + 1) {
            throw std::logic_error("GoState::to_sgf encountered non-sequential move numbers in history");
        }

        const int action = infer_action_from_transition(previous, next, mover);
        stream << ';' << (mover == kBlack ? 'B' : 'W') << '[';
        if (action != kPassAction) {
            stream << action_to_sgf_coordinate(action);
        }
        stream << ']';
    }

    stream << ')';
    return stream.str();
}

std::string GoState::actions_to_sgf(const std::vector<int>& action_history, const std::string& result, float komi) {
    if (!std::isfinite(komi)) {
        throw std::invalid_argument("SGF export requires finite komi");
    }

    GoPosition starting_position{};
    starting_position.komi = komi;
    GoState state(starting_position);

    for (std::size_t ply = 0; ply < action_history.size(); ++ply) {
        const int action = action_history[ply];

        std::unique_ptr<GameState> next_base;
        try {
            next_base = state.apply_action(action);
        } catch (const std::exception&) {
            throw std::invalid_argument(
                "SGF export encountered illegal action index " + std::to_string(action) + " at ply " +
                std::to_string(ply));
        }

        auto* typed = dynamic_cast<GoState*>(next_base.get());
        if (typed == nullptr) {
            throw std::logic_error("GoState::actions_to_sgf expected GoState transition result");
        }
        state = *typed;
    }

    return state.to_sgf(result);
}

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
