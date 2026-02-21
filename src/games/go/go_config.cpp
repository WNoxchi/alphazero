#include "games/go/go_config.h"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "games/go/go_state.h"

namespace alphazero::go {
namespace {

class GoDihedralSymmetryTransform final : public SymmetryTransform {
public:
    GoDihedralSymmetryTransform(int quarter_turns, bool reflect)
        : quarter_turns_(quarter_turns),
          reflect_(reflect) {
        if (quarter_turns_ < 0 || quarter_turns_ > 3) {
            throw std::invalid_argument("Go symmetry quarter_turns must be in [0, 3]");
        }
    }

    void transform_board(float* board, int channels, int rows, int cols) const override {
        if (board == nullptr) {
            throw std::invalid_argument("Go symmetry board pointer must be non-null");
        }
        if (channels <= 0 || rows <= 0 || cols <= 0) {
            throw std::invalid_argument("Go symmetry board dimensions must be positive");
        }
        if (rows != cols) {
            throw std::invalid_argument("Go symmetry transforms require square board tensors");
        }

        const auto total_values = static_cast<std::size_t>(channels) * static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(cols);
        std::vector<float> source(total_values);
        std::copy_n(board, total_values, source.data());

        for (int channel = 0; channel < channels; ++channel) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    const auto [transformed_row, transformed_col] = transform_square_coordinates(row, col, rows);
                    board[tensor_index(channel, transformed_row, transformed_col, rows, cols)] =
                        source[tensor_index(channel, row, col, rows, cols)];
                }
            }
        }
    }

    void transform_policy(float* policy, int action_space_size) const override {
        if (policy == nullptr) {
            throw std::invalid_argument("Go symmetry policy pointer must be non-null");
        }
        if (action_space_size != kActionSpaceSize) {
            throw std::invalid_argument("Go symmetry policy size must match Go action space");
        }

        std::array<float, kActionSpaceSize> source{};
        std::copy_n(policy, kActionSpaceSize, source.begin());

        for (int action = 0; action < kBoardArea; ++action) {
            const int row = intersection_row(action);
            const int col = intersection_col(action);
            const auto [transformed_row, transformed_col] = transform_square_coordinates(row, col, kBoardSize);
            const int transformed_action = to_intersection(transformed_row, transformed_col);
            policy[transformed_action] = source[static_cast<std::size_t>(action)];
        }

        policy[kPassAction] = source[static_cast<std::size_t>(kPassAction)];
    }

private:
    [[nodiscard]] static std::size_t tensor_index(int channel, int row, int col, int rows, int cols) {
        return ((static_cast<std::size_t>(channel) * static_cast<std::size_t>(rows)) +
                   static_cast<std::size_t>(row)) *
                static_cast<std::size_t>(cols) +
            static_cast<std::size_t>(col);
    }

    [[nodiscard]] static std::pair<int, int> rotate_coordinates(int row, int col, int size, int quarter_turns) {
        switch (quarter_turns) {
            case 0:
                return {row, col};
            case 1:
                return {col, (size - 1) - row};
            case 2:
                return {(size - 1) - row, (size - 1) - col};
            case 3:
                return {(size - 1) - col, row};
            default:
                break;
        }

        return {row, col};
    }

    [[nodiscard]] std::pair<int, int> transform_square_coordinates(int row, int col, int size) const {
        auto [transformed_row, transformed_col] = rotate_coordinates(row, col, size, quarter_turns_);
        if (reflect_) {
            transformed_col = (size - 1) - transformed_col;
        }
        return {transformed_row, transformed_col};
    }

    int quarter_turns_;
    bool reflect_;
};

}  // namespace

GoGameConfig::GoGameConfig() {
    name = "go";

    board_rows = kBoardSize;
    board_cols = kBoardSize;

    planes_per_step = GoState::kPlanesPerStep;
    num_history_steps = GoState::kHistorySteps;
    constant_planes = GoState::kConstantPlanes;
    total_input_channels = GoState::kTotalInputChannels;

    action_space_size = kActionSpaceSize;

    dirichlet_alpha = 0.03F;
    max_game_length = GoState::kMaxGameLength;

    value_head_type = ValueHeadType::SCALAR;

    supports_symmetry = true;
    num_symmetries = 8;
}

std::unique_ptr<GameState> GoGameConfig::new_game() const { return std::make_unique<GoState>(); }

std::vector<std::unique_ptr<SymmetryTransform>> GoGameConfig::get_symmetries() const {
    std::vector<std::unique_ptr<SymmetryTransform>> transforms;
    transforms.reserve(8);
    for (int quarter_turns = 0; quarter_turns < 4; ++quarter_turns) {
        transforms.emplace_back(std::make_unique<GoDihedralSymmetryTransform>(quarter_turns, false));
        transforms.emplace_back(std::make_unique<GoDihedralSymmetryTransform>(quarter_turns, true));
    }
    return transforms;
}

const GoGameConfig& go_game_config() {
    static const GoGameConfig config{};
    return config;
}

}  // namespace alphazero::go
