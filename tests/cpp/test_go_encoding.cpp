#include "games/go/go_config.h"
#include "games/go/go_state.h"

#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::go::GoGameConfig;
using alphazero::go::kActionSpaceSize;
using alphazero::go::kBoardArea;
using alphazero::go::kBoardSize;
using alphazero::go::kPassAction;

[[nodiscard]] constexpr int tensor_index(int channel, int row, int col, int rows, int cols) {
    return ((channel * rows) + row) * cols + col;
}

[[nodiscard]] constexpr int row_from_action(int action) { return action / kBoardSize; }
[[nodiscard]] constexpr int col_from_action(int action) { return action % kBoardSize; }
[[nodiscard]] constexpr int action_from_row_col(int row, int col) { return (row * kBoardSize) + col; }

[[nodiscard]] constexpr int apply_dihedral_transform(int action, int transform_id) {
    const int row = row_from_action(action);
    const int col = col_from_action(action);

    switch (transform_id) {
        case 0:  // Identity
            return action_from_row_col(row, col);
        case 1:  // Rotate 90 degrees clockwise
            return action_from_row_col(col, (kBoardSize - 1) - row);
        case 2:  // Rotate 180 degrees
            return action_from_row_col((kBoardSize - 1) - row, (kBoardSize - 1) - col);
        case 3:  // Rotate 270 degrees clockwise
            return action_from_row_col((kBoardSize - 1) - col, row);
        case 4:  // Reflect vertically
            return action_from_row_col(row, (kBoardSize - 1) - col);
        case 5:  // Reflect horizontally
            return action_from_row_col((kBoardSize - 1) - row, col);
        case 6:  // Reflect over main diagonal
            return action_from_row_col(col, row);
        case 7:  // Reflect over anti-diagonal
            return action_from_row_col((kBoardSize - 1) - col, (kBoardSize - 1) - row);
        default:
            break;
    }

    return -1;
}

[[nodiscard]] std::set<std::vector<int>> expected_policy_permutations() {
    std::set<std::vector<int>> expected;

    for (int transform_id = 0; transform_id < 8; ++transform_id) {
        std::vector<int> transformed(kActionSpaceSize, -1);
        for (int action = 0; action < kBoardArea; ++action) {
            const int mapped_action = apply_dihedral_transform(action, transform_id);
            transformed[mapped_action] = action;
        }
        transformed[kPassAction] = kPassAction;
        expected.insert(std::move(transformed));
    }

    return expected;
}

[[nodiscard]] std::vector<int> policy_as_integer_vector(const std::vector<float>& policy) {
    std::vector<int> as_int(policy.size(), -1);
    for (std::size_t i = 0; i < policy.size(); ++i) {
        as_int[i] = static_cast<int>(std::lround(policy[i]));
    }
    return as_int;
}

}  // namespace

// WHY: Go augmentation relies on exactly the D4 symmetry group; wrong permutations silently corrupt training labels.
TEST(GoEncodingTest, SymmetriesExposeAllEightDihedralPolicyPermutationsWithInvariantPass) {
    const GoGameConfig config{};
    std::vector<std::unique_ptr<alphazero::SymmetryTransform>> transforms = config.get_symmetries();
    ASSERT_EQ(transforms.size(), 8U);

    std::vector<float> source_policy(kActionSpaceSize, 0.0F);
    for (int action = 0; action < kActionSpaceSize; ++action) {
        source_policy[action] = static_cast<float>(action);
    }

    std::set<std::vector<int>> observed;
    for (const auto& transform : transforms) {
        std::vector<float> transformed_policy = source_policy;
        transform->transform_policy(transformed_policy.data(), kActionSpaceSize);
        EXPECT_FLOAT_EQ(transformed_policy[kPassAction], source_policy[kPassAction]);

        std::vector<int> as_int = policy_as_integer_vector(transformed_policy);
        observed.insert(std::move(as_int));
    }

    const std::set<std::vector<int>> expected = expected_policy_permutations();
    EXPECT_EQ(observed, expected);
}

// WHY: Board and policy must stay synchronized under augmentation; mismatched transforms break policy supervision.
TEST(GoEncodingTest, BoardAndPolicyTransformsStayConsistentAcrossChannels) {
    const GoGameConfig config{};
    std::vector<std::unique_ptr<alphazero::SymmetryTransform>> transforms = config.get_symmetries();
    ASSERT_EQ(transforms.size(), 8U);

    constexpr int kChannels = 3;
    std::vector<float> source_board(kChannels * kBoardArea, 0.0F);
    std::vector<float> source_policy(kActionSpaceSize, 0.0F);
    for (int action = 0; action < kBoardArea; ++action) {
        source_board[tensor_index(0, row_from_action(action), col_from_action(action), kBoardSize, kBoardSize)] =
            static_cast<float>(action);
        source_board[tensor_index(1, row_from_action(action), col_from_action(action), kBoardSize, kBoardSize)] =
            static_cast<float>(1000 + action);
        source_board[tensor_index(2, row_from_action(action), col_from_action(action), kBoardSize, kBoardSize)] =
            static_cast<float>(2000 + action);
        source_policy[action] = static_cast<float>(action);
    }
    source_policy[kPassAction] = 12345.0F;

    for (const auto& transform : transforms) {
        std::vector<float> transformed_board = source_board;
        std::vector<float> transformed_policy = source_policy;
        transform->transform_board(transformed_board.data(), kChannels, kBoardSize, kBoardSize);
        transform->transform_policy(transformed_policy.data(), kActionSpaceSize);

        for (int action = 0; action < kBoardArea; ++action) {
            const int row = row_from_action(action);
            const int col = col_from_action(action);
            const float board_ch0 =
                transformed_board[tensor_index(0, row, col, kBoardSize, kBoardSize)];
            const float board_ch1 =
                transformed_board[tensor_index(1, row, col, kBoardSize, kBoardSize)];
            const float board_ch2 =
                transformed_board[tensor_index(2, row, col, kBoardSize, kBoardSize)];

            EXPECT_FLOAT_EQ(board_ch0, transformed_policy[action]);
            EXPECT_FLOAT_EQ(board_ch1, board_ch0 + 1000.0F);
            EXPECT_FLOAT_EQ(board_ch2, board_ch0 + 2000.0F);
        }
        EXPECT_FLOAT_EQ(transformed_policy[kPassAction], source_policy[kPassAction]);
    }
}

// WHY: Defensive argument validation avoids undefined behavior when augmentation is wired incorrectly.
TEST(GoEncodingTest, SymmetryTransformsRejectInvalidInputs) {
    const GoGameConfig config{};
    std::vector<std::unique_ptr<alphazero::SymmetryTransform>> transforms = config.get_symmetries();
    ASSERT_FALSE(transforms.empty());

    std::vector<float> board(kBoardArea, 0.0F);
    std::vector<float> policy(kActionSpaceSize, 0.0F);

    EXPECT_THROW(transforms.front()->transform_board(nullptr, 1, kBoardSize, kBoardSize), std::invalid_argument);
    EXPECT_THROW(transforms.front()->transform_board(board.data(), 1, kBoardSize, kBoardSize - 1), std::invalid_argument);
    EXPECT_THROW(transforms.front()->transform_policy(nullptr, kActionSpaceSize), std::invalid_argument);
    EXPECT_THROW(transforms.front()->transform_policy(policy.data(), kActionSpaceSize - 1), std::invalid_argument);
}
