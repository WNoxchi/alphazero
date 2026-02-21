#include "games/go/go_state.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::go::GoPosition;
using alphazero::go::GoState;
using alphazero::go::kActionSpaceSize;
using alphazero::go::kBlack;
using alphazero::go::kBoardSize;
using alphazero::go::kPassAction;
using alphazero::go::kWhite;

[[nodiscard]] constexpr int I(int row, int col) { return alphazero::go::to_intersection(row, col); }

[[nodiscard]] int intersection_from_sgf(const std::string& coordinate) {
    if (coordinate.size() != 2) {
        return -1;
    }

    const int col = coordinate[0] - 'a';
    const int sgf_row = coordinate[1] - 'a';
    const int row = (kBoardSize - 1) - sgf_row;
    return alphazero::go::to_intersection(row, col);
}

[[nodiscard]] GoState apply_action_or_throw(const GoState& state, int action) {
    std::unique_ptr<alphazero::GameState> next_base = state.apply_action(action);
    auto* typed = dynamic_cast<GoState*>(next_base.get());
    if (typed == nullptr) {
        throw std::runtime_error("Expected GoState transition result");
    }
    return *typed;
}

}  // namespace

// WHY: SGF export/import is the Go replay artifact; round-trip identity prevents silent corruption in analysis workflows.
TEST(GoSerializationTest, ActionsToSgfRoundTripPreservesStateAndMetadata) {
    const std::vector<int> actions = {
        I(3, 3),
        I(3, 4),
        I(4, 3),
        kPassAction,
        I(4, 4),
        kPassAction,
        kPassAction,
    };

    GoPosition start{};
    start.komi = 6.5F;
    GoState expected(start);
    for (int action : actions) {
        expected = apply_action_or_throw(expected, action);
    }

    const std::string sgf = GoState::actions_to_sgf(actions, "B+R", 6.5F);
    EXPECT_NE(sgf.find("GM[1]"), std::string::npos);
    EXPECT_NE(sgf.find("SZ[19]"), std::string::npos);
    EXPECT_NE(sgf.find("KM[6.5]"), std::string::npos);
    EXPECT_NE(sgf.find("RE[B+R]"), std::string::npos);
    EXPECT_NE(sgf.find(";W[]"), std::string::npos);

    const GoState parsed = GoState::from_sgf(sgf);
    EXPECT_EQ(parsed.hash(), expected.hash());
    EXPECT_EQ(parsed.position().move_number, expected.position().move_number);
    EXPECT_EQ(parsed.position().consecutive_passes, expected.position().consecutive_passes);
    EXPECT_FLOAT_EQ(parsed.position().komi, 6.5F);

    const GoState reparsed = GoState::from_sgf(parsed.to_sgf("B+R"));
    EXPECT_EQ(reparsed.hash(), parsed.hash());
}

// WHY: Debug SGF files commonly start from setup stones, so import must handle AB/AW/PL/KM root properties correctly.
TEST(GoSerializationTest, FromSgfParsesRootSetupAndSideToMove) {
    const std::string sgf =
        "(;GM[1]FF[4]SZ[19]KM[0.5]AB[dd][pd]AW[qp]PL[W];W[dc];B[])";

    const GoState state = GoState::from_sgf(sgf);
    EXPECT_FLOAT_EQ(state.position().komi, 0.5F);
    EXPECT_EQ(state.position().move_number, 2);
    EXPECT_EQ(state.position().consecutive_passes, 1);
    EXPECT_EQ(state.position().side_to_move, kWhite);
    EXPECT_EQ(alphazero::go::stone_at(state.position(), intersection_from_sgf("dd")), kBlack);
    EXPECT_EQ(alphazero::go::stone_at(state.position(), intersection_from_sgf("pd")), kBlack);
    EXPECT_EQ(alphazero::go::stone_at(state.position(), intersection_from_sgf("qp")), kWhite);
    EXPECT_EQ(alphazero::go::stone_at(state.position(), intersection_from_sgf("dc")), kWhite);

    const std::string exported = state.to_sgf("?");
    EXPECT_NE(exported.find("PL[W]"), std::string::npos);
    EXPECT_NE(exported.find("AB["), std::string::npos);
    EXPECT_NE(exported.find("AW["), std::string::npos);
}

// WHY: Strict parse-time validation prevents malformed SGF files from poisoning debugging and regression fixtures.
TEST(GoSerializationTest, FromSgfRejectsMalformedOrUnsupportedInputs) {
    const std::vector<std::string> invalid_sgfs = {
        "GM[1]SZ[19];B[aa]",
        "(;GM[1]SZ[9];B[aa])",
        "(;GM[2]SZ[19];B[aa])",
        "(;GM[1]SZ[19];B[zz])",
        "(;GM[1]SZ[19]PL[W];B[aa])",
        "(;GM[1]SZ[19];B[aa](;W[bb]))",
    };

    for (const std::string& sgf : invalid_sgfs) {
        EXPECT_THROW(
            {
                const GoState ignored = GoState::from_sgf(sgf);
                (void)ignored;
            },
            std::invalid_argument)
            << "sgf=" << sgf;
    }
}

// WHY: Export-time validation ensures broken move traces fail immediately instead of producing invalid SGF logs.
TEST(GoSerializationTest, ActionsToSgfRejectsIllegalActionHistories) {
    EXPECT_THROW(
        {
            const std::string ignored = GoState::actions_to_sgf({kActionSpaceSize}, "?");
            (void)ignored;
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            const std::string ignored = GoState::actions_to_sgf({I(3, 3), I(3, 3)}, "?");
            (void)ignored;
        },
        std::invalid_argument);
}

// WHY: Passes are legal Go actions and must serialize as empty coordinates (`[]`) for SGF compatibility.
TEST(GoSerializationTest, ToSgfEncodesPassMovesAsEmptyCoordinates) {
    GoState state{};
    state = apply_action_or_throw(state, I(10, 10));
    state = apply_action_or_throw(state, kPassAction);

    const std::string sgf = state.to_sgf("W+0.5");
    EXPECT_NE(sgf.find(";B["), std::string::npos);
    EXPECT_NE(sgf.find(";W[]"), std::string::npos);

    const GoState parsed = GoState::from_sgf(sgf);
    EXPECT_EQ(parsed.hash(), state.hash());
    EXPECT_EQ(parsed.position().move_number, state.position().move_number);
}
