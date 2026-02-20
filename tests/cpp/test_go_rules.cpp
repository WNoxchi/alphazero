#include "games/go/go_rules.h"
#include "games/go/scoring.h"
#include "games/go/go_state.h"

#include <algorithm>
#include <cstdint>

#include <gtest/gtest.h>

namespace {

using alphazero::go::GoPosition;
using alphazero::go::MoveStatus;
using alphazero::go::StoneGroup;
using alphazero::go::kBlack;
using alphazero::go::kBoardArea;
using alphazero::go::kBoardSize;
using alphazero::go::kDefaultKomi;
using alphazero::go::kEmpty;
using alphazero::go::kPassAction;
using alphazero::go::kWhite;

[[nodiscard]] constexpr int I(int row, int col) { return alphazero::go::to_intersection(row, col); }

}  // namespace

// WHY: Core defaults and indexing invariants are prerequisites for every Go rule operation built on this state.
TEST(GoStateRepresentationTest, PositionDefaultsAndIntersectionHelpersMatchSpec) {
    const GoPosition position{};
    EXPECT_EQ(position.side_to_move, kBlack);
    EXPECT_EQ(position.ko_point, -1);
    EXPECT_FLOAT_EQ(position.komi, kDefaultKomi);
    EXPECT_EQ(position.move_number, 0);
    EXPECT_EQ(position.consecutive_passes, 0);
    EXPECT_TRUE(position.position_history.empty());

    for (int row = 0; row < kBoardSize; ++row) {
        for (int col = 0; col < kBoardSize; ++col) {
            EXPECT_EQ(alphazero::go::stone_at(position, row, col), kEmpty);
            const int intersection = alphazero::go::to_intersection(row, col);
            EXPECT_EQ(alphazero::go::intersection_row(intersection), row);
            EXPECT_EQ(alphazero::go::intersection_col(intersection), col);
            EXPECT_TRUE(alphazero::go::is_valid_intersection(intersection));
        }
    }

    EXPECT_FALSE(alphazero::go::is_valid_intersection(-1));
    EXPECT_FALSE(alphazero::go::is_valid_intersection(kBoardArea));
    EXPECT_EQ(alphazero::go::to_intersection(-1, 0), -1);
    EXPECT_EQ(alphazero::go::to_intersection(0, kBoardSize), -1);
}

// WHY: Deterministic hashing is required so repeated states can be reliably recognized in superko tracking.
TEST(GoStateRepresentationTest, ZobristHashIsStableForEquivalentStates) {
    GoPosition first{};
    alphazero::go::set_stone(&first, 3, 3, kBlack);
    alphazero::go::set_stone(&first, 3, 4, kWhite);
    first.side_to_move = kWhite;
    first.ko_point = alphazero::go::to_intersection(10, 10);

    GoPosition second = first;
    EXPECT_EQ(alphazero::go::zobrist_board_hash(first), alphazero::go::zobrist_board_hash(second));
    EXPECT_EQ(alphazero::go::zobrist_hash(first), alphazero::go::zobrist_hash(second));
}

// WHY: Hash sensitivity guards against false positives in position deduplication and transposition handling.
TEST(GoStateRepresentationTest, ZobristHashChangesWhenBoardSideOrKoChanges) {
    GoPosition baseline{};
    alphazero::go::set_stone(&baseline, 9, 9, kBlack);
    alphazero::go::set_stone(&baseline, 9, 10, kWhite);
    baseline.side_to_move = kBlack;
    baseline.ko_point = -1;

    const std::uint64_t board_hash = alphazero::go::zobrist_board_hash(baseline);
    const std::uint64_t full_hash = alphazero::go::zobrist_hash(baseline);

    GoPosition piece_changed = baseline;
    alphazero::go::set_stone(&piece_changed, 10, 9, kBlack);
    EXPECT_NE(alphazero::go::zobrist_board_hash(piece_changed), board_hash);
    EXPECT_NE(alphazero::go::zobrist_hash(piece_changed), full_hash);

    GoPosition side_changed = baseline;
    side_changed.side_to_move = kWhite;
    EXPECT_EQ(alphazero::go::zobrist_board_hash(side_changed), board_hash);
    EXPECT_NE(alphazero::go::zobrist_hash(side_changed), full_hash);

    GoPosition ko_changed = baseline;
    ko_changed.ko_point = alphazero::go::to_intersection(9, 8);
    EXPECT_EQ(alphazero::go::zobrist_board_hash(ko_changed), board_hash);
    EXPECT_NE(alphazero::go::zobrist_hash(ko_changed), full_hash);
}

// WHY: Rules code will place/capture stones incrementally, so XOR updates must match full recomputation exactly.
TEST(GoStateRepresentationTest, IncrementalStoneHashUpdatesMatchFullBoardHash) {
    GoPosition position{};
    std::uint64_t incremental_hash = alphazero::go::zobrist_board_hash(position);

    const int c4 = alphazero::go::to_intersection(3, 2);
    const int d4 = alphazero::go::to_intersection(3, 3);
    const int e4 = alphazero::go::to_intersection(3, 4);

    // Place black stone at D4.
    incremental_hash = alphazero::go::zobrist_update_for_stone(incremental_hash, d4, kBlack);
    alphazero::go::set_stone(&position, d4, kBlack);
    EXPECT_EQ(incremental_hash, alphazero::go::zobrist_board_hash(position));

    // Place white stones around it.
    incremental_hash = alphazero::go::zobrist_update_for_stone(incremental_hash, c4, kWhite);
    alphazero::go::set_stone(&position, c4, kWhite);
    EXPECT_EQ(incremental_hash, alphazero::go::zobrist_board_hash(position));

    incremental_hash = alphazero::go::zobrist_update_for_stone(incremental_hash, e4, kWhite);
    alphazero::go::set_stone(&position, e4, kWhite);
    EXPECT_EQ(incremental_hash, alphazero::go::zobrist_board_hash(position));

    // Capture/removal toggles with the same stone key.
    incremental_hash = alphazero::go::zobrist_update_for_stone(incremental_hash, d4, kBlack);
    alphazero::go::set_stone(&position, d4, kEmpty);
    EXPECT_EQ(incremental_hash, alphazero::go::zobrist_board_hash(position));

    // Toggling the same key twice is a no-op.
    const std::uint64_t before_toggle_twice = incremental_hash;
    incremental_hash = alphazero::go::zobrist_update_for_stone(incremental_hash, e4, kWhite);
    incremental_hash = alphazero::go::zobrist_update_for_stone(incremental_hash, e4, kWhite);
    EXPECT_EQ(incremental_hash, before_toggle_twice);
}

// WHY: Position-history membership is the mechanism positional superko checks depend on.
TEST(GoStateRepresentationTest, PositionHistoryTracksRepeatedBoardHashes) {
    GoPosition position{};
    const std::uint64_t initial_hash = alphazero::go::zobrist_board_hash(position);
    position.position_history.insert(initial_hash);

    const int move = alphazero::go::to_intersection(10, 10);
    alphazero::go::set_stone(&position, move, kBlack);
    const std::uint64_t advanced_hash = alphazero::go::zobrist_board_hash(position);
    position.position_history.insert(advanced_hash);

    alphazero::go::set_stone(&position, move, kEmpty);
    const std::uint64_t returned_hash = alphazero::go::zobrist_board_hash(position);

    EXPECT_TRUE(position.position_history.contains(initial_hash));
    EXPECT_TRUE(position.position_history.contains(advanced_hash));
    EXPECT_TRUE(position.position_history.contains(returned_hash));
    EXPECT_EQ(returned_hash, initial_hash);
}

// WHY: Union-find group metadata underpins liberty tracking and must report unique liberties for connected chains.
TEST(GoRulesEngineTest, UnionFindLibertyTrackingReportsConnectedGroupsAndUniqueLiberties) {
    GoPosition position{};
    alphazero::go::set_stone(&position, I(3, 3), kBlack);
    alphazero::go::set_stone(&position, I(3, 4), kBlack);
    alphazero::go::set_stone(&position, I(2, 3), kWhite);

    const std::vector<StoneGroup> groups = alphazero::go::compute_stone_groups(position);
    const auto two_stone_group = std::find_if(groups.begin(), groups.end(), [](const StoneGroup& group) {
        return group.stone_count == 2;
    });

    ASSERT_NE(two_stone_group, groups.end());
    EXPECT_EQ(two_stone_group->liberty_count, 5);
    EXPECT_EQ(alphazero::go::liberties_for_intersection(position, I(3, 3)), 5);
    EXPECT_EQ(alphazero::go::liberties_for_intersection(position, I(3, 4)), 5);
}

// WHY: Single-stone capture correctness is the baseline tactical rule required for all gameplay.
TEST(GoRulesEngineTest, PlayActionCapturesSingleStoneAndUpdatesTurnState) {
    GoPosition position{};
    position.side_to_move = kBlack;
    alphazero::go::set_stone(&position, I(10, 10), kWhite);
    alphazero::go::set_stone(&position, I(9, 10), kBlack);
    alphazero::go::set_stone(&position, I(10, 9), kBlack);
    alphazero::go::set_stone(&position, I(10, 11), kBlack);

    const auto result = alphazero::go::play_action(position, I(11, 10));
    ASSERT_TRUE(result.legal());
    EXPECT_EQ(result.status, MoveStatus::kLegal);
    EXPECT_EQ(result.captured_stones, 1);
    EXPECT_EQ(alphazero::go::stone_at(result.position, I(10, 10)), kEmpty);
    EXPECT_EQ(alphazero::go::stone_at(result.position, I(11, 10)), kBlack);
    EXPECT_EQ(result.position.side_to_move, kWhite);
    EXPECT_EQ(result.position.move_number, 1);
    EXPECT_EQ(result.position.consecutive_passes, 0);
    EXPECT_EQ(result.position.ko_point, -1);
}

// WHY: Multi-stone capture ensures the engine removes entire groups, not just directly-adjacent stones.
TEST(GoRulesEngineTest, PlayActionCapturesMultiStoneGroupInOneMove) {
    GoPosition position{};
    position.side_to_move = kBlack;
    alphazero::go::set_stone(&position, I(10, 10), kWhite);
    alphazero::go::set_stone(&position, I(10, 11), kWhite);
    alphazero::go::set_stone(&position, I(11, 10), kWhite);
    alphazero::go::set_stone(&position, I(9, 10), kBlack);
    alphazero::go::set_stone(&position, I(10, 9), kBlack);
    alphazero::go::set_stone(&position, I(9, 11), kBlack);
    alphazero::go::set_stone(&position, I(10, 12), kBlack);
    alphazero::go::set_stone(&position, I(11, 9), kBlack);
    alphazero::go::set_stone(&position, I(12, 10), kBlack);

    const auto result = alphazero::go::play_action(position, I(11, 11));
    ASSERT_TRUE(result.legal());
    EXPECT_EQ(result.captured_stones, 3);
    EXPECT_EQ(alphazero::go::stone_at(result.position, I(10, 10)), kEmpty);
    EXPECT_EQ(alphazero::go::stone_at(result.position, I(10, 11)), kEmpty);
    EXPECT_EQ(alphazero::go::stone_at(result.position, I(11, 10)), kEmpty);
    EXPECT_EQ(alphazero::go::stone_at(result.position, I(11, 11)), kBlack);
}

// WHY: Ko tracking must forbid immediate single-stone recapture that would recreate the previous board.
TEST(GoRulesEngineTest, KoRecaptureIsBlockedAtTrackedKoPoint) {
    GoPosition position{};
    position.side_to_move = kBlack;
    alphazero::go::set_stone(&position, I(9, 10), kWhite);
    alphazero::go::set_stone(&position, I(11, 10), kWhite);
    alphazero::go::set_stone(&position, I(10, 9), kWhite);
    alphazero::go::set_stone(&position, I(10, 11), kWhite);
    alphazero::go::set_stone(&position, I(9, 11), kBlack);
    alphazero::go::set_stone(&position, I(11, 11), kBlack);
    alphazero::go::set_stone(&position, I(10, 12), kBlack);

    const auto capture = alphazero::go::play_action(position, I(10, 10));
    ASSERT_TRUE(capture.legal());
    ASSERT_EQ(capture.position.ko_point, I(10, 11));
    EXPECT_EQ(capture.captured_stones, 1);

    const auto illegal_recapture = alphazero::go::play_action(capture.position, I(10, 11));
    EXPECT_FALSE(illegal_recapture.legal());
    EXPECT_EQ(illegal_recapture.status, MoveStatus::kKoViolation);
    EXPECT_EQ(alphazero::go::stone_at(illegal_recapture.position, I(10, 11)), kEmpty);
}

// WHY: Positional superko must reject any move that recreates a historical board state, not just immediate ko.
TEST(GoRulesEngineTest, PositionalSuperkoRejectsMovesThatRecreatePriorBoardHash) {
    GoPosition position{};
    position.side_to_move = kBlack;

    GoPosition historical = position;
    alphazero::go::set_stone(&historical, I(6, 6), kBlack);
    position.position_history.insert(alphazero::go::zobrist_board_hash(historical));

    const auto result = alphazero::go::play_action(position, I(6, 6));
    EXPECT_FALSE(result.legal());
    EXPECT_EQ(result.status, MoveStatus::kSuperkoViolation);
    EXPECT_EQ(alphazero::go::stone_at(result.position, I(6, 6)), kEmpty);
}

// WHY: Suicide prevention is a core legality rule and must reject moves that leave own group without liberties.
TEST(GoRulesEngineTest, SelfCaptureIsRejectedWhenNoOpponentGroupIsCaptured) {
    GoPosition position{};
    position.side_to_move = kBlack;
    alphazero::go::set_stone(&position, I(9, 10), kWhite);
    alphazero::go::set_stone(&position, I(11, 10), kWhite);
    alphazero::go::set_stone(&position, I(10, 9), kWhite);
    alphazero::go::set_stone(&position, I(10, 11), kWhite);

    const auto result = alphazero::go::play_action(position, I(10, 10));
    EXPECT_FALSE(result.legal());
    EXPECT_EQ(result.status, MoveStatus::kSelfCapture);
    EXPECT_EQ(alphazero::go::stone_at(result.position, I(10, 10)), kEmpty);
}

// WHY: Pass handling drives Go game termination; two consecutive passes must be detectable as terminal.
TEST(GoRulesEngineTest, PassesIncrementCounterAndTwoPassesEndTheGame) {
    GoPosition position{};
    position.side_to_move = kBlack;

    const auto first_pass = alphazero::go::play_action(position, kPassAction);
    ASSERT_TRUE(first_pass.legal());
    EXPECT_EQ(first_pass.position.side_to_move, kWhite);
    EXPECT_EQ(first_pass.position.consecutive_passes, 1);
    EXPECT_EQ(first_pass.position.ko_point, -1);
    EXPECT_FALSE(alphazero::go::passes_end_game(first_pass.position));

    const auto second_pass = alphazero::go::play_action(first_pass.position, kPassAction);
    ASSERT_TRUE(second_pass.legal());
    EXPECT_EQ(second_pass.position.side_to_move, kBlack);
    EXPECT_EQ(second_pass.position.consecutive_passes, 2);
    EXPECT_TRUE(alphazero::go::passes_end_game(second_pass.position));
}

// WHY: Tromp-Taylor scoring must combine occupied points, enclosed territory ownership, and komi exactly.
TEST(GoScoringTest, CountsOccupiedAndExclusiveTerritoryAndAppliesKomi) {
    GoPosition position{};
    position.komi = 3.5F;

    // Black encloses a single-point territory at (3, 3).
    alphazero::go::set_stone(&position, I(2, 3), kBlack);
    alphazero::go::set_stone(&position, I(3, 2), kBlack);
    alphazero::go::set_stone(&position, I(3, 4), kBlack);
    alphazero::go::set_stone(&position, I(4, 3), kBlack);

    // White encloses a single-point territory at (10, 10).
    alphazero::go::set_stone(&position, I(9, 10), kWhite);
    alphazero::go::set_stone(&position, I(10, 9), kWhite);
    alphazero::go::set_stone(&position, I(10, 11), kWhite);
    alphazero::go::set_stone(&position, I(11, 10), kWhite);

    const auto score = alphazero::go::compute_tromp_taylor_score(position);

    EXPECT_EQ(score.black_points, 5);
    EXPECT_EQ(score.white_points, 5);
    EXPECT_FLOAT_EQ(score.komi, 3.5F);
    EXPECT_FLOAT_EQ(score.final_score, -3.5F);
    EXPECT_EQ(score.winner(), kWhite);
}

// WHY: Empty regions adjacent to both colors are neutral and must not be awarded to either side.
TEST(GoScoringTest, SharedEmptyRegionIsNeutral) {
    GoPosition position{};
    position.komi = 0.0F;

    // Surround a single empty point with both colors.
    alphazero::go::set_stone(&position, I(9, 8), kBlack);
    alphazero::go::set_stone(&position, I(8, 9), kBlack);
    alphazero::go::set_stone(&position, I(9, 10), kWhite);
    alphazero::go::set_stone(&position, I(10, 9), kWhite);

    const auto score = alphazero::go::compute_tromp_taylor_score(position);

    EXPECT_EQ(score.black_points, 2);
    EXPECT_EQ(score.white_points, 2);
    EXPECT_FLOAT_EQ(score.final_score, 0.0F);
    EXPECT_EQ(score.winner(), kEmpty);
}

// WHY: Known extreme positions protect against off-by-one territory/occupancy bugs in flood-fill scoring.
TEST(GoScoringTest, EmptyAndFullBoardKnownResults) {
    GoPosition empty{};
    const auto empty_score = alphazero::go::compute_tromp_taylor_score(empty);
    EXPECT_EQ(empty_score.black_points, 0);
    EXPECT_EQ(empty_score.white_points, 0);
    EXPECT_FLOAT_EQ(empty_score.final_score, -kDefaultKomi);
    EXPECT_EQ(empty_score.winner(), kWhite);

    GoPosition full_black{};
    for (int intersection = 0; intersection < kBoardArea; ++intersection) {
        alphazero::go::set_stone(&full_black, intersection, kBlack);
    }
    const auto full_black_score = alphazero::go::compute_tromp_taylor_score(full_black);
    EXPECT_EQ(full_black_score.black_points, kBoardArea);
    EXPECT_EQ(full_black_score.white_points, 0);
    EXPECT_FLOAT_EQ(full_black_score.final_score, static_cast<float>(kBoardArea) - kDefaultKomi);
    EXPECT_EQ(full_black_score.winner(), kBlack);
}
