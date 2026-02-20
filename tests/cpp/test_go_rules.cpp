#include "games/go/go_state.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace {

using alphazero::go::GoPosition;
using alphazero::go::kBlack;
using alphazero::go::kBoardArea;
using alphazero::go::kBoardSize;
using alphazero::go::kDefaultKomi;
using alphazero::go::kEmpty;
using alphazero::go::kWhite;

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

// WHY: Position-history membership is the mechanism positional superko checks will depend on in the next task.
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
