#include "selfplay/replay_buffer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::selfplay::ReplayBuffer;
using alphazero::selfplay::CompactReplayPosition;
using alphazero::selfplay::ReplayPosition;

constexpr std::size_t kStateSize = 4U;
constexpr std::size_t kPolicySize = 6U;

[[nodiscard]] float scalar_value_for(const std::uint32_t game_id, const std::uint16_t move_number) {
    const std::uint32_t bucket = (game_id + move_number) % 3U;
    if (bucket == 0U) {
        return 1.0F;
    }
    if (bucket == 1U) {
        return 0.0F;
    }
    return -1.0F;
}

[[nodiscard]] float training_weight_for(const std::uint32_t game_id, const std::uint16_t move_number) {
    const std::uint32_t bucket = ((game_id * 13U) + move_number) % 11U;
    return 0.25F + (static_cast<float>(bucket) * 0.125F);
}

[[nodiscard]] std::array<float, ReplayPosition::kWdlSize> wdl_for(const float value) {
    if (value > 0.0F) {
        return {1.0F, 0.0F, 0.0F};
    }
    if (value < 0.0F) {
        return {0.0F, 0.0F, 1.0F};
    }
    return {0.0F, 1.0F, 0.0F};
}

[[nodiscard]] ReplayPosition make_position(const std::uint32_t game_id, const std::uint16_t move_number) {
    const float signature = static_cast<float>(game_id * 1000U + move_number);
    const float value = scalar_value_for(game_id, move_number);
    const std::array<float, kStateSize> state{
        signature,
        static_cast<float>(move_number),
        signature + 2.0F,
        signature + 3.0F,
    };
    const std::array<float, kPolicySize> policy{
        signature + 0.5F,
        static_cast<float>(game_id),
        static_cast<float>(move_number),
        signature + 4.0F,
        signature + 5.0F,
        signature + 6.0F,
    };
    return ReplayPosition::make(
        state,
        policy,
        value,
        wdl_for(value),
        game_id,
        move_number,
        training_weight_for(game_id, move_number));
}

[[nodiscard]] std::uint64_t signature_of(const ReplayPosition& position) {
    return static_cast<std::uint64_t>(position.game_id) * 1000ULL + static_cast<std::uint64_t>(position.move_number);
}

void expect_position_is_consistent(const ReplayPosition& position) {
    ASSERT_EQ(position.encoded_state_size, kStateSize);
    ASSERT_EQ(position.policy_size, kPolicySize);

    const std::uint64_t expected_signature = signature_of(position);
    EXPECT_FLOAT_EQ(position.encoded_state[0], static_cast<float>(expected_signature));
    EXPECT_FLOAT_EQ(position.encoded_state[1], static_cast<float>(position.move_number));
    EXPECT_FLOAT_EQ(position.policy[0], static_cast<float>(expected_signature) + 0.5F);
    EXPECT_FLOAT_EQ(position.policy[1], static_cast<float>(position.game_id));
    EXPECT_FLOAT_EQ(position.policy[2], static_cast<float>(position.move_number));

    const float expected_value = scalar_value_for(position.game_id, position.move_number);
    EXPECT_FLOAT_EQ(position.value, expected_value);
    EXPECT_FLOAT_EQ(position.training_weight, training_weight_for(position.game_id, position.move_number));
    const auto expected_wdl = wdl_for(expected_value);
    EXPECT_EQ(position.value_wdl, expected_wdl);
}

[[nodiscard]] std::vector<ReplayPosition> make_game(
    const std::uint32_t game_id,
    const std::uint16_t move_count,
    const std::uint16_t move_offset = 0U) {
    std::vector<ReplayPosition> positions;
    positions.reserve(move_count);
    for (std::uint16_t move = 0U; move < move_count; ++move) {
        positions.push_back(make_position(game_id, static_cast<std::uint16_t>(move_offset + move)));
    }
    return positions;
}

}  // namespace

// WHY: Phase-1 replay compression depends on a fixed-size compact record with deterministic defaults so the
// compression path can safely populate only touched fields while full-playout samples default to weight 1.0.
TEST(ReplayBufferTest, CompactReplayPositionDefinesExpectedConstantsAndZeroDefaults) {
    CompactReplayPosition position{};

    EXPECT_EQ(CompactReplayPosition::kMaxBinaryPlanes, 117U);
    EXPECT_EQ(CompactReplayPosition::kMaxFloatPlanes, 2U);
    EXPECT_EQ(CompactReplayPosition::kMaxSparsePolicy, 64U);
    EXPECT_EQ(CompactReplayPosition::kWdlSize, ReplayPosition::kWdlSize);

    EXPECT_TRUE(std::all_of(position.bitpacked_planes.begin(), position.bitpacked_planes.end(), [](std::uint64_t v) {
        return v == 0U;
    }));
    EXPECT_TRUE(std::all_of(
        position.quantized_float_planes.begin(),
        position.quantized_float_planes.end(),
        [](std::uint8_t v) { return v == 0U; }));
    EXPECT_TRUE(std::all_of(position.policy_actions.begin(), position.policy_actions.end(), [](std::uint16_t v) {
        return v == 0U;
    }));
    EXPECT_TRUE(std::all_of(position.policy_probs_fp16.begin(), position.policy_probs_fp16.end(), [](std::uint16_t v) {
        return v == 0U;
    }));
    EXPECT_EQ(position.num_policy_entries, 0U);

    EXPECT_FLOAT_EQ(position.value, 0.0F);
    EXPECT_FLOAT_EQ(position.training_weight, 1.0F);
    EXPECT_EQ(position.value_wdl, (std::array<float, ReplayPosition::kWdlSize>{0.0F, 0.0F, 0.0F}));
    EXPECT_EQ(position.game_id, 0U);
    EXPECT_EQ(position.move_number, 0U);
    EXPECT_EQ(position.num_binary_planes, 0U);
    EXPECT_EQ(position.num_float_planes, 0U);
    EXPECT_EQ(position.policy_size, 0U);
}

// WHY: Replay-capacity scaling assumes compact records are far smaller than dense ReplayPosition entries.
TEST(ReplayBufferTest, CompactReplayPositionIsSubstantiallySmallerThanDenseReplayPosition) {
    EXPECT_LE(sizeof(CompactReplayPosition), 1300U);
    EXPECT_GT(sizeof(ReplayPosition), sizeof(CompactReplayPosition));
}

// WHY: Construction and input-shape guards prevent silent corruption from invalid capacity or malformed replay entries.
TEST(ReplayBufferTest, ValidatesCapacityAndPositionShapes) {
    EXPECT_THROW(static_cast<void>(ReplayBuffer(0U)), std::invalid_argument);

    ReplayBuffer buffer(16U, 1234U);
    EXPECT_EQ(buffer.capacity(), 16U);
    EXPECT_EQ(buffer.size(), 0U);

    const std::vector<float> empty;
    const std::vector<float> state = {1.0F, 2.0F};
    const std::vector<float> policy = {0.1F, 0.2F};
    EXPECT_THROW(
        static_cast<void>(ReplayPosition::make(empty, policy, 0.0F, {0.0F, 1.0F, 0.0F}, 1U, 0U)),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(ReplayPosition::make(state, empty, 0.0F, {0.0F, 1.0F, 0.0F}, 1U, 0U)),
        std::invalid_argument);

    ReplayPosition malformed = make_position(3U, 2U);
    malformed.policy_size = 0U;
    EXPECT_THROW(buffer.add_game({malformed}), std::invalid_argument);
    EXPECT_EQ(buffer.size(), 0U);
}

// WHY: Ring-buffer overwrite semantics are the core retention policy; once full, only the most recent N positions
// must remain sampleable.
TEST(ReplayBufferTest, WrapsAndRetainsMostRecentPositions) {
    ReplayBuffer buffer(6U, 0xA55AULL);

    buffer.add_game(make_game(0U, 1U));
    buffer.add_game(make_game(1U, 1U));
    buffer.add_game(make_game(2U, 1U));
    buffer.add_game(make_game(3U, 1U));
    buffer.add_game(make_game(4U, 1U));
    buffer.add_game(make_game(5U, 1U));
    buffer.add_game(make_game(6U, 1U));
    buffer.add_game(make_game(7U, 1U));

    EXPECT_EQ(buffer.size(), 6U);
    EXPECT_EQ(buffer.write_head(), 2U);

    const std::vector<ReplayPosition> sampled = buffer.sample(6U);
    ASSERT_EQ(sampled.size(), 6U);

    std::unordered_set<std::uint32_t> observed_game_ids;
    for (const ReplayPosition& position : sampled) {
        expect_position_is_consistent(position);
        observed_game_ids.insert(position.game_id);
    }
    ASSERT_EQ(observed_game_ids.size(), 6U);
    for (std::uint32_t game_id = 2U; game_id <= 7U; ++game_id) {
        EXPECT_TRUE(observed_game_ids.contains(game_id));
    }
}

// WHY: Self-play writers and the training reader run concurrently; this verifies thread safety, no corruption, and no
// dropped entries when capacity is sufficient to avoid overwrite.
TEST(ReplayBufferTest, SupportsConcurrentWritersAndReaderWithoutDataLoss) {
    constexpr int kWriterThreads = 6;
    constexpr int kGamesPerWriter = 20;
    constexpr std::uint16_t kPositionsPerGame = 5U;
    constexpr std::size_t kTotalPositions =
        static_cast<std::size_t>(kWriterThreads) * static_cast<std::size_t>(kGamesPerWriter) *
        static_cast<std::size_t>(kPositionsPerGame);

    ReplayBuffer buffer(kTotalPositions, 0xD15EA5EULL);

    std::atomic<bool> stop_reader{false};
    std::atomic<int> read_iterations{0};

    std::thread reader([&] {
        while (!stop_reader.load(std::memory_order_acquire)) {
            const std::size_t current_size = buffer.size();
            if (current_size == 0U) {
                std::this_thread::yield();
                continue;
            }

            const std::size_t batch_size = std::min<std::size_t>(16U, current_size);
            const std::vector<ReplayPosition> batch = buffer.sample(batch_size);
            for (const ReplayPosition& position : batch) {
                expect_position_is_consistent(position);
            }
            read_iterations.fetch_add(1, std::memory_order_relaxed);
        }
    });

    std::vector<std::thread> writers;
    writers.reserve(kWriterThreads);
    for (int writer = 0; writer < kWriterThreads; ++writer) {
        writers.emplace_back([&buffer, writer] {
            for (int game = 0; game < kGamesPerWriter; ++game) {
                const std::uint32_t game_id = static_cast<std::uint32_t>(writer * 100 + game);
                buffer.add_game(make_game(game_id, kPositionsPerGame));
            }
        });
    }

    for (auto& writer : writers) {
        writer.join();
    }
    stop_reader.store(true, std::memory_order_release);
    reader.join();

    EXPECT_GT(read_iterations.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(buffer.size(), kTotalPositions);

    const std::vector<ReplayPosition> all_positions = buffer.sample(kTotalPositions);
    ASSERT_EQ(all_positions.size(), kTotalPositions);

    std::unordered_set<std::uint64_t> observed_signatures;
    observed_signatures.reserve(kTotalPositions * 2U);
    for (const ReplayPosition& position : all_positions) {
        expect_position_is_consistent(position);
        observed_signatures.insert(signature_of(position));
    }

    ASSERT_EQ(observed_signatures.size(), kTotalPositions);
    for (int writer = 0; writer < kWriterThreads; ++writer) {
        for (int game = 0; game < kGamesPerWriter; ++game) {
            const std::uint32_t game_id = static_cast<std::uint32_t>(writer * 100 + game);
            for (std::uint16_t move = 0U; move < kPositionsPerGame; ++move) {
                const std::uint64_t signature =
                    static_cast<std::uint64_t>(game_id) * 1000ULL + static_cast<std::uint64_t>(move);
                EXPECT_TRUE(observed_signatures.contains(signature));
            }
        }
    }
}

// WHY: Training batches depend on uniform sampling; repeated single-item draws should not systematically bias toward
// specific entries.
TEST(ReplayBufferTest, SamplesApproximatelyUniformlyForSingleDraws) {
    ReplayBuffer buffer(5U, 0xBADC0FFEEULL);
    for (std::uint32_t game_id = 10U; game_id < 15U; ++game_id) {
        buffer.add_game(make_game(game_id, 1U));
    }

    constexpr int kDrawCount = 50'000;
    std::array<int, 5> counts{0, 0, 0, 0, 0};
    for (int draw = 0; draw < kDrawCount; ++draw) {
        const std::vector<ReplayPosition> sampled = buffer.sample(1U);
        ASSERT_EQ(sampled.size(), 1U);
        const ReplayPosition& position = sampled.front();
        expect_position_is_consistent(position);
        const std::size_t bucket = static_cast<std::size_t>(position.game_id - 10U);
        ASSERT_LT(bucket, counts.size());
        ++counts[bucket];
    }

    const double expected = static_cast<double>(kDrawCount) / static_cast<double>(counts.size());
    for (const int count : counts) {
        const double relative_error = std::abs(static_cast<double>(count) - expected) / expected;
        EXPECT_LT(relative_error, 0.08);  // 8% tolerance keeps the test stable while catching major bias.
    }
}

// WHY: Early training can request a batch larger than current fill level; sampling should still return batch_size
// entries by drawing with replacement from available data.
TEST(ReplayBufferTest, SamplesWithReplacementWhenBatchExceedsCurrentSize) {
    ReplayBuffer buffer(3U, 7U);
    buffer.add_game(make_game(21U, 1U));
    buffer.add_game(make_game(22U, 1U));
    buffer.add_game(make_game(23U, 1U));

    const std::vector<ReplayPosition> batch = buffer.sample(10U);
    ASSERT_EQ(batch.size(), 10U);
    for (const ReplayPosition& position : batch) {
        expect_position_is_consistent(position);
        EXPECT_TRUE(position.game_id >= 21U && position.game_id <= 23U);
    }
}

// WHY: The packed sampler is the C++ hot path consumed by Python training; rows must stay internally consistent and
// preserve scalar/WDL targets while avoiding per-position Python unpacking.
TEST(ReplayBufferTest, SampleBatchPacksConsistentRowsForScalarAndWdlValues) {
    ReplayBuffer buffer(8U, 0xC0FFEEULL);
    buffer.add_game(make_game(31U, 2U));
    buffer.add_game(make_game(32U, 2U));
    buffer.add_game(make_game(33U, 2U));

    const alphazero::selfplay::SampledBatch scalar_batch = buffer.sample_batch(12U, kStateSize, kPolicySize, 1U);
    ASSERT_EQ(scalar_batch.batch_size, 12U);
    ASSERT_EQ(scalar_batch.states.size(), 12U * kStateSize);
    ASSERT_EQ(scalar_batch.policies.size(), 12U * kPolicySize);
    ASSERT_EQ(scalar_batch.values.size(), 12U);
    ASSERT_EQ(scalar_batch.weights.size(), 12U);

    for (std::size_t sample_index = 0U; sample_index < scalar_batch.batch_size; ++sample_index) {
        const float signature = scalar_batch.states[sample_index * kStateSize];
        const float state_move = scalar_batch.states[(sample_index * kStateSize) + 1U];
        const float policy_game = scalar_batch.policies[(sample_index * kPolicySize) + 1U];
        const float policy_move = scalar_batch.policies[(sample_index * kPolicySize) + 2U];
        const auto game_id = static_cast<std::uint32_t>(std::lround(policy_game));
        const auto move_number = static_cast<std::uint16_t>(std::lround(policy_move));
        const float expected_signature = static_cast<float>(game_id * 1000U + move_number);

        EXPECT_FLOAT_EQ(signature, expected_signature);
        EXPECT_FLOAT_EQ(state_move, static_cast<float>(move_number));
        EXPECT_FLOAT_EQ(scalar_batch.policies[sample_index * kPolicySize], expected_signature + 0.5F);
        EXPECT_FLOAT_EQ(scalar_batch.values[sample_index], scalar_value_for(game_id, move_number));
        EXPECT_FLOAT_EQ(scalar_batch.weights[sample_index], training_weight_for(game_id, move_number));
    }

    const alphazero::selfplay::SampledBatch wdl_batch =
        buffer.sample_batch(12U, kStateSize, kPolicySize, ReplayPosition::kWdlSize);
    ASSERT_EQ(wdl_batch.batch_size, 12U);
    ASSERT_EQ(wdl_batch.values.size(), 12U * ReplayPosition::kWdlSize);
    ASSERT_EQ(wdl_batch.weights.size(), 12U);

    for (std::size_t sample_index = 0U; sample_index < wdl_batch.batch_size; ++sample_index) {
        const float policy_game = wdl_batch.policies[(sample_index * kPolicySize) + 1U];
        const float policy_move = wdl_batch.policies[(sample_index * kPolicySize) + 2U];
        const auto game_id = static_cast<std::uint32_t>(std::lround(policy_game));
        const auto move_number = static_cast<std::uint16_t>(std::lround(policy_move));
        const auto expected_wdl = wdl_for(scalar_value_for(game_id, move_number));
        const std::size_t value_offset = sample_index * ReplayPosition::kWdlSize;

        EXPECT_FLOAT_EQ(wdl_batch.values[value_offset], expected_wdl[0]);
        EXPECT_FLOAT_EQ(wdl_batch.values[value_offset + 1U], expected_wdl[1]);
        EXPECT_FLOAT_EQ(wdl_batch.values[value_offset + 2U], expected_wdl[2]);
        EXPECT_FLOAT_EQ(wdl_batch.weights[sample_index], training_weight_for(game_id, move_number));
    }
}

// WHY: Packed sampling must fail fast for invalid target shapes/dimensions so training does not silently consume
// malformed tensors.
TEST(ReplayBufferTest, SampleBatchValidatesRequestedShapesAndDimensions) {
    ReplayBuffer buffer(4U, 0x1234ULL);
    buffer.add_game(make_game(9U, 1U));

    EXPECT_THROW(static_cast<void>(buffer.sample_batch(1U, 0U, kPolicySize, 1U)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(buffer.sample_batch(1U, kStateSize, 0U, 1U)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(buffer.sample_batch(1U, kStateSize, kPolicySize, 2U)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(buffer.sample_batch(1U, kStateSize + 1U, kPolicySize, 1U)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(buffer.sample_batch(1U, kStateSize, kPolicySize + 1U, 1U)), std::invalid_argument);

    ReplayBuffer empty_buffer(4U, 0x5678ULL);
    EXPECT_THROW(static_cast<void>(empty_buffer.sample_batch(1U, kStateSize, kPolicySize, 1U)), std::runtime_error);
}
