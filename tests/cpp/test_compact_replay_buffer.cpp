#include "selfplay/compact_replay_buffer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::selfplay::CompactReplayBuffer;
using alphazero::selfplay::ReplayPosition;
using alphazero::selfplay::SamplingStrategy;
using alphazero::selfplay::SampledBatch;

constexpr std::size_t kSquaresPerPlane = 64U;
constexpr std::size_t kTotalPlanes = 4U;
constexpr std::size_t kFloatPlaneIndex = 1U;
constexpr std::size_t kNumBinaryPlanes = 3U;
constexpr std::size_t kNumFloatPlanes = 1U;
constexpr std::size_t kStateSize = kTotalPlanes * kSquaresPerPlane;
constexpr std::size_t kPolicySize = 10U;

enum class PolicyVariant {
    kDense,
    kAllZero,
    kSingleMove,
};

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
    const std::uint32_t bucket = ((game_id * 17U) + move_number) % 9U;
    return 0.2F + (static_cast<float>(bucket) * 0.1F);
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

[[nodiscard]] float make_binary_plane_value(
    const std::size_t plane,
    const std::size_t square,
    const std::uint32_t game_id,
    const std::uint16_t move_number) {
    const std::size_t pattern =
        (plane * 19U) + (square * 7U) + static_cast<std::size_t>(game_id) + static_cast<std::size_t>(move_number);
    if ((pattern % 5U) == 0U) {
        return 0.5F;
    }
    return (pattern % 2U) == 0U ? 0.9F : 0.1F;
}

[[nodiscard]] float make_float_plane_value(const std::uint32_t game_id, const std::uint16_t move_number) {
    const std::size_t numerator = ((static_cast<std::size_t>(game_id) * 11U) + move_number) % 17U;
    return static_cast<float>(numerator) / 16.0F;
}

[[nodiscard]] std::vector<float> make_state(const std::uint32_t game_id, const std::uint16_t move_number) {
    std::vector<float> state(kStateSize, 0.0F);
    for (std::size_t plane = 0U; plane < kTotalPlanes; ++plane) {
        const std::size_t base = plane * kSquaresPerPlane;
        if (plane == kFloatPlaneIndex) {
            const float value = make_float_plane_value(game_id, move_number);
            std::fill_n(state.data() + base, kSquaresPerPlane, value);
            continue;
        }

        for (std::size_t square = 0U; square < kSquaresPerPlane; ++square) {
            state[base + square] = make_binary_plane_value(plane, square, game_id, move_number);
        }
    }
    return state;
}

[[nodiscard]] std::vector<float> make_policy(
    const std::uint32_t game_id,
    const std::uint16_t move_number,
    const PolicyVariant variant) {
    std::vector<float> policy(kPolicySize, 0.0F);
    if (variant == PolicyVariant::kAllZero) {
        return policy;
    }
    if (variant == PolicyVariant::kSingleMove) {
        policy[7] = 1.0F;
        return policy;
    }

    const float offset = static_cast<float>((game_id + move_number) % 10U) * 0.001F;
    policy[0] = 0.41F + offset;
    policy[2] = 0.29F + offset;
    policy[5] = 0.20F + offset;
    policy[9] = 0.10F + offset;
    return policy;
}

[[nodiscard]] ReplayPosition make_position(
    const std::uint32_t game_id,
    const std::uint16_t move_number,
    const PolicyVariant variant = PolicyVariant::kDense) {
    const std::vector<float> state = make_state(game_id, move_number);
    const std::vector<float> policy = make_policy(game_id, move_number, variant);
    const float value = scalar_value_for(game_id, move_number);
    return ReplayPosition::make(
        state,
        policy,
        value,
        wdl_for(value),
        game_id,
        move_number,
        training_weight_for(game_id, move_number));
}

[[nodiscard]] std::vector<ReplayPosition> make_game(
    const std::uint32_t game_id,
    const std::uint16_t move_count,
    const PolicyVariant variant = PolicyVariant::kDense) {
    std::vector<ReplayPosition> positions;
    positions.reserve(move_count);
    for (std::uint16_t move = 0U; move < move_count; ++move) {
        positions.push_back(make_position(game_id, move, variant));
    }
    return positions;
}

[[nodiscard]] std::uint64_t signature_of(const ReplayPosition& position) {
    return (static_cast<std::uint64_t>(position.game_id) * 1000ULL) + static_cast<std::uint64_t>(position.move_number);
}

void expect_roundtrip_match(const ReplayPosition& expected, const ReplayPosition& observed) {
    ASSERT_EQ(observed.encoded_state_size, kStateSize);
    ASSERT_EQ(observed.policy_size, kPolicySize);
    EXPECT_EQ(observed.game_id, expected.game_id);
    EXPECT_EQ(observed.move_number, expected.move_number);
    EXPECT_FLOAT_EQ(observed.value, expected.value);
    EXPECT_FLOAT_EQ(observed.training_weight, expected.training_weight);
    EXPECT_EQ(observed.value_wdl, expected.value_wdl);

    for (std::size_t plane = 0U; plane < kTotalPlanes; ++plane) {
        const std::size_t base = plane * kSquaresPerPlane;
        if (plane == kFloatPlaneIndex) {
            const float quantized = std::round(std::clamp(expected.encoded_state[base], 0.0F, 1.0F) * 255.0F) / 255.0F;
            for (std::size_t square = 0U; square < kSquaresPerPlane; ++square) {
                EXPECT_NEAR(observed.encoded_state[base + square], quantized, 1.0F / 255.0F);
            }
            continue;
        }

        for (std::size_t square = 0U; square < kSquaresPerPlane; ++square) {
            const float binary = expected.encoded_state[base + square] >= 0.5F ? 1.0F : 0.0F;
            EXPECT_FLOAT_EQ(observed.encoded_state[base + square], binary);
        }
    }

    for (std::size_t action = 0U; action < kPolicySize; ++action) {
        EXPECT_NEAR(observed.policy[action], expected.policy[action], 1.0e-3F);
    }
}

}  // namespace

// WHY: CompactReplayBuffer must losslessly preserve metadata and binary planes while bounding quantization/fp16
// error so downstream training receives stable targets.
TEST(CompactReplayBufferTest, RoundtripSamplePreservesPayloadWithinCompressionTolerance) {
    CompactReplayBuffer buffer(
        /*capacity=*/16U,
        /*num_binary_planes=*/kNumBinaryPlanes,
        /*num_float_planes=*/kNumFloatPlanes,
        /*float_plane_indices=*/{kFloatPlaneIndex},
        /*full_policy_size=*/kPolicySize,
        /*random_seed=*/1234U);

    const ReplayPosition first = make_position(11U, 2U, PolicyVariant::kDense);
    const ReplayPosition second = make_position(12U, 3U, PolicyVariant::kDense);
    buffer.add_game({first, second});

    const std::vector<ReplayPosition> sampled = buffer.sample(2U);
    ASSERT_EQ(sampled.size(), 2U);

    std::unordered_set<std::uint64_t> observed_signatures;
    for (const ReplayPosition& position : sampled) {
        const std::uint64_t signature = signature_of(position);
        observed_signatures.insert(signature);
        if (signature == signature_of(first)) {
            expect_roundtrip_match(first, position);
        } else if (signature == signature_of(second)) {
            expect_roundtrip_match(second, position);
        } else {
            FAIL() << "Unexpected sampled signature: " << signature;
        }
    }
    EXPECT_EQ(observed_signatures.size(), 2U);
}

// WHY: Replay semantics require fixed-capacity ring behavior; once full, oldest positions must be evicted.
TEST(CompactReplayBufferTest, RingBufferRetainsMostRecentPositionsAfterWrap) {
    CompactReplayBuffer buffer(
        /*capacity=*/3U,
        /*num_binary_planes=*/kNumBinaryPlanes,
        /*num_float_planes=*/kNumFloatPlanes,
        /*float_plane_indices=*/{kFloatPlaneIndex},
        /*full_policy_size=*/kPolicySize,
        /*random_seed=*/9988U);

    for (std::uint32_t game_id = 0U; game_id < 5U; ++game_id) {
        buffer.add_game(make_game(game_id, 1U));
    }

    EXPECT_EQ(buffer.size(), 3U);
    EXPECT_EQ(buffer.write_head(), 2U);

    const std::vector<ReplayPosition> sampled = buffer.sample(3U);
    ASSERT_EQ(sampled.size(), 3U);

    std::unordered_set<std::uint32_t> game_ids;
    for (const ReplayPosition& position : sampled) {
        game_ids.insert(position.game_id);
    }
    EXPECT_EQ(game_ids.size(), 3U);
    EXPECT_TRUE(game_ids.contains(2U));
    EXPECT_TRUE(game_ids.contains(3U));
    EXPECT_TRUE(game_ids.contains(4U));
}

// WHY: Edge policies (all-zero and single legal move) appear in defensive paths; compression must not corrupt them.
TEST(CompactReplayBufferTest, SupportsZeroAndSingleMovePolicies) {
    CompactReplayBuffer buffer(
        /*capacity=*/8U,
        /*num_binary_planes=*/kNumBinaryPlanes,
        /*num_float_planes=*/kNumFloatPlanes,
        /*float_plane_indices=*/{kFloatPlaneIndex},
        /*full_policy_size=*/kPolicySize,
        /*random_seed=*/5533U);

    const ReplayPosition zero_policy = make_position(30U, 0U, PolicyVariant::kAllZero);
    const ReplayPosition single_policy = make_position(31U, 0U, PolicyVariant::kSingleMove);
    buffer.add_game({zero_policy, single_policy});

    const std::vector<ReplayPosition> sampled = buffer.sample(2U);
    ASSERT_EQ(sampled.size(), 2U);

    bool saw_zero_policy = false;
    bool saw_single_policy = false;
    for (const ReplayPosition& position : sampled) {
        if (position.game_id == 30U) {
            saw_zero_policy = true;
            for (std::size_t action = 0U; action < kPolicySize; ++action) {
                EXPECT_FLOAT_EQ(position.policy[action], 0.0F);
            }
        } else if (position.game_id == 31U) {
            saw_single_policy = true;
            for (std::size_t action = 0U; action < kPolicySize; ++action) {
                const float expected = action == 7U ? 1.0F : 0.0F;
                EXPECT_NEAR(position.policy[action], expected, 1.0e-3F);
            }
        }
    }

    EXPECT_TRUE(saw_zero_policy);
    EXPECT_TRUE(saw_single_policy);
}

// WHY: Recency-weighted sampling is intended to prioritize fresh self-play data to improve policy freshness.
TEST(CompactReplayBufferTest, RecencyWeightedSamplingBiasesTowardNewestEntries) {
    constexpr std::size_t kPopulation = 100U;
    constexpr std::size_t kSampleCount = 20'000U;
    constexpr std::size_t kQuartileWidth = kPopulation / 4U;

    CompactReplayBuffer buffer(
        /*capacity=*/kPopulation,
        /*num_binary_planes=*/kNumBinaryPlanes,
        /*num_float_planes=*/kNumFloatPlanes,
        /*float_plane_indices=*/{kFloatPlaneIndex},
        /*full_policy_size=*/kPolicySize,
        /*random_seed=*/24680U,
        /*sampling_strategy=*/SamplingStrategy::kRecencyWeighted,
        /*recency_weight_lambda=*/4.0F);

    for (std::uint32_t game_id = 0U; game_id < kPopulation; ++game_id) {
        buffer.add_game(make_game(game_id, 1U));
    }

    const std::vector<ReplayPosition> sampled = buffer.sample(kSampleCount);
    ASSERT_EQ(sampled.size(), kSampleCount);

    std::array<std::size_t, kPopulation> counts{};
    for (const ReplayPosition& position : sampled) {
        ASSERT_LT(position.game_id, kPopulation);
        ++counts[position.game_id];
    }

    std::size_t oldest_quartile = 0U;
    for (std::size_t i = 0U; i < kQuartileWidth; ++i) {
        oldest_quartile += counts[i];
    }
    std::size_t newest_quartile = 0U;
    for (std::size_t i = kPopulation - kQuartileWidth; i < kPopulation; ++i) {
        newest_quartile += counts[i];
    }

    EXPECT_GT(newest_quartile, oldest_quartile * 2U);
    EXPECT_GT(counts[kPopulation - 1U], counts[0U]);
}

// WHY: Invalid recency-weight hyperparameters should fail at construction to prevent undefined sampling behavior.
TEST(CompactReplayBufferTest, RejectsNegativeRecencyWeightLambda) {
    EXPECT_THROW(
        static_cast<void>(CompactReplayBuffer(
            /*capacity=*/8U,
            /*num_binary_planes=*/kNumBinaryPlanes,
            /*num_float_planes=*/kNumFloatPlanes,
            /*float_plane_indices=*/{kFloatPlaneIndex},
            /*full_policy_size=*/kPolicySize,
            /*random_seed=*/1234U,
            /*sampling_strategy=*/SamplingStrategy::kRecencyWeighted,
            /*recency_weight_lambda=*/-0.01F)),
        std::invalid_argument);
}

// WHY: Self-play writers and training readers run concurrently; this ensures compact storage remains thread-safe.
TEST(CompactReplayBufferTest, ConcurrentWritesAndSamplingRemainConsistent) {
    constexpr int kWriterThreads = 4;
    constexpr int kGamesPerWriter = 20;
    constexpr std::uint16_t kPositionsPerGame = 2U;
    constexpr std::size_t kTotalPositions =
        static_cast<std::size_t>(kWriterThreads) * static_cast<std::size_t>(kGamesPerWriter) *
        static_cast<std::size_t>(kPositionsPerGame);

    CompactReplayBuffer buffer(
        /*capacity=*/kTotalPositions,
        /*num_binary_planes=*/kNumBinaryPlanes,
        /*num_float_planes=*/kNumFloatPlanes,
        /*float_plane_indices=*/{kFloatPlaneIndex},
        /*full_policy_size=*/kPolicySize,
        /*random_seed=*/7777U);

    std::atomic<bool> stop_reader{false};
    std::atomic<int> reader_iterations{0};
    std::thread reader([&] {
        while (!stop_reader.load(std::memory_order_acquire)) {
            const std::size_t current_size = buffer.size();
            if (current_size == 0U) {
                std::this_thread::yield();
                continue;
            }

            const std::size_t batch_size = std::min<std::size_t>(8U, current_size);
            const SampledBatch batch = buffer.sample_batch(batch_size, kStateSize, kPolicySize, 1U);
            EXPECT_EQ(batch.states.size(), batch.batch_size * kStateSize);
            EXPECT_EQ(batch.policies.size(), batch.batch_size * kPolicySize);
            EXPECT_EQ(batch.values.size(), batch.batch_size);
            EXPECT_EQ(batch.weights.size(), batch.batch_size);
            reader_iterations.fetch_add(1, std::memory_order_relaxed);
        }
    });

    std::vector<std::thread> writers;
    writers.reserve(kWriterThreads);
    for (int writer = 0; writer < kWriterThreads; ++writer) {
        writers.emplace_back([&buffer, writer] {
            for (int game = 0; game < kGamesPerWriter; ++game) {
                const std::uint32_t game_id = static_cast<std::uint32_t>(writer * 100U + game);
                buffer.add_game(make_game(game_id, kPositionsPerGame));
            }
        });
    }

    for (auto& writer : writers) {
        writer.join();
    }
    stop_reader.store(true, std::memory_order_release);
    reader.join();

    EXPECT_GT(reader_iterations.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(buffer.size(), kTotalPositions);

    const std::vector<ReplayPosition> sampled = buffer.sample(kTotalPositions);
    ASSERT_EQ(sampled.size(), kTotalPositions);

    std::unordered_set<std::uint64_t> signatures;
    signatures.reserve(kTotalPositions * 2U);
    for (const ReplayPosition& position : sampled) {
        signatures.insert(signature_of(position));
    }
    EXPECT_EQ(signatures.size(), kTotalPositions);
}

// WHY: Checkpoint I/O depends on export/import compatibility; compact buffers must roundtrip flat arrays safely.
TEST(CompactReplayBufferTest, ExportImportRoundtripPreservesLogicalContents) {
    CompactReplayBuffer source(
        /*capacity=*/16U,
        /*num_binary_planes=*/kNumBinaryPlanes,
        /*num_float_planes=*/kNumFloatPlanes,
        /*float_plane_indices=*/{kFloatPlaneIndex},
        /*full_policy_size=*/kPolicySize,
        /*random_seed=*/9090U);
    source.add_game(make_game(40U, 3U));
    source.add_game(make_game(41U, 2U));

    const std::size_t count = source.size();
    ASSERT_GT(count, 0U);

    std::vector<float> states(count * kStateSize, 0.0F);
    std::vector<float> policies(count * kPolicySize, 0.0F);
    std::vector<float> values_wdl(count * ReplayPosition::kWdlSize, 0.0F);
    std::vector<std::uint32_t> game_ids(count, 0U);
    std::vector<std::uint16_t> move_numbers(count, 0U);

    const std::size_t exported = source.export_positions(
        states.data(),
        policies.data(),
        values_wdl.data(),
        game_ids.data(),
        move_numbers.data(),
        kStateSize,
        kPolicySize);
    ASSERT_EQ(exported, count);

    CompactReplayBuffer restored(
        /*capacity=*/16U,
        /*num_binary_planes=*/kNumBinaryPlanes,
        /*num_float_planes=*/kNumFloatPlanes,
        /*float_plane_indices=*/{kFloatPlaneIndex},
        /*full_policy_size=*/kPolicySize,
        /*random_seed=*/123U);
    restored.import_positions(
        states.data(),
        policies.data(),
        values_wdl.data(),
        game_ids.data(),
        move_numbers.data(),
        count,
        kStateSize,
        kPolicySize);

    std::vector<float> states_again(count * kStateSize, 0.0F);
    std::vector<float> policies_again(count * kPolicySize, 0.0F);
    std::vector<float> values_wdl_again(count * ReplayPosition::kWdlSize, 0.0F);
    std::vector<std::uint32_t> game_ids_again(count, 0U);
    std::vector<std::uint16_t> move_numbers_again(count, 0U);

    const std::size_t exported_again = restored.export_positions(
        states_again.data(),
        policies_again.data(),
        values_wdl_again.data(),
        game_ids_again.data(),
        move_numbers_again.data(),
        kStateSize,
        kPolicySize);
    ASSERT_EQ(exported_again, count);

    for (std::size_t i = 0U; i < states.size(); ++i) {
        EXPECT_NEAR(states_again[i], states[i], 1.0F / 255.0F);
    }
    for (std::size_t i = 0U; i < policies.size(); ++i) {
        EXPECT_NEAR(policies_again[i], policies[i], 1.0e-3F);
    }
    for (std::size_t i = 0U; i < values_wdl.size(); ++i) {
        EXPECT_FLOAT_EQ(values_wdl_again[i], values_wdl[i]);
    }
    EXPECT_EQ(game_ids_again, game_ids);
    EXPECT_EQ(move_numbers_again, move_numbers);
}
