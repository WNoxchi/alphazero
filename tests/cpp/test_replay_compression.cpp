#include "selfplay/replay_compression.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::selfplay::StateCompressionLayout;
using alphazero::selfplay::compress_policy;
using alphazero::selfplay::compress_state;
using alphazero::selfplay::decompress_policy;
using alphazero::selfplay::decompress_state;
using alphazero::selfplay::float_to_fp16;
using alphazero::selfplay::fp16_to_float;

constexpr std::size_t kTotalPlanes = 119U;
constexpr std::size_t kSquaresPerPlane = 64U;
constexpr std::size_t kBinaryPlaneCount = 117U;
constexpr std::size_t kBinaryWordCount = 117U;
constexpr std::size_t kFloatPlaneCount = 2U;

[[nodiscard]] bool is_float_plane(const std::size_t plane) {
    return plane == 113U || plane == 118U;
}

[[nodiscard]] std::vector<float> make_dense_state() {
    std::vector<float> dense_state(kTotalPlanes * kSquaresPerPlane, 0.0F);
    for (std::size_t plane = 0U; plane < kTotalPlanes; ++plane) {
        const std::size_t base = plane * kSquaresPerPlane;
        if (plane == 113U) {
            std::fill_n(dense_state.data() + base, kSquaresPerPlane, 0.314F);
            continue;
        }
        if (plane == 118U) {
            std::fill_n(dense_state.data() + base, kSquaresPerPlane, 0.907F);
            continue;
        }

        for (std::size_t square = 0U; square < kSquaresPerPlane; ++square) {
            const std::size_t pattern = ((plane * 37U) + (square * 17U)) % 9U;
            if (pattern <= 2U) {
                dense_state[base + square] = 0.95F;
            } else if (pattern == 3U) {
                // Ensure threshold behavior at exactly 0.5.
                dense_state[base + square] = 0.5F;
            } else {
                dense_state[base + square] = 0.1F;
            }
        }
    }
    return dense_state;
}

}  // namespace

// WHY: Compact replay storage depends on exact binary-plane reconstruction and bounded quantization error on the two
// chess float planes; this verifies the core state roundtrip behavior before wiring into a full buffer.
TEST(ReplayCompressionTest, StateCompressionRoundtripPreservesBinaryAndQuantizedPlanes) {
    const std::vector<float> dense_state = make_dense_state();
    const std::array<std::size_t, kFloatPlaneCount> float_plane_indices{113U, 118U};
    std::array<std::uint64_t, kBinaryPlaneCount> bitpacked_planes{};
    std::array<std::uint8_t, kFloatPlaneCount> quantized_float_planes{};

    const StateCompressionLayout layout = compress_state(
        dense_state,
        float_plane_indices,
        kSquaresPerPlane,
        bitpacked_planes,
        quantized_float_planes);
    EXPECT_EQ(layout.num_binary_words, kBinaryWordCount);
    EXPECT_EQ(layout.num_binary_planes, kBinaryPlaneCount);
    EXPECT_EQ(layout.num_float_planes, kFloatPlaneCount);

    std::vector<float> restored_state(kTotalPlanes * kSquaresPerPlane, -1.0F);
    decompress_state(
        bitpacked_planes,
        quantized_float_planes,
        float_plane_indices,
        kSquaresPerPlane,
        restored_state);

    for (std::size_t plane = 0U; plane < kTotalPlanes; ++plane) {
        const std::size_t base = plane * kSquaresPerPlane;
        if (is_float_plane(plane)) {
            const float expected = std::round(std::clamp(dense_state[base], 0.0F, 1.0F) * 255.0F) / 255.0F;
            for (std::size_t square = 0U; square < kSquaresPerPlane; ++square) {
                EXPECT_NEAR(restored_state[base + square], expected, 1.0F / 255.0F);
            }
            continue;
        }

        for (std::size_t square = 0U; square < kSquaresPerPlane; ++square) {
            const float expected = dense_state[base + square] >= 0.5F ? 1.0F : 0.0F;
            EXPECT_FLOAT_EQ(restored_state[base + square], expected);
        }
    }
}

// WHY: Go uses 19x19 inputs (361 squares), which require six 64-bit words per binary plane; this guards
// multi-word packing/unpacking, including high-index bits in the sixth word.
TEST(ReplayCompressionTest, StateCompressionSupportsMultiWordPlanesForNineteenByNineteenBoards) {
    constexpr std::size_t kGoSquaresPerPlane = 361U;
    constexpr std::size_t kGoPlanes = 3U;
    constexpr std::size_t kGoWordsPerPlane = 6U;
    constexpr std::size_t kGoBinaryWords = kGoPlanes * kGoWordsPerPlane;

    std::vector<float> dense_state(kGoPlanes * kGoSquaresPerPlane, 0.0F);
    for (std::size_t square = 0U; square < kGoSquaresPerPlane; ++square) {
        if ((square % 2U) == 0U) {
            dense_state[square] = 1.0F;
        }
    }
    std::fill_n(
        dense_state.data() + kGoSquaresPerPlane,
        kGoSquaresPerPlane,
        1.0F);
    dense_state[(2U * kGoSquaresPerPlane) + 0U] = 1.0F;
    dense_state[(2U * kGoSquaresPerPlane) + 320U] = 1.0F;
    dense_state[(2U * kGoSquaresPerPlane) + 359U] = 1.0F;
    dense_state[(2U * kGoSquaresPerPlane) + 360U] = 1.0F;

    std::array<std::uint64_t, kGoBinaryWords> bitpacked_planes{};
    std::array<std::uint8_t, 0U> quantized_float_planes{};

    const StateCompressionLayout layout = compress_state(
        dense_state,
        std::span<const std::size_t>{},
        kGoSquaresPerPlane,
        bitpacked_planes,
        quantized_float_planes);

    EXPECT_EQ(layout.num_binary_words, kGoBinaryWords);
    EXPECT_EQ(layout.num_binary_planes, kGoPlanes);
    EXPECT_EQ(layout.num_float_planes, 0U);

    EXPECT_EQ(bitpacked_planes[kGoWordsPerPlane + 4U], std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(
        bitpacked_planes[kGoWordsPerPlane + 5U],
        (std::uint64_t{1} << 41U) - std::uint64_t{1});
    EXPECT_NE(bitpacked_planes[(2U * kGoWordsPerPlane) + 5U], 0U);

    std::vector<float> restored_state(dense_state.size(), -1.0F);
    decompress_state(
        bitpacked_planes,
        quantized_float_planes,
        std::span<const std::size_t>{},
        kGoSquaresPerPlane,
        restored_state);

    for (std::size_t square = 0U; square < kGoSquaresPerPlane; ++square) {
        const float expected = (square % 2U) == 0U ? 1.0F : 0.0F;
        EXPECT_FLOAT_EQ(restored_state[square], expected);
        EXPECT_FLOAT_EQ(restored_state[kGoSquaresPerPlane + square], 1.0F);
    }
    for (std::size_t square = 0U; square < kGoSquaresPerPlane; ++square) {
        const bool on_bit = square == 0U || square == 320U || square == 359U || square == 360U;
        EXPECT_FLOAT_EQ(restored_state[(2U * kGoSquaresPerPlane) + square], on_bit ? 1.0F : 0.0F);
    }
}

// WHY: Sparse policy storage is only useful if it keeps the highest-probability actions and preserves retained mass
// within fp16 precision limits, so training targets stay faithful after compression.
TEST(ReplayCompressionTest, PolicyCompressionKeepsTopEntriesAndRoundtripsFp16Values) {
    std::vector<float> dense_policy(128U, 0.0F);
    for (std::size_t index = 0U; index < 80U; ++index) {
        dense_policy[index] = static_cast<float>(80U - index) / 100.0F;
    }
    dense_policy[120U] = std::numeric_limits<float>::quiet_NaN();
    dense_policy[121U] = -0.25F;

    std::array<std::uint16_t, 64U> actions{};
    std::array<std::uint16_t, 64U> probs_fp16{};
    const std::uint8_t count = compress_policy(dense_policy, actions, probs_fp16);
    ASSERT_EQ(count, 64U);

    for (std::size_t i = 0U; i < count; ++i) {
        EXPECT_EQ(actions[i], i);
    }

    std::vector<float> restored_policy(dense_policy.size(), -1.0F);
    decompress_policy(actions, probs_fp16, count, restored_policy);

    for (std::size_t i = 0U; i < 64U; ++i) {
        EXPECT_NEAR(restored_policy[i], dense_policy[i], 1.0e-3F);
    }
    for (std::size_t i = 64U; i < restored_policy.size(); ++i) {
        EXPECT_FLOAT_EQ(restored_policy[i], 0.0F);
    }

    const float original_topk_sum =
        std::accumulate(dense_policy.begin(), dense_policy.begin() + 64, 0.0F);
    const float restored_sum = std::accumulate(restored_policy.begin(), restored_policy.end(), 0.0F);
    EXPECT_NEAR(restored_sum, original_topk_sum, 2.0e-2F);
}

// WHY: Zero-policy rows can occur in defensive paths, and compression must produce an explicit empty sparse payload.
TEST(ReplayCompressionTest, PolicyCompressionHandlesAllZeroPolicy) {
    const std::vector<float> dense_policy(4672U, 0.0F);
    std::array<std::uint16_t, 64U> actions{};
    std::array<std::uint16_t, 64U> probs_fp16{};

    const std::uint8_t count = compress_policy(dense_policy, actions, probs_fp16);
    EXPECT_EQ(count, 0U);
    EXPECT_TRUE(std::all_of(actions.begin(), actions.end(), [](std::uint16_t value) { return value == 0U; }));
    EXPECT_TRUE(std::all_of(probs_fp16.begin(), probs_fp16.end(), [](std::uint16_t value) { return value == 0U; }));
}

// WHY: FP16 conversion is shared by policy compression/decompression; bounded roundtrip error here protects sparse
// replay values from unexpected drift.
TEST(ReplayCompressionTest, Float16ConversionRoundtripStaysWithinExpectedTolerance) {
    const std::array<float, 6U> values{
        0.0F,
        0.03125F,
        0.1F,
        0.33333334F,
        0.5F,
        1.0F,
    };

    for (const float value : values) {
        const std::uint16_t half = float_to_fp16(value);
        const float restored = fp16_to_float(half);
        EXPECT_NEAR(restored, value, 1.0e-3F);
    }
}

// WHY: Detecting malformed sparse payloads prevents silent corruption if checkpoint data is truncated or duplicated.
TEST(ReplayCompressionTest, PolicyDecompressionRejectsDuplicateActions) {
    const std::array<std::uint16_t, 2U> actions{4U, 4U};
    const std::array<std::uint16_t, 2U> probs_fp16{
        float_to_fp16(0.5F),
        float_to_fp16(0.5F),
    };
    std::vector<float> dense_policy(16U, 0.0F);

    EXPECT_THROW(
        decompress_policy(actions, probs_fp16, /*num_entries=*/2U, dense_policy),
        std::invalid_argument);
}
