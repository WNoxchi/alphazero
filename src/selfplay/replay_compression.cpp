#include "selfplay/replay_compression.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace alphazero::selfplay {
namespace {

[[nodiscard]] std::vector<bool> make_float_plane_mask(
    const std::size_t total_planes,
    const std::span<const std::size_t> float_plane_indices) {
    std::vector<bool> mask(total_planes, false);
    for (const std::size_t plane_index : float_plane_indices) {
        if (plane_index >= total_planes) {
            throw std::invalid_argument("Float plane index is out of range for encoded state");
        }
        if (mask[plane_index]) {
            throw std::invalid_argument("Float plane indices must be unique");
        }
        mask[plane_index] = true;
    }
    return mask;
}

}  // namespace

StateCompressionLayout compress_state(
    const std::span<const float> dense_state,
    const std::span<const std::size_t> float_plane_indices,
    const std::size_t squares_per_plane,
    const std::span<std::uint64_t> out_bitpacked_planes,
    const std::span<std::uint8_t> out_quantized_float_planes) {
    if (dense_state.empty()) {
        throw std::invalid_argument("compress_state requires a non-empty dense state");
    }
    if (squares_per_plane == 0U) {
        throw std::invalid_argument("compress_state squares_per_plane must be greater than zero");
    }
    if (dense_state.size() % squares_per_plane != 0U) {
        throw std::invalid_argument("compress_state dense state must be a multiple of squares_per_plane");
    }
    if (float_plane_indices.size() > out_quantized_float_planes.size()) {
        throw std::invalid_argument("compress_state float-plane output capacity is too small");
    }

    const std::size_t total_planes = dense_state.size() / squares_per_plane;
    if (float_plane_indices.size() > total_planes) {
        throw std::invalid_argument("compress_state has more float planes than total planes");
    }

    const std::size_t required_binary_planes = total_planes - float_plane_indices.size();
    const std::size_t words_per_plane = (squares_per_plane + 63U) / 64U;
    const std::size_t required_binary_words = required_binary_planes * words_per_plane;
    if (required_binary_words > out_bitpacked_planes.size()) {
        throw std::invalid_argument("compress_state binary-plane output capacity is too small");
    }

    std::fill(out_bitpacked_planes.begin(), out_bitpacked_planes.end(), 0U);
    std::fill(out_quantized_float_planes.begin(), out_quantized_float_planes.end(), 0U);

    const std::vector<bool> float_plane_mask = make_float_plane_mask(total_planes, float_plane_indices);

    std::size_t binary_index = 0U;
    std::size_t float_index = 0U;
    for (std::size_t plane = 0U; plane < total_planes; ++plane) {
        const std::size_t plane_offset = plane * squares_per_plane;
        if (float_plane_mask[plane]) {
            const float clamped_value = std::clamp(dense_state[plane_offset], 0.0F, 1.0F);
            const long quantized = std::lround(clamped_value * 255.0F);
            out_quantized_float_planes[float_index++] = static_cast<std::uint8_t>(quantized);
            continue;
        }

        for (std::size_t word = 0U; word < words_per_plane; ++word) {
            std::uint64_t bits = 0U;
            const std::size_t bit_start = word * 64U;
            const std::size_t bit_end = std::min(bit_start + 64U, squares_per_plane);
            for (std::size_t square = bit_start; square < bit_end; ++square) {
                if (dense_state[plane_offset + square] >= 0.5F) {
                    bits |= (std::uint64_t{1} << (square - bit_start));
                }
            }
            out_bitpacked_planes[binary_index++] = bits;
        }
    }

    return StateCompressionLayout{
        .num_binary_words = binary_index,
        .num_binary_planes = required_binary_planes,
        .num_float_planes = float_index,
    };
}

void decompress_state(
    const std::span<const std::uint64_t> bitpacked_planes,
    const std::span<const std::uint8_t> quantized_float_planes,
    const std::span<const std::size_t> float_plane_indices,
    const std::size_t squares_per_plane,
    const std::span<float> out_dense_state) {
    if (squares_per_plane == 0U) {
        throw std::invalid_argument("decompress_state squares_per_plane must be greater than zero");
    }
    if (float_plane_indices.size() != quantized_float_planes.size()) {
        throw std::invalid_argument("decompress_state float-plane index count must match quantized plane count");
    }
    const std::size_t words_per_plane = (squares_per_plane + 63U) / 64U;
    if (words_per_plane == 0U || (bitpacked_planes.size() % words_per_plane) != 0U) {
        throw std::invalid_argument(
            "decompress_state binary word count must align with squares_per_plane");
    }
    const std::size_t num_binary_planes = bitpacked_planes.size() / words_per_plane;
    const std::size_t total_planes = num_binary_planes + quantized_float_planes.size();
    if (total_planes == 0U) {
        if (out_dense_state.empty()) {
            return;
        }
        throw std::invalid_argument("decompress_state output must be empty when no planes are provided");
    }
    if (out_dense_state.size() != total_planes * squares_per_plane) {
        throw std::invalid_argument("decompress_state output size does not match plane counts");
    }

    const std::vector<bool> float_plane_mask = make_float_plane_mask(total_planes, float_plane_indices);

    std::size_t binary_index = 0U;
    std::size_t float_index = 0U;
    for (std::size_t plane = 0U; plane < total_planes; ++plane) {
        const std::size_t plane_offset = plane * squares_per_plane;
        if (float_plane_mask[plane]) {
            const float value = static_cast<float>(quantized_float_planes[float_index++]) / 255.0F;
            std::fill_n(out_dense_state.data() + plane_offset, squares_per_plane, value);
            continue;
        }

        for (std::size_t word = 0U; word < words_per_plane; ++word) {
            const std::uint64_t bits = bitpacked_planes[binary_index++];
            const std::size_t bit_start = word * 64U;
            const std::size_t bit_end = std::min(bit_start + 64U, squares_per_plane);
            for (std::size_t square = bit_start; square < bit_end; ++square) {
                out_dense_state[plane_offset + square] =
                    ((bits >> (square - bit_start)) & std::uint64_t{1}) != 0U ? 1.0F : 0.0F;
            }
        }
    }
}

std::uint16_t float_to_fp16(const float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
    std::uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa != 0U) {
            return static_cast<std::uint16_t>(sign | 0x7E00U);
        }
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }

    const std::int32_t half_exponent = static_cast<std::int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    }

    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return sign;
        }

        mantissa |= 0x800000U;
        const std::int32_t shift = 14 - half_exponent;
        std::uint16_t half_mantissa = static_cast<std::uint16_t>(mantissa >> shift);
        const std::uint32_t remainder = mantissa & ((std::uint32_t{1} << shift) - 1U);
        const std::uint32_t halfway = std::uint32_t{1} << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 0x1U) != 0U)) {
            ++half_mantissa;
        }

        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint16_t half_exponent_bits = static_cast<std::uint16_t>(half_exponent) << 10U;
    std::uint16_t half_mantissa = static_cast<std::uint16_t>(mantissa >> 13U);
    const std::uint32_t remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 0x1U) != 0U)) {
        ++half_mantissa;
        if (half_mantissa == 0x400U) {
            half_mantissa = 0U;
            half_exponent_bits = static_cast<std::uint16_t>(half_exponent_bits + 0x400U);
            if (half_exponent_bits >= 0x7C00U) {
                return static_cast<std::uint16_t>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<std::uint16_t>(sign | half_exponent_bits | half_mantissa);
}

float fp16_to_float(const std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1FU;
    const std::uint32_t mantissa = value & 0x03FFU;

    std::uint32_t bits = 0U;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            std::uint32_t normalized_mantissa = mantissa;
            std::int32_t normalized_exponent = 127 - 15 + 1;
            while ((normalized_mantissa & 0x0400U) == 0U) {
                normalized_mantissa <<= 1U;
                --normalized_exponent;
            }
            normalized_mantissa &= 0x03FFU;
            bits = sign | (static_cast<std::uint32_t>(normalized_exponent) << 23U) | (normalized_mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        const std::uint32_t float_exponent = (exponent + (127U - 15U)) << 23U;
        bits = sign | float_exponent | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

std::uint8_t compress_policy(
    const std::span<const float> dense_policy,
    const std::span<std::uint16_t> out_actions,
    const std::span<std::uint16_t> out_probs_fp16) {
    if (out_actions.size() != out_probs_fp16.size()) {
        throw std::invalid_argument("compress_policy action and probability arrays must have equal size");
    }
    if (dense_policy.size() > (static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1U)) {
        throw std::invalid_argument("compress_policy policy size exceeds uint16 action-index capacity");
    }

    std::fill(out_actions.begin(), out_actions.end(), 0U);
    std::fill(out_probs_fp16.begin(), out_probs_fp16.end(), 0U);

    struct PolicyEntry {
        std::uint16_t action = 0U;
        float probability = 0.0F;
    };

    std::vector<PolicyEntry> non_zero_entries;
    non_zero_entries.reserve(dense_policy.size());
    for (std::size_t action = 0U; action < dense_policy.size(); ++action) {
        const float probability = dense_policy[action];
        if (std::isfinite(probability) && probability > 0.0F) {
            non_zero_entries.push_back(
                PolicyEntry{.action = static_cast<std::uint16_t>(action), .probability = probability});
        }
    }

    if (non_zero_entries.empty() || out_actions.empty()) {
        return 0U;
    }

    std::sort(
        non_zero_entries.begin(),
        non_zero_entries.end(),
        [](const PolicyEntry& lhs, const PolicyEntry& rhs) {
            if (lhs.probability == rhs.probability) {
                return lhs.action < rhs.action;
            }
            return lhs.probability > rhs.probability;
        });

    const std::size_t count = std::min(
        {non_zero_entries.size(), out_actions.size(), static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max())});
    const double retained_sum = std::accumulate(
        non_zero_entries.begin(),
        non_zero_entries.begin() + static_cast<std::ptrdiff_t>(count),
        0.0,
        [](const double total, const PolicyEntry& entry) { return total + static_cast<double>(entry.probability); });
    if (!std::isfinite(retained_sum) || retained_sum <= 0.0) {
        return 0U;
    }

    double normalized_sum = 0.0;
    for (std::size_t i = 0U; i < count; ++i) {
        normalized_sum += static_cast<double>(non_zero_entries[i].probability) / retained_sum;
    }
    if (!std::isfinite(normalized_sum) || normalized_sum <= 0.0) {
        return 0U;
    }
    const double restore_scale = retained_sum / normalized_sum;

    for (std::size_t i = 0U; i < count; ++i) {
        const double normalized_probability = static_cast<double>(non_zero_entries[i].probability) / retained_sum;
        const float restored_probability = static_cast<float>(normalized_probability * restore_scale);
        out_actions[i] = non_zero_entries[i].action;
        out_probs_fp16[i] = float_to_fp16(restored_probability);
    }
    return static_cast<std::uint8_t>(count);
}

void decompress_policy(
    const std::span<const std::uint16_t> actions,
    const std::span<const std::uint16_t> probs_fp16,
    const std::uint8_t num_entries,
    const std::span<float> out_dense_policy) {
    if (actions.size() != probs_fp16.size()) {
        throw std::invalid_argument("decompress_policy action and probability arrays must have equal size");
    }
    if (num_entries > actions.size()) {
        throw std::invalid_argument("decompress_policy entry count exceeds sparse array size");
    }

    std::fill(out_dense_policy.begin(), out_dense_policy.end(), 0.0F);
    if (num_entries == 0U) {
        return;
    }

    std::vector<bool> seen_action(out_dense_policy.size(), false);
    for (std::size_t i = 0U; i < num_entries; ++i) {
        const std::size_t action = actions[i];
        if (action >= out_dense_policy.size()) {
            throw std::invalid_argument("decompress_policy action index is out of range");
        }
        if (seen_action[action]) {
            throw std::invalid_argument("decompress_policy sparse actions must be unique");
        }
        seen_action[action] = true;
        out_dense_policy[action] = fp16_to_float(probs_fp16[i]);
    }
}

void compress_ownership(
    const std::span<const float> ownership,
    const std::size_t board_area,
    const std::span<std::uint64_t> out_bitpacked) {
    if (board_area == 0U) {
        throw std::invalid_argument("compress_ownership board_area must be greater than zero");
    }
    if (ownership.size() != board_area) {
        throw std::invalid_argument("compress_ownership ownership size must match board_area");
    }

    const std::size_t words_per_plane = (board_area + 63U) / 64U;
    const std::size_t required_words = 2U * words_per_plane;
    if (out_bitpacked.size() < required_words) {
        throw std::invalid_argument("compress_ownership output capacity is too small");
    }
    std::fill(out_bitpacked.begin(), out_bitpacked.end(), 0U);

    for (std::size_t intersection = 0U; intersection < board_area; ++intersection) {
        const float value = ownership[intersection];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("compress_ownership values must be finite");
        }
        if (value == 0.0F) {
            continue;
        }

        const std::size_t word_index = intersection / 64U;
        const std::uint64_t bit = std::uint64_t{1} << (intersection % 64U);
        if (value > 0.0F) {
            out_bitpacked[word_index] |= bit;
        } else {
            out_bitpacked[words_per_plane + word_index] |= bit;
        }
    }
}

void decompress_ownership(
    const std::span<const std::uint64_t> bitpacked,
    const std::size_t board_area,
    const std::span<float> out_ownership) {
    if (board_area == 0U) {
        throw std::invalid_argument("decompress_ownership board_area must be greater than zero");
    }
    if (out_ownership.size() != board_area) {
        throw std::invalid_argument("decompress_ownership output size must match board_area");
    }

    const std::size_t words_per_plane = (board_area + 63U) / 64U;
    const std::size_t required_words = 2U * words_per_plane;
    if (bitpacked.size() < required_words) {
        throw std::invalid_argument("decompress_ownership input size is too small");
    }
    std::fill(out_ownership.begin(), out_ownership.end(), 0.0F);

    for (std::size_t intersection = 0U; intersection < board_area; ++intersection) {
        const std::size_t word_index = intersection / 64U;
        const std::uint64_t bit = std::uint64_t{1} << (intersection % 64U);
        const bool black_owned = (bitpacked[word_index] & bit) != 0U;
        const bool white_owned = (bitpacked[words_per_plane + word_index] & bit) != 0U;
        if (black_owned && white_owned) {
            throw std::invalid_argument("decompress_ownership found conflicting ownership planes");
        }
        if (black_owned) {
            out_ownership[intersection] = 1.0F;
        } else if (white_owned) {
            out_ownership[intersection] = -1.0F;
        }
    }
}

}  // namespace alphazero::selfplay
