#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace alphazero::selfplay {

struct StateCompressionLayout {
    std::size_t num_binary_words = 0U;
    std::size_t num_binary_planes = 0U;
    std::size_t num_float_planes = 0U;
};

// Compress dense board encoding into bitpacked binary planes + quantized float planes.
[[nodiscard]] StateCompressionLayout compress_state(
    std::span<const float> dense_state,
    std::span<const std::size_t> float_plane_indices,
    std::size_t squares_per_plane,
    std::span<std::uint64_t> out_bitpacked_planes,
    std::span<std::uint8_t> out_quantized_float_planes);

// Decompress bitpacked/quantized state back into dense [planes * squares_per_plane] layout.
void decompress_state(
    std::span<const std::uint64_t> bitpacked_planes,
    std::span<const std::uint8_t> quantized_float_planes,
    std::span<const std::size_t> float_plane_indices,
    std::size_t squares_per_plane,
    std::span<float> out_dense_state);

// IEEE-754 binary16 conversion helpers used for sparse policy storage.
[[nodiscard]] std::uint16_t float_to_fp16(float value) noexcept;
[[nodiscard]] float fp16_to_float(std::uint16_t value) noexcept;

// Compress dense policy into sparse (action, fp16 probability) arrays.
[[nodiscard]] std::uint8_t compress_policy(
    std::span<const float> dense_policy,
    std::span<std::uint16_t> out_actions,
    std::span<std::uint16_t> out_probs_fp16);

// Decompress sparse policy entries into a dense policy vector.
void decompress_policy(
    std::span<const std::uint16_t> actions,
    std::span<const std::uint16_t> probs_fp16,
    std::uint8_t num_entries,
    std::span<float> out_dense_policy);

// Compress ternary ownership targets (+1, 0, -1) into two bitplanes.
void compress_ownership(
    std::span<const float> ownership,
    std::size_t board_area,
    std::span<std::uint64_t> out_bitpacked);

// Decompress two ownership bitplanes into dense ternary ownership targets.
void decompress_ownership(
    std::span<const std::uint64_t> bitpacked,
    std::size_t board_area,
    std::span<float> out_ownership);

}  // namespace alphazero::selfplay
