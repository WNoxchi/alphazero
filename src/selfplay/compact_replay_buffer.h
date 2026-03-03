#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "selfplay/replay_buffer.h"

static_assert(
    std::is_trivially_copyable_v<alphazero::selfplay::CompactReplayPosition>,
    "CompactReplayPosition must be trivially copyable for binary serialization");

namespace alphazero::selfplay {

enum class SamplingStrategy : std::uint8_t {
    kUniform = 0U,
    kRecencyWeighted = 1U,
};

class CompactReplayBuffer {
public:
    static constexpr std::size_t kDefaultCapacity = ReplayBuffer::kDefaultCapacity;

    explicit CompactReplayBuffer(
        std::size_t capacity,
        std::size_t num_binary_planes,
        std::size_t num_float_planes,
        std::vector<std::size_t> float_plane_indices,
        std::size_t full_policy_size,
        std::uint64_t random_seed = 0x9E3779B97F4A7C15ULL,
        SamplingStrategy sampling_strategy = SamplingStrategy::kUniform,
        float recency_weight_lambda = 1.0F,
        std::size_t squares_per_plane = 64U);

    void add_game(const std::vector<ReplayPosition>& positions);

    [[nodiscard]] std::vector<ReplayPosition> sample(std::size_t batch_size) const;
    [[nodiscard]] SampledBatch sample_batch(
        std::size_t batch_size,
        std::size_t encoded_state_size,
        std::size_t policy_size,
        std::size_t value_dim) const;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t write_head() const noexcept;
    [[nodiscard]] std::size_t ownership_payload_size() const;
    [[nodiscard]] SamplingStrategy sampling_strategy() const noexcept;
    [[nodiscard]] float recency_weight_lambda() const noexcept;

    std::size_t export_positions(
        float* out_states,
        float* out_policies,
        float* out_values_wdl,
        std::uint32_t* out_game_ids,
        std::uint16_t* out_move_numbers,
        std::size_t encoded_state_size,
        std::size_t policy_size,
        float* out_ownership = nullptr,
        std::size_t ownership_size = 0U) const;

    void import_positions(
        const float* states,
        const float* policies,
        const float* values_wdl,
        const std::uint32_t* game_ids,
        const std::uint16_t* move_numbers,
        std::size_t count,
        std::size_t encoded_state_size,
        std::size_t policy_size,
        const float* ownership = nullptr,
        std::size_t ownership_size = 0U);

    void save_to_file(const std::string& path) const;
    std::size_t load_from_file(const std::string& path);

private:
    [[nodiscard]] static bool has_valid_shape(const ReplayPosition& position) noexcept;
    [[nodiscard]] bool has_valid_compact_shape(const CompactReplayPosition& position) const noexcept;

    [[nodiscard]] std::vector<std::size_t> sample_logical_indices(
        std::size_t population_size,
        std::size_t sample_size) const;
    [[nodiscard]] std::size_t uniform_index(std::size_t upper_bound_exclusive) const;
    [[nodiscard]] std::size_t recency_weighted_index(std::size_t population_size) const;
    [[nodiscard]] double uniform_unit_interval() const;
    [[nodiscard]] std::size_t to_physical_index(
        std::size_t logical_index,
        std::size_t current_count,
        std::size_t current_write_head) const noexcept;

    std::vector<CompactReplayPosition> buffer_;
    std::atomic<std::size_t> write_head_{0U};
    std::atomic<std::size_t> count_{0U};
    mutable std::shared_mutex mutex_;

    mutable std::mutex rng_mutex_;
    mutable std::mt19937_64 rng_;

    std::vector<std::size_t> float_plane_indices_;
    std::size_t squares_per_plane_ = 64U;
    std::size_t words_per_plane_ = 1U;
    std::size_t num_binary_words_ = 0U;
    std::size_t encoded_state_size_ = 0U;
    std::size_t num_binary_planes_ = 0U;
    std::size_t num_float_planes_ = 0U;
    std::size_t full_policy_size_ = 0U;
    SamplingStrategy sampling_strategy_ = SamplingStrategy::kUniform;
    float recency_weight_lambda_ = 1.0F;
    double recency_weight_expm1_ = 0.0;
};

}  // namespace alphazero::selfplay
