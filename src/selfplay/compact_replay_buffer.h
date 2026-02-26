#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <vector>

#include "selfplay/replay_buffer.h"

namespace alphazero::selfplay {

class CompactReplayBuffer {
public:
    static constexpr std::size_t kDefaultCapacity = ReplayBuffer::kDefaultCapacity;

    explicit CompactReplayBuffer(
        std::size_t capacity,
        std::size_t num_binary_planes,
        std::size_t num_float_planes,
        std::vector<std::size_t> float_plane_indices,
        std::size_t full_policy_size,
        std::uint64_t random_seed = 0x9E3779B97F4A7C15ULL);

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

    std::size_t export_positions(
        float* out_states,
        float* out_policies,
        float* out_values_wdl,
        std::uint32_t* out_game_ids,
        std::uint16_t* out_move_numbers,
        std::size_t encoded_state_size,
        std::size_t policy_size) const;

    void import_positions(
        const float* states,
        const float* policies,
        const float* values_wdl,
        const std::uint32_t* game_ids,
        const std::uint16_t* move_numbers,
        std::size_t count,
        std::size_t encoded_state_size,
        std::size_t policy_size);

private:
    [[nodiscard]] static bool has_valid_shape(const ReplayPosition& position) noexcept;
    [[nodiscard]] bool has_valid_compact_shape(const CompactReplayPosition& position) const noexcept;

    [[nodiscard]] std::vector<std::size_t> sample_logical_indices(
        std::size_t population_size,
        std::size_t sample_size) const;
    [[nodiscard]] std::size_t uniform_index(std::size_t upper_bound_exclusive) const;
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
    std::size_t encoded_state_size_ = 0U;
    std::size_t num_binary_planes_ = 0U;
    std::size_t num_float_planes_ = 0U;
    std::size_t full_policy_size_ = 0U;
};

}  // namespace alphazero::selfplay
