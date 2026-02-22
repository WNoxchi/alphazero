#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <span>
#include <vector>

namespace alphazero::selfplay {

struct ReplayPosition {
    // Max NN input size for supported games in v1 (chess: 119*8*8, go: 17*19*19).
    static constexpr std::size_t kMaxEncodedStateSize = 119U * 8U * 8U;
    // Max action-space size for supported games in v1 (chess: 4672, go: 362).
    static constexpr std::size_t kMaxPolicySize = 4672U;
    static constexpr std::size_t kWdlSize = 3U;

    std::array<float, kMaxEncodedStateSize> encoded_state{};
    std::array<float, kMaxPolicySize> policy{};
    float value = 0.0F;
    std::array<float, kWdlSize> value_wdl{0.0F, 0.0F, 0.0F};
    std::uint32_t game_id = 0U;
    std::uint16_t move_number = 0U;
    std::uint16_t encoded_state_size = 0U;
    std::uint16_t policy_size = 0U;

    [[nodiscard]] static ReplayPosition make(
        std::span<const float> encoded_state_values,
        std::span<const float> policy_values,
        float value,
        const std::array<float, kWdlSize>& value_wdl,
        std::uint32_t game_id,
        std::uint16_t move_number);
};

struct SampledBatch {
    std::vector<float> states;
    std::vector<float> policies;
    std::vector<float> values;
    std::size_t batch_size = 0U;
};

class ReplayBuffer {
public:
    static constexpr std::size_t kDefaultCapacity = 1'000'000U;

    explicit ReplayBuffer(
        std::size_t capacity = kDefaultCapacity,
        std::uint64_t random_seed = 0x9E3779B97F4A7C15ULL);

    // Thread-safe write path used by concurrent self-play workers.
    void add_game(const std::vector<ReplayPosition>& positions);

    // Thread-safe uniform random sampling used by training.
    [[nodiscard]] std::vector<ReplayPosition> sample(std::size_t batch_size) const;
    [[nodiscard]] SampledBatch sample_batch(
        std::size_t batch_size,
        std::size_t encoded_state_size,
        std::size_t policy_size,
        std::size_t value_dim) const;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t write_head() const noexcept;

private:
    [[nodiscard]] static bool has_valid_shape(const ReplayPosition& position) noexcept;
    [[nodiscard]] std::vector<std::size_t> sample_logical_indices(
        std::size_t population_size,
        std::size_t sample_size) const;
    [[nodiscard]] std::size_t uniform_index(std::size_t upper_bound_exclusive) const;
    [[nodiscard]] std::size_t to_physical_index(
        std::size_t logical_index,
        std::size_t current_count,
        std::size_t current_write_head) const noexcept;

    // Contiguous storage; on target hardware this maps naturally to unified memory.
    std::vector<ReplayPosition> buffer_;
    std::atomic<std::size_t> write_head_{0U};
    std::atomic<std::size_t> count_{0U};
    mutable std::shared_mutex mutex_;

    mutable std::mutex rng_mutex_;
    mutable std::mt19937_64 rng_;
};

}  // namespace alphazero::selfplay
