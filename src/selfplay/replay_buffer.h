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
    // Max board area for supported ownership targets (19x19 Go).
    static constexpr std::size_t kMaxBoardArea = 361U;
    static constexpr std::size_t kWdlSize = 3U;

    std::array<float, kMaxEncodedStateSize> encoded_state{};
    std::array<float, kMaxPolicySize> policy{};
    float value = 0.0F;
    float training_weight = 1.0F;
    std::array<float, kWdlSize> value_wdl{0.0F, 0.0F, 0.0F};
    std::array<float, kMaxBoardArea> ownership{};
    std::uint32_t game_id = 0U;
    std::uint16_t move_number = 0U;
    std::uint16_t encoded_state_size = 0U;
    std::uint16_t policy_size = 0U;
    std::uint16_t ownership_size = 0U;

    [[nodiscard]] static ReplayPosition make(
        std::span<const float> encoded_state_values,
        std::span<const float> policy_values,
        float value,
        const std::array<float, kWdlSize>& value_wdl,
        std::uint32_t game_id,
        std::uint16_t move_number,
        float training_weight = 1.0F);
};

struct CompactReplayPosition {
    static constexpr std::size_t kMaxBinaryWords = 117U;
    static constexpr std::size_t kMaxFloatPlanes = 2U;
    static constexpr std::size_t kMaxSparsePolicy = 64U;
    static constexpr std::size_t kMaxOwnershipWords = 12U;
    static constexpr std::size_t kWdlSize = 3U;

    // Binary state planes bitpacked as words (words_per_plane * num_binary_planes total).
    std::array<std::uint64_t, kMaxBinaryWords> bitpacked_planes{};
    // Constant-valued float planes quantized to [0, 255].
    std::array<std::uint8_t, kMaxFloatPlanes> quantized_float_planes{};

    // Sparse policy representation (action index + FP16 probability).
    std::array<std::uint16_t, kMaxSparsePolicy> policy_actions{};
    std::array<std::uint16_t, kMaxSparsePolicy> policy_probs_fp16{};
    std::uint8_t num_policy_entries = 0U;

    // Ownership encoded as two bitplanes: black-owned and white-owned.
    std::array<std::uint64_t, kMaxOwnershipWords> bitpacked_ownership{};

    // Metadata and value targets matching ReplayPosition semantics.
    float value = 0.0F;
    float training_weight = 1.0F;
    std::array<float, kWdlSize> value_wdl{0.0F, 0.0F, 0.0F};
    std::uint32_t game_id = 0U;
    std::uint16_t move_number = 0U;
    std::uint16_t num_binary_words = 0U;
    std::uint16_t num_binary_planes = 0U;
    std::uint16_t num_float_planes = 0U;
    std::uint16_t policy_size = 0U;
    std::uint16_t num_ownership_words = 0U;
};

struct SampledBatch {
    std::vector<float> states;
    std::vector<float> policies;
    std::vector<float> values;
    std::vector<float> weights;
    std::vector<float> ownership;
    std::size_t batch_size = 0U;
    std::size_t ownership_size = 0U;
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

    /// Export all valid positions into pre-allocated flat arrays (logical order).
    /// Returns the number of positions exported.  Caller must allocate arrays
    /// of at least size() * field_width elements.
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

    /// Import positions from flat arrays into the buffer.
    /// Appends to the current buffer (does not clear existing data).
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
