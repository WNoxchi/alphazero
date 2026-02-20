#include "selfplay/replay_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace alphazero::selfplay {

ReplayPosition ReplayPosition::make(
    const std::span<const float> encoded_state_values,
    const std::span<const float> policy_values,
    const float scalar_value,
    const std::array<float, kWdlSize>& wdl_value,
    const std::uint32_t game_id_value,
    const std::uint16_t move_number_value) {
    if (encoded_state_values.empty()) {
        throw std::invalid_argument("ReplayPosition encoded_state must be non-empty");
    }
    if (policy_values.empty()) {
        throw std::invalid_argument("ReplayPosition policy must be non-empty");
    }
    if (encoded_state_values.size() > kMaxEncodedStateSize) {
        throw std::invalid_argument("ReplayPosition encoded_state exceeds maximum supported size");
    }
    if (policy_values.size() > kMaxPolicySize) {
        throw std::invalid_argument("ReplayPosition policy exceeds maximum supported size");
    }

    ReplayPosition position{};
    std::copy(encoded_state_values.begin(), encoded_state_values.end(), position.encoded_state.begin());
    std::copy(policy_values.begin(), policy_values.end(), position.policy.begin());
    position.value = scalar_value;
    position.value_wdl = wdl_value;
    position.game_id = game_id_value;
    position.move_number = move_number_value;
    position.encoded_state_size = static_cast<std::uint16_t>(encoded_state_values.size());
    position.policy_size = static_cast<std::uint16_t>(policy_values.size());
    return position;
}

ReplayBuffer::ReplayBuffer(const std::size_t capacity, const std::uint64_t random_seed)
    : buffer_(capacity),
      rng_(random_seed) {
    if (capacity == 0U) {
        throw std::invalid_argument("ReplayBuffer capacity must be greater than zero");
    }
}

void ReplayBuffer::add_game(const std::vector<ReplayPosition>& positions) {
    if (positions.empty()) {
        return;
    }

    for (const ReplayPosition& position : positions) {
        if (!has_valid_shape(position)) {
            throw std::invalid_argument("ReplayBuffer add_game received malformed ReplayPosition");
        }
    }

    std::unique_lock lock(mutex_);
    std::size_t head = write_head_.load(std::memory_order_relaxed);
    std::size_t current_count = count_.load(std::memory_order_relaxed);

    for (const ReplayPosition& position : positions) {
        buffer_[head] = position;
        head = (head + 1U) % buffer_.size();
        if (current_count < buffer_.size()) {
            ++current_count;
        }
    }

    write_head_.store(head, std::memory_order_release);
    count_.store(current_count, std::memory_order_release);
}

std::vector<ReplayPosition> ReplayBuffer::sample(const std::size_t batch_size) const {
    if (batch_size == 0U) {
        return {};
    }

    std::shared_lock lock(mutex_);
    const std::size_t current_count = count_.load(std::memory_order_relaxed);
    if (current_count == 0U) {
        throw std::runtime_error("ReplayBuffer sample requested from an empty buffer");
    }
    const std::size_t current_head = write_head_.load(std::memory_order_relaxed);

    const std::vector<std::size_t> logical_indices = sample_logical_indices(current_count, batch_size);
    std::vector<ReplayPosition> batch;
    batch.reserve(logical_indices.size());

    for (const std::size_t logical_index : logical_indices) {
        const std::size_t physical_index = to_physical_index(logical_index, current_count, current_head);
        batch.push_back(buffer_[physical_index]);
    }
    return batch;
}

std::size_t ReplayBuffer::size() const noexcept { return count_.load(std::memory_order_acquire); }

std::size_t ReplayBuffer::capacity() const noexcept { return buffer_.size(); }

std::size_t ReplayBuffer::write_head() const noexcept { return write_head_.load(std::memory_order_acquire); }

bool ReplayBuffer::has_valid_shape(const ReplayPosition& position) noexcept {
    return position.encoded_state_size > 0U && position.encoded_state_size <= ReplayPosition::kMaxEncodedStateSize &&
           position.policy_size > 0U && position.policy_size <= ReplayPosition::kMaxPolicySize;
}

std::vector<std::size_t> ReplayBuffer::sample_logical_indices(
    const std::size_t population_size,
    const std::size_t sample_size) const {
    std::vector<std::size_t> logical_indices;
    logical_indices.reserve(sample_size);

    if (sample_size <= population_size) {
        std::unordered_set<std::size_t> selected;
        selected.reserve(sample_size * 2U);

        const std::size_t start = population_size - sample_size;
        for (std::size_t j = start; j < population_size; ++j) {
            const std::size_t draw = uniform_index(j + 1U);
            if (!selected.insert(draw).second) {
                selected.insert(j);
            }
        }

        logical_indices.assign(selected.begin(), selected.end());
        return logical_indices;
    }

    for (std::size_t i = 0; i < sample_size; ++i) {
        logical_indices.push_back(uniform_index(population_size));
    }
    return logical_indices;
}

std::size_t ReplayBuffer::uniform_index(const std::size_t upper_bound_exclusive) const {
    if (upper_bound_exclusive == 0U) {
        throw std::invalid_argument("ReplayBuffer uniform_index upper bound must be positive");
    }

    const std::uint64_t upper = static_cast<std::uint64_t>(upper_bound_exclusive);
    const std::uint64_t rejection_threshold = (std::numeric_limits<std::uint64_t>::max() - upper + 1U) % upper;

    std::lock_guard lock(rng_mutex_);
    while (true) {
        const std::uint64_t candidate = rng_();
        if (candidate >= rejection_threshold) {
            return static_cast<std::size_t>(candidate % upper);
        }
    }
}

std::size_t ReplayBuffer::to_physical_index(
    const std::size_t logical_index,
    const std::size_t current_count,
    const std::size_t current_write_head) const noexcept {
    const std::size_t base = (current_count == buffer_.size()) ? current_write_head : 0U;
    return (base + logical_index) % buffer_.size();
}

}  // namespace alphazero::selfplay
