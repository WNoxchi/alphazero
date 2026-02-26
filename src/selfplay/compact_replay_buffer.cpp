#include "selfplay/compact_replay_buffer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "selfplay/replay_compression.h"

namespace alphazero::selfplay {

namespace {

[[nodiscard]] std::size_t checked_flat_size(
    const std::size_t batch_size,
    const std::size_t row_width,
    const char* field_name) {
    if (batch_size == 0U || row_width == 0U) {
        return 0U;
    }
    if (batch_size > (std::numeric_limits<std::size_t>::max() / row_width)) {
        throw std::overflow_error(
            std::string("CompactReplayBuffer overflowed while sizing ") + field_name);
    }
    return batch_size * row_width;
}

[[nodiscard]] std::uint16_t checked_u16(const std::size_t value, const char* field_name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())) {
        throw std::overflow_error(std::string("CompactReplayBuffer ") + field_name + " exceeds uint16 range");
    }
    return static_cast<std::uint16_t>(value);
}

}  // namespace

CompactReplayBuffer::CompactReplayBuffer(
    const std::size_t capacity,
    const std::size_t num_binary_planes,
    const std::size_t num_float_planes,
    std::vector<std::size_t> float_plane_indices,
    const std::size_t full_policy_size,
    const std::uint64_t random_seed)
    : buffer_(capacity),
      rng_(random_seed),
      float_plane_indices_(std::move(float_plane_indices)),
      num_binary_planes_(num_binary_planes),
      num_float_planes_(num_float_planes),
      full_policy_size_(full_policy_size) {
    if (capacity == 0U) {
        throw std::invalid_argument("CompactReplayBuffer capacity must be greater than zero");
    }
    if (num_binary_planes_ == 0U) {
        throw std::invalid_argument("CompactReplayBuffer num_binary_planes must be greater than zero");
    }
    if (num_binary_planes_ > CompactReplayPosition::kMaxBinaryPlanes) {
        throw std::invalid_argument("CompactReplayBuffer num_binary_planes exceeds supported maximum");
    }
    if (num_float_planes_ > CompactReplayPosition::kMaxFloatPlanes) {
        throw std::invalid_argument("CompactReplayBuffer num_float_planes exceeds supported maximum");
    }
    if (num_float_planes_ != float_plane_indices_.size()) {
        throw std::invalid_argument(
            "CompactReplayBuffer num_float_planes must match float_plane_indices length");
    }
    if (full_policy_size_ == 0U || full_policy_size_ > ReplayPosition::kMaxPolicySize) {
        throw std::invalid_argument("CompactReplayBuffer full_policy_size is out of range");
    }
    if (full_policy_size_ > (static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1U)) {
        throw std::invalid_argument("CompactReplayBuffer full_policy_size exceeds sparse policy action-index range");
    }

    const std::size_t total_planes = num_binary_planes_ + num_float_planes_;
    if (total_planes > (ReplayPosition::kMaxEncodedStateSize / 64U)) {
        throw std::invalid_argument("CompactReplayBuffer total plane count exceeds supported encoded-state size");
    }
    encoded_state_size_ = total_planes * 64U;

    std::vector<bool> seen(total_planes, false);
    for (const std::size_t plane : float_plane_indices_) {
        if (plane >= total_planes) {
            throw std::invalid_argument("CompactReplayBuffer float_plane_indices contains an out-of-range entry");
        }
        if (seen[plane]) {
            throw std::invalid_argument("CompactReplayBuffer float_plane_indices must be unique");
        }
        seen[plane] = true;
    }
}

void CompactReplayBuffer::add_game(const std::vector<ReplayPosition>& positions) {
    if (positions.empty()) {
        return;
    }

    std::vector<CompactReplayPosition> compact_positions;
    compact_positions.reserve(positions.size());

    for (const ReplayPosition& position : positions) {
        if (!has_valid_shape(position)) {
            throw std::invalid_argument("CompactReplayBuffer add_game received malformed ReplayPosition");
        }
        if (static_cast<std::size_t>(position.encoded_state_size) != encoded_state_size_) {
            throw std::invalid_argument(
                "CompactReplayBuffer add_game encoded_state_size does not match configured shape");
        }
        if (static_cast<std::size_t>(position.policy_size) != full_policy_size_) {
            throw std::invalid_argument(
                "CompactReplayBuffer add_game policy_size does not match configured shape");
        }

        CompactReplayPosition compact{};
        const StateCompressionLayout layout = compress_state(
            std::span<const float>(position.encoded_state.data(), encoded_state_size_),
            float_plane_indices_,
            compact.bitpacked_planes,
            compact.quantized_float_planes);
        if (layout.num_binary_planes != num_binary_planes_ || layout.num_float_planes != num_float_planes_) {
            throw std::invalid_argument("CompactReplayBuffer add_game produced an unexpected compression layout");
        }

        compact.num_binary_planes = checked_u16(layout.num_binary_planes, "num_binary_planes");
        compact.num_float_planes = checked_u16(layout.num_float_planes, "num_float_planes");
        compact.num_policy_entries = compress_policy(
            std::span<const float>(position.policy.data(), full_policy_size_),
            compact.policy_actions,
            compact.policy_probs_fp16);
        compact.policy_size = checked_u16(full_policy_size_, "policy_size");
        compact.value = position.value;
        compact.value_wdl = position.value_wdl;
        compact.game_id = position.game_id;
        compact.move_number = position.move_number;
        compact_positions.push_back(compact);
    }

    std::unique_lock lock(mutex_);
    std::size_t head = write_head_.load(std::memory_order_relaxed);
    std::size_t current_count = count_.load(std::memory_order_relaxed);

    for (const CompactReplayPosition& compact : compact_positions) {
        buffer_[head] = compact;
        head = (head + 1U) % buffer_.size();
        if (current_count < buffer_.size()) {
            ++current_count;
        }
    }

    write_head_.store(head, std::memory_order_release);
    count_.store(current_count, std::memory_order_release);
}

std::vector<ReplayPosition> CompactReplayBuffer::sample(const std::size_t batch_size) const {
    if (batch_size == 0U) {
        return {};
    }

    std::shared_lock lock(mutex_);
    const std::size_t current_count = count_.load(std::memory_order_relaxed);
    if (current_count == 0U) {
        throw std::runtime_error("CompactReplayBuffer sample requested from an empty buffer");
    }
    const std::size_t current_head = write_head_.load(std::memory_order_relaxed);

    const std::vector<std::size_t> logical_indices = sample_logical_indices(current_count, batch_size);
    std::vector<ReplayPosition> batch;
    batch.reserve(logical_indices.size());

    for (const std::size_t logical_index : logical_indices) {
        const std::size_t physical_index = to_physical_index(logical_index, current_count, current_head);
        const CompactReplayPosition& compact = buffer_[physical_index];
        if (!has_valid_compact_shape(compact)) {
            throw std::invalid_argument("CompactReplayBuffer sample encountered malformed CompactReplayPosition");
        }

        ReplayPosition dense{};
        decompress_state(
            std::span<const std::uint64_t>(compact.bitpacked_planes.data(), num_binary_planes_),
            std::span<const std::uint8_t>(compact.quantized_float_planes.data(), num_float_planes_),
            float_plane_indices_,
            std::span<float>(dense.encoded_state.data(), encoded_state_size_));
        decompress_policy(
            compact.policy_actions,
            compact.policy_probs_fp16,
            compact.num_policy_entries,
            std::span<float>(dense.policy.data(), full_policy_size_));

        dense.value = compact.value;
        dense.value_wdl = compact.value_wdl;
        dense.game_id = compact.game_id;
        dense.move_number = compact.move_number;
        dense.encoded_state_size = checked_u16(encoded_state_size_, "encoded_state_size");
        dense.policy_size = checked_u16(full_policy_size_, "policy_size");
        batch.push_back(dense);
    }

    return batch;
}

SampledBatch CompactReplayBuffer::sample_batch(
    const std::size_t batch_size,
    const std::size_t encoded_state_size,
    const std::size_t policy_size,
    const std::size_t value_dim) const {
    if (batch_size == 0U) {
        return {};
    }
    if (encoded_state_size != encoded_state_size_) {
        throw std::invalid_argument("CompactReplayBuffer sample_batch encoded_state_size does not match configuration");
    }
    if (policy_size != full_policy_size_) {
        throw std::invalid_argument("CompactReplayBuffer sample_batch policy_size does not match configuration");
    }
    if (value_dim != 1U && value_dim != ReplayPosition::kWdlSize) {
        throw std::invalid_argument("CompactReplayBuffer sample_batch value_dim must be 1 (scalar) or 3 (wdl)");
    }

    std::shared_lock lock(mutex_);
    const std::size_t current_count = count_.load(std::memory_order_relaxed);
    if (current_count == 0U) {
        throw std::runtime_error("CompactReplayBuffer sample_batch requested from an empty buffer");
    }
    const std::size_t current_head = write_head_.load(std::memory_order_relaxed);

    const std::vector<std::size_t> logical_indices = sample_logical_indices(current_count, batch_size);
    SampledBatch packed{};
    packed.batch_size = logical_indices.size();
    packed.states.resize(checked_flat_size(packed.batch_size, encoded_state_size_, "states"));
    packed.policies.resize(checked_flat_size(packed.batch_size, full_policy_size_, "policies"));
    packed.values.resize(checked_flat_size(packed.batch_size, value_dim, "values"));

    for (std::size_t sample_index = 0U; sample_index < logical_indices.size(); ++sample_index) {
        const std::size_t logical_index = logical_indices[sample_index];
        const std::size_t physical_index = to_physical_index(logical_index, current_count, current_head);
        const CompactReplayPosition& compact = buffer_[physical_index];
        if (!has_valid_compact_shape(compact)) {
            throw std::invalid_argument("CompactReplayBuffer sample_batch encountered malformed CompactReplayPosition");
        }

        float* const state_row = packed.states.data() + (sample_index * encoded_state_size_);
        decompress_state(
            std::span<const std::uint64_t>(compact.bitpacked_planes.data(), num_binary_planes_),
            std::span<const std::uint8_t>(compact.quantized_float_planes.data(), num_float_planes_),
            float_plane_indices_,
            std::span<float>(state_row, encoded_state_size_));

        float* const policy_row = packed.policies.data() + (sample_index * full_policy_size_);
        decompress_policy(
            compact.policy_actions,
            compact.policy_probs_fp16,
            compact.num_policy_entries,
            std::span<float>(policy_row, full_policy_size_));

        float* const value_row = packed.values.data() + (sample_index * value_dim);
        if (value_dim == 1U) {
            value_row[0] = compact.value;
        } else {
            std::copy_n(compact.value_wdl.begin(), ReplayPosition::kWdlSize, value_row);
        }
    }

    return packed;
}

std::size_t CompactReplayBuffer::export_positions(
    float* const out_states,
    float* const out_policies,
    float* const out_values_wdl,
    std::uint32_t* const out_game_ids,
    std::uint16_t* const out_move_numbers,
    const std::size_t encoded_state_size,
    const std::size_t policy_size) const {
    if (encoded_state_size != encoded_state_size_) {
        throw std::invalid_argument("export_positions: encoded_state_size does not match CompactReplayBuffer");
    }
    if (policy_size != full_policy_size_) {
        throw std::invalid_argument("export_positions: policy_size does not match CompactReplayBuffer");
    }

    std::shared_lock lock(mutex_);
    const std::size_t current_count = count_.load(std::memory_order_relaxed);
    const std::size_t current_head = write_head_.load(std::memory_order_relaxed);

    for (std::size_t i = 0U; i < current_count; ++i) {
        const std::size_t physical = to_physical_index(i, current_count, current_head);
        const CompactReplayPosition& compact = buffer_[physical];
        if (!has_valid_compact_shape(compact)) {
            throw std::invalid_argument("export_positions: encountered malformed CompactReplayPosition");
        }

        float* const state_row = out_states + (i * encoded_state_size_);
        decompress_state(
            std::span<const std::uint64_t>(compact.bitpacked_planes.data(), num_binary_planes_),
            std::span<const std::uint8_t>(compact.quantized_float_planes.data(), num_float_planes_),
            float_plane_indices_,
            std::span<float>(state_row, encoded_state_size_));

        float* const policy_row = out_policies + (i * full_policy_size_);
        decompress_policy(
            compact.policy_actions,
            compact.policy_probs_fp16,
            compact.num_policy_entries,
            std::span<float>(policy_row, full_policy_size_));

        std::copy_n(
            compact.value_wdl.begin(),
            ReplayPosition::kWdlSize,
            out_values_wdl + (i * ReplayPosition::kWdlSize));
        out_game_ids[i] = compact.game_id;
        out_move_numbers[i] = compact.move_number;
    }

    return current_count;
}

void CompactReplayBuffer::import_positions(
    const float* const states,
    const float* const policies,
    const float* const values_wdl,
    const std::uint32_t* const game_ids,
    const std::uint16_t* const move_numbers,
    const std::size_t count,
    const std::size_t encoded_state_size,
    const std::size_t policy_size) {
    if (count == 0U) {
        return;
    }
    if (encoded_state_size != encoded_state_size_) {
        throw std::invalid_argument("import_positions: encoded_state_size does not match CompactReplayBuffer");
    }
    if (policy_size != full_policy_size_) {
        throw std::invalid_argument("import_positions: policy_size does not match CompactReplayBuffer");
    }

    std::vector<CompactReplayPosition> compact_positions;
    compact_positions.reserve(count);
    for (std::size_t i = 0U; i < count; ++i) {
        CompactReplayPosition compact{};
        const StateCompressionLayout layout = compress_state(
            std::span<const float>(states + (i * encoded_state_size_), encoded_state_size_),
            float_plane_indices_,
            compact.bitpacked_planes,
            compact.quantized_float_planes);
        if (layout.num_binary_planes != num_binary_planes_ || layout.num_float_planes != num_float_planes_) {
            throw std::invalid_argument("import_positions: compressed state layout does not match configuration");
        }

        compact.num_binary_planes = checked_u16(layout.num_binary_planes, "num_binary_planes");
        compact.num_float_planes = checked_u16(layout.num_float_planes, "num_float_planes");
        compact.num_policy_entries = compress_policy(
            std::span<const float>(policies + (i * full_policy_size_), full_policy_size_),
            compact.policy_actions,
            compact.policy_probs_fp16);
        compact.policy_size = checked_u16(full_policy_size_, "policy_size");

        std::copy_n(
            values_wdl + (i * ReplayPosition::kWdlSize),
            ReplayPosition::kWdlSize,
            compact.value_wdl.begin());
        compact.value =
            values_wdl[i * ReplayPosition::kWdlSize] - values_wdl[(i * ReplayPosition::kWdlSize) + 2U];
        compact.game_id = game_ids[i];
        compact.move_number = move_numbers[i];
        compact_positions.push_back(compact);
    }

    std::unique_lock lock(mutex_);
    std::size_t head = write_head_.load(std::memory_order_relaxed);
    std::size_t current_count = count_.load(std::memory_order_relaxed);

    for (const CompactReplayPosition& compact : compact_positions) {
        buffer_[head] = compact;
        head = (head + 1U) % buffer_.size();
        if (current_count < buffer_.size()) {
            ++current_count;
        }
    }

    write_head_.store(head, std::memory_order_release);
    count_.store(current_count, std::memory_order_release);
}

std::size_t CompactReplayBuffer::size() const noexcept { return count_.load(std::memory_order_acquire); }

std::size_t CompactReplayBuffer::capacity() const noexcept { return buffer_.size(); }

std::size_t CompactReplayBuffer::write_head() const noexcept {
    return write_head_.load(std::memory_order_acquire);
}

bool CompactReplayBuffer::has_valid_shape(const ReplayPosition& position) noexcept {
    return position.encoded_state_size > 0U && position.encoded_state_size <= ReplayPosition::kMaxEncodedStateSize &&
           position.policy_size > 0U && position.policy_size <= ReplayPosition::kMaxPolicySize;
}

bool CompactReplayBuffer::has_valid_compact_shape(const CompactReplayPosition& position) const noexcept {
    return position.num_binary_planes == num_binary_planes_ && position.num_float_planes == num_float_planes_ &&
           position.policy_size == full_policy_size_ &&
           position.num_policy_entries <= CompactReplayPosition::kMaxSparsePolicy;
}

std::vector<std::size_t> CompactReplayBuffer::sample_logical_indices(
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

std::size_t CompactReplayBuffer::uniform_index(const std::size_t upper_bound_exclusive) const {
    if (upper_bound_exclusive == 0U) {
        throw std::invalid_argument("CompactReplayBuffer uniform_index upper bound must be positive");
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

std::size_t CompactReplayBuffer::to_physical_index(
    const std::size_t logical_index,
    const std::size_t current_count,
    const std::size_t current_write_head) const noexcept {
    const std::size_t base = (current_count == buffer_.size()) ? current_write_head : 0U;
    return (base + logical_index) % buffer_.size();
}

}  // namespace alphazero::selfplay
