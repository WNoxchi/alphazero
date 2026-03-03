#include "selfplay/compact_replay_buffer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
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
    const std::uint64_t random_seed,
    const SamplingStrategy sampling_strategy,
    const float recency_weight_lambda,
    const std::size_t squares_per_plane)
    : buffer_(capacity),
      rng_(random_seed),
      float_plane_indices_(std::move(float_plane_indices)),
      squares_per_plane_(squares_per_plane),
      num_binary_planes_(num_binary_planes),
      num_float_planes_(num_float_planes),
      full_policy_size_(full_policy_size),
      sampling_strategy_(sampling_strategy),
      recency_weight_lambda_(recency_weight_lambda) {
    if (capacity == 0U) {
        throw std::invalid_argument("CompactReplayBuffer capacity must be greater than zero");
    }
    if (num_binary_planes_ == 0U) {
        throw std::invalid_argument("CompactReplayBuffer num_binary_planes must be greater than zero");
    }
    if (squares_per_plane_ == 0U) {
        throw std::invalid_argument("CompactReplayBuffer squares_per_plane must be greater than zero");
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
    if (!std::isfinite(recency_weight_lambda_) || recency_weight_lambda_ < 0.0F) {
        throw std::invalid_argument(
            "CompactReplayBuffer recency_weight_lambda must be finite and non-negative");
    }
    if (sampling_strategy_ != SamplingStrategy::kUniform &&
        sampling_strategy_ != SamplingStrategy::kRecencyWeighted) {
        throw std::invalid_argument("CompactReplayBuffer sampling_strategy is invalid");
    }

    const std::size_t total_planes = num_binary_planes_ + num_float_planes_;
    if (total_planes > (ReplayPosition::kMaxEncodedStateSize / squares_per_plane_)) {
        throw std::invalid_argument("CompactReplayBuffer total plane count exceeds supported encoded-state size");
    }
    words_per_plane_ = (squares_per_plane_ + 63U) / 64U;
    if (num_binary_planes_ > (std::numeric_limits<std::size_t>::max() / words_per_plane_)) {
        throw std::invalid_argument("CompactReplayBuffer binary plane layout overflows");
    }
    num_binary_words_ = num_binary_planes_ * words_per_plane_;
    if (num_binary_words_ > CompactReplayPosition::kMaxBinaryWords) {
        throw std::invalid_argument("CompactReplayBuffer binary-word usage exceeds supported maximum");
    }
    const std::size_t ownership_words = 2U * words_per_plane_;
    if (ownership_words > CompactReplayPosition::kMaxOwnershipWords) {
        throw std::invalid_argument("CompactReplayBuffer ownership-word usage exceeds supported maximum");
    }
    encoded_state_size_ = total_planes * squares_per_plane_;

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

    if (sampling_strategy_ == SamplingStrategy::kRecencyWeighted && recency_weight_lambda_ > 0.0F) {
        recency_weight_expm1_ = std::expm1(static_cast<double>(recency_weight_lambda_));
        if (!std::isfinite(recency_weight_expm1_)) {
            throw std::invalid_argument("CompactReplayBuffer recency_weight_lambda is too large");
        }
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
        if (position.ownership_size > ReplayPosition::kMaxBoardArea) {
            throw std::invalid_argument("CompactReplayBuffer add_game ownership_size exceeds supported maximum");
        }
        if (position.ownership_size != 0U &&
            static_cast<std::size_t>(position.ownership_size) != squares_per_plane_) {
            throw std::invalid_argument(
                "CompactReplayBuffer add_game ownership_size does not match configured board area");
        }

        CompactReplayPosition compact{};
        const StateCompressionLayout layout = compress_state(
            std::span<const float>(position.encoded_state.data(), encoded_state_size_),
            float_plane_indices_,
            squares_per_plane_,
            std::span<std::uint64_t>(compact.bitpacked_planes.data(), num_binary_words_),
            compact.quantized_float_planes);
        if (layout.num_binary_words != num_binary_words_ || layout.num_binary_planes != num_binary_planes_ ||
            layout.num_float_planes != num_float_planes_) {
            throw std::invalid_argument("CompactReplayBuffer add_game produced an unexpected compression layout");
        }

        compact.num_binary_words = checked_u16(layout.num_binary_words, "num_binary_words");
        compact.num_binary_planes = checked_u16(layout.num_binary_planes, "num_binary_planes");
        compact.num_float_planes = checked_u16(layout.num_float_planes, "num_float_planes");
        compact.num_policy_entries = compress_policy(
            std::span<const float>(position.policy.data(), full_policy_size_),
            compact.policy_actions,
            compact.policy_probs_fp16);
        compact.policy_size = checked_u16(full_policy_size_, "policy_size");
        compact.value = position.value;
        compact.training_weight = position.training_weight;
        compact.value_wdl = position.value_wdl;
        compact.game_id = position.game_id;
        compact.move_number = position.move_number;
        compact.num_ownership_words = 0U;
        if (position.ownership_size > 0U) {
            const std::size_t ownership_words = 2U * words_per_plane_;
            compress_ownership(
                std::span<const float>(
                    position.ownership.data(),
                    static_cast<std::size_t>(position.ownership_size)),
                squares_per_plane_,
                std::span<std::uint64_t>(compact.bitpacked_ownership.data(), ownership_words));
            compact.num_ownership_words = checked_u16(ownership_words, "num_ownership_words");
        }
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
            std::span<const std::uint64_t>(compact.bitpacked_planes.data(), num_binary_words_),
            std::span<const std::uint8_t>(compact.quantized_float_planes.data(), num_float_planes_),
            float_plane_indices_,
            squares_per_plane_,
            std::span<float>(dense.encoded_state.data(), encoded_state_size_));
        decompress_policy(
            compact.policy_actions,
            compact.policy_probs_fp16,
            compact.num_policy_entries,
            std::span<float>(dense.policy.data(), full_policy_size_));
        dense.ownership.fill(0.0F);
        if (compact.num_ownership_words > 0U) {
            const std::size_t ownership_words = 2U * words_per_plane_;
            if (compact.num_ownership_words != ownership_words) {
                throw std::invalid_argument(
                    "CompactReplayBuffer sample encountered malformed ownership layout");
            }
            decompress_ownership(
                std::span<const std::uint64_t>(compact.bitpacked_ownership.data(), ownership_words),
                squares_per_plane_,
                std::span<float>(dense.ownership.data(), squares_per_plane_));
            dense.ownership_size = checked_u16(squares_per_plane_, "ownership_size");
        } else {
            dense.ownership_size = 0U;
        }

        dense.value = compact.value;
        dense.training_weight = compact.training_weight;
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
    std::vector<const CompactReplayPosition*> sampled_positions;
    sampled_positions.reserve(logical_indices.size());
    for (const std::size_t logical_index : logical_indices) {
        const std::size_t physical_index = to_physical_index(logical_index, current_count, current_head);
        sampled_positions.push_back(&buffer_[physical_index]);
    }

    bool saw_missing_ownership = false;
    bool saw_present_ownership = false;
    std::size_t packed_ownership_size = 0U;
    const std::size_t expected_ownership_words = 2U * words_per_plane_;
    for (const CompactReplayPosition* const compact : sampled_positions) {
        if (compact == nullptr) {
            throw std::logic_error("CompactReplayBuffer sample_batch encountered a null sampled position");
        }
        if (!has_valid_compact_shape(*compact)) {
            throw std::invalid_argument("CompactReplayBuffer sample_batch encountered malformed CompactReplayPosition");
        }
        if (compact->num_ownership_words == 0U) {
            saw_missing_ownership = true;
            continue;
        }
        saw_present_ownership = true;
        if (compact->num_ownership_words != expected_ownership_words) {
            throw std::invalid_argument("CompactReplayBuffer sample_batch encountered malformed ownership layout");
        }
        packed_ownership_size = squares_per_plane_;
    }
    if (saw_missing_ownership || !saw_present_ownership) {
        packed_ownership_size = 0U;
    }

    SampledBatch packed{};
    packed.batch_size = logical_indices.size();
    packed.states.resize(checked_flat_size(packed.batch_size, encoded_state_size_, "states"));
    packed.policies.resize(checked_flat_size(packed.batch_size, full_policy_size_, "policies"));
    packed.values.resize(checked_flat_size(packed.batch_size, value_dim, "values"));
    packed.weights.resize(packed.batch_size, 1.0F);
    packed.ownership_size = packed_ownership_size;
    if (packed_ownership_size > 0U) {
        packed.ownership.resize(checked_flat_size(packed.batch_size, packed_ownership_size, "ownership"));
    }

    for (std::size_t sample_index = 0U; sample_index < sampled_positions.size(); ++sample_index) {
        const CompactReplayPosition& compact = *sampled_positions[sample_index];

        float* const state_row = packed.states.data() + (sample_index * encoded_state_size_);
        decompress_state(
            std::span<const std::uint64_t>(compact.bitpacked_planes.data(), num_binary_words_),
            std::span<const std::uint8_t>(compact.quantized_float_planes.data(), num_float_planes_),
            float_plane_indices_,
            squares_per_plane_,
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

        packed.weights[sample_index] = compact.training_weight;
        if (packed_ownership_size > 0U) {
            if (compact.num_ownership_words != expected_ownership_words) {
                throw std::invalid_argument(
                    "CompactReplayBuffer sample_batch encountered a row without ownership data in an ownership batch");
            }
            float* const ownership_row = packed.ownership.data() + (sample_index * packed_ownership_size);
            decompress_ownership(
                std::span<const std::uint64_t>(compact.bitpacked_ownership.data(), expected_ownership_words),
                squares_per_plane_,
                std::span<float>(ownership_row, packed_ownership_size));
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
    const std::size_t policy_size,
    float* const out_ownership,
    const std::size_t ownership_size) const {
    if (encoded_state_size != encoded_state_size_) {
        throw std::invalid_argument("export_positions: encoded_state_size does not match CompactReplayBuffer");
    }
    if (policy_size != full_policy_size_) {
        throw std::invalid_argument("export_positions: policy_size does not match CompactReplayBuffer");
    }
    if (ownership_size > ReplayPosition::kMaxBoardArea) {
        throw std::invalid_argument("export_positions: ownership_size is out of range");
    }
    if (ownership_size > 0U && out_ownership == nullptr) {
        throw std::invalid_argument("export_positions: ownership output must be non-null when ownership_size > 0");
    }
    if (ownership_size > 0U && ownership_size != squares_per_plane_) {
        throw std::invalid_argument("export_positions: ownership_size does not match CompactReplayBuffer");
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
            std::span<const std::uint64_t>(compact.bitpacked_planes.data(), num_binary_words_),
            std::span<const std::uint8_t>(compact.quantized_float_planes.data(), num_float_planes_),
            float_plane_indices_,
            squares_per_plane_,
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
        if (ownership_size > 0U) {
            const std::size_t ownership_words = 2U * words_per_plane_;
            if (compact.num_ownership_words != ownership_words) {
                throw std::invalid_argument("export_positions: encountered position without ownership payload");
            }
            decompress_ownership(
                std::span<const std::uint64_t>(compact.bitpacked_ownership.data(), ownership_words),
                squares_per_plane_,
                std::span<float>(out_ownership + (i * ownership_size), ownership_size));
        }
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
    const std::size_t policy_size,
    const float* const ownership,
    const std::size_t ownership_size) {
    if (count == 0U) {
        return;
    }
    if (encoded_state_size != encoded_state_size_) {
        throw std::invalid_argument("import_positions: encoded_state_size does not match CompactReplayBuffer");
    }
    if (policy_size != full_policy_size_) {
        throw std::invalid_argument("import_positions: policy_size does not match CompactReplayBuffer");
    }
    if (ownership_size > ReplayPosition::kMaxBoardArea) {
        throw std::invalid_argument("import_positions: ownership_size is out of range");
    }
    if (ownership_size > 0U && ownership == nullptr) {
        throw std::invalid_argument("import_positions: ownership input must be non-null when ownership_size > 0");
    }
    if (ownership_size > 0U && ownership_size != squares_per_plane_) {
        throw std::invalid_argument("import_positions: ownership_size does not match CompactReplayBuffer");
    }

    std::vector<CompactReplayPosition> compact_positions;
    compact_positions.reserve(count);
    for (std::size_t i = 0U; i < count; ++i) {
        CompactReplayPosition compact{};
        const StateCompressionLayout layout = compress_state(
            std::span<const float>(states + (i * encoded_state_size_), encoded_state_size_),
            float_plane_indices_,
            squares_per_plane_,
            std::span<std::uint64_t>(compact.bitpacked_planes.data(), num_binary_words_),
            compact.quantized_float_planes);
        if (layout.num_binary_words != num_binary_words_ || layout.num_binary_planes != num_binary_planes_ ||
            layout.num_float_planes != num_float_planes_) {
            throw std::invalid_argument("import_positions: compressed state layout does not match configuration");
        }

        compact.num_binary_words = checked_u16(layout.num_binary_words, "num_binary_words");
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
        compact.training_weight = 1.0F;
        compact.game_id = game_ids[i];
        compact.move_number = move_numbers[i];
        compact.num_ownership_words = 0U;
        if (ownership_size > 0U) {
            const std::size_t ownership_words = 2U * words_per_plane_;
            compress_ownership(
                std::span<const float>(ownership + (i * ownership_size), ownership_size),
                squares_per_plane_,
                std::span<std::uint64_t>(compact.bitpacked_ownership.data(), ownership_words));
            compact.num_ownership_words = checked_u16(ownership_words, "num_ownership_words");
        }
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

// --- Binary serialization (compact format) ---
//
// File layout:
//   [magic: 4 bytes "AZRB"]
//   [version: uint32]
//   [count: uint64]
//   [sizeof_position: uint64]
//   [squares_per_plane: uint32]   // version 2+
//   [positions: count × CompactReplayPosition, oldest first]

namespace {

constexpr char kFileMagic[4] = {'A', 'Z', 'R', 'B'};
constexpr std::uint32_t kFileVersionV1 = 1U;
constexpr std::uint32_t kFileVersionV2 = 2U;
constexpr std::uint32_t kFileVersion = 3U;

struct FileHeaderPrefix {
    char magic[4]{};
    std::uint32_t version = 0U;
};

struct FileHeaderCommon {
    std::uint64_t count = 0U;
    std::uint64_t sizeof_position = 0U;
};

struct FileHeaderV2 {
    char magic[4]{};
    std::uint32_t version = 0U;
    std::uint64_t count = 0U;
    std::uint64_t sizeof_position = 0U;
    std::uint32_t squares_per_plane = 64U;
};

struct LegacyCompactReplayPositionV1 {
    std::array<std::uint64_t, CompactReplayPosition::kMaxBinaryWords> bitpacked_planes{};
    std::array<std::uint8_t, CompactReplayPosition::kMaxFloatPlanes> quantized_float_planes{};
    std::array<std::uint16_t, CompactReplayPosition::kMaxSparsePolicy> policy_actions{};
    std::array<std::uint16_t, CompactReplayPosition::kMaxSparsePolicy> policy_probs_fp16{};
    std::uint8_t num_policy_entries = 0U;
    float value = 0.0F;
    float training_weight = 1.0F;
    std::array<float, CompactReplayPosition::kWdlSize> value_wdl{0.0F, 0.0F, 0.0F};
    std::uint32_t game_id = 0U;
    std::uint16_t move_number = 0U;
    std::uint16_t num_binary_planes = 0U;
    std::uint16_t num_float_planes = 0U;
    std::uint16_t policy_size = 0U;
};

static_assert(
    std::is_trivially_copyable_v<LegacyCompactReplayPositionV1>,
    "LegacyCompactReplayPositionV1 must be trivially copyable");

struct LegacyCompactReplayPositionV2 {
    std::array<std::uint64_t, CompactReplayPosition::kMaxBinaryWords> bitpacked_planes{};
    std::array<std::uint8_t, CompactReplayPosition::kMaxFloatPlanes> quantized_float_planes{};
    std::array<std::uint16_t, CompactReplayPosition::kMaxSparsePolicy> policy_actions{};
    std::array<std::uint16_t, CompactReplayPosition::kMaxSparsePolicy> policy_probs_fp16{};
    std::uint8_t num_policy_entries = 0U;
    float value = 0.0F;
    float training_weight = 1.0F;
    std::array<float, CompactReplayPosition::kWdlSize> value_wdl{0.0F, 0.0F, 0.0F};
    std::uint32_t game_id = 0U;
    std::uint16_t move_number = 0U;
    std::uint16_t num_binary_words = 0U;
    std::uint16_t num_binary_planes = 0U;
    std::uint16_t num_float_planes = 0U;
    std::uint16_t policy_size = 0U;
};

static_assert(
    std::is_trivially_copyable_v<LegacyCompactReplayPositionV2>,
    "LegacyCompactReplayPositionV2 must be trivially copyable");

}  // namespace

void CompactReplayBuffer::save_to_file(const std::string& path) const {
    std::shared_lock lock(mutex_);
    const std::size_t current_count = count_.load(std::memory_order_relaxed);
    const std::size_t current_head = write_head_.load(std::memory_order_relaxed);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("CompactReplayBuffer save_to_file: cannot open " + path);
    }

    FileHeaderV2 header{};
    std::memcpy(header.magic, kFileMagic, 4);
    header.version = kFileVersion;
    header.count = static_cast<std::uint64_t>(current_count);
    header.sizeof_position = static_cast<std::uint64_t>(sizeof(CompactReplayPosition));
    header.squares_per_plane = static_cast<std::uint32_t>(squares_per_plane_);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write positions in logical order (oldest → newest).
    for (std::size_t i = 0U; i < current_count; ++i) {
        const std::size_t physical = to_physical_index(i, current_count, current_head);
        out.write(reinterpret_cast<const char*>(&buffer_[physical]), sizeof(CompactReplayPosition));
    }

    if (!out) {
        throw std::runtime_error("CompactReplayBuffer save_to_file: write error on " + path);
    }
}

std::size_t CompactReplayBuffer::load_from_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("CompactReplayBuffer load_from_file: cannot open " + path);
    }

    FileHeaderPrefix prefix{};
    in.read(reinterpret_cast<char*>(&prefix), sizeof(prefix));
    if (!in) {
        throw std::runtime_error("CompactReplayBuffer load_from_file: truncated header in " + path);
    }
    if (std::memcmp(prefix.magic, kFileMagic, 4) != 0) {
        throw std::runtime_error("CompactReplayBuffer load_from_file: invalid magic in " + path);
    }

    FileHeaderCommon common{};
    std::uint32_t file_squares_per_plane = 64U;
    std::size_t serialized_position_size = 0U;
    bool is_legacy_v1 = false;
    bool is_legacy_v2 = false;
    if (prefix.version == kFileVersionV1) {
        in.read(reinterpret_cast<char*>(&common), sizeof(common));
        serialized_position_size = sizeof(LegacyCompactReplayPositionV1);
        is_legacy_v1 = true;
    } else if (prefix.version == kFileVersionV2 || prefix.version == kFileVersion) {
        struct FileHeaderV2Tail {
            std::uint64_t count = 0U;
            std::uint64_t sizeof_position = 0U;
            std::uint32_t squares_per_plane = 64U;
        } tail;
        in.read(reinterpret_cast<char*>(&tail), sizeof(tail));
        common.count = tail.count;
        common.sizeof_position = tail.sizeof_position;
        file_squares_per_plane = tail.squares_per_plane;
        if (prefix.version == kFileVersionV2) {
            serialized_position_size = sizeof(LegacyCompactReplayPositionV2);
            is_legacy_v2 = true;
        } else {
            serialized_position_size = sizeof(CompactReplayPosition);
        }
    } else {
        throw std::runtime_error("CompactReplayBuffer load_from_file: unsupported version in " + path);
    }
    if (!in) {
        throw std::runtime_error("CompactReplayBuffer load_from_file: truncated header in " + path);
    }
    if (file_squares_per_plane != squares_per_plane_) {
        throw std::runtime_error(
            "CompactReplayBuffer load_from_file: squares_per_plane mismatch in " + path +
            " (file=" + std::to_string(file_squares_per_plane) +
            ", runtime=" + std::to_string(squares_per_plane_) + ")");
    }
    if (common.sizeof_position != static_cast<std::uint64_t>(serialized_position_size)) {
        throw std::runtime_error(
            "CompactReplayBuffer load_from_file: sizeof mismatch in " + path +
            " (file=" + std::to_string(common.sizeof_position) +
            ", runtime=" + std::to_string(serialized_position_size) + ")");
    }

    if (common.count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("CompactReplayBuffer load_from_file: count exceeds platform limits in " + path);
    }
    const auto file_count = static_cast<std::size_t>(common.count);
    // If the file has more positions than our capacity, skip the oldest entries.
    const std::size_t to_load = std::min(file_count, buffer_.size());
    const std::size_t to_skip = file_count - to_load;
    if (to_skip > 0U) {
        in.seekg(
            static_cast<std::streamoff>(to_skip * serialized_position_size),
            std::ios::cur);
    }

    std::unique_lock lock(mutex_);
    for (std::size_t i = 0U; i < to_load; ++i) {
        if (is_legacy_v1) {
            LegacyCompactReplayPositionV1 legacy{};
            in.read(reinterpret_cast<char*>(&legacy), sizeof(legacy));
            if (!in) {
                throw std::runtime_error("CompactReplayBuffer load_from_file: truncated data in " + path);
            }

            CompactReplayPosition converted{};
            converted.bitpacked_planes = legacy.bitpacked_planes;
            converted.quantized_float_planes = legacy.quantized_float_planes;
            converted.policy_actions = legacy.policy_actions;
            converted.policy_probs_fp16 = legacy.policy_probs_fp16;
            converted.num_policy_entries = legacy.num_policy_entries;
            converted.value = legacy.value;
            converted.training_weight = legacy.training_weight;
            converted.value_wdl = legacy.value_wdl;
            converted.game_id = legacy.game_id;
            converted.move_number = legacy.move_number;
            converted.num_binary_words = legacy.num_binary_planes;
            converted.num_binary_planes = legacy.num_binary_planes;
            converted.num_float_planes = legacy.num_float_planes;
            converted.policy_size = legacy.policy_size;
            converted.num_ownership_words = 0U;
            buffer_[i] = converted;
        } else if (is_legacy_v2) {
            LegacyCompactReplayPositionV2 legacy{};
            in.read(reinterpret_cast<char*>(&legacy), sizeof(legacy));
            if (!in) {
                throw std::runtime_error("CompactReplayBuffer load_from_file: truncated data in " + path);
            }

            CompactReplayPosition converted{};
            converted.bitpacked_planes = legacy.bitpacked_planes;
            converted.quantized_float_planes = legacy.quantized_float_planes;
            converted.policy_actions = legacy.policy_actions;
            converted.policy_probs_fp16 = legacy.policy_probs_fp16;
            converted.num_policy_entries = legacy.num_policy_entries;
            converted.value = legacy.value;
            converted.training_weight = legacy.training_weight;
            converted.value_wdl = legacy.value_wdl;
            converted.game_id = legacy.game_id;
            converted.move_number = legacy.move_number;
            converted.num_binary_words = legacy.num_binary_words;
            converted.num_binary_planes = legacy.num_binary_planes;
            converted.num_float_planes = legacy.num_float_planes;
            converted.policy_size = legacy.policy_size;
            converted.num_ownership_words = 0U;
            buffer_[i] = converted;
        } else {
            in.read(reinterpret_cast<char*>(&buffer_[i]), sizeof(CompactReplayPosition));
            if (!in) {
                throw std::runtime_error("CompactReplayBuffer load_from_file: truncated data in " + path);
            }
        }
    }

    for (std::size_t i = 0U; i < to_load; ++i) {
        if (!has_valid_compact_shape(buffer_[i])) {
            throw std::runtime_error("CompactReplayBuffer load_from_file: incompatible compact payload in " + path);
        }
    }
    for (std::size_t i = to_load; i < buffer_.size(); ++i) {
        buffer_[i] = CompactReplayPosition{};
    }

    if (!in) {
        throw std::runtime_error("CompactReplayBuffer load_from_file: truncated data in " + path);
    }

    write_head_.store(to_load % buffer_.size(), std::memory_order_release);
    count_.store(to_load, std::memory_order_release);
    return to_load;
}

std::size_t CompactReplayBuffer::size() const noexcept { return count_.load(std::memory_order_acquire); }

std::size_t CompactReplayBuffer::capacity() const noexcept { return buffer_.size(); }

std::size_t CompactReplayBuffer::write_head() const noexcept {
    return write_head_.load(std::memory_order_acquire);
}

std::size_t CompactReplayBuffer::ownership_payload_size() const {
    std::shared_lock lock(mutex_);
    const std::size_t current_count = count_.load(std::memory_order_relaxed);
    if (current_count == 0U) {
        return 0U;
    }
    const std::size_t current_head = write_head_.load(std::memory_order_relaxed);
    const std::size_t expected_ownership_words = 2U * words_per_plane_;

    bool saw_missing_ownership = false;
    bool saw_present_ownership = false;

    for (std::size_t logical_index = 0U; logical_index < current_count; ++logical_index) {
        const std::size_t physical_index = to_physical_index(logical_index, current_count, current_head);
        const CompactReplayPosition& position = buffer_[physical_index];
        if (!has_valid_compact_shape(position)) {
            throw std::invalid_argument(
                "CompactReplayBuffer ownership_payload_size encountered malformed CompactReplayPosition");
        }
        if (position.num_ownership_words == 0U) {
            saw_missing_ownership = true;
            continue;
        }
        if (position.num_ownership_words != expected_ownership_words) {
            throw std::invalid_argument(
                "CompactReplayBuffer ownership_payload_size encountered inconsistent ownership layout");
        }
        saw_present_ownership = true;
    }

    if (saw_missing_ownership && saw_present_ownership) {
        throw std::invalid_argument(
            "CompactReplayBuffer ownership_payload_size encountered mixed ownership presence");
    }
    if (!saw_present_ownership) {
        return 0U;
    }
    return squares_per_plane_;
}

SamplingStrategy CompactReplayBuffer::sampling_strategy() const noexcept { return sampling_strategy_; }

float CompactReplayBuffer::recency_weight_lambda() const noexcept { return recency_weight_lambda_; }

bool CompactReplayBuffer::has_valid_shape(const ReplayPosition& position) noexcept {
    return position.encoded_state_size > 0U && position.encoded_state_size <= ReplayPosition::kMaxEncodedStateSize &&
           position.policy_size > 0U && position.policy_size <= ReplayPosition::kMaxPolicySize;
}

bool CompactReplayBuffer::has_valid_compact_shape(const CompactReplayPosition& position) const noexcept {
    const std::size_t expected_ownership_words = 2U * words_per_plane_;
    return position.num_binary_words == num_binary_words_ && position.num_binary_planes == num_binary_planes_ &&
           position.num_float_planes == num_float_planes_ && position.policy_size == full_policy_size_ &&
           position.num_policy_entries <= CompactReplayPosition::kMaxSparsePolicy &&
           (position.num_ownership_words == 0U || position.num_ownership_words == expected_ownership_words) &&
           std::isfinite(position.training_weight) && position.training_weight >= 0.0F;
}

std::vector<std::size_t> CompactReplayBuffer::sample_logical_indices(
    const std::size_t population_size,
    const std::size_t sample_size) const {
    std::vector<std::size_t> logical_indices;
    logical_indices.reserve(sample_size);

    if (sampling_strategy_ == SamplingStrategy::kRecencyWeighted) {
        for (std::size_t i = 0U; i < sample_size; ++i) {
            logical_indices.push_back(recency_weighted_index(population_size));
        }
        return logical_indices;
    }

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

std::size_t CompactReplayBuffer::recency_weighted_index(const std::size_t population_size) const {
    if (population_size == 0U) {
        throw std::invalid_argument("CompactReplayBuffer recency_weighted_index population must be positive");
    }
    if (recency_weight_lambda_ <= 0.0F) {
        return uniform_index(population_size);
    }

    const double sample = std::log1p(uniform_unit_interval() * recency_weight_expm1_) /
                          static_cast<double>(recency_weight_lambda_);
    std::size_t index = static_cast<std::size_t>(sample * static_cast<double>(population_size));
    if (index >= population_size) {
        index = population_size - 1U;
    }
    return index;
}

double CompactReplayBuffer::uniform_unit_interval() const {
    std::lock_guard lock(rng_mutex_);
    return std::ldexp(static_cast<double>(rng_()), -64);
}

std::size_t CompactReplayBuffer::to_physical_index(
    const std::size_t logical_index,
    const std::size_t current_count,
    const std::size_t current_write_head) const noexcept {
    const std::size_t base = (current_count == buffer_.size()) ? current_write_head : 0U;
    return (base + logical_index) % buffer_.size();
}

}  // namespace alphazero::selfplay
