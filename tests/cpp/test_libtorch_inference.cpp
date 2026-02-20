#include "nn/libtorch_inference.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#if ALPHAZERO_HAS_TORCH
#include <torch/script.h>
#endif

#include <gtest/gtest.h>

namespace {

using alphazero::GameConfig;
using alphazero::GameState;
using alphazero::nn::LibTorchInference;

class InferenceTestConfig final : public GameConfig {
public:
    InferenceTestConfig(
        const int input_channels,
        const int rows,
        const int cols,
        const int action_space,
        const ValueHeadType value_type) {
        name = "inference-test";
        board_rows = rows;
        board_cols = cols;
        planes_per_step = input_channels;
        num_history_steps = 1;
        constant_planes = 0;
        total_input_channels = input_channels;
        action_space_size = action_space;
        dirichlet_alpha = 0.3F;
        max_game_length = 1;
        value_head_type = value_type;
        supports_symmetry = false;
        num_symmetries = 1;
    }

    [[nodiscard]] std::unique_ptr<GameState> new_game() const override { return nullptr; }
};

#if ALPHAZERO_HAS_TORCH
class ScopedTempFile {
public:
    explicit ScopedTempFile(std::filesystem::path path) : path_(std::move(path)) {}

    ~ScopedTempFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    ScopedTempFile(const ScopedTempFile&) = delete;
    ScopedTempFile& operator=(const ScopedTempFile&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

[[nodiscard]] std::string float_literal(const float value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

[[nodiscard]] std::filesystem::path unique_model_path(const std::string& stem) {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream file_name;
    file_name << stem << "_" << std::hex << ticks << ".pt";
    return std::filesystem::temp_directory_path() / file_name.str();
}

[[nodiscard]] ScopedTempFile write_scalar_model(
    const std::string& stem,
    const int action_space_size,
    const float policy_bias,
    const float value_bias) {
    const std::filesystem::path path = unique_model_path(stem);

    torch::jit::Module module("ScalarModel");
    std::ostringstream source;
    source << "def forward(self, x):\n";
    source << "    flat = x.reshape(x.size(0), -1)\n";
    source << "    policy = flat[:, 0:" << action_space_size << "] + " << float_literal(policy_bias) << "\n";
    source << "    value = flat.mean(dim=1, keepdim=True) + " << float_literal(value_bias) << "\n";
    source << "    return policy, value\n";
    module.define(source.str());
    module.save(path.string());

    return ScopedTempFile(path);
}

[[nodiscard]] ScopedTempFile write_wdl_model(const std::string& stem, const int action_space_size) {
    const std::filesystem::path path = unique_model_path(stem);

    torch::jit::Module module("WdlModel");
    std::ostringstream source;
    source << "def forward(self, x):\n";
    source << "    flat = x.reshape(x.size(0), -1)\n";
    source << "    policy = flat[:, 0:" << action_space_size << "]\n";
    source << "    v0 = flat[:, 0]\n";
    source << "    v1 = flat[:, 1]\n";
    source << "    v2 = flat[:, 2]\n";
    source << "    value = torch.stack((v0, v1, v2), dim=1)\n";
    source << "    return policy, value\n";
    module.define(source.str());
    module.save(path.string());

    return ScopedTempFile(path);
}
#endif

}  // namespace

// WHY: Constructor-level validation prevents silent shape mismatches that would otherwise fail deep in inference.
TEST(LibTorchInferenceTest, RejectsInvalidGameConfiguration) {
    InferenceTestConfig invalid_channels(0, 2, 2, 4, GameConfig::ValueHeadType::SCALAR);
    EXPECT_THROW(static_cast<void>(LibTorchInference(invalid_channels)), std::invalid_argument);

    InferenceTestConfig invalid_rows(2, 0, 2, 4, GameConfig::ValueHeadType::SCALAR);
    EXPECT_THROW(static_cast<void>(LibTorchInference(invalid_rows)), std::invalid_argument);

    InferenceTestConfig invalid_actions(2, 2, 2, 0, GameConfig::ValueHeadType::SCALAR);
    EXPECT_THROW(static_cast<void>(LibTorchInference(invalid_actions)), std::invalid_argument);
}

// WHY: Input-pointer and batch-size guards must fail early before touching backend state to avoid undefined behavior.
TEST(LibTorchInferenceTest, ValidatesInferArguments) {
    InferenceTestConfig config(2, 2, 2, 4, GameConfig::ValueHeadType::SCALAR);
    LibTorchInference inference(config);

    std::array<float, 8> input{};
    std::array<float, 4> policy_out{};
    std::array<float, 1> value_out{};

    EXPECT_THROW(inference.infer(nullptr, 1, policy_out.data(), value_out.data()), std::invalid_argument);
    EXPECT_THROW(inference.infer(input.data(), 1, nullptr, value_out.data()), std::invalid_argument);
    EXPECT_THROW(inference.infer(input.data(), 1, policy_out.data(), nullptr), std::invalid_argument);
    EXPECT_THROW(inference.infer(input.data(), 0, policy_out.data(), value_out.data()), std::invalid_argument);
}

#if ALPHAZERO_HAS_TORCH
// WHY: Scalar-head inference is the hot path for Go; this verifies batched policy/value extraction and output layout.
TEST(LibTorchInferenceTest, RunsScalarBatchInferenceWithKnownOutputs) {
    InferenceTestConfig config(2, 2, 2, 5, GameConfig::ValueHeadType::SCALAR);
    LibTorchInference inference(config);
    const ScopedTempFile model = write_scalar_model("az_scalar", config.action_space_size, 0.0F, 0.0F);
    inference.load_weights(model.path().string());

    const std::array<float, 16> input = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
        10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F,
    };
    std::array<float, 10> policy_out{};
    std::array<float, 2> value_out{};

    inference.infer(input.data(), 2, policy_out.data(), value_out.data());

    const std::array<float, 10> expected_policy = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
        10.0F, 11.0F, 12.0F, 13.0F, 14.0F,
    };
    for (std::size_t i = 0; i < expected_policy.size(); ++i) {
        EXPECT_FLOAT_EQ(policy_out[i], expected_policy[i]);
    }

    EXPECT_FLOAT_EQ(value_out[0], 4.5F);
    EXPECT_FLOAT_EQ(value_out[1], 13.5F);
}

// WHY: Chess inference depends on 3-way WDL outputs; this verifies shape handling for non-scalar value heads.
TEST(LibTorchInferenceTest, RunsWdlBatchInferenceWithKnownOutputs) {
    InferenceTestConfig config(1, 2, 2, 4, GameConfig::ValueHeadType::WDL);
    LibTorchInference inference(config);
    const ScopedTempFile model = write_wdl_model("az_wdl", config.action_space_size);
    inference.load_weights(model.path().string());

    const std::array<float, 8> input = {
        2.0F, 4.0F, 6.0F, 8.0F,
        1.0F, 3.0F, 5.0F, 7.0F,
    };
    std::array<float, 8> policy_out{};
    std::array<float, 6> value_out{};

    inference.infer(input.data(), 2, policy_out.data(), value_out.data());

    const std::array<float, 8> expected_policy = {
        2.0F, 4.0F, 6.0F, 8.0F,
        1.0F, 3.0F, 5.0F, 7.0F,
    };
    for (std::size_t i = 0; i < expected_policy.size(); ++i) {
        EXPECT_FLOAT_EQ(policy_out[i], expected_policy[i]);
    }

    const std::array<float, 6> expected_value = {
        2.0F, 4.0F, 6.0F,
        1.0F, 3.0F, 5.0F,
    };
    for (std::size_t i = 0; i < expected_value.size(); ++i) {
        EXPECT_FLOAT_EQ(value_out[i], expected_value[i]);
    }
}

// WHY: Training periodically refreshes inference checkpoints; load_weights must replace outputs immediately.
TEST(LibTorchInferenceTest, ReplacesModelOutputsAfterWeightReload) {
    InferenceTestConfig config(1, 2, 2, 3, GameConfig::ValueHeadType::SCALAR);
    LibTorchInference inference(config);

    const ScopedTempFile first_model = write_scalar_model("az_reload_a", config.action_space_size, 0.0F, 0.0F);
    const ScopedTempFile second_model = write_scalar_model("az_reload_b", config.action_space_size, 5.0F, -2.0F);

    const std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 3> policy_out{};
    std::array<float, 1> value_out{};

    inference.load_weights(first_model.path().string());
    inference.infer(input.data(), 1, policy_out.data(), value_out.data());
    EXPECT_FLOAT_EQ(policy_out[0], 1.0F);
    EXPECT_FLOAT_EQ(policy_out[1], 2.0F);
    EXPECT_FLOAT_EQ(policy_out[2], 3.0F);
    EXPECT_FLOAT_EQ(value_out[0], 2.5F);

    inference.load_weights(second_model.path().string());
    inference.infer(input.data(), 1, policy_out.data(), value_out.data());
    EXPECT_FLOAT_EQ(policy_out[0], 6.0F);
    EXPECT_FLOAT_EQ(policy_out[1], 7.0F);
    EXPECT_FLOAT_EQ(policy_out[2], 8.0F);
    EXPECT_FLOAT_EQ(value_out[0], 0.5F);
}
#else
// WHY: In no-LibTorch builds, the class should fail fast with actionable errors instead of pretending inference works.
TEST(LibTorchInferenceTest, ReportsUnavailableBackendWhenLibTorchIsMissing) {
    InferenceTestConfig config(1, 2, 2, 4, GameConfig::ValueHeadType::SCALAR);
    LibTorchInference inference(config);

    std::array<float, 4> input = {1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 4> policy_out{};
    std::array<float, 1> value_out{};

    EXPECT_FALSE(inference.torch_available());
    EXPECT_FALSE(inference.using_cuda());
    EXPECT_THROW(inference.load_weights("/tmp/nonexistent_model.pt"), std::runtime_error);
    EXPECT_THROW(inference.infer(input.data(), 1, policy_out.data(), value_out.data()), std::runtime_error);
}
#endif
