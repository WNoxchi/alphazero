#pragma once

#include <shared_mutex>
#include <string>
#include <utility>

#include "games/game_config.h"
#include "nn/nn_inference.h"

#ifndef ALPHAZERO_HAS_TORCH
#define ALPHAZERO_HAS_TORCH 0
#endif

#if ALPHAZERO_HAS_TORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace alphazero::nn {

class LibTorchInference final : public NeuralNetInference {
public:
    explicit LibTorchInference(const GameConfig& game_config, std::string weights_path = {});

    void infer(const float* input, int batch_size, float* policy_out, float* value_out) override;
    void load_weights(const std::string& path) override;

    [[nodiscard]] int input_channels() const noexcept { return input_channels_; }
    [[nodiscard]] int board_rows() const noexcept { return board_rows_; }
    [[nodiscard]] int board_cols() const noexcept { return board_cols_; }
    [[nodiscard]] int action_space_size() const noexcept { return action_space_size_; }
    [[nodiscard]] int value_output_size() const noexcept { return value_output_size_; }

    [[nodiscard]] bool torch_available() const noexcept { return kTorchAvailable; }
    [[nodiscard]] bool using_cuda() const noexcept;

private:
    static constexpr bool kTorchAvailable = ALPHAZERO_HAS_TORCH != 0;

    [[nodiscard]] static int value_output_size_for(const GameConfig& game_config);

#if ALPHAZERO_HAS_TORCH
    [[nodiscard]] static std::pair<torch::Tensor, torch::Tensor> unpack_forward_output(const torch::jit::IValue& output);
    [[nodiscard]] static torch::Device select_device();
#endif

    int input_channels_ = 0;
    int board_rows_ = 0;
    int board_cols_ = 0;
    int action_space_size_ = 0;
    int value_output_size_ = 0;

    mutable std::shared_mutex module_mutex_;
    bool weights_loaded_ = false;

#if ALPHAZERO_HAS_TORCH
    torch::jit::script::Module module_;
    torch::Device device_{torch::kCPU};
#endif
};

}  // namespace alphazero::nn
