#include "nn/libtorch_inference.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if ALPHAZERO_HAS_TORCH
#include <torch/cuda.h>
#endif

namespace alphazero::nn {

LibTorchInference::LibTorchInference(const GameConfig& game_config, std::string weights_path)
    : input_channels_(game_config.total_input_channels),
      board_rows_(game_config.board_rows),
      board_cols_(game_config.board_cols),
      action_space_size_(game_config.action_space_size),
      value_output_size_(value_output_size_for(game_config))
#if ALPHAZERO_HAS_TORCH
      ,
      device_(select_device())
#endif
{
    if (input_channels_ <= 0) {
        throw std::invalid_argument("LibTorchInference requires positive input channels");
    }
    if (board_rows_ <= 0 || board_cols_ <= 0) {
        throw std::invalid_argument("LibTorchInference requires positive board dimensions");
    }
    if (action_space_size_ <= 0) {
        throw std::invalid_argument("LibTorchInference requires a positive action space size");
    }
    if (value_output_size_ <= 0) {
        throw std::invalid_argument("LibTorchInference requires a valid value output size");
    }

    if (!weights_path.empty()) {
        load_weights(weights_path);
    }
}

void LibTorchInference::infer(const float* input, const int batch_size, float* policy_out, float* value_out) {
    if (input == nullptr) {
        throw std::invalid_argument("LibTorchInference input pointer must be non-null");
    }
    if (policy_out == nullptr) {
        throw std::invalid_argument("LibTorchInference policy_out pointer must be non-null");
    }
    if (value_out == nullptr) {
        throw std::invalid_argument("LibTorchInference value_out pointer must be non-null");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("LibTorchInference batch_size must be greater than zero");
    }

    const auto batch = static_cast<std::int64_t>(batch_size);
    const std::size_t policy_elements =
        static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(action_space_size_);
    const std::size_t value_elements = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(value_output_size_);

    if (policy_elements > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        value_elements > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error("LibTorchInference output tensor size exceeds supported bounds");
    }

#if ALPHAZERO_HAS_TORCH
    std::shared_lock model_lock(module_mutex_);
    if (!weights_loaded_) {
        throw std::logic_error("LibTorchInference weights are not loaded");
    }

    try {
        torch::NoGradGuard no_grad;
        const std::vector<std::int64_t> input_shape{
            batch,
            static_cast<std::int64_t>(input_channels_),
            static_cast<std::int64_t>(board_rows_),
            static_cast<std::int64_t>(board_cols_),
        };

        torch::Tensor input_cpu = torch::from_blob(
            const_cast<float*>(input),
            input_shape,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        torch::Tensor model_input = device_.is_cuda() ? input_cpu.to(device_, torch::kFloat32, false, false) : input_cpu;

        auto [policy_tensor, value_tensor] = unpack_forward_output(module_.forward({model_input}));

        if (policy_tensor.dim() == 1 && batch_size == 1 &&
            policy_tensor.size(0) == static_cast<std::int64_t>(action_space_size_)) {
            policy_tensor = policy_tensor.unsqueeze(0);
        }
        if (policy_tensor.dim() != 2 || policy_tensor.size(0) != batch ||
            policy_tensor.size(1) != static_cast<std::int64_t>(action_space_size_)) {
            throw std::runtime_error("LibTorchInference model policy output has an unexpected shape");
        }

        if (value_tensor.dim() == 1 && value_output_size_ == 1 && value_tensor.size(0) == batch) {
            value_tensor = value_tensor.unsqueeze(1);
        }
        if (value_tensor.dim() != 2 || value_tensor.size(0) != batch ||
            value_tensor.size(1) != static_cast<std::int64_t>(value_output_size_)) {
            throw std::runtime_error("LibTorchInference model value output has an unexpected shape");
        }

        policy_tensor = policy_tensor.to(torch::kCPU, torch::kFloat32, false, false).contiguous();
        value_tensor = value_tensor.to(torch::kCPU, torch::kFloat32, false, false).contiguous();

        std::copy_n(policy_tensor.data_ptr<float>(), policy_elements, policy_out);
        std::copy_n(value_tensor.data_ptr<float>(), value_elements, value_out);
    } catch (const c10::Error& error) {
        throw std::runtime_error(std::string("LibTorchInference inference failed: ") + error.what());
    }
#else
    static_cast<void>(batch);
    throw std::runtime_error("LibTorchInference was built without LibTorch support");
#endif
}

void LibTorchInference::load_weights(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("LibTorchInference weights path must be non-empty");
    }

#if ALPHAZERO_HAS_TORCH
    try {
        torch::jit::script::Module loaded = torch::jit::load(path, device_);
        loaded.eval();
        loaded.to(device_);

        std::unique_lock lock(module_mutex_);
        module_ = std::move(loaded);
        weights_loaded_ = true;
    } catch (const c10::Error& error) {
        throw std::runtime_error(
            std::string("LibTorchInference failed to load weights from '") + path + "': " + error.what());
    }
#else
    static_cast<void>(path);
    throw std::runtime_error("LibTorchInference was built without LibTorch support");
#endif
}

bool LibTorchInference::using_cuda() const noexcept {
#if ALPHAZERO_HAS_TORCH
    return device_.is_cuda();
#else
    return false;
#endif
}

int LibTorchInference::value_output_size_for(const GameConfig& game_config) {
    switch (game_config.value_head_type) {
        case GameConfig::ValueHeadType::SCALAR:
            return 1;
        case GameConfig::ValueHeadType::WDL:
            return 3;
        default:
            break;
    }
    throw std::invalid_argument("LibTorchInference encountered an unknown value head type");
}

#if ALPHAZERO_HAS_TORCH
std::pair<torch::Tensor, torch::Tensor> LibTorchInference::unpack_forward_output(const torch::jit::IValue& output) {
    if (output.isTuple()) {
        const auto& elements = output.toTupleRef().elements();
        if (elements.size() != 2U) {
            throw std::runtime_error("LibTorchInference expected model output tuple of size 2");
        }
        if (!elements[0].isTensor() || !elements[1].isTensor()) {
            throw std::runtime_error("LibTorchInference expected tensor outputs for policy/value");
        }
        return {elements[0].toTensor(), elements[1].toTensor()};
    }

    if (output.isList()) {
        const c10::List<c10::IValue> list = output.toList();
        if (list.size() != 2U) {
            throw std::runtime_error("LibTorchInference expected model output list of size 2");
        }
        if (!list.get(0).isTensor() || !list.get(1).isTensor()) {
            throw std::runtime_error("LibTorchInference expected tensor outputs for policy/value");
        }
        return {list.get(0).toTensor(), list.get(1).toTensor()};
    }

    throw std::runtime_error("LibTorchInference expected model output as (policy, value)");
}

torch::Device LibTorchInference::select_device() {
    if (torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
}
#endif

}  // namespace alphazero::nn
