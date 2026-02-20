#pragma once

#include <string>

namespace alphazero::nn {

class NeuralNetInference {
public:
    virtual ~NeuralNetInference() = default;

    // Batch inference over encoded states.
    //
    // input:
    //   Shape (batch_size, input_channels, board_rows, board_cols)
    // policy_out:
    //   Shape (batch_size, action_space_size), raw logits over the full action space.
    // value_out:
    //   Shape (batch_size, 1) for scalar value heads, or (batch_size, 3) for WDL heads.
    virtual void infer(const float* input, int batch_size, float* policy_out, float* value_out) = 0;

    // Load model weights from a serialized checkpoint (e.g., TorchScript file).
    virtual void load_weights(const std::string& path) = 0;
};

}  // namespace alphazero::nn
