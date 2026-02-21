#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <semaphore>
#include <vector>

namespace alphazero::mcts {

struct EvalResult {
    std::vector<float> policy_logits;
    float value = 0.0F;
};

struct EvalQueueConfig {
    std::size_t batch_size = 256;
    std::chrono::microseconds flush_timeout{100};
    /// Maximum time process_batch() will block waiting for the first request.
    /// Default 100ms.  A finite timeout is essential because MCTS simulations
    /// that reach terminal game states complete without submitting an eval
    /// request, so the caller must be able to re-enter the loop promptly.
    std::chrono::microseconds wait_timeout{100'000};
};

class EvalQueue {
public:
    using BatchEvaluator = std::function<std::vector<EvalResult>(const std::vector<const float*>&)>;

    explicit EvalQueue(BatchEvaluator evaluator, EvalQueueConfig config = {});
    ~EvalQueue();

    EvalQueue(const EvalQueue&) = delete;
    EvalQueue& operator=(const EvalQueue&) = delete;
    EvalQueue(EvalQueue&&) = delete;
    EvalQueue& operator=(EvalQueue&&) = delete;

    [[nodiscard]] EvalResult submit_and_wait(const float* encoded_state);
    void process_batch();
    void stop();

private:
    struct PendingRequest {
        explicit PendingRequest(const float* state) : encoded_state(state) {}

        const float* encoded_state = nullptr;
        EvalResult result;
        std::exception_ptr error;
        std::binary_semaphore done{0};
    };

    BatchEvaluator evaluator_;
    EvalQueueConfig config_;

    std::mutex pending_mutex_;
    std::condition_variable pending_cv_;
    std::deque<std::shared_ptr<PendingRequest>> pending_;
    bool stop_requested_ = false;
};

}  // namespace alphazero::mcts
