#include "mcts/eval_queue.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace alphazero::mcts {

EvalQueue::EvalQueue(BatchEvaluator evaluator, EvalQueueConfig config)
    : evaluator_(std::move(evaluator)),
      config_(config) {
    if (!evaluator_) {
        throw std::invalid_argument("EvalQueue requires a non-null batch evaluator callback");
    }
    if (config_.batch_size == 0U) {
        throw std::invalid_argument("EvalQueue batch_size must be greater than zero");
    }
    if (config_.flush_timeout.count() < 0) {
        throw std::invalid_argument("EvalQueue flush_timeout must be non-negative");
    }
}

EvalQueue::~EvalQueue() { stop(); }

EvaluateFn make_eval_queue_evaluator(
    EvalQueue& queue,
    std::size_t encoded_state_size,
    int action_space_size) {
    if (encoded_state_size == 0U) {
        throw std::invalid_argument("make_eval_queue_evaluator encoded_state_size must be greater than zero");
    }
    if (action_space_size <= 0) {
        throw std::invalid_argument("make_eval_queue_evaluator action_space_size must be positive");
    }

    return [&queue, encoded_state_size, action_space_size](const GameState& state) -> EvaluationResult {
        thread_local std::vector<float> encoded_state_buffer;
        if (encoded_state_buffer.size() != encoded_state_size) {
            encoded_state_buffer.resize(encoded_state_size);
        }

        state.encode(encoded_state_buffer.data());
        EvalResult queue_result = queue.submit_and_wait(encoded_state_buffer.data());
        if (queue_result.policy_logits.size() != static_cast<std::size_t>(action_space_size)) {
            throw std::runtime_error("make_eval_queue_evaluator received policy with unexpected size");
        }

        EvaluationResult result;
        result.policy = std::move(queue_result.policy_logits);
        result.value = queue_result.value;
        result.policy_is_logits = true;
        return result;
    };
}

EvalResult EvalQueue::submit_and_wait(const float* encoded_state) {
    if (encoded_state == nullptr) {
        throw std::invalid_argument("EvalQueue encoded_state must be non-null");
    }

    auto request = std::make_shared<PendingRequest>(encoded_state);
    {
        std::lock_guard lock(pending_mutex_);
        if (stop_requested_) {
            throw std::runtime_error("EvalQueue is stopped");
        }
        pending_.push_back(request);
    }
    pending_cv_.notify_one();

    request->done.acquire();
    if (request->error) {
        std::rethrow_exception(request->error);
    }
    return std::move(request->result);
}

void EvalQueue::process_batch() {
    std::vector<std::shared_ptr<PendingRequest>> batch;
    {
        std::unique_lock lock(pending_mutex_);
        if (config_.wait_timeout.count() > 0) {
            pending_cv_.wait_for(lock, config_.wait_timeout, [this] {
                return stop_requested_ || !pending_.empty();
            });
        } else {
            pending_cv_.wait(lock, [this] { return stop_requested_ || !pending_.empty(); });
        }
        if (pending_.empty()) {
            return;
        }

        const auto deadline = std::chrono::steady_clock::now() + config_.flush_timeout;
        while (!stop_requested_ && pending_.size() < config_.batch_size) {
            const bool wake_condition = pending_cv_.wait_until(lock, deadline, [this] {
                return stop_requested_ || pending_.size() >= config_.batch_size;
            });
            if (wake_condition) {
                break;
            }
            // Timeout elapsed before batch_size was reached.
            break;
        }

        const std::size_t take_count = std::min(config_.batch_size, pending_.size());
        batch.reserve(take_count);
        for (std::size_t i = 0; i < take_count; ++i) {
            batch.push_back(std::move(pending_.front()));
            pending_.pop_front();
        }
    }

    std::vector<const float*> inputs;
    inputs.reserve(batch.size());
    for (const auto& request : batch) {
        inputs.push_back(request->encoded_state);
    }

    std::vector<EvalResult> outputs;
    std::exception_ptr batch_error;
    try {
        outputs = evaluator_(inputs);
        if (outputs.size() != batch.size()) {
            throw std::runtime_error("EvalQueue evaluator output size does not match request count");
        }
    } catch (...) {
        batch_error = std::current_exception();
    }

    for (std::size_t i = 0; i < batch.size(); ++i) {
        if (batch_error) {
            batch[i]->error = batch_error;
        } else {
            batch[i]->result = std::move(outputs[i]);
        }
        batch[i]->done.release();
    }
}

void EvalQueue::stop() {
    std::deque<std::shared_ptr<PendingRequest>> pending_to_fail;
    std::exception_ptr stop_error;
    {
        std::lock_guard lock(pending_mutex_);
        if (stop_requested_) {
            return;
        }
        stop_requested_ = true;
        pending_to_fail.swap(pending_);
        stop_error = std::make_exception_ptr(std::runtime_error("EvalQueue stopped before processing request"));
    }

    for (const auto& request : pending_to_fail) {
        request->error = stop_error;
        request->done.release();
    }
    pending_cv_.notify_all();
}

}  // namespace alphazero::mcts
