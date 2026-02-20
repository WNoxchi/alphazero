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
        pending_cv_.wait(lock, [this] { return stop_requested_ || !pending_.empty(); });
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
