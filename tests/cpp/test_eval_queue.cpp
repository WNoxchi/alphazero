#include "mcts/eval_queue.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using alphazero::mcts::EvalQueue;
using alphazero::mcts::EvalQueueConfig;
using alphazero::mcts::EvalResult;

class EvalQueueConsumer {
public:
    explicit EvalQueueConsumer(EvalQueue& queue) : queue_(queue), thread_([this] { run(); }) {}

    ~EvalQueueConsumer() { stop(); }

    EvalQueueConsumer(const EvalQueueConsumer&) = delete;
    EvalQueueConsumer& operator=(const EvalQueueConsumer&) = delete;

    void stop() {
        bool expected = false;
        if (stop_requested_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            queue_.stop();
            if (thread_.joinable()) {
                thread_.join();
            }
        }
    }

private:
    void run() {
        while (!stop_requested_.load(std::memory_order_acquire)) {
            queue_.process_batch();
        }
    }

    EvalQueue& queue_;
    std::atomic<bool> stop_requested_{false};
    std::thread thread_;
};

}  // namespace

// WHY: The queue is the bridge between many MCTS workers and one GPU worker, so per-request result routing must
// remain correct under concurrent multi-producer submission.
TEST(EvalQueueTest, DispatchesResultsForConcurrentProducers) {
    EvalQueueConfig config{};
    config.batch_size = 4;
    config.flush_timeout = std::chrono::milliseconds(10);

    EvalQueue queue(
        [](const std::vector<const float*>& inputs) -> std::vector<EvalResult> {
            std::vector<EvalResult> outputs;
            outputs.reserve(inputs.size());
            for (const float* input : inputs) {
                const float id = input[0];
                outputs.push_back(EvalResult{
                    .policy_logits = {id, id + 1.0F},
                    .value = id * 2.0F,
                });
            }
            return outputs;
        },
        config);
    EvalQueueConsumer consumer(queue);

    constexpr int kProducerCount = 8;
    constexpr int kRequestsPerProducer = 12;
    constexpr int kTotalRequests = kProducerCount * kRequestsPerProducer;

    std::vector<float> observed_values(static_cast<std::size_t>(kTotalRequests), -1.0F);
    std::vector<float> observed_policy0(static_cast<std::size_t>(kTotalRequests), -1.0F);
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);

    for (int producer = 0; producer < kProducerCount; ++producer) {
        producers.emplace_back([&queue, &observed_values, &observed_policy0, producer] {
            for (int i = 0; i < kRequestsPerProducer; ++i) {
                const int id = producer * kRequestsPerProducer + i;
                float encoded_state[1] = {static_cast<float>(id)};
                const EvalResult result = queue.submit_and_wait(encoded_state);
                observed_values[static_cast<std::size_t>(id)] = result.value;
                observed_policy0[static_cast<std::size_t>(id)] =
                    result.policy_logits.empty() ? -1.0F : result.policy_logits.front();
            }
        });
    }

    for (auto& producer : producers) {
        producer.join();
    }

    consumer.stop();

    for (int id = 0; id < kTotalRequests; ++id) {
        EXPECT_FLOAT_EQ(observed_values[static_cast<std::size_t>(id)], static_cast<float>(id) * 2.0F);
        EXPECT_FLOAT_EQ(observed_policy0[static_cast<std::size_t>(id)], static_cast<float>(id));
    }
}

// WHY: Timeout flushing prevents tail-latency stalls when pending work never fills a full batch, which is critical at
// the end of moves/games.
TEST(EvalQueueTest, FlushesPartialBatchAfterTimeout) {
    EvalQueueConfig config{};
    config.batch_size = 16;
    config.flush_timeout = std::chrono::milliseconds(40);

    std::mutex batch_mutex;
    std::vector<std::size_t> observed_batch_sizes;

    EvalQueue queue(
        [&batch_mutex, &observed_batch_sizes](const std::vector<const float*>& inputs) -> std::vector<EvalResult> {
            {
                std::lock_guard lock(batch_mutex);
                observed_batch_sizes.push_back(inputs.size());
            }

            std::vector<EvalResult> outputs;
            outputs.reserve(inputs.size());
            for (const float* input : inputs) {
                outputs.push_back(EvalResult{
                    .policy_logits = {*input},
                    .value = input[0],
                });
            }
            return outputs;
        },
        config);
    EvalQueueConsumer consumer(queue);

    const auto start = std::chrono::steady_clock::now();
    float encoded_state[1] = {7.0F};
    const EvalResult result = queue.submit_and_wait(encoded_state);
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    consumer.stop();

    EXPECT_FLOAT_EQ(result.value, 7.0F);
    EXPECT_GE(elapsed_ms, 15);
    EXPECT_LT(elapsed_ms, 300);

    std::lock_guard lock(batch_mutex);
    ASSERT_EQ(observed_batch_sizes.size(), 1U);
    EXPECT_EQ(observed_batch_sizes.front(), 1U);
}

// WHY: Tree-parallel MCTS can flood the queue at move boundaries; high-contention behavior must be deadlock-free and
// process every request exactly once.
TEST(EvalQueueTest, ProcessesAllRequestsUnderHighContention) {
    EvalQueueConfig config{};
    config.batch_size = 32;
    config.flush_timeout = std::chrono::milliseconds(5);

    std::atomic<int> processed_requests{0};
    EvalQueue queue(
        [&processed_requests](const std::vector<const float*>& inputs) -> std::vector<EvalResult> {
            processed_requests.fetch_add(static_cast<int>(inputs.size()), std::memory_order_relaxed);
            std::vector<EvalResult> outputs;
            outputs.reserve(inputs.size());
            for (const float* input : inputs) {
                outputs.push_back(EvalResult{
                    .policy_logits = {input[0]},
                    .value = input[0] + 1.0F,
                });
            }
            return outputs;
        },
        config);
    EvalQueueConsumer consumer(queue);

    constexpr int kProducerCount = 20;
    constexpr int kRequestsPerProducer = 40;
    constexpr int kTotalRequests = kProducerCount * kRequestsPerProducer;

    std::vector<float> observed_values(static_cast<std::size_t>(kTotalRequests), -1.0F);
    std::atomic<int> next_id{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);

    for (int producer = 0; producer < kProducerCount; ++producer) {
        producers.emplace_back([&queue, &observed_values, &next_id] {
            for (int i = 0; i < kRequestsPerProducer; ++i) {
                const int id = next_id.fetch_add(1, std::memory_order_relaxed);
                float encoded_state[1] = {static_cast<float>(id)};
                const EvalResult result = queue.submit_and_wait(encoded_state);
                observed_values[static_cast<std::size_t>(id)] = result.value;
            }
        });
    }

    for (auto& producer : producers) {
        producer.join();
    }

    consumer.stop();

    EXPECT_EQ(next_id.load(std::memory_order_relaxed), kTotalRequests);
    EXPECT_EQ(processed_requests.load(std::memory_order_relaxed), kTotalRequests);
    for (int id = 0; id < kTotalRequests; ++id) {
        EXPECT_FLOAT_EQ(observed_values[static_cast<std::size_t>(id)], static_cast<float>(id) + 1.0F);
    }
}

// WHY: Shutdown paths are exercised during process teardown and must fail fast instead of blocking producer threads.
TEST(EvalQueueTest, RejectsSubmissionsAfterStop) {
    EvalQueue queue(
        [](const std::vector<const float*>& inputs) -> std::vector<EvalResult> {
            return std::vector<EvalResult>(inputs.size());
        });

    queue.stop();

    float encoded_state[1] = {1.0F};
    EXPECT_THROW(static_cast<void>(queue.submit_and_wait(encoded_state)), std::runtime_error);
}
