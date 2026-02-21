#pragma once

#include <cstddef>

#include "mcts/mcts_node.h"

namespace alphazero::mcts {

class NodeStore {
public:
    virtual ~NodeStore() = default;

    [[nodiscard]] virtual NodeId allocate() = 0;

    [[nodiscard]] virtual MCTSNode& get(NodeId id) = 0;
    [[nodiscard]] virtual const MCTSNode& get(NodeId id) const = 0;

    virtual void release_subtree(NodeId root) = 0;
    virtual void reset() = 0;

    [[nodiscard]] virtual std::size_t nodes_allocated() const = 0;
    [[nodiscard]] virtual std::size_t memory_used_bytes() const = 0;
};

}  // namespace alphazero::mcts
