#pragma once

#include <cstddef>

#include "mcts/mcts_node.h"

namespace alphazero::mcts {

template <typename NodeType>
class NodeStoreT {
public:
    virtual ~NodeStoreT() = default;

    [[nodiscard]] virtual NodeId allocate() = 0;

    [[nodiscard]] virtual NodeType& get(NodeId id) = 0;
    [[nodiscard]] virtual const NodeType& get(NodeId id) const = 0;

    virtual void release_subtree(NodeId root) = 0;
    virtual void reset() = 0;

    [[nodiscard]] virtual std::size_t nodes_allocated() const = 0;
    [[nodiscard]] virtual std::size_t memory_used_bytes() const = 0;
};

using NodeStore = NodeStoreT<MCTSNode>;
using ChessNodeStore = NodeStoreT<ChessMCTSNode>;
using GoNodeStore = NodeStoreT<GoMCTSNode>;

}  // namespace alphazero::mcts
