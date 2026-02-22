#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "mcts/node_store.h"

namespace alphazero::mcts {

template <typename NodeType>
class ArenaNodeStoreT final : public NodeStoreT<NodeType> {
public:
    static constexpr std::size_t kDefaultCapacity = 8192;

    explicit ArenaNodeStoreT(std::size_t capacity = kDefaultCapacity);

    [[nodiscard]] NodeId allocate() override;
    [[nodiscard]] NodeType& get(NodeId id) override;
    [[nodiscard]] const NodeType& get(NodeId id) const override;

    void release_subtree(NodeId root) override;
    void reset() override;

    [[nodiscard]] std::size_t nodes_allocated() const override;
    [[nodiscard]] std::size_t memory_used_bytes() const override;

    // Preserve one child subtree when changing the root after selecting a move.
    [[nodiscard]] NodeId reuse_subtree(NodeId old_root, NodeId preserved_child);

    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    [[nodiscard]] bool is_within_capacity(NodeId id) const noexcept;
    [[nodiscard]] bool is_live(NodeId id) const noexcept;
    void validate_live_node(NodeId id) const;
    void release_single_node(NodeId id);

    std::vector<NodeType> arena_;
    std::vector<std::uint32_t> node_epoch_;
    std::vector<NodeId> free_list_;

    NodeId next_free_ = 0;
    std::size_t live_nodes_ = 0;
    std::uint32_t epoch_ = 1;
};

using ArenaNodeStore = ArenaNodeStoreT<MCTSNode>;
using ChessArenaNodeStore = ArenaNodeStoreT<ChessMCTSNode>;
using GoArenaNodeStore = ArenaNodeStoreT<GoMCTSNode>;

}  // namespace alphazero::mcts
