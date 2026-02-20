#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "games/chess/bitboard.h"
#include "games/game_state.h"

namespace alphazero::chess {

class ChessState final : public GameState {
public:
    static constexpr int kHistorySteps = 8;
    static constexpr int kPlanesPerStep = 14;
    static constexpr int kConstantPlanes = 7;
    static constexpr int kTotalInputChannels = (kHistorySteps * kPlanesPerStep) + kConstantPlanes;
    static constexpr int kMaxGameLength = 512;

    ChessState();
    explicit ChessState(const ChessPosition& position);
    [[nodiscard]] static ChessState from_fen(const std::string& fen);
    [[nodiscard]] std::string to_fen() const;
    [[nodiscard]] static std::string actions_to_pgn(
        const std::vector<int>& action_history,
        const std::string& result,
        const std::string& starting_fen = "");

    [[nodiscard]] std::unique_ptr<GameState> apply_action(int action) const override;
    [[nodiscard]] std::vector<int> legal_actions() const override;
    [[nodiscard]] bool is_terminal() const override;
    [[nodiscard]] float outcome(int player) const override;
    [[nodiscard]] int current_player() const override;
    void encode(float* buffer) const override;
    [[nodiscard]] std::unique_ptr<GameState> clone() const override;
    [[nodiscard]] std::uint64_t hash() const override;
    [[nodiscard]] std::string to_string() const override;

    [[nodiscard]] const ChessPosition& position() const { return position_; }
    [[nodiscard]] int history_size() const { return history_size_; }
    [[nodiscard]] int ply_count() const { return ply_count_; }
    [[nodiscard]] const ChessPosition& history_position(int steps_ago) const;

private:
    ChessState(
        ChessPosition position,
        std::array<ChessPosition, kHistorySteps> history,
        int history_size,
        std::unordered_map<std::uint64_t, int> repetition_table,
        int ply_count);

    [[nodiscard]] bool is_insufficient_material() const;
    [[nodiscard]] static int orient_square_for_side(int square, int side_to_move);
    [[nodiscard]] float normalized_move_count() const;
    [[nodiscard]] float normalized_no_progress_count() const;
    static void encode_position_planes(
        const ChessPosition& encoded_position,
        int perspective_color,
        int history_index,
        float* buffer);

    ChessPosition position_{};
    std::array<ChessPosition, kHistorySteps> history_{};
    int history_size_ = 0;
    std::unordered_map<std::uint64_t, int> repetition_table_;
    int ply_count_ = 0;
};

}  // namespace alphazero::chess
