#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "games/chess/chess_config.h"
#include "games/chess/movegen.h"
#include "games/chess/chess_state.h"
#include "games/game_config.h"
#include "games/game_state.h"
#include "games/go/go_config.h"
#include "games/go/go_state.h"
#include "mcts/arena_node_store.h"
#include "mcts/eval_queue.h"
#include "mcts/mcts_search.h"
#include "selfplay/replay_buffer.h"
#include "selfplay/self_play_game.h"
#include "selfplay/self_play_manager.h"

namespace py = pybind11;

namespace {

using alphazero::GameConfig;
using alphazero::GameState;
using alphazero::chess::ChessState;
using alphazero::go::GoState;
using alphazero::mcts::EvaluationResult;
using alphazero::mcts::EvalResult;
using alphazero::selfplay::ReplayPosition;
using alphazero::selfplay::SelfPlayManager;

[[nodiscard]] std::vector<float> cast_float_sequence(const py::handle& handle, const char* field_name) {
    if (py::isinstance<py::str>(handle) || py::isinstance<py::bytes>(handle)) {
        throw std::invalid_argument(std::string(field_name) + " must be a sequence of floats");
    }
    if (!py::isinstance<py::sequence>(handle)) {
        throw std::invalid_argument(std::string(field_name) + " must be a sequence of floats");
    }

    const py::sequence sequence = py::reinterpret_borrow<py::sequence>(handle);
    std::vector<float> values;
    values.reserve(sequence.size());
    for (const py::handle item : sequence) {
        values.push_back(py::cast<float>(item));
    }
    return values;
}

[[nodiscard]] std::array<float, ReplayPosition::kWdlSize> cast_wdl_sequence(const py::handle& handle) {
    const std::vector<float> values = cast_float_sequence(handle, "value_wdl");
    if (values.size() != ReplayPosition::kWdlSize) {
        throw std::invalid_argument("value_wdl must contain exactly 3 floats: [win, draw, loss]");
    }

    return {
        values[0],
        values[1],
        values[2],
    };
}

[[nodiscard]] ReplayPosition make_replay_position_from_python(
    const py::handle& encoded_state,
    const py::handle& policy,
    const float value,
    const py::handle& value_wdl,
    const std::uint32_t game_id,
    const std::uint16_t move_number) {
    const std::vector<float> encoded_state_values = cast_float_sequence(encoded_state, "encoded_state");
    const std::vector<float> policy_values = cast_float_sequence(policy, "policy");
    const std::array<float, ReplayPosition::kWdlSize> wdl = cast_wdl_sequence(value_wdl);

    return ReplayPosition::make(
        std::span<const float>(encoded_state_values.data(), encoded_state_values.size()),
        std::span<const float>(policy_values.data(), policy_values.size()),
        value,
        wdl,
        game_id,
        move_number);
}

[[nodiscard]] py::array_t<float> replay_position_encoded_state_view(const ReplayPosition& position) {
    py::array_t<float> array(static_cast<py::ssize_t>(position.encoded_state_size));
    std::copy_n(position.encoded_state.begin(), position.encoded_state_size, array.mutable_data());
    return array;
}

[[nodiscard]] py::array_t<float> replay_position_policy_view(const ReplayPosition& position) {
    py::array_t<float> array(static_cast<py::ssize_t>(position.policy_size));
    std::copy_n(position.policy.begin(), position.policy_size, array.mutable_data());
    return array;
}

[[nodiscard]] py::array_t<float> replay_position_wdl_view(const ReplayPosition& position) {
    py::array_t<float> array(static_cast<py::ssize_t>(ReplayPosition::kWdlSize));
    std::copy_n(position.value_wdl.begin(), ReplayPosition::kWdlSize, array.mutable_data());
    return array;
}

template <typename StateType>
[[nodiscard]] std::unique_ptr<StateType> downcast_state_unique_ptr(
    std::unique_ptr<GameState> base_ptr,
    const char* source_method) {
    if (base_ptr == nullptr) {
        throw std::logic_error(std::string(source_method) + " returned a null GameState pointer");
    }

    StateType* derived_ptr = dynamic_cast<StateType*>(base_ptr.release());
    if (derived_ptr == nullptr) {
        throw std::logic_error(std::string(source_method) + " returned a GameState with an unexpected dynamic type");
    }

    return std::unique_ptr<StateType>(derived_ptr);
}

[[nodiscard]] std::size_t encoded_state_size(const GameState& state) {
    if (dynamic_cast<const ChessState*>(&state) != nullptr) {
        return static_cast<std::size_t>(ChessState::kTotalInputChannels) * 8U * 8U;
    }
    if (dynamic_cast<const GoState*>(&state) != nullptr) {
        return static_cast<std::size_t>(GoState::kTotalInputChannels) * 19U * 19U;
    }
    throw std::invalid_argument(
        "encode() requires a known concrete GameState type (ChessState or GoState) for shape inference");
}

[[nodiscard]] std::vector<float> encode_state_flat(const GameState& state) {
    const std::size_t value_count = encoded_state_size(state);
    std::vector<float> encoded(value_count, 0.0F);
    state.encode(encoded.data());
    return encoded;
}

[[nodiscard]] int parse_chess_uci_square(const std::string_view token) {
    if (token.size() != 2U) {
        return -1;
    }

    const char file = static_cast<char>(std::tolower(static_cast<unsigned char>(token[0])));
    const char rank = token[1];
    if (file < 'a' || file > 'h' || rank < '1' || rank > '8') {
        return -1;
    }

    const int file_index = static_cast<int>(file - 'a');
    const int rank_index = static_cast<int>(rank - '1');
    return (rank_index * 8) + file_index;
}

[[nodiscard]] std::string chess_square_to_uci(const int square) {
    if (!alphazero::chess::is_valid_square(square)) {
        throw std::invalid_argument("Chess square index out of range for UCI conversion");
    }
    const int file = square % 8;
    const int rank = square / 8;
    std::string uci;
    uci.reserve(2);
    uci.push_back(static_cast<char>('a' + file));
    uci.push_back(static_cast<char>('1' + rank));
    return uci;
}

[[nodiscard]] char promotion_piece_to_uci_char(const int piece_type) {
    using alphazero::chess::kBishop;
    using alphazero::chess::kKnight;
    using alphazero::chess::kQueen;
    using alphazero::chess::kRook;
    switch (piece_type) {
        case kQueen:
            return 'q';
        case kRook:
            return 'r';
        case kBishop:
            return 'b';
        case kKnight:
            return 'n';
        default:
            throw std::invalid_argument("Unsupported promotion piece in chess move");
    }
}

[[nodiscard]] int uci_char_to_promotion_piece(const char promotion_char) {
    using alphazero::chess::kBishop;
    using alphazero::chess::kKnight;
    using alphazero::chess::kQueen;
    using alphazero::chess::kRook;
    switch (static_cast<char>(std::tolower(static_cast<unsigned char>(promotion_char)))) {
        case 'q':
            return kQueen;
        case 'r':
            return kRook;
        case 'b':
            return kBishop;
        case 'n':
            return kKnight;
        default:
            return -1;
    }
}

[[nodiscard]] std::string chess_move_to_uci(const alphazero::chess::Move& move) {
    std::string uci = chess_square_to_uci(move.from) + chess_square_to_uci(move.to);
    if (move.promotion >= 0) {
        uci.push_back(promotion_piece_to_uci_char(move.promotion));
    }
    return uci;
}

[[nodiscard]] std::string chess_action_to_uci(const ChessState& state, const int action) {
    const std::optional<alphazero::chess::Move> move =
        alphazero::chess::action_index_to_semantic_move(state.position(), action);
    if (!move.has_value()) {
        throw std::invalid_argument("Action is illegal for the current chess state");
    }
    return chess_move_to_uci(*move);
}

[[nodiscard]] std::vector<std::pair<int, std::string>> chess_legal_actions_uci(const ChessState& state) {
    const std::vector<int> legal_actions = state.legal_actions();
    std::vector<std::pair<int, std::string>> mappings;
    mappings.reserve(legal_actions.size());

    for (const int action : legal_actions) {
        mappings.emplace_back(action, chess_action_to_uci(state, action));
    }
    return mappings;
}

[[nodiscard]] int chess_uci_to_action(const ChessState& state, const std::string& uci_move_text) {
    const std::string_view uci(uci_move_text);
    if (uci.size() != 4U && uci.size() != 5U) {
        throw std::invalid_argument("Chess UCI move must have form e2e4 or e7e8q");
    }

    const int from = parse_chess_uci_square(uci.substr(0, 2));
    const int to = parse_chess_uci_square(uci.substr(2, 2));
    if (from < 0 || to < 0) {
        throw std::invalid_argument("Chess UCI move has an invalid source or destination square");
    }

    const int promotion = uci.size() == 5U ? uci_char_to_promotion_piece(uci[4]) : -1;
    if (uci.size() == 5U && promotion < 0) {
        throw std::invalid_argument("Chess UCI promotion suffix must be one of q/r/b/n");
    }

    const std::vector<int> legal_actions = state.legal_actions();
    for (const int action : legal_actions) {
        const std::optional<alphazero::chess::Move> move =
            alphazero::chess::action_index_to_semantic_move(state.position(), action);
        if (!move.has_value()) {
            continue;
        }

        if (move->from != from || move->to != to) {
            continue;
        }
        if (promotion >= 0 && move->promotion != promotion) {
            continue;
        }
        if (promotion < 0 && move->promotion >= 0) {
            continue;
        }
        return action;
    }

    throw std::invalid_argument("Chess UCI move is illegal in the current position");
}

template <typename StateType>
[[nodiscard]] py::array_t<float> encode_state_tensor(
    const StateType& state,
    const int channels,
    const int rows,
    const int cols) {
    py::array_t<float> encoded(
        {static_cast<py::ssize_t>(channels), static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    state.encode(encoded.mutable_data());
    return encoded;
}

[[nodiscard]] EvalResult parse_eval_queue_result(const py::handle& handle) {
    if (py::isinstance<EvalResult>(handle)) {
        return py::cast<EvalResult>(handle);
    }

    if (py::isinstance<py::dict>(handle)) {
        const py::dict result = py::reinterpret_borrow<py::dict>(handle);
        if (!result.contains("policy_logits") || !result.contains("value")) {
            throw std::invalid_argument("EvalQueue evaluator dict result must contain 'policy_logits' and 'value'");
        }

        EvalResult parsed{};
        parsed.policy_logits = cast_float_sequence(result["policy_logits"], "policy_logits");
        parsed.value = py::cast<float>(result["value"]);
        if (!std::isfinite(parsed.value)) {
            throw std::invalid_argument("EvalQueue evaluator returned a non-finite value");
        }
        return parsed;
    }

    if (py::isinstance<py::tuple>(handle) || py::isinstance<py::list>(handle)) {
        const py::sequence sequence = py::reinterpret_borrow<py::sequence>(handle);
        if (sequence.size() != 2) {
            throw std::invalid_argument("EvalQueue evaluator tuple/list result must have shape (policy_logits, value)");
        }

        EvalResult parsed{};
        parsed.policy_logits = cast_float_sequence(sequence[0], "policy_logits");
        parsed.value = py::cast<float>(sequence[1]);
        if (!std::isfinite(parsed.value)) {
            throw std::invalid_argument("EvalQueue evaluator returned a non-finite value");
        }
        return parsed;
    }

    throw std::invalid_argument(
        "EvalQueue evaluator must return EvalResult, dict {'policy_logits','value'}, or tuple (policy_logits, value)");
}

[[nodiscard]] EvaluationResult parse_search_evaluation_result(
    const py::handle& handle,
    const int action_space_size) {
    EvaluationResult parsed{};

    if (py::isinstance<EvaluationResult>(handle)) {
        parsed = py::cast<EvaluationResult>(handle);
    } else if (py::isinstance<py::dict>(handle)) {
        const py::dict result = py::reinterpret_borrow<py::dict>(handle);
        if (!result.contains("policy") || !result.contains("value")) {
            throw std::invalid_argument("SelfPlay evaluator dict result must contain 'policy' and 'value'");
        }
        parsed.policy = cast_float_sequence(result["policy"], "policy");
        parsed.value = py::cast<float>(result["value"]);
        parsed.policy_is_logits = result.contains("policy_is_logits")
            ? py::cast<bool>(result["policy_is_logits"])
            : true;
    } else if (py::isinstance<py::tuple>(handle) || py::isinstance<py::list>(handle)) {
        const py::sequence sequence = py::reinterpret_borrow<py::sequence>(handle);
        if (sequence.size() != 2 && sequence.size() != 3) {
            throw std::invalid_argument(
                "SelfPlay evaluator tuple/list result must have shape (policy, value[, policy_is_logits])");
        }
        parsed.policy = cast_float_sequence(sequence[0], "policy");
        parsed.value = py::cast<float>(sequence[1]);
        parsed.policy_is_logits = sequence.size() == 3 ? py::cast<bool>(sequence[2]) : true;
    } else {
        throw std::invalid_argument(
            "SelfPlay evaluator must return EvaluationResult, dict {'policy','value'}, or tuple"
            " (policy, value[, policy_is_logits])");
    }

    if (action_space_size <= 0) {
        throw std::invalid_argument("SelfPlay evaluator action_space_size must be positive");
    }
    if (parsed.policy.size() != static_cast<std::size_t>(action_space_size)) {
        throw std::invalid_argument("SelfPlay evaluator returned policy with wrong action-space size");
    }
    if (!std::isfinite(parsed.value)) {
        throw std::invalid_argument("SelfPlay evaluator returned a non-finite value");
    }
    return parsed;
}

class PyEvalQueue {
public:
    PyEvalQueue(py::function evaluator, const std::size_t encoded_state_size, alphazero::mcts::EvalQueueConfig config)
        : encoded_state_size_(encoded_state_size),
          queue_(
              [evaluator = std::move(evaluator), encoded_state_size](
                  const std::vector<const float*>& inputs) -> std::vector<EvalResult> {
                  py::gil_scoped_acquire acquire_gil;

                  py::list py_inputs;
                  for (const float* input : inputs) {
                      if (input == nullptr) {
                          throw std::invalid_argument("EvalQueue received a null encoded-state pointer");
                      }

                      py::array_t<float> encoded(static_cast<py::ssize_t>(encoded_state_size));
                      std::copy_n(input, encoded_state_size, encoded.mutable_data());
                      py_inputs.append(std::move(encoded));
                  }

                  const py::object py_outputs = evaluator(py_inputs);
                  if (py::isinstance<py::str>(py_outputs) || py::isinstance<py::bytes>(py_outputs) ||
                      !py::isinstance<py::sequence>(py_outputs)) {
                      throw std::invalid_argument("EvalQueue evaluator must return a sequence of per-request results");
                  }

                  const py::sequence outputs = py::reinterpret_borrow<py::sequence>(py_outputs);
                  if (outputs.size() != static_cast<py::ssize_t>(inputs.size())) {
                      throw std::invalid_argument("EvalQueue evaluator output size does not match request batch size");
                  }

                  std::vector<EvalResult> parsed_outputs;
                  parsed_outputs.reserve(static_cast<std::size_t>(outputs.size()));
                  for (const py::handle output : outputs) {
                      parsed_outputs.push_back(parse_eval_queue_result(output));
                  }
                  return parsed_outputs;
              },
              config) {
        if (encoded_state_size_ == 0U) {
            throw std::invalid_argument("EvalQueue encoded_state_size must be greater than zero");
        }
    }

    [[nodiscard]] EvalResult submit_and_wait(const py::handle& encoded_state) {
        const std::vector<float> encoded_state_values = cast_float_sequence(encoded_state, "encoded_state");
        if (encoded_state_values.size() != encoded_state_size_) {
            throw std::invalid_argument("EvalQueue encoded_state has unexpected length");
        }

        py::gil_scoped_release release_gil;
        return queue_.submit_and_wait(encoded_state_values.data());
    }

    void process_batch() {
        py::gil_scoped_release release_gil;
        queue_.process_batch();
    }

    void stop() { queue_.stop(); }

    [[nodiscard]] std::size_t encoded_state_size() const noexcept { return encoded_state_size_; }

private:
    std::size_t encoded_state_size_ = 0U;
    alphazero::mcts::EvalQueue queue_;
};

[[nodiscard]] SelfPlayManager::EvaluateFn make_selfplay_evaluator(py::function evaluator, const int action_space_size) {
    if (action_space_size <= 0) {
        throw std::invalid_argument("SelfPlayManager requires a positive action_space_size");
    }
    return [evaluator = std::move(evaluator), action_space_size](const GameState& state) -> EvaluationResult {
        py::gil_scoped_acquire acquire_gil;
        const py::object python_state = py::cast(&state, py::return_value_policy::reference);
        const py::object python_result = evaluator(python_state);
        return parse_search_evaluation_result(python_result, action_space_size);
    };
}

class PyMctsSearch {
public:
    PyMctsSearch(
        const GameConfig& game_config,
        const alphazero::mcts::SearchConfig config,
        const std::size_t node_arena_capacity)
        : game_config_(game_config),
          node_store_(node_arena_capacity),
          search_(node_store_, game_config_, config),
          action_space_size_(game_config_.action_space_size) {
        if (node_arena_capacity == 0U) {
            throw std::invalid_argument("MctsSearch node_arena_capacity must be greater than zero");
        }
        if (action_space_size_ <= 0) {
            throw std::invalid_argument("MctsSearch requires a positive action space size");
        }
    }

    void set_root_state(const GameState& state) {
        py::gil_scoped_release release_gil;
        search_.set_root_state(state.clone());
    }

    [[nodiscard]] bool has_root() const {
        py::gil_scoped_release release_gil;
        return search_.has_root();
    }

    [[nodiscard]] std::vector<float> root_policy_target(const int move_number) const {
        py::gil_scoped_release release_gil;
        return search_.root_policy_target(move_number);
    }

    [[nodiscard]] int select_action(const int move_number) {
        py::gil_scoped_release release_gil;
        return search_.select_action(move_number);
    }

    void run_simulations(py::function evaluator, const std::optional<std::size_t> simulation_count) {
        const SelfPlayManager::EvaluateFn wrapped = make_selfplay_evaluator(std::move(evaluator), action_space_size_);
        py::gil_scoped_release release_gil;
        if (simulation_count.has_value()) {
            search_.run_simulations(*simulation_count, wrapped);
            return;
        }
        search_.run_simulations(wrapped);
    }

    void advance_root(const int action) {
        py::gil_scoped_release release_gil;
        search_.advance_root(action);
    }

    [[nodiscard]] bool should_resign() const {
        py::gil_scoped_release release_gil;
        return search_.should_resign();
    }

    [[nodiscard]] std::optional<alphazero::mcts::EdgeStats> root_edge_stats(const int action) const {
        py::gil_scoped_release release_gil;
        return search_.root_edge_stats(action);
    }

private:
    const GameConfig& game_config_;
    alphazero::mcts::ArenaNodeStore node_store_;
    alphazero::mcts::MctsSearch search_;
    int action_space_size_ = 0;
};

[[nodiscard]] SelfPlayManager::CompletionCallback make_completion_callback(const py::object& callback) {
    if (callback.is_none()) {
        return {};
    }

    py::function completion_callback = py::cast<py::function>(callback);
    return [completion_callback = std::move(completion_callback)](
               const std::size_t slot_index,
               const alphazero::selfplay::SelfPlayGameResult& result) {
        py::gil_scoped_acquire acquire_gil;
        completion_callback(slot_index, result);
    };
}

}  // namespace

PYBIND11_MODULE(alphazero_cpp, module) {
    module.doc() = "pybind11 bridge for the AlphaZero C++ engine";

    py::class_<GameState>(module, "GameState")
        .def("apply_action", &GameState::apply_action, py::arg("action"))
        .def("legal_actions", &GameState::legal_actions)
        .def("is_terminal", &GameState::is_terminal)
        .def("outcome", &GameState::outcome, py::arg("player"))
        .def("current_player", &GameState::current_player)
        .def(
            "encode",
            [](const GameState& state) { return encode_state_flat(state); },
            "Encode the state into a flat float vector.")
        .def("clone", &GameState::clone)
        .def("hash", &GameState::hash)
        .def("to_string", &GameState::to_string)
        .def("__repr__", [](const GameState& state) { return state.to_string(); })
        .def("__str__", [](const GameState& state) { return state.to_string(); });

    py::class_<ChessState, GameState>(module, "ChessState")
        .def(py::init<>())
        .def_static("from_fen", &ChessState::from_fen, py::arg("fen"))
        .def("to_fen", &ChessState::to_fen)
        .def_static(
            "actions_to_pgn",
            &ChessState::actions_to_pgn,
            py::arg("action_history"),
            py::arg("result"),
            py::arg("starting_fen") = std::string{})
        .def(
            "apply_action",
            [](const ChessState& state, const int action) {
                return downcast_state_unique_ptr<ChessState>(state.apply_action(action), "ChessState.apply_action()");
            },
            py::arg("action"))
        .def("legal_actions", &ChessState::legal_actions)
        .def("is_terminal", &ChessState::is_terminal)
        .def("outcome", &ChessState::outcome, py::arg("player"))
        .def("current_player", &ChessState::current_player)
        .def(
            "encode",
            [](const ChessState& state) {
                return encode_state_tensor(state, ChessState::kTotalInputChannels, 8, 8);
            })
        .def(
            "clone",
            [](const ChessState& state) {
                return downcast_state_unique_ptr<ChessState>(state.clone(), "ChessState.clone()");
            })
        .def("hash", &ChessState::hash)
        .def("action_to_uci", &chess_action_to_uci, py::arg("action"))
        .def("uci_to_action", &chess_uci_to_action, py::arg("uci"))
        .def("legal_actions_uci", &chess_legal_actions_uci)
        .def("to_string", &ChessState::to_string)
        .def("__repr__", [](const ChessState& state) { return state.to_string(); })
        .def("__str__", [](const ChessState& state) { return state.to_string(); });

    py::class_<GoState, GameState>(module, "GoState")
        .def(py::init<>())
        .def_static("from_sgf", &GoState::from_sgf, py::arg("sgf"))
        .def(
            "to_sgf",
            &GoState::to_sgf,
            py::arg("result") = std::string{"?"})
        .def_static(
            "actions_to_sgf",
            &GoState::actions_to_sgf,
            py::arg("action_history"),
            py::arg("result") = std::string{"?"},
            py::arg("komi") = alphazero::go::kDefaultKomi)
        .def(
            "apply_action",
            [](const GoState& state, const int action) {
                return downcast_state_unique_ptr<GoState>(state.apply_action(action), "GoState.apply_action()");
            },
            py::arg("action"))
        .def("legal_actions", &GoState::legal_actions)
        .def("is_terminal", &GoState::is_terminal)
        .def("outcome", &GoState::outcome, py::arg("player"))
        .def("current_player", &GoState::current_player)
        .def(
            "encode",
            [](const GoState& state) { return encode_state_tensor(state, GoState::kTotalInputChannels, 19, 19); })
        .def(
            "clone",
            [](const GoState& state) {
                return downcast_state_unique_ptr<GoState>(state.clone(), "GoState.clone()");
            })
        .def("hash", &GoState::hash)
        .def("to_string", &GoState::to_string)
        .def("__repr__", [](const GoState& state) { return state.to_string(); })
        .def("__str__", [](const GoState& state) { return state.to_string(); });

    py::class_<GameConfig> game_config(module, "GameConfig");
    py::enum_<GameConfig::ValueHeadType>(game_config, "ValueHeadType")
        .value("SCALAR", GameConfig::ValueHeadType::SCALAR)
        .value("WDL", GameConfig::ValueHeadType::WDL)
        .export_values();

    game_config
        .def_property_readonly("name", [](const GameConfig& config) { return config.name; })
        .def_property_readonly("board_rows", [](const GameConfig& config) { return config.board_rows; })
        .def_property_readonly("board_cols", [](const GameConfig& config) { return config.board_cols; })
        .def_property_readonly("planes_per_step", [](const GameConfig& config) { return config.planes_per_step; })
        .def_property_readonly(
            "num_history_steps",
            [](const GameConfig& config) { return config.num_history_steps; })
        .def_property_readonly(
            "constant_planes",
            [](const GameConfig& config) { return config.constant_planes; })
        .def_property_readonly(
            "total_input_channels",
            [](const GameConfig& config) { return config.total_input_channels; })
        .def_property_readonly(
            "action_space_size",
            [](const GameConfig& config) { return config.action_space_size; })
        .def_property_readonly(
            "dirichlet_alpha",
            [](const GameConfig& config) { return config.dirichlet_alpha; })
        .def_property_readonly(
            "max_game_length",
            [](const GameConfig& config) { return config.max_game_length; })
        .def_property_readonly(
            "value_head_type",
            [](const GameConfig& config) { return config.value_head_type; })
        .def_property_readonly(
            "supports_symmetry",
            [](const GameConfig& config) { return config.supports_symmetry; })
        .def_property_readonly(
            "num_symmetries",
            [](const GameConfig& config) { return config.num_symmetries; })
        .def("new_game", &GameConfig::new_game)
        .def("__repr__", [](const GameConfig& config) {
            return "<GameConfig name='" + config.name + "' action_space=" + std::to_string(config.action_space_size) +
                ">";
        });

    py::class_<alphazero::chess::ChessGameConfig, GameConfig>(module, "ChessGameConfig").def(py::init<>());
    py::class_<alphazero::go::GoGameConfig, GameConfig>(module, "GoGameConfig").def(py::init<>());

    module.def(
        "chess_game_config",
        []() -> const alphazero::chess::ChessGameConfig& { return alphazero::chess::chess_game_config(); },
        py::return_value_policy::reference);
    module.def(
        "go_game_config",
        []() -> const alphazero::go::GoGameConfig& { return alphazero::go::go_game_config(); },
        py::return_value_policy::reference);

    py::class_<EvaluationResult>(module, "EvaluationResult")
        .def(py::init<>())
        .def_readwrite("policy", &EvaluationResult::policy)
        .def_readwrite("value", &EvaluationResult::value)
        .def_readwrite("policy_is_logits", &EvaluationResult::policy_is_logits);

    py::class_<EvalResult>(module, "EvalResult")
        .def(py::init<>())
        .def_readwrite("policy_logits", &EvalResult::policy_logits)
        .def_readwrite("value", &EvalResult::value);

    py::class_<alphazero::mcts::SearchConfig>(module, "SearchConfig")
        .def(py::init<>())
        .def_readwrite("simulations_per_move", &alphazero::mcts::SearchConfig::simulations_per_move)
        .def_readwrite("c_puct", &alphazero::mcts::SearchConfig::c_puct)
        .def_readwrite("c_fpu", &alphazero::mcts::SearchConfig::c_fpu)
        .def_readwrite("enable_dirichlet_noise", &alphazero::mcts::SearchConfig::enable_dirichlet_noise)
        .def_readwrite("dirichlet_epsilon", &alphazero::mcts::SearchConfig::dirichlet_epsilon)
        .def_readwrite("dirichlet_alpha_override", &alphazero::mcts::SearchConfig::dirichlet_alpha_override)
        .def_readwrite("temperature", &alphazero::mcts::SearchConfig::temperature)
        .def_readwrite("temperature_moves", &alphazero::mcts::SearchConfig::temperature_moves)
        .def_readwrite("enable_resignation", &alphazero::mcts::SearchConfig::enable_resignation)
        .def_readwrite("resign_threshold", &alphazero::mcts::SearchConfig::resign_threshold)
        .def_readwrite("random_seed", &alphazero::mcts::SearchConfig::random_seed);

    py::class_<alphazero::mcts::EdgeStats>(module, "EdgeStats")
        .def(py::init<>())
        .def_readwrite("action", &alphazero::mcts::EdgeStats::action)
        .def_readwrite("visit_count", &alphazero::mcts::EdgeStats::visit_count)
        .def_readwrite("total_value", &alphazero::mcts::EdgeStats::total_value)
        .def_readwrite("mean_value", &alphazero::mcts::EdgeStats::mean_value)
        .def_readwrite("prior", &alphazero::mcts::EdgeStats::prior)
        .def_readwrite("virtual_loss", &alphazero::mcts::EdgeStats::virtual_loss)
        .def_readwrite("child", &alphazero::mcts::EdgeStats::child);

    py::class_<PyMctsSearch>(module, "MctsSearch")
        .def(
            py::init<const GameConfig&, alphazero::mcts::SearchConfig, std::size_t>(),
            py::arg("game_config"),
            py::arg("config") = alphazero::mcts::SearchConfig{},
            py::arg("node_arena_capacity") = alphazero::mcts::ArenaNodeStore::kDefaultCapacity,
            py::keep_alive<1, 2>())
        .def("set_root_state", &PyMctsSearch::set_root_state, py::arg("state"))
        .def("has_root", &PyMctsSearch::has_root)
        .def("root_policy_target", &PyMctsSearch::root_policy_target, py::arg("move_number"))
        .def("select_action", &PyMctsSearch::select_action, py::arg("move_number"))
        .def(
            "run_simulations",
            &PyMctsSearch::run_simulations,
            py::arg("evaluator"),
            py::arg("simulation_count") = py::none(),
            py::keep_alive<1, 2>())
        .def("advance_root", &PyMctsSearch::advance_root, py::arg("action"))
        .def("should_resign", &PyMctsSearch::should_resign)
        .def("root_edge_stats", &PyMctsSearch::root_edge_stats, py::arg("action"));

    py::class_<alphazero::mcts::EvalQueueConfig>(module, "EvalQueueConfig")
        .def(py::init<>())
        .def_readwrite("batch_size", &alphazero::mcts::EvalQueueConfig::batch_size)
        .def_property(
            "flush_timeout_us",
            [](const alphazero::mcts::EvalQueueConfig& config) { return config.flush_timeout.count(); },
            [](alphazero::mcts::EvalQueueConfig* config, const std::int64_t timeout_us) {
                if (timeout_us < 0) {
                    throw std::invalid_argument("EvalQueueConfig.flush_timeout_us must be non-negative");
                }
                config->flush_timeout = std::chrono::microseconds(timeout_us);
            })
        .def_property(
            "wait_timeout_us",
            [](const alphazero::mcts::EvalQueueConfig& config) { return config.wait_timeout.count(); },
            [](alphazero::mcts::EvalQueueConfig* config, const std::int64_t timeout_us) {
                if (timeout_us < 0) {
                    throw std::invalid_argument("EvalQueueConfig.wait_timeout_us must be non-negative");
                }
                config->wait_timeout = std::chrono::microseconds(timeout_us);
            });

    py::class_<PyEvalQueue>(module, "EvalQueue")
        .def(
            py::init<py::function, std::size_t, alphazero::mcts::EvalQueueConfig>(),
            py::arg("evaluator"),
            py::arg("encoded_state_size"),
            py::arg("config") = alphazero::mcts::EvalQueueConfig{},
            py::keep_alive<1, 2>())
        .def("submit_and_wait", &PyEvalQueue::submit_and_wait, py::arg("encoded_state"))
        .def("process_batch", &PyEvalQueue::process_batch)
        .def("stop", &PyEvalQueue::stop)
        .def_property_readonly("encoded_state_size", &PyEvalQueue::encoded_state_size);

    py::class_<ReplayPosition> replay_position(module, "ReplayPosition");
    replay_position
        .def(py::init<>())
        .def_static(
            "make",
            &make_replay_position_from_python,
            py::arg("encoded_state"),
            py::arg("policy"),
            py::arg("value"),
            py::arg("value_wdl"),
            py::arg("game_id"),
            py::arg("move_number"))
        .def_property_readonly("encoded_state", &replay_position_encoded_state_view)
        .def_property_readonly("policy", &replay_position_policy_view)
        .def_readwrite("value", &ReplayPosition::value)
        .def_property_readonly("value_wdl", &replay_position_wdl_view)
        .def_readwrite("game_id", &ReplayPosition::game_id)
        .def_readwrite("move_number", &ReplayPosition::move_number)
        .def_readwrite("encoded_state_size", &ReplayPosition::encoded_state_size)
        .def_readwrite("policy_size", &ReplayPosition::policy_size);

    replay_position.attr("MAX_ENCODED_STATE_SIZE") = py::int_(ReplayPosition::kMaxEncodedStateSize);
    replay_position.attr("MAX_POLICY_SIZE") = py::int_(ReplayPosition::kMaxPolicySize);
    replay_position.attr("WDL_SIZE") = py::int_(ReplayPosition::kWdlSize);

    py::class_<alphazero::selfplay::ReplayBuffer>(module, "ReplayBuffer")
        .def(
            py::init<std::size_t, std::uint64_t>(),
            py::arg("capacity") = alphazero::selfplay::ReplayBuffer::kDefaultCapacity,
            py::arg("random_seed") = 0x9E3779B97F4A7C15ULL)
        .def("add_game", &alphazero::selfplay::ReplayBuffer::add_game, py::arg("positions"))
        .def("sample", &alphazero::selfplay::ReplayBuffer::sample, py::arg("batch_size"))
        .def("size", &alphazero::selfplay::ReplayBuffer::size)
        .def("capacity", &alphazero::selfplay::ReplayBuffer::capacity)
        .def("write_head", &alphazero::selfplay::ReplayBuffer::write_head);

    py::enum_<alphazero::selfplay::GameTerminationReason>(module, "GameTerminationReason")
        .value("NATURAL", alphazero::selfplay::GameTerminationReason::kNatural)
        .value("RESIGNATION", alphazero::selfplay::GameTerminationReason::kResignation)
        .value("MAX_LENGTH_ADJUDICATION", alphazero::selfplay::GameTerminationReason::kMaxLengthAdjudication)
        .export_values();

    py::class_<alphazero::selfplay::SelfPlayGameConfig>(module, "SelfPlayGameConfig")
        .def(py::init<>())
        .def_readwrite("simulations_per_move", &alphazero::selfplay::SelfPlayGameConfig::simulations_per_move)
        .def_readwrite("mcts_threads", &alphazero::selfplay::SelfPlayGameConfig::mcts_threads)
        .def_readwrite("node_arena_capacity", &alphazero::selfplay::SelfPlayGameConfig::node_arena_capacity)
        .def_readwrite("c_puct", &alphazero::selfplay::SelfPlayGameConfig::c_puct)
        .def_readwrite("c_fpu", &alphazero::selfplay::SelfPlayGameConfig::c_fpu)
        .def_readwrite("enable_dirichlet_noise", &alphazero::selfplay::SelfPlayGameConfig::enable_dirichlet_noise)
        .def_readwrite("dirichlet_epsilon", &alphazero::selfplay::SelfPlayGameConfig::dirichlet_epsilon)
        .def_readwrite("dirichlet_alpha_override", &alphazero::selfplay::SelfPlayGameConfig::dirichlet_alpha_override)
        .def_readwrite("temperature", &alphazero::selfplay::SelfPlayGameConfig::temperature)
        .def_readwrite("temperature_moves", &alphazero::selfplay::SelfPlayGameConfig::temperature_moves)
        .def_readwrite("enable_resignation", &alphazero::selfplay::SelfPlayGameConfig::enable_resignation)
        .def_readwrite("resign_threshold", &alphazero::selfplay::SelfPlayGameConfig::resign_threshold)
        .def_readwrite("resign_disable_fraction", &alphazero::selfplay::SelfPlayGameConfig::resign_disable_fraction)
        .def_readwrite("random_seed", &alphazero::selfplay::SelfPlayGameConfig::random_seed);

    py::class_<alphazero::selfplay::SelfPlayManagerConfig>(module, "SelfPlayManagerConfig")
        .def(py::init<>())
        .def_readwrite("concurrent_games", &alphazero::selfplay::SelfPlayManagerConfig::concurrent_games)
        .def_readwrite("max_games_per_slot", &alphazero::selfplay::SelfPlayManagerConfig::max_games_per_slot)
        .def_readwrite("initial_game_id", &alphazero::selfplay::SelfPlayManagerConfig::initial_game_id)
        .def_readwrite("random_seed", &alphazero::selfplay::SelfPlayManagerConfig::random_seed)
        .def_readwrite("game_config", &alphazero::selfplay::SelfPlayManagerConfig::game_config);

    py::class_<alphazero::selfplay::SelfPlayGameResult>(module, "SelfPlayGameResult")
        .def(py::init<>())
        .def_readwrite("game_id", &alphazero::selfplay::SelfPlayGameResult::game_id)
        .def_readwrite("move_count", &alphazero::selfplay::SelfPlayGameResult::move_count)
        .def_readwrite(
            "replay_positions_written",
            &alphazero::selfplay::SelfPlayGameResult::replay_positions_written)
        .def_readwrite("reused_subtree_count", &alphazero::selfplay::SelfPlayGameResult::reused_subtree_count)
        .def_readwrite(
            "simulation_batches_executed",
            &alphazero::selfplay::SelfPlayGameResult::simulation_batches_executed)
        .def_readwrite("termination_reason", &alphazero::selfplay::SelfPlayGameResult::termination_reason)
        .def_readwrite(
            "resignation_was_disabled",
            &alphazero::selfplay::SelfPlayGameResult::resignation_was_disabled)
        .def_readwrite(
            "resignation_would_have_triggered",
            &alphazero::selfplay::SelfPlayGameResult::resignation_would_have_triggered)
        .def_readwrite(
            "resignation_candidate_player",
            &alphazero::selfplay::SelfPlayGameResult::resignation_candidate_player)
        .def_readwrite("outcome_player0", &alphazero::selfplay::SelfPlayGameResult::outcome_player0)
        .def_readwrite("outcome_player1", &alphazero::selfplay::SelfPlayGameResult::outcome_player1)
        .def_readwrite("action_history", &alphazero::selfplay::SelfPlayGameResult::action_history);

    py::class_<alphazero::selfplay::SelfPlayMetricsSnapshot>(module, "SelfPlayMetricsSnapshot")
        .def(py::init<>())
        .def_readwrite("configured_slots", &alphazero::selfplay::SelfPlayMetricsSnapshot::configured_slots)
        .def_readwrite("threads_per_game", &alphazero::selfplay::SelfPlayMetricsSnapshot::threads_per_game)
        .def_readwrite("active_slots", &alphazero::selfplay::SelfPlayMetricsSnapshot::active_slots)
        .def_readwrite("games_completed", &alphazero::selfplay::SelfPlayMetricsSnapshot::games_completed)
        .def_readwrite(
            "replay_positions_written",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::replay_positions_written)
        .def_readwrite("total_moves", &alphazero::selfplay::SelfPlayMetricsSnapshot::total_moves)
        .def_readwrite("total_simulations", &alphazero::selfplay::SelfPlayMetricsSnapshot::total_simulations)
        .def_readwrite(
            "natural_terminations",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::natural_terminations)
        .def_readwrite(
            "resignation_terminations",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::resignation_terminations)
        .def_readwrite(
            "max_length_adjudications",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::max_length_adjudications)
        .def_readwrite(
            "resignation_disabled_games",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::resignation_disabled_games)
        .def_readwrite(
            "resignation_false_positive_games",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::resignation_false_positive_games)
        .def_readwrite("has_latest_game", &alphazero::selfplay::SelfPlayMetricsSnapshot::has_latest_game)
        .def_readwrite("latest_game_id", &alphazero::selfplay::SelfPlayMetricsSnapshot::latest_game_id)
        .def_readwrite("latest_slot", &alphazero::selfplay::SelfPlayMetricsSnapshot::latest_slot)
        .def_readwrite(
            "latest_game_length",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::latest_game_length)
        .def_readwrite(
            "latest_outcome_player0",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::latest_outcome_player0)
        .def_readwrite(
            "latest_game_resigned",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::latest_game_resigned)
        .def_readwrite(
            "latest_resignation_disabled",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::latest_resignation_disabled)
        .def_readwrite(
            "latest_resignation_false_positive",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::latest_resignation_false_positive)
        .def_readwrite(
            "average_game_length",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::average_game_length)
        .def_readwrite(
            "average_outcome_player0",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::average_outcome_player0)
        .def_readwrite("moves_per_second", &alphazero::selfplay::SelfPlayMetricsSnapshot::moves_per_second)
        .def_readwrite("games_per_hour", &alphazero::selfplay::SelfPlayMetricsSnapshot::games_per_hour)
        .def_readwrite(
            "avg_simulations_per_second",
            &alphazero::selfplay::SelfPlayMetricsSnapshot::avg_simulations_per_second)
        .def_readwrite("worker_failed", &alphazero::selfplay::SelfPlayMetricsSnapshot::worker_failed);

    py::class_<SelfPlayManager>(module, "SelfPlayManager")
        .def(
            py::init(
                [](const GameConfig& game_config,
                   alphazero::selfplay::ReplayBuffer& replay_buffer,
                   py::function evaluator,
                   alphazero::selfplay::SelfPlayManagerConfig config,
                   py::object completion_callback) {
                    return std::make_unique<SelfPlayManager>(
                        game_config,
                        replay_buffer,
                        make_selfplay_evaluator(std::move(evaluator), game_config.action_space_size),
                        config,
                        make_completion_callback(completion_callback));
                }),
            py::arg("game_config"),
            py::arg("replay_buffer"),
            py::arg("evaluator"),
            py::arg("config") = alphazero::selfplay::SelfPlayManagerConfig{},
            py::arg("completion_callback") = py::none(),
            py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>(),
            py::keep_alive<1, 4>(),
            py::keep_alive<1, 6>())
        .def("start", &SelfPlayManager::start, py::call_guard<py::gil_scoped_release>())
        .def("stop", &SelfPlayManager::stop, py::call_guard<py::gil_scoped_release>())
        .def("is_running", &SelfPlayManager::is_running)
        .def("metrics", &SelfPlayManager::metrics);
}
