"""Integration learning test for a tractable Connect Four harness."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
import random
import sys
from typing import Sequence
import unittest


ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if _TORCH_AVAILABLE:
    import torch
    from torch import nn

    from alphazero.config import GameConfig
    from alphazero.training.lr_schedule import StepDecayLRSchedule
    from alphazero.training.trainer import (
        create_optimizer,
        prepare_replay_batch,
        train_one_step,
    )


_EMPTY = 0
_PLAYER_ZERO_STONE = 1
_PLAYER_ONE_STONE = -1


class _ConnectFourState:
    """Small Connect Four state with gravity and terminal/outcome logic."""

    __slots__ = (
        "_rows",
        "_cols",
        "_connect_n",
        "_board",
        "_current_player",
        "_winner",
        "_moves_played",
    )

    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        connect_n: int,
        board: tuple[int, ...] | None = None,
        current_player: int = 0,
        winner: int | None = None,
        moves_played: int = 0,
    ) -> None:
        self._rows = rows
        self._cols = cols
        self._connect_n = connect_n
        self._board = board if board is not None else (_EMPTY,) * (rows * cols)
        self._current_player = current_player
        self._winner = winner
        self._moves_played = moves_played

    @classmethod
    def new_game(cls, *, rows: int = 4, cols: int = 4, connect_n: int = 4) -> "_ConnectFourState":
        return cls(rows=rows, cols=cols, connect_n=connect_n)

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def action_space_size(self) -> int:
        return self._cols

    def current_player(self) -> int:
        return self._current_player

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        actions: list[int] = []
        for col in range(self._cols):
            if self._board[self._index(0, col)] == _EMPTY:
                actions.append(col)
        return actions

    def is_terminal(self) -> bool:
        return self._winner is not None or self._moves_played >= self._rows * self._cols

    def outcome(self, player: int) -> float:
        if not self.is_terminal():
            raise ValueError("Outcome is only valid on terminal states")
        if self._winner is None:
            return 0.0
        return 1.0 if player == self._winner else -1.0

    def apply_action(self, action: int) -> "_ConnectFourState":
        if self.is_terminal():
            raise ValueError("Cannot apply an action to a terminal position")
        if action < 0 or action >= self._cols:
            raise ValueError(f"Action {action} is out of range")

        landing_row = self._find_landing_row(action)
        if landing_row is None:
            raise ValueError(f"Column {action} is full")

        stone = _PLAYER_ZERO_STONE if self._current_player == 0 else _PLAYER_ONE_STONE
        board_list = list(self._board)
        board_list[self._index(landing_row, action)] = stone
        updated_board = tuple(board_list)

        winner = self._current_player if self._is_winning_drop(updated_board, landing_row, action, stone) else None
        return _ConnectFourState(
            rows=self._rows,
            cols=self._cols,
            connect_n=self._connect_n,
            board=updated_board,
            current_player=1 - self._current_player,
            winner=winner,
            moves_played=self._moves_played + 1,
        )

    def encode(self) -> list[float]:
        """Encode perspective-relative planes: current stones, opponent stones, color plane."""

        current_stone = _PLAYER_ZERO_STONE if self._current_player == 0 else _PLAYER_ONE_STONE
        opponent_stone = -current_stone
        board_area = self._rows * self._cols

        current_plane = [0.0] * board_area
        opponent_plane = [0.0] * board_area
        for index, value in enumerate(self._board):
            if value == current_stone:
                current_plane[index] = 1.0
            elif value == opponent_stone:
                opponent_plane[index] = 1.0

        color_value = 1.0 if self._current_player == 0 else 0.0
        color_plane = [color_value] * board_area
        return current_plane + opponent_plane + color_plane

    def _find_landing_row(self, col: int) -> int | None:
        for row in range(self._rows - 1, -1, -1):
            if self._board[self._index(row, col)] == _EMPTY:
                return row
        return None

    def _is_winning_drop(self, board: tuple[int, ...], row: int, col: int, stone: int) -> bool:
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for d_row, d_col in directions:
            length = 1
            length += self._count_direction(board, row, col, d_row, d_col, stone)
            length += self._count_direction(board, row, col, -d_row, -d_col, stone)
            if length >= self._connect_n:
                return True
        return False

    def _count_direction(
        self,
        board: tuple[int, ...],
        row: int,
        col: int,
        d_row: int,
        d_col: int,
        stone: int,
    ) -> int:
        count = 0
        r = row + d_row
        c = col + d_col
        while 0 <= r < self._rows and 0 <= c < self._cols and board[self._index(r, c)] == stone:
            count += 1
            r += d_row
            c += d_col
        return count

    def _index(self, row: int, col: int) -> int:
        return row * self._cols + col


if _TORCH_AVAILABLE:

    @dataclass(slots=True)
    class _ReplayPosition:
        encoded_state: list[float]
        policy: list[float]
        value: float
        value_wdl: list[float]
        game_id: int
        move_number: int
        encoded_state_size: int
        policy_size: int


if _TORCH_AVAILABLE:

    class _TinyConnectFourNetwork(nn.Module):
        """Small network suitable for CPU-only test training."""

        def __init__(self, game_config: GameConfig) -> None:
            super().__init__()
            rows, cols = game_config.board_shape
            self._conv1 = nn.Conv2d(game_config.input_channels, 24, kernel_size=3, padding=1)
            self._conv2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
            self._relu = nn.ReLU()
            self._flatten = nn.Flatten()
            self._hidden = nn.Linear(24 * rows * cols, 64)
            self._policy = nn.Linear(64, game_config.action_space_size)
            self._value = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = self._relu(self._conv1(x))
            features = self._relu(self._conv2(features))
            features = self._relu(self._hidden(self._flatten(features)))
            policy_logits = self._policy(features)
            value = torch.tanh(self._value(features))
            return policy_logits, value


if _TORCH_AVAILABLE:

    @dataclass(slots=True)
    class _MctsNode:
        state: _ConnectFourState
        prior: float
        visit_count: int = 0
        value_sum: float = 0.0
        children: dict[int, "_MctsNode"] = field(default_factory=dict)

        def mean_value(self) -> float:
            if self.visit_count == 0:
                return 0.0
            return self.value_sum / self.visit_count


if _TORCH_AVAILABLE:

    def _masked_policy_from_logits(logits: torch.Tensor, legal_actions: Sequence[int]) -> list[float]:
        masked = torch.full_like(logits, fill_value=-1e9)
        for action in legal_actions:
            masked[action] = logits[action]
        probabilities = torch.softmax(masked, dim=0)
        return [float(value) for value in probabilities.tolist()]


if _TORCH_AVAILABLE:

    def _evaluate_state(
        model: nn.Module,
        state: _ConnectFourState,
        *,
        device: torch.device,
    ) -> tuple[list[float], float]:
        encoded = torch.as_tensor(
            state.encode(),
            dtype=torch.float32,
            device=device,
        ).reshape(1, 3, state.rows, state.cols)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            policy_logits, value = model(encoded)
        if was_training:
            model.train()

        legal_actions = state.legal_actions()
        policy = _masked_policy_from_logits(policy_logits[0], legal_actions)
        scalar_value = float(value.reshape(-1)[0].item())
        return policy, scalar_value


if _TORCH_AVAILABLE:

    def _add_dirichlet_noise(
        probabilities: list[float],
        legal_actions: Sequence[int],
        *,
        rng: random.Random,
        alpha: float = 0.3,
        epsilon: float = 0.25,
    ) -> list[float]:
        if not legal_actions:
            return probabilities
        noise = [rng.gammavariate(alpha, 1.0) for _ in legal_actions]
        total_noise = sum(noise)
        if total_noise <= 0.0:
            return probabilities
        normalized_noise = [value / total_noise for value in noise]

        updated = probabilities.copy()
        for local_index, action in enumerate(legal_actions):
            updated[action] = (
                (1.0 - epsilon) * probabilities[action]
                + epsilon * normalized_noise[local_index]
            )
        return updated


if _TORCH_AVAILABLE:

    def _select_child(node: _MctsNode, *, c_puct: float) -> tuple[int, _MctsNode]:
        if not node.children:
            raise ValueError("Cannot select a child from an unexpanded node")

        sqrt_total_visits = math.sqrt(max(1, node.visit_count))
        best_action = -1
        best_score = -float("inf")
        best_child: _MctsNode | None = None

        for action, child in node.children.items():
            q_value = -child.mean_value()
            exploration = c_puct * child.prior * sqrt_total_visits / (1 + child.visit_count)
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS failed to choose a child")
        return best_action, best_child


if _TORCH_AVAILABLE:

    def _expand_node(
        node: _MctsNode,
        prior_probabilities: Sequence[float],
    ) -> None:
        if node.children:
            return
        for action in node.state.legal_actions():
            child_state = node.state.apply_action(action)
            node.children[action] = _MctsNode(
                state=child_state,
                prior=float(prior_probabilities[action]),
            )


if _TORCH_AVAILABLE:

    def _run_mcts(
        model: nn.Module,
        root_state: _ConnectFourState,
        *,
        device: torch.device,
        rng: random.Random,
        simulations: int,
        c_puct: float,
        add_root_noise: bool,
    ) -> _MctsNode:
        root = _MctsNode(state=root_state, prior=1.0)
        if root_state.is_terminal():
            return root

        root_policy, _ = _evaluate_state(model, root_state, device=device)
        if add_root_noise:
            root_policy = _add_dirichlet_noise(
                root_policy,
                root_state.legal_actions(),
                rng=rng,
            )
        _expand_node(root, root_policy)

        for _ in range(simulations):
            node = root
            path = [node]

            while node.children and not node.state.is_terminal():
                _selected_action, node = _select_child(node, c_puct=c_puct)
                path.append(node)

            if node.state.is_terminal():
                leaf_value = node.state.outcome(node.state.current_player())
            else:
                leaf_policy, leaf_value = _evaluate_state(model, node.state, device=device)
                _expand_node(node, leaf_policy)

            backed_up_value = leaf_value
            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += backed_up_value
                backed_up_value = -backed_up_value

        return root


if _TORCH_AVAILABLE:

    def _policy_from_visits(
        root: _MctsNode,
        *,
        action_space_size: int,
        temperature: float,
    ) -> list[float]:
        policy = [0.0] * action_space_size
        if not root.children:
            return policy

        actions = list(root.children.keys())
        visit_counts = [root.children[action].visit_count for action in actions]

        if temperature <= 1e-6:
            best_action = max(actions, key=lambda action: root.children[action].visit_count)
            policy[best_action] = 1.0
            return policy

        adjusted = [count ** (1.0 / temperature) for count in visit_counts]
        normalizer = sum(adjusted)
        if normalizer <= 0.0:
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                policy[action] = uniform_prob
            return policy

        for action, probability_mass in zip(actions, adjusted):
            policy[action] = probability_mass / normalizer
        return policy


if _TORCH_AVAILABLE:

    def _sample_action_from_policy(
        policy: Sequence[float],
        *,
        legal_actions: Sequence[int],
        rng: random.Random,
    ) -> int:
        total = sum(policy[action] for action in legal_actions)
        if total <= 0.0:
            return rng.choice(list(legal_actions))

        threshold = rng.random() * total
        cumulative = 0.0
        for action in legal_actions:
            cumulative += policy[action]
            if cumulative >= threshold:
                return action
        return legal_actions[-1]


if _TORCH_AVAILABLE:

    def _self_play_game(
        model: nn.Module,
        *,
        device: torch.device,
        rng: random.Random,
        game_id: int,
        simulations_per_move: int,
        c_puct: float,
        temperature_moves: int,
    ) -> list[_ReplayPosition]:
        state = _ConnectFourState.new_game()
        trajectory: list[tuple[list[float], list[float], int, int]] = []
        move_number = 0

        while not state.is_terminal():
            root = _run_mcts(
                model,
                state,
                device=device,
                rng=rng,
                simulations=simulations_per_move,
                c_puct=c_puct,
                add_root_noise=True,
            )
            temperature = 1.0 if move_number < temperature_moves else 0.0
            policy = _policy_from_visits(
                root,
                action_space_size=state.action_space_size,
                temperature=temperature,
            )
            legal_actions = state.legal_actions()
            if temperature <= 1e-6:
                action = max(legal_actions, key=lambda candidate: policy[candidate])
            else:
                action = _sample_action_from_policy(
                    policy,
                    legal_actions=legal_actions,
                    rng=rng,
                )

            encoded_state = state.encode()
            trajectory.append((encoded_state, policy, state.current_player(), move_number + 1))
            state = state.apply_action(action)
            move_number += 1

        replay_positions: list[_ReplayPosition] = []
        for encoded_state, policy, player_to_move, sample_move_number in trajectory:
            replay_positions.append(
                _ReplayPosition(
                    encoded_state=encoded_state,
                    policy=policy,
                    value=state.outcome(player_to_move),
                    value_wdl=[0.0, 1.0, 0.0],
                    game_id=game_id,
                    move_number=sample_move_number,
                    encoded_state_size=len(encoded_state),
                    policy_size=len(policy),
                )
            )
        return replay_positions


if _TORCH_AVAILABLE:

    def _make_grad_scaler(*, enabled: bool) -> object:
        try:
            return torch.amp.GradScaler(device="cpu", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=False)


if _TORCH_AVAILABLE:

    def _train_short_connect_four_run() -> tuple[nn.Module, GameConfig, list[float]]:
        game_config = GameConfig(
            name="connect-four-mini",
            board_shape=(4, 4),
            input_channels=3,
            action_space_size=4,
            value_head_type="scalar",
            supports_symmetry=False,
            num_symmetries=1,
        )
        device = torch.device("cpu")
        model = _TinyConnectFourNetwork(game_config).to(device=device)
        lr_schedule = StepDecayLRSchedule(entries=((0, 0.05), (300, 0.01)))
        optimizer = create_optimizer(
            model,
            lr_schedule=lr_schedule,
            momentum=0.9,
        )
        scaler = _make_grad_scaler(enabled=False)

        replay_capacity = 4096
        replay: list[_ReplayPosition] = []
        rng = random.Random(2026)
        torch.manual_seed(2026)

        losses: list[float] = []
        global_step = 0

        for game_id in range(1, 61):
            replay.extend(
                _self_play_game(
                    model,
                    device=device,
                    rng=rng,
                    game_id=game_id,
                    simulations_per_move=24,
                    c_puct=1.5,
                    temperature_moves=4,
                )
            )
            if len(replay) > replay_capacity:
                del replay[: len(replay) - replay_capacity]

            if len(replay) < 128:
                continue

            for _ in range(2):
                batch = [rng.choice(replay) for _ in range(64)]
                states, target_policy, target_value = prepare_replay_batch(
                    batch,
                    game_config,
                    device=device,
                )
                metrics = train_one_step(
                    model,
                    optimizer,
                    states=states,
                    target_policy=target_policy,
                    target_value=target_value,
                    game_config=game_config,
                    lr_schedule=lr_schedule,
                    global_step=global_step,
                    l2_reg=1e-4,
                    scaler=scaler,
                    use_mixed_precision=False,
                )
                losses.append(metrics.loss_total)
                global_step += 1

        return model, game_config, losses


if _TORCH_AVAILABLE:

    def _evaluate_vs_random_player(
        model: nn.Module,
        game_config: GameConfig,
        *,
        games: int,
    ) -> float:
        rng = random.Random(99)
        device = torch.device("cpu")
        wins = 0

        for _ in range(games):
            state = _ConnectFourState.new_game(
                rows=game_config.board_shape[0],
                cols=game_config.board_shape[1],
                connect_n=4,
            )
            while not state.is_terminal():
                if state.current_player() == 0:
                    root = _run_mcts(
                        model,
                        state,
                        device=device,
                        rng=rng,
                        simulations=32,
                        c_puct=1.5,
                        add_root_noise=False,
                    )
                    policy = _policy_from_visits(
                        root,
                        action_space_size=game_config.action_space_size,
                        temperature=0.0,
                    )
                    legal_actions = state.legal_actions()
                    action = max(legal_actions, key=lambda candidate: policy[candidate])
                else:
                    action = rng.choice(state.legal_actions())
                state = state.apply_action(action)

            if state.outcome(0) > 0.0:
                wins += 1

        return wins / games


@unittest.skipUnless(_TORCH_AVAILABLE, "torch is required for Connect Four learning tests")
class ConnectFourLearningTests(unittest.TestCase):
    def test_connect_four_state_enforces_gravity_and_terminal_outcomes(self) -> None:
        """WHY: Protects the game harness rules so the learning test measures strategy, not broken mechanics."""
        state = _ConnectFourState.new_game(rows=4, cols=4, connect_n=4)

        for action in [0, 1, 0, 1, 0, 1]:
            state = state.apply_action(action)

        self.assertFalse(state.is_terminal())
        winning_state = state.apply_action(0)
        self.assertTrue(winning_state.is_terminal())
        self.assertAlmostEqual(winning_state.outcome(0), 1.0)
        self.assertAlmostEqual(winning_state.outcome(1), -1.0)

        filled_column_state = _ConnectFourState.new_game(rows=4, cols=4, connect_n=4)
        for action in [2, 2, 2, 2]:
            filled_column_state = filled_column_state.apply_action(action)
        self.assertNotIn(2, filled_column_state.legal_actions())
        with self.assertRaises(ValueError):
            _ = filled_column_state.apply_action(2)

        encoded = state.encode()
        self.assertEqual(len(encoded), 3 * 4 * 4)

    def test_short_alphazero_training_beats_random_connect_four_above_ninety_percent(self) -> None:
        """WHY: Covers TASK-093 by verifying short self-play training yields >90% wins over a random opponent."""
        model, game_config, losses = _train_short_connect_four_run()
        self.assertGreaterEqual(len(losses), 20)
        self.assertLess(losses[-1], losses[0])

        win_rate = _evaluate_vs_random_player(
            model,
            game_config,
            games=40,
        )
        self.assertGreater(win_rate, 0.9)


if __name__ == "__main__":
    unittest.main()
