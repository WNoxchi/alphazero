# External Elo Assessment

## Goal

Benchmark AlphaZero checkpoints against external engines with known Elo ratings to
track absolute strength over the course of training.

## What We Already Have

- **`scripts/play.py`**: Has `UciEngineClient` (UCI **client**) and
  `play_engine_match()` that plays AlphaZero vs any UCI engine. Supports
  `--opponent`, `--games`, `--simulations`, `--engine-time-ms`, and alternates
  colors each game. This is the critical piece — it already drives external engines
  via UCI.
- **`web/watch_engine.py`**: `WatchEngine` plays two of our own models against each
  other (used by the web UI watch mode). Demonstrates the same MCTS flow:
  `set_root_state()` → `run_simulations()` → `select_action()` → `apply_action()`.
- **`python/alphazero/pipeline/evaluation.py`**: Has `PeriodicEloEvaluator`,
  `estimate_elo_from_score()`, `MatchOutcome`, and `EloEvaluationResult` — all the
  Elo math and scheduling, but no concrete `MatchRunner` and not wired into the
  orchestrator.
- **C++ bindings for chess**: `ChessState.from_fen()`, `to_fen()`,
  `action_to_uci()`, `uci_to_action()`, `legal_actions_uci()`, `apply_action()` —
  all the state management needed for UCI protocol handling.

## Key Architectural Insight: Client vs Server

UCI is the universal protocol for chess engines. Stockfish, LC0, and Maia (via LC0)
all speak it. There are two sides:

- **UCI client** (we drive an external engine): `play.py`'s `UciEngineClient`
  already does this. This is what we need for Elo benchmarking — launch stockfish or
  lc0 as a subprocess and exchange moves over stdin/stdout.

- **UCI server** (external tools drive our engine): A separate adapter that makes
  our engine look like a standard UCI engine. Needed for cutechess-cli, chess GUIs,
  or tournament entry, but **not required for Elo benchmarking**.

**For Elo measurement, the shortest path is extending the existing client side**
(play.py). The UCI server adapter is a nice-to-have for interoperability.

## Calibrated Opponents

### Maia Chess (Elo 1100-1900) — human-like play at specific ratings

- Neural nets trained to mimic human play at specific Elo brackets (1100, 1200, ..., 1900).
- Weights are LC0-format `.pb.gz` files, publicly available at:
  https://github.com/CSSLab/maia-chess
- Run with the `lc0` engine at `--nodes=1` (raw policy, no tree search).
- Unlike weakened Stockfish, Maia makes human-typical mistakes (positional
  misunderstandings, tactical oversights) rather than random blunders.
- Best for: early-to-mid training benchmarking.

### Stockfish with UCI_LimitStrength (Elo ~1320-3190)

- Set UCI options `UCI_LimitStrength=true` and `UCI_Elo=<value>` for a continuous
  Elo scale.
- The weakening mechanism introduces deliberate errors — plays strong moves most of
  the time, then occasionally makes a catastrophic blunder. Not human-like, but
  numerically calibrated.
- Approximate Skill Level mapping (for reference):

  | Skill Level | ~Elo  | Skill Level | ~Elo  |
  |-------------|-------|-------------|-------|
  | 0           | 800   | 10          | 1950  |
  | 1           | 1050  | 11          | 2050  |
  | 2           | 1150  | 12          | 2150  |
  | 3           | 1250  | 13          | 2250  |
  | 4           | 1350  | 14          | 2350  |
  | 5           | 1450  | 15          | 2450  |
  | 6           | 1550  | 16          | 2550  |
  | 7           | 1650  | 17          | 2650  |
  | 8           | 1750  | 18          | 2750  |
  | 9           | 1850  | 19          | 2850  |
  |             |       | 20          | 3500+ |

- `UCI_LimitStrength`/`UCI_Elo` is preferred over `Skill Level` — more precise and
  continuous.
- Best for: broad-range benchmarking across the full training curve.

### LC0 (Leela Chess Zero) — most architecturally similar

- Same MCTS + neural network approach as our engine. Fairest apples-to-apples
  comparison of network quality.
- Networks at various strengths: https://lczero.org/play/networks/bestnets/
- Strongest nets (T80, BT4 series): ~3500+ Elo. Early/small nets: ~1500+.
- Control strength via `--nodes=N`, smaller network architectures, or earlier
  training-run networks.
- Comparing at equal node counts isolates network quality from search budget.
- Best for: measuring neural network quality independent of search.

### Other Options

| Engine      | Elo Range  | Notes                                  |
|-------------|------------|----------------------------------------|
| Rodent IV   | 800-3000+  | Highly configurable strength/style     |
| GNU Chess   | ~2200      | Classic, stable                        |
| Crafty      | ~2600      | Well-known older engine                |
| Fruit 2.1   | ~2800      | Famous historical engine               |
| TSCP        | ~1800      | Very simple, good baseline             |

CCRL (https://ccrl.chessdom.com/) has Elo ratings for hundreds of engines.

## How External Engines Are Installed and Invoked

| Engine | Install | How play.py drives it |
|---|---|---|
| Stockfish | `apt install stockfish` or download binary | `--opponent stockfish` (needs `--engine-option` for `UCI_LimitStrength`) |
| LC0 + Maia | Install `lc0` binary + download `.pb.gz` weights from GitHub | `--opponent "lc0 --weights=maia-1500.pb.gz --nodes=1"` |
| LC0 (strong) | Same `lc0` binary + download a strong net from lczero.org | `--opponent "lc0 --weights=t80-net.pb.gz --nodes=800"` |

All communication happens over UCI protocol via stdin/stdout of the subprocess.

## Running Matches

### Using existing play.py (works today)

```bash
# AlphaZero vs Stockfish at Elo 1500, 50 games
PYTHONPATH=build/src:$PYTHONPATH python scripts/play.py \
  --game chess \
  --model checkpoints/milestone_00050000.pt \
  --opponent "stockfish" \
  --games 50 \
  --simulations 800 \
  --engine-time-ms 1000

# AlphaZero vs Maia 1500 (via lc0)
PYTHONPATH=build/src:$PYTHONPATH python scripts/play.py \
  --game chess \
  --model checkpoints/milestone_00050000.pt \
  --opponent "lc0 --weights=maia-1500.pb.gz --nodes=1" \
  --games 50 \
  --simulations 800 \
  --engine-time-ms 5000
```

Limitation: `play.py` doesn't pass UCI options to the opponent engine (e.g.,
`UCI_LimitStrength`). To use Stockfish at a specific Elo, you'd currently need a
wrapper script or we need to extend `play.py` to support `--engine-option` flags.

### Using cutechess-cli / fastchess (requires UCI server adapter for our engine)

The standard tool for automated engine-vs-engine matches. Requires our engine to
speak UCI as a **server** (act as a responder, not a driver). This is a separate
piece of work from the play.py extensions.

```bash
cutechess-cli \
  -engine name=AlphaZero cmd=./scripts/uci_engine.py proto=uci \
  -engine name=SF1500 cmd=stockfish proto=uci \
      option.UCI_LimitStrength=true option.UCI_Elo=1500 \
  -each tc=60+0.6 \
  -rounds 200 \
  -openings file=openings.pgn format=pgn order=random \
  -pgnout results.pgn \
  -recover -repeat
```

Advantages over play.py:
- Handles UCI option passing natively.
- Can set time controls, use opening books, run concurrent games.
- Built-in SPRT for efficient Elo estimation.
- Outputs PGN, Elo estimates, and statistical confidence intervals.
- Download: https://github.com/cutechess/cutechess
- Alternative: https://github.com/Disservin/fastchess (faster, simpler)

### Statistical requirements

- ~200 games: rough estimate (error bars ~50-100 Elo)
- ~500 games: reasonable precision (~20-30 Elo error)
- Use opening books to reduce variance (e.g., `noob_3moves.epd`, `8moves_v3.pgn`).
- Alternate colors each game (play.py already does this).

## What Needs to Be Built

### Phase 1: Elo benchmarking (client-side, extend play.py)

This is the shortest path to measuring Elo against external engines.

**1a. `--engine-option` support in play.py**

Allow passing UCI options to the opponent engine so we can configure Stockfish's
strength level:

```bash
python scripts/play.py \
  --opponent stockfish \
  --engine-option UCI_LimitStrength=true \
  --engine-option UCI_Elo=1500 \
  ...
```

Small change: send `setoption name X value Y` commands in `UciEngineClient.initialize()`.

**1b. Benchmark ladder script (`scripts/benchmark_elo.py`)**

Convenience script that runs a checkpoint against a ladder of opponents:

1. Load a checkpoint.
2. Play N games against each opponent in the ladder.
3. Compute win/draw/loss and Elo difference for each (reuse `evaluation.py` math).
4. Estimate absolute Elo by anchoring to the opponent's known rating.
5. Output a summary table and optionally log to TensorBoard.

### Phase 2: UCI server adapter (nice-to-have, for interoperability)

Makes our engine act as a standard UCI engine so external tools can drive it.

**What it enables:** cutechess-cli orchestration, chess GUI compatibility, tournament
entry.

**Minimal UCI commands to implement:**

```
uci              -> id name AlphaZero\nid author ...\nuciok
isready          -> readyok
ucinewgame       -> reset MCTS
position ...     -> set up board position
go ...           -> run MCTS, respond "bestmove <uci_move>"
quit             -> exit
```

**Building blocks already exist** in `play.py` — model loading, evaluator setup,
`MctsSearch`, `ChessState.from_fen()`, `action_to_uci()` / `uci_to_action()`. The
adapter is ~150-200 lines of UCI protocol parsing on top.

**Search strategy: fixed simulations vs interruptible.**

Our `MctsSearch.run_simulations()` is a blocking C++ loop with no interruption
mechanism. Two options:

- **Fixed simulations (no C++ changes):** Always run N simulations, ignore time
  controls and `stop` commands. Works fine for benchmarking (we control both sides)
  and with cutechess-cli using `nodes` or `movetime` time controls. play.py and
  WatchEngine already work this way.

- **Interruptible search (requires C++ change):** Add an atomic stop flag to the
  `run_simulations` loop so it can bail early. Needed for real clock time controls
  (e.g., "5 min per side") or tournament play. Small change but touches the C++ hot
  path.

**For Elo benchmarking, fixed simulations is the right choice.** We're measuring
network quality at a fixed search budget, not testing time management.
Interruptible search is only needed for time-controlled competition.

### Phase 3: Wire into training loop (optional)

Implement a concrete `MatchRunner` that calls `play_engine_match()` (or a lighter
internal version) and plug it into `PeriodicEloEvaluator` in the orchestrator.
Would enable automated Elo tracking during training. Lower priority — the
standalone benchmark script is more immediately useful.

## Suggested Opponent Ladder

| Stage             | Opponent                              | Expected Elo |
|-------------------|---------------------------------------|--------------|
| Very early        | Maia 1100 (lc0 --nodes=1)            | ~1100        |
| Early             | Maia 1500 (lc0 --nodes=1)            | ~1500        |
| Intermediate      | Stockfish UCI_Elo=1800                | ~1800        |
| Club player       | Stockfish UCI_Elo=2200                | ~2200        |
| Strong            | Stockfish UCI_Elo=2600                | ~2600        |
| Expert            | Stockfish UCI_Elo=3000                | ~3000        |
| Top engine        | LC0 (strong net, 800 nodes)           | ~3200+       |
| Full strength     | Stockfish 17 (full)                   | ~3500+       |
