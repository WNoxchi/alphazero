# Game Abstraction Layer

## 1. Design Principles

The game abstraction layer encapsulates all game-specific knowledge behind a clean interface. MCTS, neural network inference, training, and the self-play pipeline are entirely game-agnostic. Adding a new game (e.g., Shogi, Othello, Connect Four) requires only implementing this interface — no changes to any other component.

## 2. Abstract Game Interface (C++)

### GameState

The core abstraction. Represents a single game position with full state needed for play and encoding.

```cpp
class GameState {
public:
    virtual ~GameState() = default;

    // --- Core game mechanics ---

    // Apply an action, returning the resulting state.
    // The action is an index in [0, action_space_size).
    // Precondition: action is in legal_actions().
    virtual std::unique_ptr<GameState> apply_action(int action) const = 0;

    // Return all legal action indices in the current position.
    virtual std::vector<int> legal_actions() const = 0;

    // Is the game over?
    virtual bool is_terminal() const = 0;

    // Terminal outcome from the perspective of the given player.
    // Only valid when is_terminal() == true.
    // Returns: +1 (win), 0 (draw), -1 (loss).
    virtual float outcome(int player) const = 0;

    // Which player is to move: 0 or 1.
    virtual int current_player() const = 0;

    // --- Neural network encoding ---

    // Encode the current position (including history) as an input tensor.
    // Output shape: (total_input_channels, board_rows, board_cols)
    // The tensor is allocated by the caller; this method fills it.
    virtual void encode(float* buffer) const = 0;

    // --- Utilities ---

    // Deep copy of this state.
    virtual std::unique_ptr<GameState> clone() const = 0;

    // Hash of the position (for transposition tables, deduplication).
    virtual uint64_t hash() const = 0;

    // Human-readable string representation (for debugging/logging).
    virtual std::string to_string() const = 0;
};
```

### GameConfig

Static configuration for a game type. Known at compile time or at initialization; does not change during play.

```cpp
struct GameConfig {
    std::string name;                // "chess" or "go"

    // Board geometry
    int board_rows;                  // 8 (chess) or 19 (go)
    int board_cols;                  // 8 (chess) or 19 (go)

    // Neural network encoding
    int planes_per_step;             // M: feature planes per history step
    int num_history_steps;           // T: number of history steps (8)
    int constant_planes;             // L: constant-valued planes (color, castling, etc.)
    int total_input_channels;        // M * T + L

    // Action space
    int action_space_size;           // total possible action indices

    // MCTS parameters
    float dirichlet_alpha;           // Dirichlet noise parameter
    int max_game_length;             // resign/terminate after this many moves

    // Value head
    enum class ValueHeadType { SCALAR, WDL };
    ValueHeadType value_head_type;

    // Symmetry
    bool supports_symmetry;          // true for Go, false for chess
    int num_symmetries;              // 8 for Go, 1 for chess

    // Factory
    virtual std::unique_ptr<GameState> new_game() const = 0;
};
```

### Symmetry Interface

For training data augmentation (Go only). Transforms both the board encoding and the policy vector.

```cpp
struct SymmetryTransform {
    // Apply symmetry to a board tensor in-place.
    // Shape: (channels, rows, cols)
    virtual void transform_board(float* board, int channels, int rows, int cols) const = 0;

    // Apply the corresponding symmetry to a policy vector in-place.
    // Length: action_space_size
    virtual void transform_policy(float* policy, int action_space_size) const = 0;
};

// Returns all symmetry transforms for this game (including identity).
// Chess: returns {identity}
// Go: returns 8 transforms (4 rotations x 2 reflections)
virtual std::vector<std::unique_ptr<SymmetryTransform>> get_symmetries() const;
```

## 3. Action Representation

Actions are represented as **flat integer indices** throughout the system. The game implementation handles the bidirectional mapping between semantic moves and indices.

MCTS, the eval queue, and the replay buffer only see integers in `[0, action_space_size)`. The neural network policy head outputs logits of size `action_space_size`. Spatial encoding of actions (e.g., 8x8x73 planes for chess) exists only inside the NN architecture as an implementation detail.

```cpp
// Inside each game implementation:
int semantic_move_to_action_index(const Move& move) const;
Move action_index_to_semantic_move(int action) const;
```

### Illegal Move Masking

The policy output from the NN includes logits for all `action_space_size` actions, including illegal ones. Before use in MCTS:

1. Get `legal_actions()` from the game state.
2. Set logits for all illegal actions to `-infinity`.
3. Apply softmax over remaining legal actions.

This masking is performed at the boundary between NN inference and MCTS, not inside the game or NN.

## 4. History Management

AlphaZero encodes the last T=8 board positions as input to the NN. History is **internal to GameState**:

- Each `GameState` stores a reference or copy of the previous T-1 board positions.
- The `encode()` method produces the full `(M*T + L)` channel tensor.
- For time steps before the start of the game (t < 0), the corresponding planes are filled with zeros.

Implementation options (game-specific choice):
- **Copy-on-write linked list**: Each state points to its parent. `encode()` walks back T steps. Memory-efficient; no redundant copies.
- **Inline ring buffer**: Each state stores the last T board arrays. Constant-time encoding; more memory.

Recommendation: copy-on-write linked list for Go (large boards, many moves), inline buffer for chess (small boards).

## 5. Chess Implementation

### Board Representation: Bitboards

Chess positions are represented using **bitboards** — one 64-bit integer per piece type per color, where each bit corresponds to a square.

```cpp
struct ChessPosition {
    // 12 bitboards: 6 piece types x 2 colors
    uint64_t pieces[2][6];  // [color][piece_type]
    // color: 0=white, 1=black
    // piece_type: 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king

    // Side to move
    int side_to_move;        // 0=white, 1=black

    // Castling rights (4 bits)
    uint8_t castling;        // bit 0: white kingside, bit 1: white queenside,
                             // bit 2: black kingside, bit 3: black queenside

    // En passant target square (-1 if none)
    int en_passant_square;

    // Halfmove clock (for 50-move rule)
    int halfmove_clock;

    // Fullmove number
    int fullmove_number;

    // Repetition count for this position
    int repetition_count;
};
```

Bitboard operations (population count, bit scan, shift) map to single ARM instructions and are extremely fast for legal move generation.

### Input Encoding (from AlphaZero paper)

The input to the NN is an `8 x 8 x 119` tensor.

**Per-timestep feature planes (M=14 planes, repeated for T=8 history steps = 112 planes):**

| Planes | Count | Description |
|---|---|---|
| P1 pieces | 6 | One plane per piece type (pawn, knight, bishop, rook, queen, king) for current player |
| P2 pieces | 6 | One plane per piece type for opponent |
| Repetitions | 2 | Binary: position has occurred 1 time / 2+ times |

**Constant-valued planes (L=7 planes):**

| Planes | Count | Description |
|---|---|---|
| Color | 1 | All 1s if current player is white, all 0s if black |
| Total move count | 1 | Scalar value broadcast to all squares (normalized) |
| P1 kingside castling | 1 | All 1s if right exists, all 0s otherwise |
| P1 queenside castling | 1 | All 1s if right exists, all 0s otherwise |
| P2 kingside castling | 1 | Same for opponent |
| P2 queenside castling | 1 | Same for opponent |
| No-progress count | 1 | Halfmove clock (50-move rule), normalized |

**Total: 14 * 8 + 7 = 119 input planes.**

The board is always oriented from the perspective of the current player: if black is to move, the board is flipped vertically so that black's pieces appear in the "first player" planes and the board layout matches black's perspective.

### Action Encoding

Chess actions are encoded as a flat index in `[0, 4672)`, derived from the AlphaZero paper's spatial encoding:

**Conceptually an 8 x 8 x 73 tensor** (from-square x move-type):

| Move type | Planes | Description |
|---|---|---|
| Queen moves | 56 | 7 distances x 8 directions (N, NE, E, SE, S, SW, W, NW) |
| Knight moves | 8 | 8 possible L-shaped knight moves |
| Underpromotions | 9 | 3 piece types (knight, bishop, rook) x 3 directions (forward, capture-left, capture-right) |

Normal pawn promotions to queen are encoded as queen moves. The 73 planes x 64 squares = 4,672 action indices.

```
action_index = from_square * 73 + move_type_index
```

When the current player is black, the from-square and move direction are mirrored to match the flipped board perspective.

### Move Generation

Implement standard bitboard move generation:
- **Sliding pieces** (bishop, rook, queen): Magic bitboards or kogge-stone for ray attack generation.
- **Knights, kings**: Precomputed attack tables.
- **Pawns**: Direction-dependent push/capture masks; en passant; promotion.
- **Castling**: Check for rights, empty squares, non-attacked squares.
- **Legality**: Generate pseudo-legal moves, filter those that leave the king in check.

### Terminal Conditions

A chess game ends when:
1. **Checkmate**: Current player's king is in check and no legal moves exist. Outcome: loss for current player.
2. **Stalemate**: Current player has no legal moves but is not in check. Outcome: draw.
3. **50-move rule**: `halfmove_clock >= 100` (50 full moves). Outcome: draw.
4. **Threefold repetition**: Position has occurred 3 times. Outcome: draw.
5. **Insufficient material**: Neither side can checkmate (e.g., K vs K, K+B vs K). Outcome: draw.
6. **Max game length**: Exceeded `max_game_length` (512 moves). Outcome: draw.

### Serialization

Support FEN (Forsyth-Edwards Notation) for position import/export and PGN for game records. Useful for debugging, evaluation against external engines, and logging.

## 6. Go Implementation

### Board Representation

Go positions are represented as a flat array of intersections:

```cpp
struct GoPosition {
    // Board state: 0=empty, 1=black, 2=white
    uint8_t board[19][19];

    // Side to move: 1=black, 2=white
    int side_to_move;

    // Ko point: illegal recapture point (-1 if none)
    int ko_point;

    // Komi: compensation points for white (typically 7.5)
    float komi;

    // Move count
    int move_number;

    // Pass count: consecutive passes (game ends at 2)
    int consecutive_passes;

    // Superko history (set of position hashes for positional superko)
    // Used to detect and prohibit repeated positions.
    std::unordered_set<uint64_t> position_history;
};
```

### Zobrist Hashing

Use Zobrist hashing for fast position comparison and superko detection:
- Pre-generate random 64-bit values for each (intersection, color) pair.
- XOR values incrementally as stones are placed/captured.
- Hash comparison for repetition detection is O(1).

### Input Encoding (from AlphaGo Zero paper)

The input to the NN is a `19 x 19 x 17` tensor.

**Per-timestep feature planes (M=2 planes, repeated for T=8 history steps = 16 planes):**

| Planes | Count | Description |
|---|---|---|
| Current player stones | 1 | Binary: 1 if intersection has a stone of the current player's color |
| Opponent stones | 1 | Binary: 1 if intersection has an opponent's stone |

**Constant-valued planes (L=1 plane):**

| Planes | Count | Description |
|---|---|---|
| Color | 1 | All 1s if black to play, all 0s if white to play |

**Total: 2 * 8 + 1 = 17 input planes.**

### Action Encoding

Go actions are a flat index in `[0, 362)`:
- Indices `0..360`: Place a stone at intersection `(row * 19 + col)`.
- Index `361`: Pass.

```
action_index = row * 19 + col     (for stone placement)
action_index = 361                 (for pass)
```

### Go Rules Implementation

Go rules are deceptively complex. The implementation must handle:

1. **Liberties**: A group of connected same-color stones survives if it has at least one empty adjacent intersection (liberty). Groups with zero liberties are captured and removed.

2. **Capture**: When a stone is placed, first check if any opponent groups lose their last liberty. If so, remove those groups. Then check if the placed stone's own group has liberties (self-capture is illegal in standard rules).

3. **Ko**: A single-stone recapture that would restore the previous board position is prohibited. Track the ko point after a single-stone capture.

4. **Superko** (positional): No board position may be repeated. Track all previous board hashes. A move that would recreate any previous position is illegal.

5. **Scoring**: Use **Tromp-Taylor rules** for self-play and training (as per AlphaGo Zero). Tromp-Taylor is simpler to implement and avoids ambiguities around dead stone removal:
   - A point scores for a color if it is either occupied by that color or is only reachable (via empty intersections) by that color.
   - Final score = black_points - white_points - komi.
   - Evaluation games can use Chinese rules (compatible with Tromp-Taylor for correctly finished games).

6. **Pass**: A player may pass instead of placing a stone. Two consecutive passes end the game.

7. **Self-capture**: Illegal (standard rules). A move that would result in the player's own group having zero liberties (without capturing any opponent stones) is prohibited.

### Liberty Tracking Optimization

For performance, maintain liberty counts incrementally using a **union-find (disjoint set)** data structure for stone groups:

```cpp
struct StoneGroup {
    int representative;    // union-find root
    int liberty_count;     // number of unique liberties
    int stone_count;       // number of stones in group
    // Liberty set can be tracked via a hash or small set
};
```

When a stone is placed:
1. Merge the new stone with adjacent same-color groups (union-find merge).
2. Subtract liberties from adjacent opponent groups.
3. Remove any opponent groups with zero liberties.
4. Verify the new stone's group has at least one liberty.

This makes move application O(board_perimeter) in the worst case, but O(1) amortized for most moves.

### Terminal Conditions

A Go game ends when:
1. **Two consecutive passes**: Game is scored by Tromp-Taylor rules.
2. **Resignation**: Value falls below resignation threshold.
3. **Max game length**: Exceeded `max_game_length` (722 = 19 * 19 * 2). Outcome: scored by Tromp-Taylor.

### Symmetry

Go is invariant under the 8 symmetries of the square (dihedral group D4):
- 4 rotations (0, 90, 180, 270 degrees)
- 4 reflections (horizontal, vertical, and two diagonal axes)

These are used for **training data augmentation only** (not during MCTS). Given a training sample `(board, policy, outcome)`, generate 8 equivalent samples by applying each symmetry to both the board encoding and the policy vector.

The policy transform for Go is straightforward: permute the 361 intersection probabilities according to the same rotation/reflection. The pass action (index 361) is invariant.

### Serialization

Support SGF (Smart Game Format) for game records. Useful for analysis and compatibility with Go tools.

## 7. Python-Side Game Configuration

The Python training code needs game configuration to construct the neural network:

```python
@dataclass
class GameConfig:
    name: str
    board_shape: tuple[int, int]       # (8, 8) or (19, 19)
    input_channels: int                 # 119 or 17
    action_space_size: int              # 4672 or 362
    value_head_type: str                # "scalar" or "wdl"
    supports_symmetry: bool
    num_symmetries: int                 # 1 or 8

CHESS_CONFIG = GameConfig(
    name="chess",
    board_shape=(8, 8),
    input_channels=119,
    action_space_size=4672,
    value_head_type="wdl",
    supports_symmetry=False,
    num_symmetries=1,
)

GO_CONFIG = GameConfig(
    name="go",
    board_shape=(19, 19),
    input_channels=17,
    action_space_size=362,
    value_head_type="scalar",
    supports_symmetry=True,
    num_symmetries=8,
)
```

This config is passed to the network factory, replay buffer, and training loop. It is the single source of truth for game-specific dimensions.

## 8. Testing Strategy for Game Implementations

### Chess
- **Perft testing**: Compare move generation counts at various depths against known-correct perft results. This is the standard correctness test for chess engines.
- **FEN round-trip**: Encode → decode → encode and verify identity.
- **Known positions**: Test checkmate, stalemate, en passant, castling, promotion, 50-move rule, threefold repetition against known positions.
- **Encoding verification**: Compare input tensor encoding against a reference implementation for known positions.

### Go
- **Liberty counting**: Test capture scenarios, including ko, snapback, large captures.
- **Tromp-Taylor scoring**: Verify scoring against known game results.
- **Superko detection**: Test known superko positions.
- **Symmetry correctness**: Apply all 8 transforms and verify the game state is equivalent.
- **Encoding verification**: Compare input tensor encoding against reference for known positions.
