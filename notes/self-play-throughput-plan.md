# Self-Play Diversity and Throughput Improvements

## Motivation

At 80K training steps on a single NVIDIA GB10 (128GB unified memory), the chess model generates only ~8K unique self-play games. The replay buffer holds 750K positions, meaning each position is sampled ~870 times during training (80K steps × 8192 batch size / 750K positions). This extreme oversampling limits the model's ability to learn from diverse game states and caps effective strength around ~1200-1600 Elo.

The root cause is the `ReplayPosition` struct in `src/selfplay/replay_buffer.h`, which stores each position as ~48KB of dense float arrays (7616 floats for state + 4672 floats for policy). At 48KB/position, 750K positions consume ~36GB — nearly all available headroom in the 128GB system. Meanwhile, the data is highly compressible: chess board encodings are 117 binary planes (0/1 values stored as float32) plus 2 float planes, and MCTS policy distributions are extremely sparse (~5-30 non-zero entries out of 4672).

**Goal**: 40x buffer capacity increase (750K → 30M positions) through in-memory compression, combined with ~2x game throughput increase via playout cap randomization. This reduces per-position sampling from ~870x to ~2x, fundamentally shifting from overfitting to healthy data diversity.

## Current Architecture

### Data Flow

1. **Self-play** (`src/selfplay/self_play_game.cpp`): `SelfPlayGame::play()` generates `PendingSample` vectors with `encoded_state` (from `GameState::encode()`) and `policy` (from `MctsSearch::root_policy_target()`). Calls `ReplayPosition::make()` and `ReplayBuffer::add_game()` when a game completes.

2. **Storage** (`src/selfplay/replay_buffer.{h,cpp}`): Ring buffer of fixed-size `ReplayPosition` structs. Pre-allocates `std::vector<ReplayPosition>` at capacity. Thread-safe: `std::shared_mutex` for reads, exclusive lock for writes.

3. **Sampling** (`replay_buffer.cpp`): `sample_batch()` copies sampled positions into flat `SampledBatch` vectors of floats, returned to Python via pybind11 bindings in `src/bindings/python_bindings.cpp`.

4. **Training** (`python/alphazero/training/trainer.py`): `sample_replay_batch_tensors()` calls `sample_batch()`, reshapes into PyTorch tensors.

5. **Checkpoints** (`python/alphazero/utils/checkpoint.py`): `save_replay_buffer_state()` exports buffer via `export_positions()` into flat numpy arrays, saved as `.replay.npz`.

### Chess Encoding Layout

From `src/games/chess/chess_state.{h,cpp}`:

- `kHistorySteps = 8`, `kPlanesPerStep = 14`, `kConstantPlanes = 7`
- `kTotalInputChannels = 8 * 14 + 7 = 119` planes, each 8×8 = 64 squares
- **Per history step** (14 planes each): 12 piece planes (sparse binary, individual squares set to 1.0 via bitboard iteration in `encode_position_planes()` at line 748) + 2 repetition planes (constant-fill binary via `fill_plane()` at lines 773-774)
- **Constant planes** (7 total, starting at `constant_offset = 112`):
  - Plane 112: color (binary 0.0 or 1.0) — `fill_plane()` line 616
  - Plane 113: `normalized_move_count()` (float in [0,1]) — `fill_plane()` line 617
  - Plane 114-117: castling rights (4 binary planes) — `fill_plane()` lines 629-632
  - Plane 118: `normalized_no_progress_count()` (float in [0,1]) — `fill_plane()` line 633
- **Summary**: 117 binary planes (indices 0-112, 114-117) + 2 float planes (indices 113, 118)
- All constant/repetition planes use `fill_plane()` which sets all 64 squares to the same value

### Current Config (`configs/chess.yaml`)

```yaml
mcts:
  simulations_per_move: 200
  concurrent_games: 384
  threads_per_game: 1
  batch_size: 384
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
  temperature_moves: 30
  resign_threshold: -0.9
  resign_disable_fraction: 0.1
replay_buffer:
  capacity: 750000
training:
  batch_size: 8192
pipeline:
  inference_batches_per_cycle: 100
  training_steps_per_cycle: 1
```

### Current Performance

- Self-play: ~670 games/hr, avg 32 moves/game
- GPU split: ~85% inference, ~15% training
- Buffer fills to 750K after ~35 hours of self-play
- Training throughput: ~0.31 train steps/sec

---

## Phase 1: Compact Replay Buffer

**Impact**: 40x capacity increase (750K → 30M positions in same ~36GB)

### 1.1 New `CompactReplayPosition` struct

Status (2026-02-26): Completed. Added `CompactReplayPosition` to `src/selfplay/replay_buffer.h` with fixed-capacity
bitpacked state planes, quantized float planes, sparse policy fields, and replay metadata; covered by
`ReplayBufferTest.CompactReplayPositionDefinesExpectedConstantsAndZeroDefaults` and
`ReplayBufferTest.CompactReplayPositionIsSubstantiallySmallerThanDenseReplayPosition`.

Add to `src/selfplay/replay_buffer.h` alongside existing `ReplayPosition` (do not remove it):

```cpp
struct CompactReplayPosition {
    static constexpr std::size_t kMaxBinaryPlanes = 117U;    // chess: 117
    static constexpr std::size_t kMaxFloatPlanes = 2U;       // chess: 2
    static constexpr std::size_t kMaxSparsePolicy = 64U;     // top-K policy entries
    static constexpr std::size_t kWdlSize = 3U;

    // State: 117 binary planes bitpacked (1 bit per square → 64 bits per plane)
    std::array<std::uint64_t, kMaxBinaryPlanes> bitpacked_planes{};
    // State: 2 float planes quantized to uint8 (value * 255)
    std::array<std::uint8_t, kMaxFloatPlanes> quantized_float_planes{};

    // Sparse policy: top-K (action_index, fp16_probability) pairs
    std::array<std::uint16_t, kMaxSparsePolicy> policy_actions{};
    std::array<std::uint16_t, kMaxSparsePolicy> policy_probs_fp16{};
    std::uint8_t num_policy_entries = 0U;

    // Metadata (same as ReplayPosition)
    float value = 0.0F;
    std::array<float, kWdlSize> value_wdl{0.0F, 0.0F, 0.0F};
    std::uint32_t game_id = 0U;
    std::uint16_t move_number = 0U;
    std::uint16_t num_binary_planes = 0U;
    std::uint16_t num_float_planes = 0U;
    std::uint16_t policy_size = 0U;  // original full policy size for decompression
};
// Size: 117*8 + 2 + 64*2 + 64*2 + 1 + 4 + 12 + 4 + 2 + 2 + 2 + 2 = ~1,223 bytes
```

### 1.2 Compression/decompression helpers

Status (2026-02-26): Completed. Added standalone helpers in `src/selfplay/replay_compression.{h,cpp}` for
`compress_state`, `decompress_state`, `compress_policy`, `decompress_policy`, plus FP16 conversion utilities; wired
the implementation into `src/CMakeLists.txt` and added coverage in `tests/cpp/test_replay_compression.cpp`.

Create standalone, unit-testable functions (can live in `compact_replay_buffer.cpp` or a separate `compression.{h,cpp}`):

**State compression** (`compress_state`):
```
Input: float dense_state[119 * 64], vector<size_t> float_plane_indices = {113, 118}
Output: uint64_t bitpacked[117], uint8_t quantized[2]

binary_idx = 0, float_idx = 0
For plane_index in 0..118:
    If plane_index is in float_plane_indices:
        // All 64 values are identical (fill_plane). Read one sample.
        float val = dense_state[plane_index * 64]
        quantized[float_idx++] = uint8_t(round(clamp(val, 0, 1) * 255))
    Else:
        uint64_t bits = 0
        For sq in 0..63:
            If dense_state[plane_index * 64 + sq] >= 0.5f:
                bits |= (1ULL << sq)
        bitpacked[binary_idx++] = bits
```

**State decompression** (`decompress_state`):
```
Output: float dense_state[119 * 64]

binary_idx = 0, float_idx = 0
For plane_index in 0..118:
    If plane_index is in float_plane_indices:
        float val = quantized[float_idx++] / 255.0f
        Fill dense_state[plane_index*64 .. plane_index*64+63] with val
    Else:
        uint64_t bits = bitpacked[binary_idx++]
        For sq in 0..63:
            dense_state[plane_index * 64 + sq] = (bits >> sq) & 1 ? 1.0f : 0.0f
```

**Policy compression** (`compress_policy`):
```
Input: float policy[4672]
Output: uint16_t actions[64], uint16_t probs_fp16[64], uint8_t count

1. Collect all (index, value) pairs where value > 0
2. Sort descending by value
3. Take top min(K=64, num_nonzero) entries
4. Re-normalize so sum of kept entries equals original sum of kept entries
5. Convert each probability to float16 (use C++ <bit> or manual conversion)
6. Store in output arrays, set count
```

**Policy decompression** (`decompress_policy`):
```
Output: float policy[4672]

1. Zero-fill entire policy array
2. For i in 0..count-1:
     policy[actions[i]] = fp16_to_float(probs_fp16[i])
```

### 1.3 New `CompactReplayBuffer` class

Status (2026-02-26): Completed. Added `CompactReplayBuffer` in
`src/selfplay/compact_replay_buffer.{h,cpp}` with compression-on-write and
decompression-on-read for `add_game`, `sample`, `sample_batch`,
`export_positions`, and `import_positions`, including ring-buffer and
thread-safety behavior matching `ReplayBuffer`. Wired compilation in
`src/CMakeLists.txt` and added coverage in `tests/cpp/test_compact_replay_buffer.cpp`.

Create new files: `src/selfplay/compact_replay_buffer.h` and `src/selfplay/compact_replay_buffer.cpp`.

```cpp
class CompactReplayBuffer {
public:
    explicit CompactReplayBuffer(
        std::size_t capacity,
        std::size_t num_binary_planes,          // 117 for chess
        std::size_t num_float_planes,           // 2 for chess
        std::vector<std::size_t> float_plane_indices,  // {113, 118} for chess
        std::size_t full_policy_size,           // 4672 for chess
        std::uint64_t random_seed = 0x9E3779B97F4A7C15ULL);

    // Write path: accepts dense ReplayPosition vector, compresses internally
    void add_game(const std::vector<ReplayPosition>& positions);

    // Read path: decompresses on the fly, returns same SampledBatch format
    [[nodiscard]] SampledBatch sample_batch(
        std::size_t batch_size,
        std::size_t encoded_state_size,
        std::size_t policy_size,
        std::size_t value_dim) const;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t write_head() const noexcept;

    // Export/import: same flat numpy format for checkpoint compat
    std::size_t export_positions(
        float* out_states, float* out_policies, float* out_values_wdl,
        std::uint32_t* out_game_ids, std::uint16_t* out_move_numbers,
        std::size_t encoded_state_size, std::size_t policy_size) const;

    void import_positions(
        const float* states, const float* policies, const float* values_wdl,
        const std::uint32_t* game_ids, const std::uint16_t* move_numbers,
        std::size_t count, std::size_t encoded_state_size, std::size_t policy_size);

private:
    std::vector<CompactReplayPosition> buffer_;
    std::atomic<std::size_t> write_head_{0U};
    std::atomic<std::size_t> count_{0U};
    mutable std::shared_mutex mutex_;
    mutable std::mutex rng_mutex_;
    mutable std::mt19937_64 rng_;

    std::vector<std::size_t> float_plane_indices_;
    std::size_t num_binary_planes_;
    std::size_t num_float_planes_;
    std::size_t full_policy_size_;

    // Same helper methods as ReplayBuffer
    [[nodiscard]] std::vector<std::size_t> sample_logical_indices(...) const;
    [[nodiscard]] std::size_t to_physical_index(...) const noexcept;
};
```

Key design: the public API matches `ReplayBuffer` exactly. Self-play code (`SelfPlayGame::play()`) still generates dense `ReplayPosition` objects — compression happens inside `add_game()`. Decompression happens inside `sample_batch()`. No changes to self-play or training code.

### 1.4 CMake changes

Add to `src/CMakeLists.txt`:
```cmake
target_sources(alphazero_engine PRIVATE
    selfplay/compact_replay_buffer.cpp
)
```

### 1.5 Python bindings

In `src/bindings/python_bindings.cpp`, add `CompactReplayBuffer` binding with the same Python API as `ReplayBuffer`. The `sample_batch` binding already returns numpy arrays via `SampledBatch` — no downstream changes needed.

Constructor takes additional game-specific args:
```python
compact_buf = cpp.CompactReplayBuffer(
    capacity=5_000_000,
    num_binary_planes=117,
    num_float_planes=2,
    float_plane_indices=[113, 118],
    full_policy_size=4672,
)
```

### 1.6 Wire up in `scripts/train.py`

Modify `_build_replay_buffer()` (currently constructs `ReplayBuffer`) to construct `CompactReplayBuffer` instead, passing game-specific plane metadata derived from the game config.

The `GameConfig` struct already has `input_channels` (119), `board_shape` (8,8), `action_space_size` (4672). Need to add or derive `float_plane_indices` — either add to the game config or hardcode per game type. Recommend adding `float_plane_indices` to `GameConfig` in `src/games/chess/chess_config.cpp` and `src/games/go/go_config.cpp`.

### 1.7 Checkpoint backward compatibility

In `python/alphazero/utils/checkpoint.py`:
- `export_buffer()` on `CompactReplayBuffer` decompresses to the same flat numpy arrays (`states`, `policies`, `values_wdl`, `game_ids`, `move_numbers`). Same `.replay.npz` format.
- `import_buffer()` on `CompactReplayBuffer` accepts the same flat numpy arrays and compresses on import.
- Old checkpoints (dense float arrays from `ReplayBuffer`) load seamlessly.
- Optionally: add a compact checkpoint format later to speed up I/O.

### 1.8 Config change

In `configs/chess.yaml`, increase buffer capacity:
```yaml
replay_buffer:
  capacity: 5000000  # start conservative; can go up to 30M
```

### 1.9 Tests

New file `tests/cpp/test_compact_replay_buffer.cpp`:
- **Roundtrip compression**: compress dense state → decompress → verify bit-exact for binary planes, within 1/255 for float planes
- **Sparse policy roundtrip**: compress → decompress → verify top entries match original, sum preserved within float16 tolerance (~1e-3)
- **Ring buffer behavior**: add more than capacity, verify oldest evicted, sampling still works
- **Thread safety**: concurrent `add_game()` + `sample_batch()` from multiple threads
- **Edge cases**: all-zero state, all-zero policy, single legal move, K=64 exactly filled
- **Import/export roundtrip**: export → import into fresh buffer → sample → verify

---

## Phase 2: Playout Cap Randomization

**Impact**: ~2.3x self-play throughput increase (670 → ~1540 games/hr)

KataGo-style technique: for each move, with probability 0.25 use full simulations (200), otherwise use reduced simulations (50). Average sims/move: `0.25*200 + 0.75*50 = 87.5` (vs 200 currently).

### 2.1 Config fields

Add to `SelfPlayGameConfig` in `src/selfplay/self_play_game.h`:
```cpp
bool enable_playout_cap = false;
std::size_t reduced_simulations = 50U;
float full_playout_probability = 0.25F;
```

### 2.2 Training weight

Add `float training_weight = 1.0F` to:
- `PendingSample` in `src/selfplay/self_play_game.h`
- `ReplayPosition` in `src/selfplay/replay_buffer.h`
- `CompactReplayPosition` (4 extra bytes, negligible)

Carry through to `SampledBatch` as a new `std::vector<float> weights` field.

### 2.3 Modify `SelfPlayGame::play()`

In `src/selfplay/self_play_game.cpp`, before each move's simulations:

```cpp
bool use_full_sims = true;
std::size_t sims_this_move = config_.simulations_per_move;
if (config_.enable_playout_cap) {
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    use_full_sims = dist(rng_) < config_.full_playout_probability;
    if (!use_full_sims) {
        sims_this_move = config_.reduced_simulations;
    }
}
// Pass sims_this_move to run_simulation_batch (needs parameterization)
sample.training_weight = use_full_sims ? 1.0F
    : float(config_.reduced_simulations) / float(config_.simulations_per_move);
```

The `run_simulation_batch()` method currently reads `config_.simulations_per_move` — change to accept a `std::size_t simulations` parameter.

### 2.4 Weighted loss in training

In `python/alphazero/training/trainer.py`, `train_one_step()`:
- Extract `weights` from `SampledBatch`
- Multiply per-sample loss by weight before averaging: `loss = (per_sample_loss * weights).mean()`
- If all weights are 1.0 (playout cap disabled), this is equivalent to current behavior

### 2.5 Bindings and config

- Expose new `SelfPlayGameConfig` fields in `src/bindings/python_bindings.cpp`
- Read them in `scripts/train.py` `_build_selfplay_manager_config()`
- Add to `configs/chess.yaml`:
  ```yaml
  mcts:
    enable_playout_cap: true
    reduced_simulations: 50
    full_playout_probability: 0.25
  ```

---

## Phase 3: Temperature and Noise Tuning

**Impact**: Wider opening variety, low effort

### 3.1 Increase temperature_moves

In `configs/chess.yaml`, change `temperature_moves: 30` to `temperature_moves: 40`. No code changes — the value is read from config and passed to MCTS. This makes moves 31-40 use proportional sampling instead of greedy, producing more variety in the early middlegame.

### 3.2 Per-game Dirichlet noise variation

Add to `SelfPlayGameConfig` (`src/selfplay/self_play_game.h`):
```cpp
bool randomize_dirichlet_epsilon = false;
float dirichlet_epsilon_min = 0.15F;
float dirichlet_epsilon_max = 0.35F;
```

In `SelfPlayManager` worker loop (`src/selfplay/self_play_manager.cpp`), when starting a new game: if `randomize_dirichlet_epsilon` is true, sample epsilon uniformly from `[min, max]` and set it on the game config. This creates a mix of quiet games (low noise) and exploratory games (high noise).

---

## Phase 4: Dynamic Simulation Schedule

**Impact**: 2x throughput during early training, low effort

### 4.1 Runtime sim count update

Add to `SelfPlayManager` (`src/selfplay/self_play_manager.h`):
```cpp
void update_simulations_per_move(std::size_t new_sims);
```

Implementation stores the value atomically. Worker threads read it at the start of each new game.

### 4.2 Schedule in training loop

In `scripts/train.py`, after each training step:
```python
if step < 10000:
    runtime.self_play_manager.update_simulations_per_move(100)
else:
    runtime.self_play_manager.update_simulations_per_move(200)
```

Expose `update_simulations_per_move` via pybind11 in `src/bindings/python_bindings.cpp`.

---

## Phase 5: Recency-Weighted Sampling

**Impact**: Better training signal from recent positions, lowest priority

Add optional sampling mode to `CompactReplayBuffer`:
```cpp
enum class SamplingStrategy { kUniform, kRecencyWeighted };
```

For recency weighting: `w(i) = exp(-lambda * (N - i) / N)` where `i` is logical index (0=oldest, N-1=newest). Use inverse CDF method for O(1) per-sample generation. Default `lambda = 1.0` gives oldest position 37% weight of newest.

Lower priority than phases 1-4 because with 30M positions and a large buffer, even uniform sampling provides good diversity.

---

## Implementation Order

| Phase | Feature | Impact | Effort | Dependencies |
|-------|---------|--------|--------|--------------|
| 1 | Compact Replay Buffer | 40x capacity | 3-4 days | None |
| 2 | Playout Cap Randomization | 2.3x throughput | 1-2 days | Benefits from Phase 1 |
| 3 | Temperature/Noise Tuning | ~20% diversity | 0.5 day | None |
| 4 | Dynamic Sim Schedule | 2x early throughput | 0.5-1 day | None |
| 5 | Recency-Weighted Sampling | Better signal | 0.5 day | Requires Phase 1 |

**Do Phase 1 first** — all other phases benefit from the increased capacity. Phase 2 next — generates more positions to fill the larger buffer. Phases 3-5 can follow in any order.

**Combined impact**: With Phase 1 + Phase 2, buffer capacity goes from 750K to ~5-30M positions while game throughput increases from ~670 to ~1,540 games/hr. At 80K training steps with batch 8192, each position is sampled ~2 times instead of ~870 times.

## Verification

1. **Build**: `cmake --build build --target alphazero_cpp -j$(nproc)`
2. **C++ unit tests**: `ctest --test-dir build` — new tests for compact buffer
3. **Python tests**: `pytest tests/python/` — checkpoint compat, training integration
4. **Short training run**: `python scripts/train.py --config configs/chess.yaml` for ~1000 steps:
   - Verify buffer reports correct capacity and position count in log output
   - Verify loss decreases normally (no regression from compression artifacts)
   - Verify memory usage via `htop` (~1.2KB × capacity, not 48KB × capacity)
5. **Checkpoint roundtrip**: save checkpoint, resume from it, verify buffer contents match
