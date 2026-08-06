# Submission protocol

## Frozen MKSky path

The submission algorithm contains four stages:

1. Candidate pre-pruning using either the low-dimensional occupied-grid route
   or the global pivot route selected by the existing workload gate.
2. Quantization, 64-bit Morton encoding, radix sorting, and continuous
   coordinate reordering.
3. Endpoint-XOR MKD partitioning followed by dominance-monotone local prefix
   filtering. This stage is not described as a complete local skyline.
4. Safe block filtering followed by exact coordinate verification, including
   the equal-Morton-code tail required by quantization collisions.

The fixed-count top-k witness shortcut is disabled in the submission path. It
must not contribute to reported results unless a parameter-independent safety
proof and a separate ablation are added.

For `d <= 4`, route estimation samples at most 4096 positions using a
deterministic coprime-stride permutation. Sampling selects the grid or pivot
route only; every deletion condition still uses an occupied cell containing a
real point or a real pivot point. The low-dimensional grid target is `2n`
cells clamped to `[256, 2^22]`; the level is limited by both 7 and
`floor(22/d)`, and is increased while the next level remains within the target.

## Controlled ablations

All ablations call the same `run_projection_bound_mksky` entry point.

- `MKSky-w/o-aux-summary`: only auxiliary summaries are disabled.
- `MKSky-w/o-block-skip`: only block-level skipping is disabled.
- `MKSky-w/o-MKD`: only endpoint-XOR MKD is replaced by fixed chunks.
- `MKSky-w/o-candidate-reduction`: all points are retained before Morton
  ordering.
- `MKSky-w/o-local-prefix`: MKD intervals are retained but intra-interval
  prefix deletion is bypassed; exact verification is widened so correctness
  does not depend on the removed stage.

An ablation result is reported only when the removed component is active in the
full configuration. In particular, small 3D/4D cases that take direct exact
verification are not used to claim an MKD or block-summary benefit.

## Baseline identities

- `SkyCell-G` is a generalized CUDA grid baseline derived from SkyCell's
  candidate-cell semantics. It uses a dense occupancy grid and multidimensional
  prefix scan, not the original progressive `ShrinkKeyCells` execution.
- The original public SkyCell source is retained under `third_party/SkyCell` at
  commit `63ed9d0ce094d3d1cb5dd8204eaa1e932b19529a`. That source is specialized to
  3D and is not presented as the implementation used for higher dimensions.
- The public-source anchor sweeps layers 2 through 7 on a held-out 10003-point
  author-format input, keeps only configurations whose original index set
  matches an exact CPU skyline, selects the fastest exact layer, and freezes
  that layer for 5003 and 50003 points. The author timer and the local
  CUDA-event timer have different boundaries, so no cross-implementation
  speedup is calculated.
- `SkyAlign-R` follows the prefilter, per-dimension quartiles, bit masks, mask
  ordering, dimension-level iterations, and exact dominance verification in
  SkyAlign Algorithm 1.

The manuscript must use these exact labels and must not state that a speedup
against `SkyCell-G` is a measured speedup against the authors' original code.

## Dataset generators

All coordinates are minimized and generated with `std::mt19937(seed)`.

- Independent: each coordinate is sampled independently from `U(0,1)`, clipped
  to `[0.0001,0.9999]`, quantized to four decimal places, deduplicated, and the
  final points are shuffled.
- Correlated: draw `b ~ U(0,1)` and set each coordinate to
  `0.7 b + 0.3 U(0,1)`, followed by the same clipping, quantization,
  deduplication, and shuffle.
- Anticorrelated: draw positive random components `u_j`, normalize them by
  their sum, and set `x_j = 0.5 u_j/sum(u) + 0.5 U(0,1)`, clipped to
  `[0.00001,0.99999]`. This branch is not quantized or deduplicated.

These formulas and the asymmetric precision treatment must be disclosed in the
experimental setup because they affect skyline cardinality.

The benchmark exposes `--dataset-policy uniform4` for the post-processing
control. Under this policy, the anti-correlated branch also uses four-decimal
quantization and deduplication, and generation continues until the requested
number of distinct points has been produced.

## Frozen performance defaults

- Path sample limit: 4096 points.
- Sample survivor gate: 1/8.
- Direct exact-verification gate: 5000 candidates when `d <= 4`.
- Target chunk size: 1024 for `d < 10`, 512 for `d = 10` or `11`, and
  128/256/512 for `d >= 12` according to input scale.
- Auxiliary-summary gate: 65536 candidates.
- Block-group compatibility gate: 2048 chunks and 128 MB maximum storage.
- Quantized-field necessary condition: enabled from `d = 14`.

These values were frozen for the main matrices. They are engineering defaults,
not claimed device-independent optima. The CLI exposes each control so that a
single parameter can be changed without altering the correctness path.

The parameter provenance is reported in three groups: representation or device
limits (64-bit code, `2^22` grid cells, 128 MB compatibility mask, 256
threads), dimension/scale rules (grid side and chunk capacity), and empirical
defaults (sampling and activation thresholds). No independent calibration set
was used, so the paper does not call the empirical defaults globally optimal.

## Correctness and statistics

1. Exact-reference tests cover low, medium, and high dimensions, all three
   distributions, and multiple seeds.
2. Large runs use cross-algorithm agreement only as an additional check.
3. One warmup precedes three measured runs for the main scalability matrices.
4. Representative close-result configurations use multiple independent seeds
   and report variability across datasets as well as timing variability.
5. Input-order tests use generated, shuffled, first-dimension-sorted, and
   coordinate-sum-sorted arrays and report route, candidate counts, GPU time,
   and adapter time.
6. Targeted sensitivity points use ten measured runs and report population
   standard deviations. Chunk-size overrides are accepted only when the CSV
   confirms the requested `adaptive_local_chunk_size`.
7. Raw CSV files are retained and figures are generated from those files.

## Reporting boundary

`algorithm_avg_ms` is device-resident GPU algorithm execution time. It excludes
data generation, file I/O, and process startup. `end_to_end_avg_ms` is reported
separately and must not be mixed with the primary device-time comparison.
