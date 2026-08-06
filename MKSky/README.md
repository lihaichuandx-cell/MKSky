# MKSky submission benchmark

This Visual Studio 2017 project contains the implementation used for the MKSky submission to *The Journal of Supercomputing*. The benchmark runs one GPU algorithm at a time and writes timing, workspace, work counters, provenance, and verification mode to CSV.

## Algorithms

- `mksky`: submission MKSky; the fixed-count top-k witness shortcut is disabled.
- `ablation-a`: MKSky without auxiliary summaries.
- `ablation-b`: MKSky without block-level skipping.
- `ablation-c`: MKSky with fixed chunks replacing endpoint-XOR MKD.
- `ablation-no-prune`: MKSky without candidate reduction.
- `ablation-no-prefix`: MKSky without intra-chunk prefix filtering.
- `skycell`: `SkyCell-G`, a generalized CUDA grid baseline preserving candidate-cell semantics; it is not the authors' progressive `ShrinkKeyCells` implementation.
- `skyalign`: `SkyAlign-R`, a CUDA reimplementation of Algorithm 1 in the SkyAlign paper; author source was not publicly located.
- `cpu-sfs`: independent serial exact reference for manageable inputs.

The public 3D SkyCell source is retained under `third_party/SkyCell`. `SkyCellAnchor` adds CSV input/output, timing, and exact CPU cross-checks around the author algorithm. It is not used silently as a general-dimensional baseline.

## Build

Open a Visual Studio 2017 x64 Native Tools Command Prompt, change to this directory, and run:

```powershell
MSBuild.exe .\MKSky_new.sln /m /t:Build /p:Configuration=Release /p:Platform=x64
```

The executable is written to `bin\Release\MKSkyBenchmark.exe`. Build outputs are intentionally excluded from Online Resource 1; rebuilding them avoids distributing machine-specific paths and binary artifacts.

## Run

```powershell
.\bin\Release\MKSkyBenchmark.exe --n 5000000 --dim 6 `
  --distribution anti --seed 12345 --warmup 1 --repeat 3 `
  --algorithms mksky,skycell,skyalign --csv results\anti_d6.csv
```

Supported distributions are `ind`, `corr`, and `anti`. `--input-order generated|shuffled|sorted-d0|sorted-sum` controls input order; the default is `generated`. `--input-csv <path>` loads a coordinate matrix.

Use `--dataset-policy legacy` for the main-paper generators. Use `--dataset-policy uniform4` for the four-decimal quantized, deduplicated, resampled control.

Exposed MKSky sensitivity controls are:

- `--mksky-sample-limit` (default `4096`)
- `--mksky-survivor-divisor` (default `8`)
- `--mksky-direct-limit` (default `5000`)
- `--mksky-chunk-size` (`0` selects the dimension/scale rule)
- `--mksky-aux-threshold` (default `65536`)
- `--mksky-compat-min-chunks` (default `2048`)
- `--mksky-compat-max-mb` (default `128`)
- `--mksky-signature-min-dim` (default `14`)

The path sample uses a deterministic coprime-stride permutation. Its initial stride is `2654435761 mod n`, incremented until coprime with `n`; the offset is `2246822519 mod n`. The actual local chunk size is recorded in `adaptive_local_chunk_size`.

## Verification

- When `n <= --verify-limit`, every algorithm is compared with the independent exact CPU index set.
- For larger inputs, the first GPU result is retained only as an agreement reference; the CSV identifies that it is not independent ground truth.
- Submission validation covers several dimensions, distributions, and seeds at exact-reference sizes.

## Timing and memory

- `algorithm_avg_ms`: mean CUDA-event time for the algorithm section.
- `algorithm_std_ms`: population standard deviation across measured runs.
- `end_to_end_avg_ms`: adapter-boundary time including allocation, transfers, execution, result collection, and cleanup.
- `end_to_end_std_ms`: corresponding population standard deviation.
- `device_memory_mb`: explicit device workspace allocated by the adapter; CUDA runtime state and internal Thrust temporary storage are excluded.

See `SUBMISSION_PROTOCOL.md` for the frozen algorithm, dataset generators, ablation controls, and reporting rules.
