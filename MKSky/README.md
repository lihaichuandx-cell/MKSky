# MKSky

Source code for the manuscript:

> **MKSky: An Exact GPU-Parallel Skyline Algorithm with Morton-Ordered Implicit Partitioning**

The repository contains the proposed MKSky implementation, the comparison
algorithms used by the benchmark, correctness utilities, and the public 3D
SkyCell source retained as a separate reference. Generated binaries, local IDE
state, `.codex`, datasets, and processed results are not included.

## Repository layout

```text
MKSky/
├── MKSky.sln
├── MKSkyBenchmark.vcxproj
├── README.md
├── .gitignore
├── src/
│   ├── app/
│   │   └── main.cpp
│   ├── common/
│   │   ├── algorithm.h
│   │   ├── common.h
│   │   ├── dataset.h
│   │   ├── dataset.cpp
│   │   ├── reference.h
│   │   └── reference.cpp
│   ├── mksky/
│   │   ├── mksky_paper.cu
│   │   ├── mksky_prefix_backend.cu
│   │   ├── mksky_prefix_backend_impl.cuh
│   │   └── mksky_adaptive.cu
│   └── baselines/
│       ├── skycell.cu
│       └── skyalign.cu
├── tests/
│   └── edge_correctness_main.cpp
├── scripts/
│   └── run_dimension_compare.ps1
├── docs/
│   └── SUBMISSION_PROTOCOL.md
└── third_party/
    └── SkyCell/
```

- `src/mksky/` contains the proposed algorithm and its CUDA backends.
- `src/baselines/` contains the generalized SkyCell and SkyAlign comparison implementations.
- `src/common/` contains shared types, dataset generation/loading, and exact CPU verification.
- `third_party/SkyCell/` is the public authors' 3D implementation and is not linked into the default benchmark.
- `tests/`, `scripts/`, and `docs/` contain validation and experiment support files; they do not affect the default build.

The obsolete `current` implementation has been removed.

## Requirements

- Windows 11 x64
- Visual Studio 2017 with the Visual C++ v141 toolset
- CUDA Toolkit 10.0
- NVIDIA CUDA-capable GPU

## Build

Open `MKSky.sln`, select `Release | x64`, and build the `MKSkyBenchmark`
project. From a Visual Studio 2017 x64 Native Tools Command Prompt, the same
build can be run with:

```powershell
MSBuild.exe .\MKSky.sln /m /t:Build /p:Configuration=Release /p:Platform=x64
```

The executable is generated at:

```text
bin\Release\MKSkyBenchmark.exe
```

## Run

```powershell
.\bin\Release\MKSkyBenchmark.exe --n 1000000 --dim 6 `
  --distribution anti --seed 12345 --warmup 1 --repeat 3 `
  --algorithms mksky,skycell,skyalign --csv results\anti_d6.csv
```

Supported synthetic distributions are `ind`, `corr`, and `anti`. An external
coordinate matrix can be supplied with `--input-csv <path>`. Use `--help` for
the complete command-line interface.

## Included algorithms

- `mksky`: submission MKSky.
- `skycell`: generalized CUDA grid baseline (`SkyCell-G`).
- `skyalign`: CUDA reimplementation of SkyAlign Algorithm 1 (`SkyAlign-R`).
- `cpu-sfs`: independent serial exact reference for manageable inputs.
- The remaining MKSky names exposed by `--help` are controlled variants used for ablation and sensitivity experiments.

When the input size is no larger than `--verify-limit`, every selected GPU
algorithm is checked against the exact CPU skyline index set. See
`docs/SUBMISSION_PROTOCOL.md` for the frozen experiment and reporting rules.

## Optional experiment script

After building `Release | x64`, run the dimensional comparison from the
repository root with:

```powershell
.\scripts\run_dimension_compare.ps1
```
