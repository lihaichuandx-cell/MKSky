# MKSky

Source code and reproducibility materials for the manuscript:

> **MKSky: An Exact GPU-Parallel Skyline Algorithm with Morton-Ordered Implicit Partitioning**  
> Haichuan Li and Zhongbo Wu  
> Submitted to *The Journal of Supercomputing*

MKSky is an exact GPU-parallel skyline algorithm for large multidimensional datasets. Its objective is not merely to parallelize point-pair dominance tests, but to reduce the amount of exact verification that must be performed. Quantized Morton codes and block summaries are used only for data organization and safe exclusion; final skyline membership is always determined by strict dominance tests on the original coordinates.

## Algorithm overview

The submission implementation combines four stages:

1. **Safe candidate reduction** using either a low-dimensional occupied-grid route or a real-pivot route.
2. **Morton-ordered GPU layout** with quantization, 64-bit Morton encoding, radix sorting, and contiguous coordinate reordering.
3. **Implicit interval partitioning** using endpoint-XOR boundaries, followed by exact intra-interval prefix filtering.
4. **Hierarchical exact verification** using safe block lower bounds and conditional summaries before falling back to original-coordinate comparisons, including equal-code completion checks.

The repository also contains controlled ablations, exact CPU references, CUDA comparison implementations, processed workbooks, raw measurements, and the script used to regenerate the manuscript figures.

## Repository contents

```text
code/                         Visual Studio/CUDA benchmark project
  src/                        MKSky, baselines, dataset generators, references
  third_party/SkyCell/        Public 3D SkyCell source retained as an anchor
  README.md                   Detailed build and command-line instructions
  SUBMISSION_PROTOCOL.md      Frozen algorithms, generators, defaults, and reporting rules
data/
  raw_csv/                    Raw scalability, ablation, correctness, and sensitivity results
  report_followup_20260723/   Follow-up validation and end-to-end measurements
  workbooks/                  Processed workbooks used by the plotting script
  edge_correctness_results.txt
scripts/generate_paper_figures.py
REPRODUCIBILITY_MAP.md        Manuscript figure/table to file mapping
FILE_MANIFEST.md              Included formats and excluded build artifacts
```

The downloadable `MKSky_Online_Resource_1.zip` archive is a frozen copy of the complete source-and-data package.

## Experimental scope

The paper evaluates three synthetic distributions (independent, correlated, and anticorrelated), dimensions from 3 to 16, and input sizes up to 30 million records. The package includes:

- raw CSV measurements for the main scalability matrices;
- candidate-reduction, local-prefix, MKD-boundary, and block-summary ablations;
- parameter-sensitivity and input-order checks;
- 36 CPU-exact correctness configurations and six deterministic boundary datasets;
- a single-threaded CPU-SFS reference;
- device-resident and adapter-boundary timing measurements;
- the processed workbooks used to generate all 15 manuscript figures.

Performance gains are input dependent. MKSky is most effective on large, candidate-rich inputs, particularly anticorrelated data. When few candidates remain, preprocessing and data-organization costs can reduce or eliminate the advantage. At high dimensionality, conditional summaries introduce an explicit memory-time trade-off.

## Build environment used for the paper

- Windows 11 x64
- Visual Studio 2017, Visual C++ v141, `Release | x64`, `/O2`
- CUDA Toolkit 10.0.130
- NVIDIA driver 582.05
- NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MB

Open `code/MKSky_new.sln` in Visual Studio and build `Release | x64`, or run the following command from a Visual Studio 2017 x64 Native Tools Command Prompt:

```powershell
MSBuild.exe .\code\MKSky_new.sln /m /t:Build /p:Configuration=Release /p:Platform=x64
```

The executable is written to `code\bin\Release\MKSkyBenchmark.exe`. Build outputs are intentionally excluded from the repository package to avoid machine-specific paths and binary artifacts.

## Representative run

```powershell
.\code\bin\Release\MKSkyBenchmark.exe `
  --n 1000000 --dim 6 --distribution anti --seed 12345 `
  --warmup 1 --repeat 3 `
  --algorithms mksky,skycell,skyalign `
  --csv results\anti_d6.csv
```

Use `--dataset-policy legacy` for the main-paper generators. See `code/README.md` for all exposed sensitivity controls and verification options.

## Timing definitions

- `algorithm_avg_ms`: data-resident GPU algorithm time measured with CUDA events. It excludes data generation, file I/O, and process startup.
- `end_to_end_avg_ms`: adapter-boundary time including host packing, device allocation, transfers, execution, result collection, and cleanup.
- `device_memory_mb`: explicitly allocated, simultaneously live device workspace. CUDA context state, driver caches, and internal Thrust temporary storage are excluded.

These metrics must not be mixed. The manuscript uses data-resident timing for its main GPU curves and reports adapter-boundary timing separately.

## Recreate the manuscript figures

Install Python with `matplotlib` and `openpyxl`, then run from the repository root:

```powershell
python .\scripts\generate_paper_figures.py figures data\workbooks panels
```

Figures 1-5 are explanatory diagrams drawn directly by the script. Figures 6-15 are generated from the included workbooks. See `REPRODUCIBILITY_MAP.md` for the exact source of every figure and experimental table.

## Baseline disclosure

- `SkyCell-G` is the generalized CUDA grid baseline supplied in this package. It is not presented as the original authors' general-dimensional executable.
- `SkyAlign-R` is a CUDA reimplementation following SkyAlign Algorithm 1; author source was not publicly located.
- The original public 3D SkyCell source is retained under `code/third_party/SkyCell/` as a separately reported correctness and timing anchor.

The paper does not describe speedups against `SkyCell-G` as speedups against the original SkyCell authors' program. Full boundary and generator definitions are frozen in `code/SUBMISSION_PROTOCOL.md`.

## Correctness and reproducibility

For manageable inputs, every GPU result is compared with an independent exact CPU skyline index set. For large inputs, cross-algorithm agreement is retained as an additional check but is not described as independent ground truth. Raw measurements are preserved so that reported values, exclusions, and timing boundaries remain auditable.

## Authors

- **Haichuan Li** — algorithm design, implementation, experiments, analysis, and manuscript preparation
- **Zhongbo Wu** — supervision, method revision, analysis, manuscript review, and project administration

Corresponding author: Zhongbo Wu, `wuzhongbo@hbuas.edu.cn`  
Department of Computer Engineering, Hubei University of Arts and Science, Xiangyang, Hubei 441053, China

## Citation

The manuscript is under review. Please cite the paper title and authors above when using this repository. A formal bibliographic entry will be added after publication.
