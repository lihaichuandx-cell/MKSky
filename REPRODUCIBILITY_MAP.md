# Reproducibility map

This map links the manuscript items to the principal files in Online Resource 1. Raw CSV files preserve the recorded measurements; processed workbooks are the direct inputs to `scripts/generate_paper_figures.py`.

## Figures

| Manuscript item | Reproduction source |
|---|---|
| Figures 1–5 | Programmatically drawn by `scripts/generate_paper_figures.py`; no external data file is required. |
| Figure 6 (6D scalability) | `data/workbooks/jomyal整体时间统计_vs n or d/6维/` |
| Figure 7 (3D scalability) | `data/workbooks/3维测试（图10）/` |
| Figure 8 (dimension scalability) | The dimensional series in `data/workbooks/jomyal整体时间统计_vs n or d/6维/`. |
| Figure 9 (ablation) | `data/workbooks/jomyal基准对比+消融实验/消融实验_维度变化_500w反相关.xlsx` and `消融实验_规模变化_8d反相关.xlsx` |
| Figure 10 (filtering workload) | `data/workbooks/3维测试（图10）/` |
| Figure 11 (memory and time) | `data/workbooks/jomyal显存占用+假阳性率/myal运行情况统计_假阳性+显存占用.xlsx` |
| Figure 12 (CPU baseline) | `data/workbooks/jomyal基准对比+消融实验/myal运行情况统计_CPU基线测试.xlsx` |
| Figures 13–15 (stage timing) | `data/workbooks/jomyal局部时间统计/` |

## Tables and checks

| Manuscript item | Principal evidence |
|---|---|
| Tables 1–2 | Conceptual comparison and mathematical notation; no experimental data file is required. |
| Table 3 | Frozen defaults in `code/README.md`, `code/SUBMISSION_PROTOCOL.md`, and the supplied project sources. |
| Table 4 (public SkyCell 3D anchor) | `data/report_followup_20260723/skycell_anchor/` and `code/third_party/SkyCell/`. |
| Table 5 (uniform post-processing control) | `data/raw_csv/distribution_control/`. |
| Table 6 (five seeds) | `data/raw_csv/main/multi_seed/`. |
| Table 7 (input order) | `data/report_followup_20260723/order/`. |
| Table 8 (candidate/prefix ablation) | `data/report_followup_20260723/ablation/`. |
| Table 9 (sensitivity) | `data/report_followup_20260723/sensitivity/` and retained runs under `data/raw_csv/sensitivity/`. |
| Table 10 (three end-to-end configurations) | `data/report_followup_20260723/end_to_end/`. |
| 36 CPU-exact configurations | `data/raw_csv/correctness36/`. |
| Six deterministic boundary datasets | `data/edge_correctness_results.txt` and `code/edge_correctness_main.cpp`. |

## Measurement interpretation

`algorithm_avg_ms` is the data-resident GPU execution metric used for the main curves. `end_to_end_avg_ms` is the secondary adapter-boundary metric. `device_memory_mb` counts explicit simultaneously live device arrays and excludes CUDA context, driver cache, and unrelated process use. Baseline scope and generator details are frozen in `code/SUBMISSION_PROTOCOL.md`.
