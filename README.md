```markdown
# MKSky: A GPU Parallel Skyline Algorithm Based on Morton KD-Tree

This repository contains the official implementation for the paper: **"MKSky: A GPU Parallel Skyline Algorithm Based on Morton KD-Tree"**. 

## 📖 Introduction

Skyline queries play a pivotal role in multi-criteria decision-making. However, existing GPU-parallel algorithms often encounter performance bottlenecks when processing high-dimensional anti-correlated data, primarily due to grid explosion (e.g., in the SkyCell algorithm) or memory access divergence. 

To address these challenges, we propose **MKSky**, a novel parallel algorithm based on implicit spatial partitioning. 

Key components of this framework include:
* **Morton-order Implicit KD-partitioning (MKD):** Leverages Morton encoding to map high-dimensional space into a one-dimensional contiguous memory layout. It achieves equivalent spatial partitioning through a bitwise XOR mechanism with zero physical tree overhead.
* **Double-level Chunk-Skipping SFS Engine:** Utilizes local absolute lower bounds to implement region-level pruning, thereby substantially reducing redundant point-to-point comparisons.

Extensive experiments demonstrate that, under the same high-dimensional and large-scale benchmarks, MKSky achieves a speedup of up to approximately two orders of magnitude compared to baseline algorithms like SkyCell and SkyAlign.

## 📂 Project Structure

The project is structured as a Visual Studio solution. The core source code is located in the `jomyal` directory:

```text
.
└── jomyal/
    ├── common.h                  # Common definitions and data structures
    ├── data_generator.h          # Synthetic dataset generator for different distributions (Independent, Correlated, Anti-correlated)
    ├── gpu_brute_algorithm.h     # Baseline: GPU Brute-force scanning implementation
    ├── hybrid_algorithm.cu       # Baseline: Hybrid algorithm implementation (CUDA)
    ├── hybrid_algorithm.h        # Baseline: Hybrid algorithm header
    ├── main.cu                   # Main entry point and experimental pipeline controller
    ├── myal_algorithm.cu         # Proposed MKSky: Core CUDA implementation (Pre-pruning, Morton MKD, Chunk-Skipping)
    ├── myal_algorithm.h          # Proposed MKSky: Header file
    ├── skyalign_algorithm.cu     # Baseline: SkyAlign implementation (CUDA)
    ├── skyalign_algorithm.h      # Baseline: SkyAlign header
    ├── skycell_algorithm.cu      # Baseline: SkyCell implementation (CUDA)
    ├── skycell_algorithm.h       # Baseline: SkyCell header
    ├── jomyal.sln                # Visual Studio Solution file
    ├── jomyal.vcxproj            # Visual Studio Project file
    ├── jomyal.vcxproj.user       # Visual Studio User specific project settings
    └── vc140.pdb                 # Program database for debugging

```

*(Note: In the source code, the proposed MKSky algorithm is implemented under the `myal_algorithm` files).*

## 🛠️ Requirements

The project is developed using **C++** and **CUDA**.

* **OS:** Windows 11 (64-bit)
* **IDE/Compiler:** Microsoft Visual Studio (compatible with `.sln` and `.vcxproj` files)
* **CUDA Toolkit:** Version 12.x
* **Hardware:** NVIDIA GPU. The algorithm is heavily optimized for modern GPU architectures (e.g., NVIDIA GeForce RTX 5070 laptop GPU).

## 🚀 Quick Start

1. **Open the Project:**
Double-click the `jomyal.sln` file to open the project in Microsoft Visual Studio.
2. **Configure the Environment:**
Ensure your Visual Studio is properly configured with the CUDA Toolkit (v12.x) build customizations.
3. **Build and Run:**
* Set `main.cu` as the entry point if necessary.
* Build the solution in `Release` mode for optimal performance.
* Run the executable.



The `main.cu` file serves as the pipeline controller. By modifying the parameters within, you can evaluate the algorithms across different dimensionalities (e.g., 3D to 8D) and data scales using the built-in `data_generator.h`.

## 📧 Contact

If you have any questions, please feel free to contact the authors or open an issue in this repository.

```

```
