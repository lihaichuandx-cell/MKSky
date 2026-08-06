#pragma once

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#define MAX_DIM 16
#define EPS 1e-6f
#define BLOCK_SIZE 256

#define CHECK_CUDA_ERROR(expr)                                                    \
    do {                                                                          \
        const cudaError_t cuda_status_ = (expr);                                  \
        if (cuda_status_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(cuda_status_));                       \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

struct AlgorithmConfig {
    int dim = 3;
    int data_num = 100000;
    int fixed_grid_per_dim = 0;
    int skycell_rho = 0;
    bool collect_telemetry = false;
    bool input_normalized = false;
    // Submission experiments use these switches to isolate one component at a
    // time while keeping the remaining execution path unchanged.
    bool enable_grid_prefilter = true;
    bool enable_small_candidate_direct = true;
    bool enable_candidate_reduction = true;
    bool enable_local_prefix_filtering = true;
    bool enable_mkd_partition = true;
    bool enable_block_skipping = true;
    bool enable_aux_summaries = true;
    // Performance-only controls exposed for sensitivity experiments. None of
    // these values changes the exact dominance relation or the output set.
    int mksky_sample_limit = 4096;
    int mksky_survivor_ratio_divisor = 8;
    int mksky_direct_limit = 5000;
    int mksky_chunk_size_override = 0;
    int mksky_aux_threshold = 65536;
    int mksky_compatibility_min_chunks = 2048;
    unsigned long long mksky_compatibility_max_bytes = 128ULL * 1024ULL * 1024ULL;
    int mksky_signature_min_dim = 14;
    // The fixed-count witness shortcut is diagnostic only. It is excluded from
    // the submission algorithm until a parameter-independent proof is provided.
    bool enable_topk_witness = false;
};

struct __align__(16) MyDataPoint {
    float coords[MAX_DIM];
    int original_idx;

    __host__ __device__ bool isDominatedBy(const MyDataPoint& other, int dim) const {
        bool all_le = true;
        bool any_lt = false;
        for (int d = 0; d < dim; ++d) {
            if (other.coords[d] > coords[d]) {
                all_le = false;
                break;
            }
            any_lt = any_lt || other.coords[d] < coords[d];
        }
        return all_le && any_lt;
    }
};

#ifdef __CUDACC__
static __device__ __forceinline__ float dev_min(float a, float b) { return a < b ? a : b; }
static __device__ __forceinline__ float dev_max(float a, float b) { return a > b ? a : b; }
#endif
