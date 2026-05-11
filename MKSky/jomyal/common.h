#ifndef COMMON_H
#define COMMON_H

#define GLOBAL_DEBUG 0

#ifdef __INTELLISENSE__
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __shared__ static
#define __constant__ static
#define __syncthreads() __noop
#define KERNEL_LAUNCH(grid, block)
#else
#define KERNEL_LAUNCH(grid, block) <<<grid, block>>>
#endif

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <chrono>
#include <cmath>
#include <climits>
#include <cfloat>
#include <cstring>
#include <set>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

// ==============================================
// 🔬 算法核心参数与数据生成配置 
// ==============================================

// [1. 裁判系统开关]
// 0: 关闭，1: 开启。
//  GPU 暴力基准测试！
#define ENABLE_GPU_BRUTE_FORCE 0  

// [2. 维度控制]
// 预留最大维度上限，保持不动。
#define MAX_DIM 4
// 精度容差，保持不动。
#define EPS 1e-6f                 



// 【核心】实际运行维度。
#define TEST_DIM 3




// [3. 底层硬件与算法配置 (默认保持不动)]
// CUDA 线程块大小。
#define BLOCK_SIZE 256
// MYAL 任务队列大小。
#define MAX_TASK_QUEUE_SIZE 8192

// MYAL Z-Curve 参数。
#define MAX_RHO 6
#define kDensityThreshold 500
#define kBitsPerDim 12
// SkyCell 网格切分粒度 (自动适配高低维)。
#define kFixedGridPerDim (TEST_DIM >= 6 ? 2 : 16)

// [4. 数据集生成器]
// 【核心】测试数据总量。
// 操作提示：小规模调测填 50000；极限性能测试填 5000000。
#define CONFIG_DATA_NUM 50000

// 随机种子控制。
// 操作提示：1=固定每次生成相同数据(方便复现)；0=每次随机。
#define CONFIG_FIX_SEED 1

// 【核心】数据分布形态。
// 操作提示：0=独立均匀分布(常规基准)  1=正相关(简单)；2=反相关(极难，测极限)；。
#define DATA_DISTRIBUTION 1        

// 生成器微调参数，保持不动。
#define DATA_RANGE_MIN 0.0f
#define DATA_RANGE_MAX 1.0f
#define CORRELATION_STRENGTH 0.1f

#define CHECK_CUDA_ERROR(err) \
	do { \
		cudaError_t _err = (err); \
		if (_err != cudaSuccess) { \
			fprintf(stderr, "[CUDA FATAL] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
			exit(EXIT_FAILURE); \
		} \
	} while(0)

struct __align__(16) AlgorithmConfig {
	int dim;
	int data_num;
	int fixed_grid_per_dim;
	int skycell_rho;
};

const AlgorithmConfig DEFAULT_CONFIG = {
	TEST_DIM,
	CONFIG_DATA_NUM,
	kFixedGridPerDim,
	4
};

struct __align__(8) MyCell {
	int idx[MAX_DIM];
	int is_non_empty;
	int is_key_cell;
	int is_candidate;
};

struct __align__(8) GridLayer {
	int layer_idx;
	int cell_num;
	MyCell* cells;
};

struct __align__(16) MyDataPoint {
	float coords[MAX_DIM];
	int original_idx;

	__host__ __device__ bool isDominatedBy(const MyDataPoint& other, int dim) const {
		bool all_le = true;
		bool any_lt = false;
		for (int d = 0; d < dim; d++) {
			if (other.coords[d] > coords[d]) { all_le = false; break; }
			if (other.coords[d] < coords[d]) { any_lt = true; }
		}
		return all_le && any_lt;
	}
};

struct __align__(16) DataSOA {
	int count;
	int dim;
	float* coords[MAX_DIM];
	int* original_idx;
	int* layer_cell_start[MAX_RHO];
	int* layer_cell_end[MAX_RHO];
};

struct __align__(16) DeviceParams {
	int dim;
	int grid_per_dim;
	int total_grid_num;
	float coord_min[MAX_DIM];
	float coord_max[MAX_DIM];
	float grid_step[MAX_DIM];
	float* grid_min[MAX_DIM];
	float* grid_max[MAX_DIM];
	int* grid_is_non_empty;
	int* grid_is_dead;
};

struct __align__(16) GridInfoSOA {
	int count;
	int dim;
	float* min[MAX_DIM];
	float* max[MAX_DIM];
	int* is_non_empty;
	int* is_dead;
};

static inline float host_min(float a, float b) { return (a < b) ? a : b; }
static inline float host_max(float a, float b) { return (a > b) ? a : b; }

static inline void normalize_dataset(std::vector<MyDataPoint>& h_points, float h_min[MAX_DIM], float h_max[MAX_DIM], int dim) {
	for (int d = 0; d < dim; d++) {
		h_min[d] = h_points[0].coords[d];
		h_max[d] = h_points[0].coords[d];
	}
	for (size_t i = 0; i < h_points.size(); i++) {
		for (int d = 0; d < dim; d++) {
			h_min[d] = host_min(h_min[d], h_points[i].coords[d]);
			h_max[d] = host_max(h_max[d], h_points[i].coords[d]);
		}
	}
	for (size_t i = 0; i < h_points.size(); i++) {
		for (int d = 0; d < dim; d++) {
			float range = h_max[d] - h_min[d];
			h_points[i].coords[d] = range < 1e-7f ? 0.0f : (h_points[i].coords[d] - h_min[d]) / range;
			h_points[i].coords[d] = host_max(0.0f, host_min(1.0f, h_points[i].coords[d]));
		}
	}
}

static inline void compare_skyline_results(
	const std::vector<MyDataPoint>& base_skyline,
	const std::vector<MyDataPoint>& test_skyline,
	const char* algorithm_name,
	int dim
) {
	printf("\n=== %s 准确性校验详情 ===", algorithm_name);
	if (base_skyline.size() != test_skyline.size()) {
		printf("\n[警告] 点数不一致！基准点数=%zu，测试点数=%zu", base_skyline.size(), test_skyline.size());
	}

	std::vector<int> base_indices, test_indices;
	for (const auto& p : base_skyline) base_indices.push_back(p.original_idx);
	for (const auto& p : test_skyline) test_indices.push_back(p.original_idx);
	std::sort(base_indices.begin(), base_indices.end());
	std::sort(test_indices.begin(), test_indices.end());

	std::vector<int> missing_points, wrong_points;
	std::set_difference(base_indices.begin(), base_indices.end(), test_indices.begin(), test_indices.end(), std::back_inserter(missing_points));
	std::set_difference(test_indices.begin(), test_indices.end(), base_indices.begin(), base_indices.end(), std::back_inserter(wrong_points));

	if (!missing_points.empty()) printf("\n[错误] 丢失天际线点");
	if (!wrong_points.empty()) printf("\n[错误] 多输出非天际线点");
	if (missing_points.empty() && wrong_points.empty()) printf("\n[通过] 结果完全一致，%s 的剪枝逻辑 100%% 正确！", algorithm_name);
	printf("\n===========================================\n");
}

#ifdef __CUDACC__
static __device__ __forceinline__ float atomicMin_float(float* address, float val) {
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(__int_as_float(assumed), val)));
	} while (assumed != old);
	return __int_as_float(old);
}

static __device__ __forceinline__ float atomicMax_float(float* address, float val) {
	int* address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(__int_as_float(assumed), val)));
	} while (assumed != old);
	return __int_as_float(old);
}

static __device__ __forceinline__ int atomicAdd_int(int* addr, int val) { return atomicAdd(addr, val); }
static __device__ __forceinline__ float dev_min(float a, float b) { return (a < b) ? a : b; }
static __device__ __forceinline__ float dev_max(float a, float b) { return (a > b) ? a : b; }
static __device__ __forceinline__ int dev_min_int(int a, int b) { return (a < b) ? a : b; }
static __device__ __forceinline__ int dev_max_int(int a, int b) { return (a > b) ? a : b; }
#endif 

#endif