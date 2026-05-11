#include "skyalign_algorithm.h"
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define SAFE_MAX_DIM 16

// 1. 使用 64 位 double 计算坐标和，提供绝对无损的排序依据
static __global__ void sa_sum_kernel(int n, int dim, const float* d_coords, double* d_sums, int* d_indices) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		double sum = 0.0;
#pragma unroll
		for (int d = 0; d < dim; d++) sum += (double)__ldg(&d_coords[d * n + i]);
		d_sums[i] = sum;
		d_indices[i] = i;
	}
}

// 👑 终极手术：无分支向量化 (Branchless Vectorization) + C++ 模板展开
// 彻底剥夺高维数据的 Early-Break 偷懒特权，强制维度时间单调递增！
template <int DIM>
static __global__ void sa_sfs_pure_kernel_templated(int n, const float* d_coords, const int* d_sorted_indices, int* d_is_alive) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	int my_orig_idx = d_sorted_indices[idx];
	float my_coords[DIM]; // 数组大小在编译期绝对固定，完美驻留高速寄存器！

#pragma unroll
	for (int d = 0; d < DIM; d++) {
		my_coords[d] = __ldg(&d_coords[d * n + my_orig_idx]);
	}

	bool alive = true;
	for (int j = 0; j < idx; j++) {
		int dom_orig_idx = d_sorted_indices[j];

		// 🚀 纯布尔逻辑替代 if-break 内循环，强制 GPU 算满每一个维度！
		bool dominated = true;
		bool strict_less = false;

#pragma unroll
		for (int d = 0; d < DIM; d++) {
			float val_c = __ldg(&d_coords[d * n + dom_orig_idx]);
			// 只有所有维度都 <= 才是 true
			dominated &= (val_c <= my_coords[d]);
			// 只要有一个维度 < 就是 true
			strict_less |= (val_c < my_coords[d]);
		}

		// 外层循环依然保留 break：一旦被确认支配直接死亡，不再跟剩下的点浪费时间
		if (dominated && strict_less) {
			alive = false;
			break;
		}
	}

	if (!alive) {
		d_is_alive[my_orig_idx] = 0;
	}
}

std::tuple<int, double> run_skyalign_v2(std::vector<MyDataPoint>& h_points, const AlgorithmConfig& config) {
	CHECK_CUDA_ERROR(cudaSetDevice(0));
	int n = static_cast<int>(h_points.size());
	int dim = config.dim;

	// 回归最原始、最干净的内存分配
	float* d_coords; CHECK_CUDA_ERROR(cudaMalloc(&d_coords, n * dim * sizeof(float)));
	std::vector<float> h_coords_flat(n * dim);
	for (int d = 0; d < dim; d++) {
		for (int i = 0; i < n; i++) h_coords_flat[d * n + i] = h_points[i].coords[d];
	}
	CHECK_CUDA_ERROR(cudaMemcpy(d_coords, h_coords_flat.data(), n * dim * sizeof(float), cudaMemcpyHostToDevice));

	double* d_sums; CHECK_CUDA_ERROR(cudaMalloc(&d_sums, n * sizeof(double)));
	int* d_indices; CHECK_CUDA_ERROR(cudaMalloc(&d_indices, n * sizeof(int)));
	int* d_is_alive; CHECK_CUDA_ERROR(cudaMalloc(&d_is_alive, n * sizeof(int)));

	std::vector<int> init_alive(n, 1);
	CHECK_CUDA_ERROR(cudaMemcpy(d_is_alive, init_alive.data(), n * sizeof(int), cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	auto beg = std::chrono::high_resolution_clock::now();

	dim3 block(256);
	dim3 grid((n + 255) / 256);

	sa_sum_kernel << <grid, block >> > (n, dim, d_coords, d_sums, d_indices);

	thrust::device_ptr<double> ptr_sums(d_sums);
	thrust::device_ptr<int> ptr_indices(d_indices);
	thrust::sort_by_key(ptr_sums, ptr_sums + n, ptr_indices);

	int* raw_ptr_indices = thrust::raw_pointer_cast(ptr_indices);

	// 👑 静态路由分发器：根据动态传入的 dim，路由到写死的静态模板 Kernel
	// 这样编译器就能把 #pragma unroll 发挥到极致！
	switch (dim) {
	case 3: sa_sfs_pure_kernel_templated<3> << <grid, block >> > (n, d_coords, raw_ptr_indices, d_is_alive); break;
	case 4: sa_sfs_pure_kernel_templated<4> << <grid, block >> > (n, d_coords, raw_ptr_indices, d_is_alive); break;
	case 5: sa_sfs_pure_kernel_templated<5> << <grid, block >> > (n, d_coords, raw_ptr_indices, d_is_alive); break;
	case 6: sa_sfs_pure_kernel_templated<6> << <grid, block >> > (n, d_coords, raw_ptr_indices, d_is_alive); break;
	case 7: sa_sfs_pure_kernel_templated<7> << <grid, block >> > (n, d_coords, raw_ptr_indices, d_is_alive); break;
	case 8: sa_sfs_pure_kernel_templated<8> << <grid, block >> > (n, d_coords, raw_ptr_indices, d_is_alive); break;
	case 9: sa_sfs_pure_kernel_templated<9> << <grid, block >> > (n, d_coords, raw_ptr_indices, d_is_alive); break;
	case 10: sa_sfs_pure_kernel_templated<10> << <grid, block >> > (n, d_coords, raw_ptr_indices, d_is_alive); break;
	default:
		printf("不支持的维度!\n");
		break;
	}

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	auto end = std::chrono::high_resolution_clock::now();
	double time_ms = std::chrono::duration<double, std::milli>(end - beg).count();

	std::vector<int> h_is_alive(n);
	CHECK_CUDA_ERROR(cudaMemcpy(h_is_alive.data(), d_is_alive, n * sizeof(int), cudaMemcpyDeviceToHost));

	int final_count = 0;
	for (int i = 0; i < n; i++) if (h_is_alive[i] == 1) final_count++;

	cudaFree(d_coords); cudaFree(d_sums); cudaFree(d_indices); cudaFree(d_is_alive);

	return std::make_tuple(final_count, time_ms);
}