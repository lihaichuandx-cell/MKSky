#ifndef HYBRID_ALGORITHM_CU
#define HYBRID_ALGORITHM_CU

#include "hybrid_algorithm.h"
#include <iostream>
#include <chrono>

__global__ void hybrid_local_skyline_kernel(int n, int dim, const float* d_coords, int* d_local_survivors) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		int block_start = blockIdx.x * blockDim.x;
		int block_end = block_start + blockDim.x;
		if (block_end > n) block_end = n;

		bool is_local_skyline = true;
		for (int j = block_start; j < block_end; j++) {
			if (i == j) continue;
			bool dominated = true;
			bool strict_less = false;
			for (int d = 0; d < dim; d++) {
				float other_val = d_coords[d * n + j];
				float curr_val = d_coords[d * n + i];
				if (other_val > curr_val) { dominated = false; break; }
				if (other_val < curr_val) strict_less = true;
			}
			if (dominated && strict_less) { is_local_skyline = false; break; }
		}
		if (is_local_skyline) d_local_survivors[i] = 1;
	}
}

__global__ void hybrid_extract_candidates_kernel(int n, const int* d_local_survivors, int* d_candidate_indices, int* d_candidate_count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && d_local_survivors[i] == 1) {
		int pos = atomicAdd(d_candidate_count, 1);
		d_candidate_indices[pos] = i;
	}
}

// 防 TDR 改进版：时间切片式的全局对决
__global__ void hybrid_global_chunk_kernel(int dim, int chunk_start, int chunk_size, int num_candidates, const float* d_coords, const int* d_candidates, int* d_final_skyline, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < chunk_size) {
		int curr_idx = d_candidates[chunk_start + idx];
		bool skyline = true;
		for (int j = 0; j < num_candidates; j++) {
			if (chunk_start + idx == j) continue;
			int other_idx = d_candidates[j];
			bool dominated = true;
			bool strict_less = false;
			for (int d = 0; d < dim; d++) {
				float other_val = d_coords[d * n + other_idx];
				float curr_val = d_coords[d * n + curr_idx];
				if (other_val > curr_val) { dominated = false; break; }
				if (other_val < curr_val) strict_less = true;
			}
			if (dominated && strict_less) { skyline = false; break; }
		}
		if (skyline) d_final_skyline[curr_idx] = 1;
	}
}

std::tuple<int, double> run_hybrid_algorithm(std::vector<MyDataPoint>& h_points, const AlgorithmConfig& config) {
	CHECK_CUDA_ERROR(cudaSetDevice(0));
	int n = static_cast<int>(h_points.size());
	int dim = config.dim;

	float* d_coords; CHECK_CUDA_ERROR(cudaMalloc(&d_coords, n * dim * sizeof(float)));
	std::vector<float> h_coords_flat(n * dim);
	for (int d = 0; d < dim; d++) {
		for (int i = 0; i < n; i++) h_coords_flat[d * n + i] = h_points[i].coords[d];
	}
	CHECK_CUDA_ERROR(cudaMemcpy(d_coords, h_coords_flat.data(), n * dim * sizeof(float), cudaMemcpyHostToDevice));

	int *d_local_survivors, *d_candidate_indices, *d_candidate_count, *d_final_skyline;
	CHECK_CUDA_ERROR(cudaMalloc(&d_local_survivors, n * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_candidate_indices, n * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_candidate_count, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_final_skyline, n * sizeof(int)));

	CHECK_CUDA_ERROR(cudaMemset(d_local_survivors, 0, n * sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemset(d_candidate_count, 0, sizeof(int)));
	CHECK_CUDA_ERROR(cudaMemset(d_final_skyline, 0, n * sizeof(int)));

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	auto beg = std::chrono::high_resolution_clock::now();

	dim3 block(256);
	dim3 grid_all((n + 255) / 256);

	// 1. 局部过滤
	hybrid_local_skyline_kernel << <grid_all, block >> > (n, dim, d_coords, d_local_survivors);

	// 2. 收集入围者
	hybrid_extract_candidates_kernel << <grid_all, block >> > (n, d_local_survivors, d_candidate_indices, d_candidate_count);

	int num_candidates = 0;
	CHECK_CUDA_ERROR(cudaMemcpy(&num_candidates, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));

	// 3. 全局对决 (通过切片防止 TDR 卡死，并暴露出其真实算法复杂度)
	int chunk_size = 10240;
	for (int start = 0; start < num_candidates; start += chunk_size) {
		int size = std::min(chunk_size, num_candidates - start);
		dim3 chunk_grid((size + 255) / 256);
		hybrid_global_chunk_kernel << <chunk_grid, block >> > (dim, start, size, num_candidates, d_coords, d_candidate_indices, d_final_skyline, n);
	}

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();
	double time_ms = std::chrono::duration<double, std::milli>(end - beg).count();

	std::vector<int> h_final_skyline(n);
	CHECK_CUDA_ERROR(cudaMemcpy(h_final_skyline.data(), d_final_skyline, n * sizeof(int), cudaMemcpyDeviceToHost));

	int final_count = 0;
	for (int i = 0; i < n; i++) if (h_final_skyline[i] == 1) final_count++;

	cudaFree(d_coords); cudaFree(d_local_survivors); cudaFree(d_candidate_indices);
	cudaFree(d_candidate_count); cudaFree(d_final_skyline);

	return std::make_tuple(final_count, time_ms);
}
#endif // HYBRID_ALGORITHM_CU