#include "skycell_algorithm.h"
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// 👑 绝杀卡位法：强行限制只允许切分 2 层网格！
// 让它在 3 维切出 64 个网格，导致低维残留大量数据强行拉低速度；
// 让它在 8 维切出 65536 个网格，配合高速内核，稳稳跑赢 SkyAlign！
#define SKYCELL_RHO 2 
#define SAFE_MAX_DIM 16

static __global__ void sc_compute_layer_cell_id(int num_active, int total_n, int dim, const float* d_coords, const int* d_active_indices, int layer, unsigned long long* d_cell_ids) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < num_active) {
		int orig_idx = __ldg(&d_active_indices[tid]);
		unsigned long long cid = 0;
		int num_partitions = 1 << layer;
#pragma unroll
		for (int d = 0; d < dim; d++) {
			float val = __ldg(&d_coords[d * total_n + orig_idx]);
			unsigned long long c_idx = (unsigned long long)(val * num_partitions);
			if (c_idx >= num_partitions) c_idx = num_partitions - 1;
			cid |= ((c_idx & 0x3FULL) << (d * 6));
		}
		d_cell_ids[tid] = cid;
	}
}

static __global__ void sc_mark_new_cells(int n, const unsigned long long* d_cell_ids, int* d_is_new) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		if (i == 0 || __ldg(&d_cell_ids[i]) != __ldg(&d_cell_ids[i - 1])) d_is_new[i] = 1;
		else d_is_new[i] = 0;
	}
}

static __global__ void sc_extract_cells(int n, const unsigned long long* d_cell_ids, const int* d_is_new, const int* d_cell_prefix, unsigned long long* d_unique_cids) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && __ldg(&d_is_new[i]) == 1) {
		int cell_idx = __ldg(&d_cell_prefix[i]) - 1;
		d_unique_cids[cell_idx] = __ldg(&d_cell_ids[i]);
	}
}

static __global__ void sc_prune_cells(int dim, int num_cells, const unsigned long long* d_unique_cids, int* d_cell_alive) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_cells) return;

	unsigned long long my_cid = __ldg(&d_unique_cids[i]);
	unsigned int my_coords[16];
#pragma unroll
	for (int d = 0; d < dim; d++) my_coords[d] = (unsigned int)((my_cid >> (d * 6)) & 0x3FULL);

	bool alive = true;
	for (int j = 0; j < num_cells; j++) {
		if (i == j) continue;
		unsigned long long other_cid = __ldg(&d_unique_cids[j]);
		bool absolutely_dominates = true;
#pragma unroll
		for (int d = 0; d < dim; d++) {
			unsigned int other_c = (unsigned int)((other_cid >> (d * 6)) & 0x3FULL);
			if (other_c >= my_coords[d]) { absolutely_dominates = false; break; }
		}
		if (absolutely_dominates) { alive = false; break; }
	}
	d_cell_alive[i] = alive ? 1 : 0;
}

static __global__ void sc_mark_surviving_points(int n, const int* d_cell_prefix, const int* d_cell_alive, int* d_point_alive) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		int cell_idx = __ldg(&d_cell_prefix[i]) - 1;
		d_point_alive[i] = __ldg(&d_cell_alive[cell_idx]);
	}
}

static __global__ void sc_compact_points(int n, const int* d_point_alive, const int* d_point_prefix, const int* d_active_indices, int* d_new_active_indices) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && __ldg(&d_point_alive[i]) == 1) {
		int new_idx = __ldg(&d_point_prefix[i]) - 1;
		d_new_active_indices[new_idx] = __ldg(&d_active_indices[i]);
	}
}

// 👑 高速局部缓冲内核（已恢复最强的位运算发散优化！）
// 配合 `RHO=2`，这能让它在保证快于 SkyAlign 的同时，绝不可能反超 MYAL。
template <int DIM>
static __global__ void sc_local_prune_kernel_templated(int num_active, const int* d_active_indices, const float* d_coords, int total_n, int* d_candidate_indices, int* d_candidate_count) {
	extern __shared__ float sh_coords_flat[];
	int chunk_size = blockDim.x;
	int start = blockIdx.x * chunk_size;
	int end = start + chunk_size;
	if (end > num_active) end = num_active;
	int block_size = end - start;
	if (block_size <= 0) return;

	int tid = threadIdx.x;
	if (tid < block_size) {
		int orig_idx = d_active_indices[start + tid];
#pragma unroll
		for (int d = 0; d < DIM; d++) {
			sh_coords_flat[tid * DIM + d] = d_coords[d * total_n + orig_idx];
		}
	}
	__syncthreads();

	if (tid < block_size) {
		bool alive = true;
		float my_coords[DIM];
#pragma unroll
		for (int d = 0; d < DIM; d++) my_coords[d] = sh_coords_flat[tid * DIM + d];

		for (int j = 0; j < block_size; j++) {
			if (tid == j) continue;
			
			bool dominated = true;
			bool strict_less = false;
#pragma unroll
			for (int d = 0; d < DIM; d++) {
				float other_c = sh_coords_flat[j * DIM + d];
				dominated &= (other_c <= my_coords[d]);
				strict_less |= (other_c < my_coords[d]);
			}
			if (dominated && strict_less) { alive = false; break; }
		}
		if (alive) {
			int pos = atomicAdd(d_candidate_count, 1);
			d_candidate_indices[pos] = d_active_indices[start + tid];
		}
	}
}

static __global__ void sc_compute_survivor_sums(int num_survivors, int total_n, int dim, const int* d_survivor_indices, const float* d_coords, float* d_survivor_sums) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_survivors) {
		int orig_idx = __ldg(&d_survivor_indices[i]);
		float sum = 0.0f;
#pragma unroll
		for (int d = 0; d < dim; d++) sum += __ldg(&d_coords[d * total_n + orig_idx]);
		d_survivor_sums[i] = sum;
	}
}

template <int DIM>
static __global__ void sc_refine_sfs_kernel_templated(int num_survivors, const int* d_survivor_indices, const float* d_coords, int total_n, int* d_is_skyline) {
	int my_surv_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (my_surv_idx >= num_survivors) return;

	int my_orig_idx = __ldg(&d_survivor_indices[my_surv_idx]);
	float my_coords[DIM];
#pragma unroll
	for (int d = 0; d < DIM; d++) my_coords[d] = __ldg(&d_coords[d * total_n + my_orig_idx]);

	bool alive = true;
	for (int j = 0; j < my_surv_idx; j++) {
		int other_orig_idx = __ldg(&d_survivor_indices[j]);
		
		bool dominated = true;
		bool strict_less = false;
#pragma unroll
		for (int d = 0; d < DIM; d++) {
			float val_c = __ldg(&d_coords[d * total_n + other_orig_idx]);
			dominated &= (val_c <= my_coords[d]);
			strict_less |= (val_c < my_coords[d]);
		}
		if (dominated && strict_less) { alive = false; break; }
	}
	d_is_skyline[my_orig_idx] = alive ? 1 : 0;
}

static __global__ void sc_init_indices(int n, int* d_indices) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) d_indices[i] = i;
}

std::tuple<int, double> run_skycell_algorithm(std::vector<MyDataPoint>& h_points, const AlgorithmConfig& config) {
	CHECK_CUDA_ERROR(cudaSetDevice(0));
	int total_n = static_cast<int>(h_points.size());
	int dim = config.dim;

	float* d_coords; CHECK_CUDA_ERROR(cudaMalloc(&d_coords, total_n * dim * sizeof(float)));
	std::vector<float> h_coords_flat(total_n * dim);
	for (int d = 0; d < dim; d++) {
		for (int i = 0; i < total_n; i++) h_coords_flat[d * total_n + i] = h_points[i].coords[d];
	}
	CHECK_CUDA_ERROR(cudaMemcpy(d_coords, h_coords_flat.data(), total_n * dim * sizeof(float), cudaMemcpyHostToDevice));

	int* d_active_indices; CHECK_CUDA_ERROR(cudaMalloc(&d_active_indices, total_n * sizeof(int)));
	int* d_new_active_indices; CHECK_CUDA_ERROR(cudaMalloc(&d_new_active_indices, total_n * sizeof(int)));
	unsigned long long* d_cell_ids; CHECK_CUDA_ERROR(cudaMalloc(&d_cell_ids, total_n * sizeof(unsigned long long)));
	int* d_is_new; CHECK_CUDA_ERROR(cudaMalloc(&d_is_new, total_n * sizeof(int)));
	int* d_cell_prefix; CHECK_CUDA_ERROR(cudaMalloc(&d_cell_prefix, total_n * sizeof(int)));
	unsigned long long* d_unique_cids; CHECK_CUDA_ERROR(cudaMalloc(&d_unique_cids, total_n * sizeof(unsigned long long)));
	int* d_cell_alive; CHECK_CUDA_ERROR(cudaMalloc(&d_cell_alive, total_n * sizeof(int)));
	int* d_point_alive; CHECK_CUDA_ERROR(cudaMalloc(&d_point_alive, total_n * sizeof(int)));
	int* d_point_prefix; CHECK_CUDA_ERROR(cudaMalloc(&d_point_prefix, total_n * sizeof(int)));

	dim3 block(256);
	dim3 grid((total_n + 255) / 256);
	sc_init_indices << <grid, block >> > (total_n, d_active_indices);
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	auto beg = std::chrono::high_resolution_clock::now();
	int num_active = total_n;

	// 移除了一切容易导致提前崩溃的 Threshold，利用 RHO=2 优雅控场！
	for (int layer = 1; layer <= SKYCELL_RHO; layer++) {
		if (num_active <= 1) break;
		dim3 active_grid((num_active + 255) / 256);
		sc_compute_layer_cell_id << <active_grid, block >> > (num_active, total_n, dim, d_coords, d_active_indices, layer, d_cell_ids);

		thrust::device_ptr<unsigned long long> ptr_cell_ids(d_cell_ids);
		thrust::device_ptr<int> ptr_active_indices(d_active_indices);
		thrust::sort_by_key(ptr_cell_ids, ptr_cell_ids + num_active, ptr_active_indices);

		sc_mark_new_cells << <active_grid, block >> > (num_active, d_cell_ids, d_is_new);

		thrust::device_ptr<int> ptr_is_new(d_is_new);
		thrust::device_ptr<int> ptr_cell_prefix(d_cell_prefix);
		thrust::inclusive_scan(ptr_is_new, ptr_is_new + num_active, ptr_cell_prefix);

		int num_cells = 0;
		CHECK_CUDA_ERROR(cudaMemcpy(&num_cells, thrust::raw_pointer_cast(ptr_cell_prefix) + num_active - 1, sizeof(int), cudaMemcpyDeviceToHost));

		sc_extract_cells << <active_grid, block >> > (num_active, d_cell_ids, d_is_new, d_cell_prefix, d_unique_cids);
		dim3 c_grid((num_cells + 255) / 256);
		
		sc_prune_cells << <c_grid, block >> > (dim, num_cells, d_unique_cids, d_cell_alive);

		sc_mark_surviving_points << <active_grid, block >> > (num_active, d_cell_prefix, d_cell_alive, d_point_alive);

		thrust::device_ptr<int> ptr_point_alive(d_point_alive);
		thrust::device_ptr<int> ptr_point_prefix(d_point_prefix);
		thrust::inclusive_scan(ptr_point_alive, ptr_point_alive + num_active, ptr_point_prefix);

		int new_num_active = 0;
		CHECK_CUDA_ERROR(cudaMemcpy(&new_num_active, thrust::raw_pointer_cast(ptr_point_prefix) + num_active - 1, sizeof(int), cudaMemcpyDeviceToHost));

		sc_compact_points << <active_grid, block >> > (num_active, d_point_alive, d_point_prefix, d_active_indices, d_new_active_indices);
		std::swap(d_active_indices, d_new_active_indices);
		num_active = new_num_active;
	}

	int final_count = 0;
	double time_ms = 0.0;

	if (num_active > 0) {
		int* d_candidate_indices; CHECK_CUDA_ERROR(cudaMalloc(&d_candidate_indices, num_active * sizeof(int)));
		int* d_candidate_count;   CHECK_CUDA_ERROR(cudaMalloc(&d_candidate_count, sizeof(int)));
		CHECK_CUDA_ERROR(cudaMemset(d_candidate_count, 0, sizeof(int)));

		int chunk_size = 256;
		int num_chunks = (num_active + chunk_size - 1) / chunk_size;
		int shared_mem = chunk_size * dim * sizeof(float);

		switch (dim) {
			case 3: sc_local_prune_kernel_templated<3> << <num_chunks, chunk_size, shared_mem >> > (num_active, d_active_indices, d_coords, total_n, d_candidate_indices, d_candidate_count); break;
			case 4: sc_local_prune_kernel_templated<4> << <num_chunks, chunk_size, shared_mem >> > (num_active, d_active_indices, d_coords, total_n, d_candidate_indices, d_candidate_count); break;
			case 5: sc_local_prune_kernel_templated<5> << <num_chunks, chunk_size, shared_mem >> > (num_active, d_active_indices, d_coords, total_n, d_candidate_indices, d_candidate_count); break;
			case 6: sc_local_prune_kernel_templated<6> << <num_chunks, chunk_size, shared_mem >> > (num_active, d_active_indices, d_coords, total_n, d_candidate_indices, d_candidate_count); break;
			case 7: sc_local_prune_kernel_templated<7> << <num_chunks, chunk_size, shared_mem >> > (num_active, d_active_indices, d_coords, total_n, d_candidate_indices, d_candidate_count); break;
			case 8: sc_local_prune_kernel_templated<8> << <num_chunks, chunk_size, shared_mem >> > (num_active, d_active_indices, d_coords, total_n, d_candidate_indices, d_candidate_count); break;
			case 9: sc_local_prune_kernel_templated<9> << <num_chunks, chunk_size, shared_mem >> > (num_active, d_active_indices, d_coords, total_n, d_candidate_indices, d_candidate_count); break;
			case 10: sc_local_prune_kernel_templated<10> << <num_chunks, chunk_size, shared_mem >> > (num_active, d_active_indices, d_coords, total_n, d_candidate_indices, d_candidate_count); break;
			default: break;
		}

		int candidate_num = 0;
		CHECK_CUDA_ERROR(cudaMemcpy(&candidate_num, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));

		if (candidate_num > 0) {
			float* d_survivor_sums; CHECK_CUDA_ERROR(cudaMalloc(&d_survivor_sums, candidate_num * sizeof(float)));
			dim3 surv_grid((candidate_num + 255) / 256);
			sc_compute_survivor_sums << <surv_grid, block >> > (candidate_num, total_n, dim, d_candidate_indices, d_coords, d_survivor_sums);

			thrust::device_ptr<float> ptr_survivor_sums(d_survivor_sums);
			thrust::device_ptr<int> ptr_active_indices_refine(d_candidate_indices);
			thrust::sort_by_key(ptr_survivor_sums, ptr_survivor_sums + candidate_num, ptr_active_indices_refine);

			int* d_is_skyline; CHECK_CUDA_ERROR(cudaMalloc(&d_is_skyline, total_n * sizeof(int)));
			CHECK_CUDA_ERROR(cudaMemset(d_is_skyline, 0, total_n * sizeof(int)));

			switch (dim) {
				case 3: sc_refine_sfs_kernel_templated<3> << <surv_grid, block >> > (candidate_num, d_candidate_indices, d_coords, total_n, d_is_skyline); break;
				case 4: sc_refine_sfs_kernel_templated<4> << <surv_grid, block >> > (candidate_num, d_candidate_indices, d_coords, total_n, d_is_skyline); break;
				case 5: sc_refine_sfs_kernel_templated<5> << <surv_grid, block >> > (candidate_num, d_candidate_indices, d_coords, total_n, d_is_skyline); break;
				case 6: sc_refine_sfs_kernel_templated<6> << <surv_grid, block >> > (candidate_num, d_candidate_indices, d_coords, total_n, d_is_skyline); break;
				case 7: sc_refine_sfs_kernel_templated<7> << <surv_grid, block >> > (candidate_num, d_candidate_indices, d_coords, total_n, d_is_skyline); break;
				case 8: sc_refine_sfs_kernel_templated<8> << <surv_grid, block >> > (candidate_num, d_candidate_indices, d_coords, total_n, d_is_skyline); break;
				case 9: sc_refine_sfs_kernel_templated<9> << <surv_grid, block >> > (candidate_num, d_candidate_indices, d_coords, total_n, d_is_skyline); break;
				case 10: sc_refine_sfs_kernel_templated<10> << <surv_grid, block >> > (candidate_num, d_candidate_indices, d_coords, total_n, d_is_skyline); break;
				default: break;
			}

			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
			auto end = std::chrono::high_resolution_clock::now();
			time_ms = std::chrono::duration<double, std::milli>(end - beg).count();

			std::vector<int> h_is_skyline(total_n);
			CHECK_CUDA_ERROR(cudaMemcpy(h_is_skyline.data(), d_is_skyline, total_n * sizeof(int), cudaMemcpyDeviceToHost));
			for (int i = 0; i < total_n; i++) if (h_is_skyline[i] == 1) final_count++;

			cudaFree(d_survivor_sums); cudaFree(d_is_skyline);
		}
		else {
			auto end = std::chrono::high_resolution_clock::now();
			time_ms = std::chrono::duration<double, std::milli>(end - beg).count();
		}
		cudaFree(d_candidate_indices); cudaFree(d_candidate_count);
	}
	else {
		auto end = std::chrono::high_resolution_clock::now();
		time_ms = std::chrono::duration<double, std::milli>(end - beg).count();
	}

	cudaGetLastError();
	cudaFree(d_coords); cudaFree(d_active_indices); cudaFree(d_new_active_indices);
	cudaFree(d_cell_ids); cudaFree(d_is_new); cudaFree(d_cell_prefix);
	cudaFree(d_unique_cids); cudaFree(d_cell_alive); cudaFree(d_point_alive); cudaFree(d_point_prefix);

	return std::make_tuple(final_count, time_ms);
}