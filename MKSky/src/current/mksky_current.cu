#ifndef MYAL_ALGORITHM_CU
#define MYAL_ALGORITHM_CU

#include "mksky_current.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// 👑 保留核心 Z-Curve 灵魂
static __device__ __forceinline__ uint64_t fast_morton_encode(const float* coords, int dim, int bits_per_dim) {
	uint64_t morton = 0;
	uint64_t mask = (1ULL << bits_per_dim) - 1;
	for (int d = 0; d < dim; d++) {
		float norm = dev_max(0.0f, dev_min(1.0f - EPS, coords[d]));
		uint64_t quant = static_cast<uint64_t>(norm * mask);
#pragma unroll 20
		for (int i = 0; i < bits_per_dim; i++) {
			if (quant & (1ULL << i)) morton |= 1ULL << (i * dim + d);
		}
	}
	return morton;
}

__global__ void myal_find_pivot_kernel(const float* d_all_coords, int n, int dim, unsigned long long* d_min_pivot) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	float sum = 0.0f;
	for (int d = 0; d < dim; d++) sum += d_all_coords[d * n + idx];
	unsigned int sum_bits = __float_as_int(sum);
	unsigned long long combined = ((unsigned long long)sum_bits << 32) | (unsigned int)idx;
	atomicMin(d_min_pivot, combined);
}

__global__ void myal_filter_and_morton_kernel(const float* d_all_coords, int n, int dim, unsigned long long* d_min_pivot, uint64_t* keys, int* indices, int bits_per_dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	int pivot_idx = (int)(*d_min_pivot & 0xFFFFFFFF);
	bool dominated = true;
	bool strict_less = false;
	float coords[MAX_DIM];
	int read_d = 0;

	for (; read_d < dim; read_d++) {
		float my_val = d_all_coords[read_d * n + idx];
		coords[read_d] = my_val;
		float p_val = d_all_coords[read_d * n + pivot_idx];
		if (p_val > my_val) { dominated = false; read_d++; break; }
		if (p_val < my_val) { strict_less = true; }
	}

	if (idx != pivot_idx && dominated && strict_less) {
		keys[idx] = ULLONG_MAX;
	}
	else {
		for (; read_d < dim; read_d++) coords[read_d] = d_all_coords[read_d * n + idx];
		keys[idx] = fast_morton_encode(coords, dim, bits_per_dim);
	}
	indices[idx] = idx;
}

__global__ void myal_find_valid_n_kernel(uint64_t* keys, int total_n, int* valid_n) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		int left = 0, right = total_n - 1, ans = total_n;
		while (left <= right) {
			int mid = left + (right - left) / 2;
			if (keys[mid] == ULLONG_MAX) { ans = mid; right = mid - 1; }
			else { left = mid + 1; }
		}
		*valid_n = ans;
	}
}

__global__ void myal_reorder_points_kernel(const float* src_coords, const int* src_idx, float* dst_coords, int* dst_idx, int* sorted_indices, int valid_n, int total_n, int dim) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= valid_n) return;
	int s_idx = sorted_indices[idx];
	for (int d = 0; d < dim; d++) dst_coords[d * total_n + idx] = src_coords[d * total_n + s_idx];
	dst_idx[idx] = src_idx[s_idx];
}

template <int DIM>
__global__ void myal_block_prune_templated(const float* d_sorted_coords, int valid_n, int* candidate_indices, int* candidate_count, int chunk_size, int total_n) {
	extern __shared__ float sh_coords_flat[];
	int start = blockIdx.x * chunk_size;
	int end = start + chunk_size;
	if (end > valid_n) end = valid_n;
	int block_size = end - start;
	if (block_size <= 0) return;

	for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
		int global_idx = start + i;
#pragma unroll
		for (int d = 0; d < DIM; d++) sh_coords_flat[i * DIM + d] = d_sorted_coords[d * total_n + global_idx];
	}
	__syncthreads();

	for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
		bool is_block_skyline = true;
		float curr_coords[DIM];
#pragma unroll
		for (int d = 0; d < DIM; d++) curr_coords[d] = sh_coords_flat[i * DIM + d];

		for (int j = 0; j < block_size; j++) {
			if (i == j) continue;
			bool dominated = true;
			bool any_strict_less = false;
#pragma unroll
			for (int d = 0; d < DIM; d++) {
				float sh_val = sh_coords_flat[j * DIM + d];
				dominated &= (sh_val <= curr_coords[d]);
				any_strict_less |= (sh_val < curr_coords[d]);
			}
			if (dominated && any_strict_less) { is_block_skyline = false; break; }
		}
		if (is_block_skyline) {
			int pos = atomicAdd(candidate_count, 1);
			candidate_indices[pos] = start + i;
		}
	}
}

// 👑 新增：打包 AoS 内存
__global__ void myal_pack_candidates_aos_kernel(const float* d_sorted_coords, const int* candidate_indices, int candidate_num, float* d_packed_coords, int dim, int total_n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= candidate_num) return;
	int orig_idx = candidate_indices[idx];
	for (int d = 0; d < dim; d++) {
		d_packed_coords[idx * dim + d] = d_sorted_coords[d * total_n + orig_idx];
	}
}

// 👑 绝招 1：计算每个 Z-Chunk 的 Min-Corner (最小下界)
__global__ void myal_compute_chunk_mins_kernel(const float* d_packed_coords, int candidate_num, int dim, float* d_chunk_mins, int chunk_size) {
	int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int num_chunks = (candidate_num + chunk_size - 1) / chunk_size;
	if (chunk_idx >= num_chunks) return;

	int start = chunk_idx * chunk_size;
	int end = start + chunk_size;
	if (end > candidate_num) end = candidate_num;

	float mins[MAX_DIM];
	for (int d = 0; d < dim; d++) mins[d] = FLT_MAX; // 初始化极大值

	for (int i = start; i < end; i++) {
		for (int d = 0; d < dim; d++) {
			float val = d_packed_coords[i * dim + d];
			if (val < mins[d]) mins[d] = val; // 获取该空间块的绝对下界
		}
	}

	for (int d = 0; d < dim; d++) {
		d_chunk_mins[chunk_idx * dim + d] = mins[d];
	}
}

// 👑 绝招 2：两级跳跃 SFS (Chunk-Skipping SFS) 算法灵魂的最终拼图！
template <int DIM>
__global__ void myal_global_sfs_chunked_templated(const float* d_packed_coords, const float* d_chunk_mins, const int* d_sorted_idx, const int* candidate_indices, int candidate_num, int chunk_size, MyDataPoint* d_skyline, int* skyline_count) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= candidate_num) return;

	float curr_coords[DIM];
#pragma unroll
	for (int d = 0; d < DIM; d++) {
		curr_coords[d] = __ldg(&d_packed_coords[idx * DIM + d]);
	}

	bool is_global_skyline = true;
	int my_chunk = idx / chunk_size;

	// ==========================================
	// 阶段一：跨块粗筛 (跳跃式免检)
	// ==========================================
	for (int c = 0; c < my_chunk; c++) {
		bool chunk_can_dominate = true;

		// 与块的 Min-Corner 比对
#pragma unroll
		for (int d = 0; d < DIM; d++) {
			float min_val = __ldg(&d_chunk_mins[c * DIM + d]);
			if (min_val > curr_coords[d]) {
				chunk_can_dominate = false; // 只要有任何一维底线比我大，整个块立刻滚蛋！
				break;
			}
		}

		// 只有当块有资格支配我时，才进去做细筛
		if (chunk_can_dominate) {
			int start_j = c * chunk_size;
			int end_j = start_j + chunk_size; // 完整的块

			for (int j = start_j; j < end_j; j++) {
				bool dominated = true;
				bool strict_less = false;
#pragma unroll
				for (int d = 0; d < DIM; d++) {
					float j_coord = __ldg(&d_packed_coords[j * DIM + d]);
					if (j_coord > curr_coords[d]) { dominated = false; break; }
					if (j_coord < curr_coords[d]) strict_less = true;
				}
				if (dominated && strict_less) {
					is_global_skyline = false;
					break;
				}
			}
		}
		if (!is_global_skyline) break;
	}

	// ==========================================
	// 阶段二：同块细筛 (比对自己所在的残缺块)
	// ==========================================
	if (is_global_skyline) {
		int start_j = my_chunk * chunk_size;
		for (int j = start_j; j < idx; j++) {
			bool dominated = true;
			bool strict_less = false;
#pragma unroll
			for (int d = 0; d < DIM; d++) {
				float j_coord = __ldg(&d_packed_coords[j * DIM + d]);
				if (j_coord > curr_coords[d]) { dominated = false; break; }
				if (j_coord < curr_coords[d]) strict_less = true;
			}
			if (dominated && strict_less) {
				is_global_skyline = false;
				break;
			}
		}
	}

	// 最终存活
	if (is_global_skyline) {
		int pos = atomicAdd(skyline_count, 1);
#pragma unroll
		for (int d = 0; d < DIM; d++) d_skyline[pos].coords[d] = curr_coords[d];
		d_skyline[pos].original_idx = d_sorted_idx[candidate_indices[idx]];
	}
}

std::tuple<int, double, std::vector<MyDataPoint>, double, int>
run_myal_algorithm(std::vector<MyDataPoint>& h_points, const AlgorithmConfig& config) {
	CHECK_CUDA_ERROR(cudaSetDevice(0));
	dim3 block(BLOCK_SIZE);
	int n = static_cast<int>(h_points.size());
	int dim = config.dim;

	int bits_per_dim = 64 / dim;
	if (bits_per_dim > 20) bits_per_dim = 20;

	float* d_orig_coords; CHECK_CUDA_ERROR(cudaMalloc(&d_orig_coords, n * dim * sizeof(float)));
	int* d_orig_idx;      CHECK_CUDA_ERROR(cudaMalloc(&d_orig_idx, n * sizeof(int)));

	std::vector<float> h_orig_coords(n * dim);
	std::vector<int> h_orig_idx(n);
	for (int d = 0; d < dim; d++) {
		for (int i = 0; i < n; i++) h_orig_coords[d * n + i] = h_points[i].coords[d];
	}
	for (int i = 0; i < n; i++) h_orig_idx[i] = h_points[i].original_idx;

	CHECK_CUDA_ERROR(cudaMemcpy(d_orig_coords, h_orig_coords.data(), n * dim * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_orig_idx, h_orig_idx.data(), n * sizeof(int), cudaMemcpyHostToDevice));

	unsigned long long* d_min_pivot; CHECK_CUDA_ERROR(cudaMalloc(&d_min_pivot, sizeof(unsigned long long)));
	uint64_t* d_morton_keys;         CHECK_CUDA_ERROR(cudaMalloc(&d_morton_keys, n * sizeof(uint64_t)));
	int* d_sorted_indices;           CHECK_CUDA_ERROR(cudaMalloc(&d_sorted_indices, n * sizeof(int)));
	int* d_valid_n;                  CHECK_CUDA_ERROR(cudaMalloc(&d_valid_n, sizeof(int)));

	float* d_sorted_coords; CHECK_CUDA_ERROR(cudaMalloc(&d_sorted_coords, n * dim * sizeof(float)));
	int* d_sorted_idx;      CHECK_CUDA_ERROR(cudaMalloc(&d_sorted_idx, n * sizeof(int)));

	int* d_candidate_indices; CHECK_CUDA_ERROR(cudaMalloc(&d_candidate_indices, n * sizeof(int)));
	int* d_counter;           CHECK_CUDA_ERROR(cudaMalloc(&d_counter, sizeof(int)));
	MyDataPoint* d_skyline;   CHECK_CUDA_ERROR(cudaMalloc(&d_skyline, n * sizeof(MyDataPoint)));

	// Explicit algorithm buffers only. CUDA runtime and Thrust temporary storage are
	// intentionally excluded so this matches the paper implementation's metric.
	const std::size_t resident_device_bytes =
		static_cast<std::size_t>(n) *
			(2 * dim * sizeof(float) + 4 * sizeof(int) + sizeof(uint64_t) +
			 sizeof(MyDataPoint)) +
		sizeof(unsigned long long) + 2 * sizeof(int);
	std::size_t peak_device_bytes = resident_device_bytes;
	int candidate_num_for_result = 0;

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	auto beg = std::chrono::high_resolution_clock::now();

	dim3 grid_init((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	CHECK_CUDA_ERROR(cudaMemset(d_min_pivot, 0xFF, sizeof(unsigned long long)));

	myal_find_pivot_kernel << <grid_init, block >> > (d_orig_coords, n, dim, d_min_pivot);
	myal_filter_and_morton_kernel << <grid_init, block >> > (d_orig_coords, n, dim, d_min_pivot, d_morton_keys, d_sorted_indices, bits_per_dim);

	thrust::device_ptr<uint64_t> ptr_keys(d_morton_keys);
	thrust::device_ptr<int> ptr_vals(d_sorted_indices);
	thrust::sort_by_key(ptr_keys, ptr_keys + n, ptr_vals);

	myal_find_valid_n_kernel << <1, 1 >> > (d_morton_keys, n, d_valid_n);

	int valid_n = 0;
	CHECK_CUDA_ERROR(cudaMemcpy(&valid_n, d_valid_n, sizeof(int), cudaMemcpyDeviceToHost));

	int final_skyline_count = 0;
	if (valid_n > 0) {
		dim3 grid_reorder((valid_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
		myal_reorder_points_kernel << <grid_reorder, block >> > (d_orig_coords, d_orig_idx, d_sorted_coords, d_sorted_idx, d_sorted_indices, valid_n, n, dim);

		CHECK_CUDA_ERROR(cudaMemset(d_counter, 0, sizeof(int)));

		int chunk_size = (dim >= 12) ? 512 : 1024;
		int total_chunks = (valid_n + chunk_size - 1) / chunk_size;
		if (total_chunks > 0) {
			int shared_mem_prune = chunk_size * dim * sizeof(float);
			switch (dim) {
			case 3: myal_block_prune_templated<3> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 4: myal_block_prune_templated<4> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 5: myal_block_prune_templated<5> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 6: myal_block_prune_templated<6> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 7: myal_block_prune_templated<7> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 8: myal_block_prune_templated<8> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 9: myal_block_prune_templated<9> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 10: myal_block_prune_templated<10> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 11: myal_block_prune_templated<11> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 12: myal_block_prune_templated<12> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 13: myal_block_prune_templated<13> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 14: myal_block_prune_templated<14> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 15: myal_block_prune_templated<15> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			case 16: myal_block_prune_templated<16> << <total_chunks, block, shared_mem_prune >> > (d_sorted_coords, valid_n, d_candidate_indices, d_counter, chunk_size, n); break;
			default: break;
			}
			CHECK_CUDA_ERROR(cudaGetLastError());
		}

		int candidate_num = 0;
		CHECK_CUDA_ERROR(cudaMemcpy(&candidate_num, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
		candidate_num_for_result = candidate_num;

		if (candidate_num > 0) {
			// 👑 彻底抛弃 Sum，直接对候选索引排序，完美恢复纯正的 Morton Z-Curve 拓扑序列！
			thrust::device_ptr<int> ptr_cand(d_candidate_indices);
			thrust::sort(ptr_cand, ptr_cand + candidate_num);

			float* d_packed_coords; CHECK_CUDA_ERROR(cudaMalloc(&d_packed_coords, candidate_num * dim * sizeof(float)));
			dim3 cand_grid((candidate_num + BLOCK_SIZE - 1) / BLOCK_SIZE);
			myal_pack_candidates_aos_kernel << <cand_grid, block >> > (d_sorted_coords, d_candidate_indices, candidate_num, d_packed_coords, dim, n);

			// 👑 提取 Chunk Min-Corner
			int sfs_chunk_size = 256;
			int num_chunks_sfs = (candidate_num + sfs_chunk_size - 1) / sfs_chunk_size;
			float* d_chunk_mins; CHECK_CUDA_ERROR(cudaMalloc(&d_chunk_mins, num_chunks_sfs * dim * sizeof(float)));
			peak_device_bytes = resident_device_bytes +
				static_cast<std::size_t>(candidate_num) * dim * sizeof(float) +
				static_cast<std::size_t>(num_chunks_sfs) * dim * sizeof(float);
			dim3 chunk_grid((num_chunks_sfs + BLOCK_SIZE - 1) / BLOCK_SIZE);
			myal_compute_chunk_mins_kernel << <chunk_grid, block >> > (d_packed_coords, candidate_num, dim, d_chunk_mins, sfs_chunk_size);

			CHECK_CUDA_ERROR(cudaMemset(d_counter, 0, sizeof(int)));

			// 👑 挂载基于 Z-Curve 红利的跳跃式 SFS
			switch (dim) {
			case 3: myal_global_sfs_chunked_templated<3> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 4: myal_global_sfs_chunked_templated<4> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 5: myal_global_sfs_chunked_templated<5> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 6: myal_global_sfs_chunked_templated<6> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 7: myal_global_sfs_chunked_templated<7> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 8: myal_global_sfs_chunked_templated<8> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 9: myal_global_sfs_chunked_templated<9> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 10: myal_global_sfs_chunked_templated<10> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 11: myal_global_sfs_chunked_templated<11> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 12: myal_global_sfs_chunked_templated<12> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 13: myal_global_sfs_chunked_templated<13> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 14: myal_global_sfs_chunked_templated<14> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 15: myal_global_sfs_chunked_templated<15> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			case 16: myal_global_sfs_chunked_templated<16> << <cand_grid, block >> > (d_packed_coords, d_chunk_mins, d_sorted_idx, d_candidate_indices, candidate_num, sfs_chunk_size, d_skyline, d_counter); break;
			default: break;
			}
			CHECK_CUDA_ERROR(cudaGetLastError());

			CHECK_CUDA_ERROR(cudaMemcpy(&final_skyline_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
			cudaFree(d_packed_coords);
			cudaFree(d_chunk_mins);
		}
	}

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	auto end = std::chrono::high_resolution_clock::now();

	std::vector<MyDataPoint> h_skyline;
	if (final_skyline_count > 0) {
		h_skyline.resize(final_skyline_count);
		CHECK_CUDA_ERROR(cudaMemcpy(h_skyline.data(), d_skyline, final_skyline_count * sizeof(MyDataPoint), cudaMemcpyDeviceToHost));
	}

	cudaGetLastError();
	cudaFree(d_orig_coords); cudaFree(d_orig_idx);
	cudaFree(d_min_pivot); cudaFree(d_morton_keys); cudaFree(d_sorted_indices); cudaFree(d_valid_n);
	cudaFree(d_sorted_coords); cudaFree(d_sorted_idx);
	cudaFree(d_candidate_indices); cudaFree(d_counter); cudaFree(d_skyline);

	return std::make_tuple(
		final_skyline_count,
		std::chrono::duration<double>(end - beg).count() * 1000.0,
		h_skyline,
		static_cast<double>(peak_device_bytes) / (1024.0 * 1024.0),
		candidate_num_for_result);
}
#endif // MYAL_ALGORITHM_CU
