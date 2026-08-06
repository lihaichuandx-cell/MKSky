#include "algorithm.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace {

struct IsOne {
    __host__ __device__ bool operator()(int value) const { return value != 0; }
};

__global__ void initialize_indices(int count, int* indices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) indices[index] = index;
}

__global__ void assign_cells(const float* coords, int count, int dim, int side,
                             unsigned int* cell_ids, int* occupancy) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int cell_id = 0;
    unsigned int stride = 1;
    for (int d = 0; d < dim; ++d) {
        int coordinate = static_cast<int>(coords[index * dim + d] * side);
        coordinate = max(0, min(side - 1, coordinate));
        cell_id += static_cast<unsigned int>(coordinate) * stride;
        stride *= static_cast<unsigned int>(side);
    }
    cell_ids[index] = cell_id;
    occupancy[cell_id] = 1;
}

__global__ void prefix_scan_step(const int* input, int* output, int cell_count,
                                 int side, int stride, int offset) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cell_count) return;
    const int coordinate = (index / stride) % side;
    int value = input[index];
    if (coordinate >= offset) value += input[index - offset * stride];
    output[index] = value;
}

__global__ void mark_candidate_points(const unsigned int* cell_ids,
                                      const int* strict_prefix,
                                      int point_count, int dim, int side,
                                      int* keep) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count) return;
    unsigned int cell_id = cell_ids[index];
    unsigned int decoded = cell_id;
    unsigned int lower_index = cell_id;
    unsigned int stride = 1;
    bool has_strict_lower_region = true;
    for (int d = 0; d < dim; ++d) {
        const unsigned int coordinate = decoded % static_cast<unsigned int>(side);
        decoded /= static_cast<unsigned int>(side);
        if (coordinate == 0) {
            has_strict_lower_region = false;
            break;
        }
        lower_index -= stride;
        stride *= static_cast<unsigned int>(side);
    }
    keep[index] = !has_strict_lower_region || strict_prefix[lower_index] == 0 ? 1 : 0;
}

__global__ void compute_sums(const float* coords, int dim,
                             const int* candidates, int candidate_count,
                             double* sums) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= candidate_count) return;
    const int index = candidates[position];
    double sum = 0.0;
    for (int d = 0; d < dim; ++d) sum += coords[index * dim + d];
    sums[position] = sum;
}

__global__ void refine_candidates(const float* coords, int dim,
                                  const int* sorted_candidates,
                                  int candidate_count,
                                  const int* original_indices,
                                  int* skyline_indices,
                                  int* skyline_count) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= candidate_count) return;
    const int point_index = sorted_candidates[position];
    bool alive = true;
    for (int other_position = 0; other_position < position && alive; ++other_position) {
        const int other_index = sorted_candidates[other_position];
        bool all_le = true;
        bool any_lt = false;
        for (int d = 0; d < dim; ++d) {
            const float other_value = coords[other_index * dim + d];
            const float value = coords[point_index * dim + d];
            if (other_value > value) {
                all_le = false;
                break;
            }
            any_lt = any_lt || other_value < value;
        }
        if (all_le && any_lt) alive = false;
    }
    if (alive) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = original_indices[point_index];
    }
}

int select_partition_ratio(int count, int dim, int requested) {
    const int maximum_bits = 22;
    const int maximum_rho = std::min(7, maximum_bits / dim);
    if (requested > 0) return std::max(1, std::min(requested, maximum_rho));
    const double target_cells = std::max(256.0, std::min(4194304.0, count * 2.0));
    int rho = 1;
    while (rho < maximum_rho && std::pow(2.0, (rho + 1) * dim) <= target_cells) ++rho;
    return rho;
}

}  // namespace

AlgorithmResult run_skycell(const std::vector<MyDataPoint>& points,
                            const AlgorithmConfig& config) {
    AlgorithmResult result;
    result.name = "SkyCell-G";
    result.provenance =
        "generalized CUDA grid baseline preserving SkyCell candidate-cell "
        "semantics; not the authors' progressive ShrinkKeyCells implementation";
    result.input_count = static_cast<int>(points.size());
    const int count = result.input_count;
    const int dim = config.dim;
    const int rho = select_partition_ratio(count, dim, config.skycell_rho);
    const int side = 1 << rho;
    int cell_count = 1;
    for (int d = 0; d < dim; ++d) cell_count *= side;
    const std::size_t count_size = static_cast<std::size_t>(count);
    const auto wall_begin = std::chrono::high_resolution_clock::now();

    std::vector<float> host_coords(static_cast<std::size_t>(count) * dim);
    std::vector<int> host_original_indices(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        host_original_indices[static_cast<std::size_t>(i)] = points[i].original_idx;
        for (int d = 0; d < dim; ++d) host_coords[static_cast<std::size_t>(i) * dim + d] = points[i].coords[d];
    }

    float* coords = nullptr;
    int* original_indices = nullptr;
    unsigned int* cell_ids = nullptr;
    int* occupancy_a = nullptr;
    int* occupancy_b = nullptr;
    int* point_indices = nullptr;
    int* candidates = nullptr;
    int* flags = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&coords, host_coords.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&original_indices, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&cell_ids, static_cast<std::size_t>(count) * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&occupancy_a, static_cast<std::size_t>(cell_count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&occupancy_b, static_cast<std::size_t>(cell_count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&point_indices, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&candidates, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&flags, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(coords, host_coords.data(), host_coords.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(original_indices, host_original_indices.data(),
                                static_cast<std::size_t>(count) * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start_event;
    cudaEvent_t phase_event;
    cudaEvent_t stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&phase_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));
    CHECK_CUDA_ERROR(cudaMemset(occupancy_a, 0, static_cast<std::size_t>(cell_count) * sizeof(int)));
    const dim3 block(BLOCK_SIZE);
    const dim3 point_grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    assign_cells<<<point_grid, block>>>(coords, count, dim, side, cell_ids, occupancy_a);
    initialize_indices<<<point_grid, block>>>(count, point_indices);

    int* prefix_input = occupancy_a;
    int* prefix_output = occupancy_b;
    int stride = 1;
    const dim3 cell_grid((cell_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    for (int d = 0; d < dim; ++d) {
        for (int offset = 1; offset < side; offset <<= 1) {
            prefix_scan_step<<<cell_grid, block>>>(prefix_input, prefix_output,
                                                   cell_count, side, stride, offset);
            std::swap(prefix_input, prefix_output);
        }
        stride *= side;
    }
    mark_candidate_points<<<point_grid, block>>>(cell_ids, prefix_input, count, dim, side, flags);

    thrust::device_ptr<int> point_ptr(point_indices);
    thrust::device_ptr<int> candidate_ptr(candidates);
    thrust::device_ptr<int> flag_ptr(flags);
    const int candidate_count = static_cast<int>(
        thrust::copy_if(point_ptr, point_ptr + count, flag_ptr, candidate_ptr, IsOne()) - candidate_ptr);
    result.candidate_count = candidate_count;
    result.valid_count = candidate_count;

    double* sums = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&sums, static_cast<std::size_t>(candidate_count) * sizeof(double)));
    const dim3 candidate_grid((candidate_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    compute_sums<<<candidate_grid, block>>>(coords, dim, candidates, candidate_count, sums);
    thrust::device_ptr<double> sum_ptr(sums);
    thrust::sort_by_key(sum_ptr, sum_ptr + candidate_count, candidate_ptr);
    CHECK_CUDA_ERROR(cudaEventRecord(phase_event));

    int* skyline_indices = nullptr;
    int* skyline_count = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&skyline_indices, static_cast<std::size_t>(candidate_count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&skyline_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(skyline_count, 0, sizeof(int)));
    refine_candidates<<<candidate_grid, block>>>(coords, dim, candidates, candidate_count,
                                                 original_indices, skyline_indices, skyline_count);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    result.device_ms = elapsed_ms;
    float preprocess_ms = 0.0f;
    float core_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&preprocess_ms, start_event, phase_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&core_ms, phase_event, stop_event));
    result.preprocess_ms = preprocess_ms;
    result.core_ms = core_ms;
    const std::size_t device_bytes =
        count_size * dim * sizeof(float) + count_size * sizeof(int) +
        count_size * sizeof(unsigned int) + static_cast<std::size_t>(cell_count) * 2 * sizeof(int) +
        count_size * 3 * sizeof(int) + static_cast<std::size_t>(candidate_count) * sizeof(double) +
        static_cast<std::size_t>(candidate_count) * sizeof(int) + sizeof(int);
    result.device_memory_mb = static_cast<double>(device_bytes) / (1024.0 * 1024.0);

    int output_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&output_count, skyline_count, sizeof(int), cudaMemcpyDeviceToHost));
    result.skyline_indices.resize(static_cast<std::size_t>(output_count));
    CHECK_CUDA_ERROR(cudaMemcpy(result.skyline_indices.data(), skyline_indices,
                                static_cast<std::size_t>(output_count) * sizeof(int), cudaMemcpyDeviceToHost));
    std::sort(result.skyline_indices.begin(), result.skyline_indices.end());

    cudaFree(skyline_count);
    cudaFree(skyline_indices);
    cudaFree(sums);
    cudaFree(flags);
    cudaFree(candidates);
    cudaFree(point_indices);
    cudaFree(occupancy_b);
    cudaFree(occupancy_a);
    cudaFree(cell_ids);
    cudaFree(original_indices);
    cudaFree(coords);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(phase_event);
    cudaEventDestroy(start_event);
    const auto wall_end = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    return result;
}
