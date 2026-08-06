#include "algorithm.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <vector>

namespace {

struct IsOne {
    __host__ __device__ bool operator()(int value) const { return value != 0; }
};

__global__ void initialize_indices(int count, int* indices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) indices[index] = index;
}

__global__ void find_prefilter_threshold(const float* coords, int count, int dim,
                                         unsigned int* threshold_bits) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float maximum = coords[index * dim];
    for (int d = 1; d < dim; ++d) maximum = fmaxf(maximum, coords[index * dim + d]);
    atomicMin(threshold_bits, __float_as_uint(maximum));
}

__global__ void mark_prefilter_survivors(const float* coords, int count, int dim,
                                         const unsigned int* threshold_bits,
                                         int* keep) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float threshold = __uint_as_float(*threshold_bits);
    bool survives = false;
    for (int d = 0; d < dim; ++d) survives = survives || coords[index * dim + d] <= threshold;
    keep[index] = survives ? 1 : 0;
}

__global__ void gather_dimension(const float* coords, const int* active_indices,
                                 int active_count, int dim, int dimension,
                                 float* values) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < active_count) values[index] = coords[active_indices[index] * dim + dimension];
}

__global__ void assign_masks(const float* coords, const int* active_indices,
                             int active_count, int dim, const float* quartiles,
                             unsigned int* median_masks,
                             unsigned int* quartile_masks,
                             unsigned long long* sort_keys) {
    const int active_position = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_position >= active_count) return;
    const int point_index = active_indices[active_position];
    unsigned int median_mask = 0;
    unsigned int quartile_mask = 0;
    for (int d = 0; d < dim; ++d) {
        const float value = coords[point_index * dim + d];
        const float q1 = quartiles[d * 3];
        const float q2 = quartiles[d * 3 + 1];
        const float q3 = quartiles[d * 3 + 2];
        const bool upper_half = value >= q2;
        const bool upper_quarter = value >= (upper_half ? q3 : q1);
        if (upper_half) median_mask |= 1U << d;
        if (upper_quarter) quartile_mask |= 1U << d;
    }
    median_masks[point_index] = median_mask;
    quartile_masks[point_index] = quartile_mask;
    const unsigned int order = __popc(median_mask);
    sort_keys[active_position] = (static_cast<unsigned long long>(order) << 32) | median_mask;
}

__device__ __forceinline__ bool dominates(const float* coords, int dim,
                                          int dominator, int target) {
    bool strict = false;
    for (int d = 0; d < dim; ++d) {
        const float left = coords[dominator * dim + d];
        const float right = coords[target * dim + d];
        if (left > right) return false;
        strict = strict || left < right;
    }
    return strict;
}

__global__ void process_level(const float* coords, int dim,
                              const unsigned int* median_masks,
                              const unsigned int* quartile_masks,
                              const int* active_indices, int active_count,
                              int level, const int* original_indices,
                              int* keep_for_next, int* skyline_indices,
                              int* skyline_count) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= active_count) return;
    const int point_index = active_indices[position];
    const unsigned int my_median = median_masks[point_index];
    const unsigned int my_quartile = quartile_masks[point_index];
    const int my_order = __popc(my_median);
    bool alive = true;

    for (int other_position = 0; other_position < active_count && alive; ++other_position) {
        if (other_position == position) continue;
        const int other_index = active_indices[other_position];
        const unsigned int other_median = median_masks[other_index];
        const int other_order = __popc(other_median);
        if (other_order != level) continue;
        const unsigned int other_quartile = quartile_masks[other_index];

        if (my_order == level) {
            if (other_median != my_median) continue;
            if ((other_quartile & ~my_quartile) != 0U) continue;
        } else {
            if ((other_median & ~my_median) != 0U) continue;
            const unsigned int equal_side_dimensions = other_median | ~my_median;
            if ((equal_side_dimensions & other_quartile & ~my_quartile) != 0U) continue;
        }
        if (dominates(coords, dim, other_index, point_index)) alive = false;
    }

    if (alive && my_order == level) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = original_indices[point_index];
    }
    keep_for_next[position] = alive && my_order > level ? 1 : 0;
}

}  // namespace

AlgorithmResult run_skyalign(const std::vector<MyDataPoint>& points,
                             const AlgorithmConfig& config) {
    AlgorithmResult result;
    result.name = "SkyAlign-R";
    result.provenance =
        "CUDA reimplementation of SkyAlign Algorithm 1 from the paper; "
        "author source was not publicly located";
    result.input_count = static_cast<int>(points.size());
    const int count = result.input_count;
    const int dim = config.dim;
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
    int* active = nullptr;
    int* next_active = nullptr;
    int* flags = nullptr;
    unsigned int* threshold_bits = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&coords, host_coords.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&original_indices, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&active, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&next_active, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&flags, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&threshold_bits, sizeof(unsigned int)));
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
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    initialize_indices<<<grid, block>>>(count, active);
    CHECK_CUDA_ERROR(cudaMemset(threshold_bits, 0xff, sizeof(unsigned int)));
    find_prefilter_threshold<<<grid, block>>>(coords, count, dim, threshold_bits);
    mark_prefilter_survivors<<<grid, block>>>(coords, count, dim, threshold_bits, flags);

    thrust::device_ptr<int> active_ptr(active);
    thrust::device_ptr<int> next_ptr(next_active);
    thrust::device_ptr<int> flag_ptr(flags);
    int active_count = static_cast<int>(thrust::copy_if(active_ptr, active_ptr + count,
                                                       flag_ptr, next_ptr, IsOne()) - next_ptr);
    std::swap(active, next_active);
    result.candidate_count = active_count;
    result.valid_count = active_count;

    float* dimension_values = nullptr;
    float* quartiles = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&dimension_values, static_cast<std::size_t>(active_count) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&quartiles, static_cast<std::size_t>(dim) * 3 * sizeof(float)));
    std::vector<float> host_quartiles(static_cast<std::size_t>(dim) * 3);
    const dim3 active_grid((active_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    for (int d = 0; d < dim; ++d) {
        gather_dimension<<<active_grid, block>>>(coords, active, active_count, dim, d, dimension_values);
        thrust::device_ptr<float> values_ptr(dimension_values);
        thrust::sort(values_ptr, values_ptr + active_count);
        const int quartile_positions[3] = {
            active_count / 4,
            active_count / 2,
            (3 * active_count) / 4,
        };
        for (int q = 0; q < 3; ++q) {
            CHECK_CUDA_ERROR(cudaMemcpy(&host_quartiles[static_cast<std::size_t>(d) * 3 + q],
                                        dimension_values + std::min(quartile_positions[q], active_count - 1),
                                        sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(quartiles, host_quartiles.data(),
                                host_quartiles.size() * sizeof(float), cudaMemcpyHostToDevice));

    unsigned int* median_masks = nullptr;
    unsigned int* quartile_masks = nullptr;
    unsigned long long* sort_keys = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&median_masks, static_cast<std::size_t>(count) * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&quartile_masks, static_cast<std::size_t>(count) * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&sort_keys, static_cast<std::size_t>(active_count) * sizeof(unsigned long long)));
    assign_masks<<<active_grid, block>>>(coords, active, active_count, dim, quartiles,
                                        median_masks, quartile_masks, sort_keys);
    thrust::device_ptr<unsigned long long> key_ptr(sort_keys);
    active_ptr = thrust::device_pointer_cast(active);
    thrust::sort_by_key(key_ptr, key_ptr + active_count, active_ptr);
    CHECK_CUDA_ERROR(cudaEventRecord(phase_event));

    int* skyline_indices = nullptr;
    int* skyline_count = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&skyline_indices, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&skyline_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(skyline_count, 0, sizeof(int)));

    for (int level = 0; level <= dim && active_count > 0; ++level) {
        const dim3 level_grid((active_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        process_level<<<level_grid, block>>>(coords, dim, median_masks, quartile_masks,
                                            active, active_count, level, original_indices,
                                            flags, skyline_indices, skyline_count);
        active_ptr = thrust::device_pointer_cast(active);
        next_ptr = thrust::device_pointer_cast(next_active);
        flag_ptr = thrust::device_pointer_cast(flags);
        const int next_count = static_cast<int>(thrust::copy_if(active_ptr, active_ptr + active_count,
                                                               flag_ptr, next_ptr, IsOne()) - next_ptr);
        std::swap(active, next_active);
        active_count = next_count;
    }

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
        count_size * dim * sizeof(float) + count_size * 4 * sizeof(int) + sizeof(unsigned int) +
        static_cast<std::size_t>(result.valid_count) * sizeof(float) +
        static_cast<std::size_t>(dim) * 3 * sizeof(float) + count_size * 2 * sizeof(unsigned int) +
        static_cast<std::size_t>(result.valid_count) * sizeof(unsigned long long) +
        count_size * sizeof(int) + sizeof(int);
    result.device_memory_mb = static_cast<double>(device_bytes) / (1024.0 * 1024.0);
    int output_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&output_count, skyline_count, sizeof(int), cudaMemcpyDeviceToHost));
    result.skyline_indices.resize(static_cast<std::size_t>(output_count));
    CHECK_CUDA_ERROR(cudaMemcpy(result.skyline_indices.data(), skyline_indices,
                                static_cast<std::size_t>(output_count) * sizeof(int), cudaMemcpyDeviceToHost));
    std::sort(result.skyline_indices.begin(), result.skyline_indices.end());

    cudaFree(skyline_count);
    cudaFree(skyline_indices);
    cudaFree(sort_keys);
    cudaFree(quartile_masks);
    cudaFree(median_masks);
    cudaFree(quartiles);
    cudaFree(dimension_values);
    cudaFree(threshold_bits);
    cudaFree(flags);
    cudaFree(next_active);
    cudaFree(active);
    cudaFree(original_indices);
    cudaFree(coords);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(phase_event);
    cudaEventDestroy(start_event);
    const auto wall_end = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    return result;
}
