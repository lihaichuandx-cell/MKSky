#include "algorithm.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <vector>

namespace {

__device__ __forceinline__ unsigned long long morton_encode(const float* coords,
                                                            int dim,
                                                            int bits_per_dim) {
    unsigned long long code = 0;
    const unsigned long long mask = (1ULL << bits_per_dim) - 1ULL;
    for (int d = 0; d < dim; ++d) {
        const float normalized = dev_max(0.0f, dev_min(1.0f - EPS, coords[d]));
        const unsigned long long quantized = static_cast<unsigned long long>(normalized * mask);
        for (int bit = 0; bit < bits_per_dim; ++bit) {
            if (quantized & (1ULL << bit)) code |= 1ULL << (bit * dim + d);
        }
    }
    return code;
}

__global__ void find_pivot(const float* coords, int count, int dim,
                           unsigned long long* minimum) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) sum += coords[d * count + index];
    const unsigned long long combined =
        (static_cast<unsigned long long>(__float_as_uint(sum)) << 32) |
        static_cast<unsigned int>(index);
    atomicMin(minimum, combined);
}

__global__ void filter_and_encode(const float* coords, int count, int dim,
                                  const unsigned long long* pivot_value,
                                  unsigned long long* morton_keys,
                                  int* sorted_indices, int bits_per_dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int pivot_index = static_cast<int>(*pivot_value & 0xffffffffULL);
    float point[MAX_DIM];
    bool dominated = index != pivot_index;
    bool strict = false;
    for (int d = 0; d < dim; ++d) {
        point[d] = coords[d * count + index];
        const float pivot_coordinate = coords[d * count + pivot_index];
        if (pivot_coordinate > point[d]) dominated = false;
        if (pivot_coordinate < point[d]) strict = true;
    }
    morton_keys[index] = dominated && strict
        ? ULLONG_MAX : morton_encode(point, dim, bits_per_dim);
    sorted_indices[index] = index;
}

__global__ void reorder_points(const float* source, int total_count, int dim,
                               const int* sorted_indices, int valid_count,
                               float* ordered, int* original_indices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= valid_count) return;
    const int source_index = sorted_indices[index];
    for (int d = 0; d < dim; ++d) {
        ordered[d * valid_count + index] = source[d * total_count + source_index];
    }
    original_indices[index] = source_index;
}

__global__ void local_prune(const float* ordered, int valid_count, int dim,
                            int chunk_size, int* candidates, int* candidate_count) {
    extern __shared__ float tile[];
    const int start = blockIdx.x * chunk_size;
    const int end = min(start + chunk_size, valid_count);
    const int tile_count = end - start;
    for (int local = threadIdx.x; local < tile_count; local += blockDim.x) {
        for (int d = 0; d < dim; ++d) {
            tile[local * dim + d] = ordered[d * valid_count + start + local];
        }
    }
    __syncthreads();

    for (int local = threadIdx.x; local < tile_count; local += blockDim.x) {
        bool alive = true;
        for (int other = 0; other < tile_count && alive; ++other) {
            if (other == local) continue;
            bool all_le = true;
            bool any_lt = false;
            for (int d = 0; d < dim; ++d) {
                const float other_value = tile[other * dim + d];
                const float value = tile[local * dim + d];
                if (other_value > value) {
                    all_le = false;
                    break;
                }
                any_lt = any_lt || other_value < value;
            }
            if (all_le && any_lt) alive = false;
        }
        if (alive) {
            const int output = atomicAdd(candidate_count, 1);
            candidates[output] = start + local;
        }
    }
}

__global__ void split_mkd_intervals(const unsigned long long* morton_keys,
                                    const int* input_starts, const int* input_ends,
                                    int input_count, int target_size,
                                    int* next_starts, int* next_ends, int* next_count,
                                    int* leaf_starts, int* leaf_ends, int* leaf_count,
                                    int capacity, int* overflow) {
    const int interval = blockIdx.x * blockDim.x + threadIdx.x;
    if (interval >= input_count) return;
    const int start = input_starts[interval];
    const int end = input_ends[interval];
    const int size = end - start;
    if (size <= target_size) {
        const int output = atomicAdd(leaf_count, 1);
        if (output < capacity) {
            leaf_starts[output] = start;
            leaf_ends[output] = end;
        } else {
            atomicExch(overflow, 1);
        }
        return;
    }

    const unsigned long long difference = morton_keys[start] ^ morton_keys[end - 1];
    int split = start + size / 2;
    if (difference != 0) {
        const int split_bit = 63 - __clzll(difference);
        int left = start;
        int right = end;
        while (left < right) {
            const int middle = left + (right - left) / 2;
            if ((morton_keys[middle] >> split_bit) & 1ULL) right = middle;
            else left = middle + 1;
        }
        split = left;
    }
    if (split <= start || split >= end) split = start + size / 2;

    const int output = atomicAdd(next_count, 2);
    if (output + 1 < capacity) {
        next_starts[output] = start;
        next_ends[output] = split;
        next_starts[output + 1] = split;
        next_ends[output + 1] = end;
    } else {
        atomicExch(overflow, 1);
    }
}

template <int DIM>
__global__ void local_prune_mkd_templated(const float* ordered, int valid_count,
                                          const int* chunk_starts,
                                          const int* chunk_ends,
                                          int* candidates, int* candidate_count) {
    extern __shared__ float tile[];
    const int start = chunk_starts[blockIdx.x];
    const int end = chunk_ends[blockIdx.x];
    const int tile_count = end - start;
    for (int local = threadIdx.x; local < tile_count; local += blockDim.x) {
#pragma unroll
        for (int d = 0; d < DIM; ++d) {
            tile[local * DIM + d] = ordered[d * valid_count + start + local];
        }
    }
    __syncthreads();

    for (int local = threadIdx.x; local < tile_count; local += blockDim.x) {
        float point[DIM];
#pragma unroll
        for (int d = 0; d < DIM; ++d) point[d] = tile[local * DIM + d];
        bool alive = true;
        for (int other = 0; other < tile_count && alive; ++other) {
            if (other == local) continue;
            bool all_le = true;
            bool any_lt = false;
#pragma unroll
            for (int d = 0; d < DIM; ++d) {
                const float other_value = tile[other * DIM + d];
                if (other_value > point[d]) {
                    all_le = false;
                    break;
                }
                any_lt = any_lt || other_value < point[d];
            }
            if (all_le && any_lt) alive = false;
        }
        if (alive) {
            const int output = atomicAdd(candidate_count, 1);
            candidates[output] = start + local;
        }
    }
}

template <int DIM>
__global__ void local_prune_templated(const float* ordered, int valid_count,
                                      int chunk_size, int* candidates,
                                      int* candidate_count) {
    extern __shared__ float tile[];
    const int start = blockIdx.x * chunk_size;
    const int end = min(start + chunk_size, valid_count);
    const int tile_count = end - start;
    for (int local = threadIdx.x; local < tile_count; local += blockDim.x) {
#pragma unroll
        for (int d = 0; d < DIM; ++d) {
            tile[local * DIM + d] = ordered[d * valid_count + start + local];
        }
    }
    __syncthreads();

    for (int local = threadIdx.x; local < tile_count; local += blockDim.x) {
        float point[DIM];
#pragma unroll
        for (int d = 0; d < DIM; ++d) point[d] = tile[local * DIM + d];
        bool alive = true;
        for (int other = 0; other < tile_count && alive; ++other) {
            if (other == local) continue;
            bool all_le = true;
            bool any_lt = false;
#pragma unroll
            for (int d = 0; d < DIM; ++d) {
                const float other_value = tile[other * DIM + d];
                if (other_value > point[d]) {
                    all_le = false;
                    break;
                }
                any_lt = any_lt || other_value < point[d];
            }
            if (all_le && any_lt) alive = false;
        }
        if (alive) {
            const int output = atomicAdd(candidate_count, 1);
            candidates[output] = start + local;
        }
    }
}

__global__ void pack_candidates(const float* ordered, int valid_count, int dim,
                                const int* candidate_positions, int candidate_count,
                                const int* ordered_original_indices,
                                float* packed, int* packed_original_indices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) return;
    const int position = candidate_positions[index];
    for (int d = 0; d < dim; ++d) {
        packed[index * dim + d] = ordered[d * valid_count + position];
    }
    packed_original_indices[index] = ordered_original_indices[position];
}

__global__ void compute_leaf_mins(const float* packed, int candidate_count, int dim,
                                  int leaf_size, float* leaf_mins) {
    const int leaf = blockIdx.x * blockDim.x + threadIdx.x;
    const int leaf_count = (candidate_count + leaf_size - 1) / leaf_size;
    if (leaf >= leaf_count) return;
    const int start = leaf * leaf_size;
    const int end = min(start + leaf_size, candidate_count);
    for (int d = 0; d < dim; ++d) {
        float minimum = FLT_MAX;
        for (int i = start; i < end; ++i) minimum = fminf(minimum, packed[i * dim + d]);
        leaf_mins[leaf * dim + d] = minimum;
    }
}

__global__ void reduce_mins(const float* child_mins, int child_count, int dim,
                            int fanout, float* parent_mins) {
    const int parent = blockIdx.x * blockDim.x + threadIdx.x;
    const int parent_count = (child_count + fanout - 1) / fanout;
    if (parent >= parent_count) return;
    const int start = parent * fanout;
    const int end = min(start + fanout, child_count);
    for (int d = 0; d < dim; ++d) {
        float minimum = FLT_MAX;
        for (int i = start; i < end; ++i) minimum = fminf(minimum, child_mins[i * dim + d]);
        parent_mins[parent * dim + d] = minimum;
    }
}

__device__ __forceinline__ bool summary_can_dominate(const float* summary,
                                                     const float* point, int dim) {
    for (int d = 0; d < dim; ++d) {
        if (summary[d] > point[d]) return false;
    }
    return true;
}

__global__ void hierarchical_verify(const float* packed, int candidate_count, int dim,
                                    int leaf_size, int fanout,
                                    const float* leaf_mins,
                                    const float* group_mins,
                                    const float* super_mins,
                                    const int* original_indices,
                                    int* skyline_indices, int* skyline_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) return;
    float point[MAX_DIM];
    for (int d = 0; d < dim; ++d) point[d] = packed[index * dim + d];

    const int leaf_count = (candidate_count + leaf_size - 1) / leaf_size;
    const int group_count = (leaf_count + fanout - 1) / fanout;
    const int preceding_leaf_limit = (index + leaf_size - 1) / leaf_size;
    bool alive = true;

    for (int super = 0; super * fanout * fanout < preceding_leaf_limit && alive; ++super) {
        if (!summary_can_dominate(super_mins + super * dim, point, dim)) continue;
        const int first_group = super * fanout;
        const int last_group = min(first_group + fanout, group_count);
        for (int group = first_group; group < last_group && alive; ++group) {
            if (group * fanout >= preceding_leaf_limit) break;
            if (!summary_can_dominate(group_mins + group * dim, point, dim)) continue;
            const int first_leaf = group * fanout;
            const int last_leaf = min(first_leaf + fanout, leaf_count);
            for (int leaf = first_leaf; leaf < last_leaf && alive; ++leaf) {
                if (leaf >= preceding_leaf_limit) break;
                if (!summary_can_dominate(leaf_mins + leaf * dim, point, dim)) continue;
                const int start = leaf * leaf_size;
                const int end = min(min(start + leaf_size, candidate_count), index);
                for (int other = start; other < end; ++other) {
                    bool all_le = true;
                    bool any_lt = false;
                    for (int d = 0; d < dim; ++d) {
                        const float other_value = packed[other * dim + d];
                        if (other_value > point[d]) {
                            all_le = false;
                            break;
                        }
                        any_lt = any_lt || other_value < point[d];
                    }
                    if (all_le && any_lt) {
                        alive = false;
                        break;
                    }
                }
            }
        }
    }

    if (alive) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = original_indices[index];
    }
}

template <int DIM>
__device__ __forceinline__ bool summary_can_dominate_templated(const float* summary,
                                                               const float* point) {
#pragma unroll
    for (int d = 0; d < DIM; ++d) {
        if (summary[d] > point[d]) return false;
    }
    return true;
}

template <int DIM>
__global__ void hierarchical_verify_templated(const float* packed, int candidate_count,
                                              int leaf_size, int fanout,
                                              const float* leaf_mins,
                                              const float* group_mins,
                                              const float* super_mins,
                                              const int* original_indices,
                                              int* skyline_indices,
                                              int* skyline_count,
                                              unsigned long long* verification_triggers,
                                              unsigned long long* false_triggers,
                                              unsigned long long* exact_point_checks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) return;
    float point[DIM];
#pragma unroll
    for (int d = 0; d < DIM; ++d) point[d] = packed[index * DIM + d];
    const int leaf_count = (candidate_count + leaf_size - 1) / leaf_size;
    const int group_count = (leaf_count + fanout - 1) / fanout;
    const int preceding_leaf_limit = (index + leaf_size - 1) / leaf_size;
    bool alive = true;

    for (int super = 0; super * fanout * fanout < preceding_leaf_limit && alive; ++super) {
        if (!summary_can_dominate_templated<DIM>(super_mins + super * DIM, point)) continue;
        const int first_group = super * fanout;
        const int last_group = min(first_group + fanout, group_count);
        for (int group = first_group; group < last_group && alive; ++group) {
            if (group * fanout >= preceding_leaf_limit) break;
            if (!summary_can_dominate_templated<DIM>(group_mins + group * DIM, point)) continue;
            const int first_leaf = group * fanout;
            const int last_leaf = min(first_leaf + fanout, leaf_count);
            for (int leaf = first_leaf; leaf < last_leaf && alive; ++leaf) {
                if (leaf >= preceding_leaf_limit) break;
                if (!summary_can_dominate_templated<DIM>(leaf_mins + leaf * DIM, point)) continue;
                if (verification_triggers) atomicAdd(verification_triggers, 1ULL);
                const int start = leaf * leaf_size;
                const int end = min(min(start + leaf_size, candidate_count), index);
                bool dominated_in_leaf = false;
                for (int other = start; other < end; ++other) {
                    if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
                    bool all_le = true;
                    bool any_lt = false;
#pragma unroll
                    for (int d = 0; d < DIM; ++d) {
                        const float other_value = packed[other * DIM + d];
                        if (other_value > point[d]) {
                            all_le = false;
                            break;
                        }
                        any_lt = any_lt || other_value < point[d];
                    }
                    if (all_le && any_lt) {
                        alive = false;
                        dominated_in_leaf = true;
                        break;
                    }
                }
                if (false_triggers && !dominated_in_leaf) atomicAdd(false_triggers, 1ULL);
            }
        }
    }
    if (alive) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = original_indices[index];
    }
}

template <int DIM>
__global__ void flat_verify_templated(const float* packed, int candidate_count,
                                      int leaf_size, const float* leaf_mins,
                                      const int* original_indices,
                                      int* skyline_indices, int* skyline_count,
                                      unsigned long long* verification_triggers,
                                      unsigned long long* false_triggers,
                                      unsigned long long* exact_point_checks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) return;
    float point[DIM];
#pragma unroll
    for (int d = 0; d < DIM; ++d) point[d] = packed[index * DIM + d];
    const int preceding_leaf_limit = (index + leaf_size - 1) / leaf_size;
    bool alive = true;
    for (int leaf = 0; leaf < preceding_leaf_limit && alive; ++leaf) {
        if (!summary_can_dominate_templated<DIM>(leaf_mins + leaf * DIM, point)) continue;
        if (verification_triggers) atomicAdd(verification_triggers, 1ULL);
        const int start = leaf * leaf_size;
        const int end = min(min(start + leaf_size, candidate_count), index);
        bool dominated_in_leaf = false;
        for (int other = start; other < end; ++other) {
            if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
            bool all_le = true;
            bool any_lt = false;
#pragma unroll
            for (int d = 0; d < DIM; ++d) {
                const float other_value = packed[other * DIM + d];
                if (other_value > point[d]) {
                    all_le = false;
                    break;
                }
                any_lt = any_lt || other_value < point[d];
            }
            if (all_le && any_lt) {
                alive = false;
                dominated_in_leaf = true;
                break;
            }
        }
        if (false_triggers && !dominated_in_leaf) atomicAdd(false_triggers, 1ULL);
    }
    if (alive) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = original_indices[index];
    }
}

int select_leaf_size(int candidate_count, int dim) {
    int leaf_size = dim <= 4 ? 512 : (dim <= 8 ? 256 : (dim <= 12 ? 128 : 64));
    if (candidate_count < 4096) leaf_size = 64;
    else if (candidate_count < 32768) leaf_size = std::min(leaf_size, 128);
    return leaf_size;
}

int select_local_chunk_size(int input_count, int valid_count, int dim) {
    const double survivor_ratio = static_cast<double>(valid_count) /
                                  static_cast<double>(input_count);
    if (input_count >= 100000 && survivor_ratio >= 0.20) {
        if (dim <= 6) return 512;
        if (dim <= 10) return 512;
    }
    return dim >= 12 ? 512 : 1024;
}

struct MkdChunks {
    int* starts = nullptr;
    int* ends = nullptr;
    int count = 0;
    int capacity = 0;
    std::size_t peak_working_bytes = 0;
};

MkdChunks build_mkd_chunks(const unsigned long long* morton_keys, int valid_count,
                           int target_size) {
    MkdChunks result;
    if (valid_count <= 0) return result;
    int capacity = std::min(valid_count, std::max(1024,
        16 * ((valid_count + target_size - 1) / target_size) + 64));

    for (;;) {
        int* queue_starts = nullptr;
        int* queue_ends = nullptr;
        int* next_starts = nullptr;
        int* next_ends = nullptr;
        int* leaf_starts = nullptr;
        int* leaf_ends = nullptr;
        int* next_count_device = nullptr;
        int* leaf_count_device = nullptr;
        int* overflow_device = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc(&queue_starts, static_cast<std::size_t>(capacity) * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&queue_ends, static_cast<std::size_t>(capacity) * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&next_starts, static_cast<std::size_t>(capacity) * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&next_ends, static_cast<std::size_t>(capacity) * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&leaf_starts, static_cast<std::size_t>(capacity) * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&leaf_ends, static_cast<std::size_t>(capacity) * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&next_count_device, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&leaf_count_device, sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&overflow_device, sizeof(int)));
        const int zero = 0;
        const int initial_start = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(queue_starts, &initial_start, sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(queue_ends, &valid_count, sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(leaf_count_device, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(overflow_device, &zero, sizeof(int), cudaMemcpyHostToDevice));

        int queue_count = 1;
        bool overflow = false;
        while (queue_count > 0) {
            CHECK_CUDA_ERROR(cudaMemset(next_count_device, 0, sizeof(int)));
            split_mkd_intervals<<<(queue_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                morton_keys, queue_starts, queue_ends, queue_count, target_size,
                next_starts, next_ends, next_count_device,
                leaf_starts, leaf_ends, leaf_count_device, capacity, overflow_device);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaMemcpy(&queue_count, next_count_device, sizeof(int),
                                        cudaMemcpyDeviceToHost));
            int overflow_value = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&overflow_value, overflow_device, sizeof(int),
                                        cudaMemcpyDeviceToHost));
            if (overflow_value != 0 || queue_count > capacity) {
                overflow = true;
                break;
            }
            std::swap(queue_starts, next_starts);
            std::swap(queue_ends, next_ends);
        }

        int leaf_count = 0;
        if (!overflow) {
            CHECK_CUDA_ERROR(cudaMemcpy(&leaf_count, leaf_count_device, sizeof(int),
                                        cudaMemcpyDeviceToHost));
            if (leaf_count > capacity) overflow = true;
        }

        cudaFree(overflow_device);
        cudaFree(leaf_count_device);
        cudaFree(next_count_device);
        cudaFree(next_ends);
        cudaFree(next_starts);
        cudaFree(queue_ends);
        cudaFree(queue_starts);

        if (!overflow) {
            result.starts = leaf_starts;
            result.ends = leaf_ends;
            result.count = leaf_count;
            result.capacity = capacity;
            result.peak_working_bytes = static_cast<std::size_t>(capacity) * 6 * sizeof(int) +
                                        3 * sizeof(int);
            return result;
        }
        cudaFree(leaf_ends);
        cudaFree(leaf_starts);
        if (capacity == valid_count) {
            std::fprintf(stderr, "MKD interval capacity exhausted\n");
            std::exit(EXIT_FAILURE);
        }
        capacity = std::min(valid_count, capacity * 4);
    }
}

}  // namespace

static AlgorithmResult run_mksky_experiment(const std::vector<MyDataPoint>& points,
                                             const AlgorithmConfig& config,
                                             bool use_mkd_partition,
                                             bool use_adaptive_mkd_chunk_size,
                                             bool use_flat_verification,
                                             bool use_fixed_chunk_ablation) {
    AlgorithmResult result;
    if (use_flat_verification) {
        result.name = "MKSky-mkd-flat";
        result.provenance = "MKSky-mkd ablation without the group/super-level summary hierarchy";
    } else if (use_fixed_chunk_ablation) {
        result.name = "MKSky-mkd-fixed";
        result.provenance = "MKSky-mkd ablation using equal-size local chunks instead of endpoint-XOR boundaries";
    } else if (!use_mkd_partition) {
        result.name = "MKSky-adaptive";
        result.provenance = "MKSky with adaptive local chunks and a three-level Min-Corner hierarchy";
    } else if (use_adaptive_mkd_chunk_size) {
        result.name = "MKSky-adaptive-mkd";
        result.provenance = "MKSky with adaptive MKD chunk limit and endpoint-XOR boundaries";
    } else {
        result.name = "MKSky-mkd";
        result.provenance = "MKSky with endpoint-XOR MKD intervals and binary boundary location";
    }
    result.input_count = static_cast<int>(points.size());
    const int count = result.input_count;
    const int dim = config.dim;
    const std::size_t count_size = static_cast<std::size_t>(count);
    const std::size_t initial_device_bytes =
        count_size * dim * sizeof(float) +
        count_size * sizeof(unsigned long long) +
        count_size * sizeof(int) + sizeof(unsigned long long);
    std::size_t peak_device_bytes = initial_device_bytes;
    const auto wall_begin = std::chrono::high_resolution_clock::now();

    std::vector<float> host_coords(static_cast<std::size_t>(count) * dim);
    for (int d = 0; d < dim; ++d) {
        for (int i = 0; i < count; ++i) host_coords[static_cast<std::size_t>(d) * count + i] = points[i].coords[d];
    }

    float* device_coords = nullptr;
    unsigned long long* morton_keys = nullptr;
    int* sorted_indices = nullptr;
    unsigned long long* pivot = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&device_coords, host_coords.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&morton_keys, static_cast<std::size_t>(count) * sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMalloc(&sorted_indices, static_cast<std::size_t>(count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&pivot, sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMemcpy(device_coords, host_coords.data(), host_coords.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start_event;
    cudaEvent_t phase_event;
    cudaEvent_t stop_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&phase_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));

    CHECK_CUDA_ERROR(cudaMemset(pivot, 0xff, sizeof(unsigned long long)));
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    find_pivot<<<grid, block>>>(device_coords, count, dim, pivot);
    const int bits_per_dim = std::min(20, 64 / dim);
    filter_and_encode<<<grid, block>>>(device_coords, count, dim, pivot,
                                      morton_keys, sorted_indices, bits_per_dim);
    thrust::device_ptr<unsigned long long> key_ptr(morton_keys);
    thrust::device_ptr<int> index_ptr(sorted_indices);
    thrust::sort_by_key(key_ptr, key_ptr + count, index_ptr);

    unsigned long long tail = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&tail, morton_keys + count - 1,
                                sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    int valid_count = count;
    if (tail == ULLONG_MAX) {
        int left = 0;
        int right = count;
        while (left < right) {
            const int middle = left + (right - left) / 2;
            unsigned long long probe = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&probe, morton_keys + middle,
                                        sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            if (probe == ULLONG_MAX) right = middle;
            else left = middle + 1;
        }
        valid_count = left;
    }
    result.valid_count = valid_count;

    float* ordered = nullptr;
    int* ordered_indices = nullptr;
    int* candidate_positions = nullptr;
    int* candidate_counter = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&ordered, static_cast<std::size_t>(valid_count) * dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&ordered_indices, static_cast<std::size_t>(valid_count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&candidate_positions, static_cast<std::size_t>(valid_count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&candidate_counter, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(candidate_counter, 0, sizeof(int)));
    const dim3 valid_grid((valid_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    reorder_points<<<valid_grid, block>>>(device_coords, count, dim, sorted_indices, valid_count,
                                          ordered, ordered_indices);

    const int local_chunk_size = (use_mkd_partition && !use_adaptive_mkd_chunk_size) ||
                                 use_fixed_chunk_ablation
        ? (dim >= 12 ? 512 : 1024)
        : select_local_chunk_size(count, valid_count, dim);
    result.adaptive_local_chunk_size = local_chunk_size;
    MkdChunks mkd_chunks;
    const int local_chunk_count = use_mkd_partition
        ? (mkd_chunks = build_mkd_chunks(morton_keys, valid_count, local_chunk_size),
           mkd_chunks.count)
        : (valid_count + local_chunk_size - 1) / local_chunk_size;
    result.mkd_chunk_count = use_mkd_partition ? local_chunk_count : 0;
    const std::size_t ordered_stage_bytes = initial_device_bytes +
        static_cast<std::size_t>(valid_count) * dim * sizeof(float) +
        static_cast<std::size_t>(valid_count) * 2 * sizeof(int) + sizeof(int);
    peak_device_bytes = std::max(peak_device_bytes, ordered_stage_bytes +
        (use_mkd_partition ? mkd_chunks.peak_working_bytes : 0));
    const std::size_t local_shared_bytes =
        static_cast<std::size_t>(local_chunk_size) * dim * sizeof(float);
#define LAUNCH_LOCAL(D) local_prune_templated<D><<<local_chunk_count, block, local_shared_bytes>>>( \
        ordered, valid_count, local_chunk_size, candidate_positions, candidate_counter)
#define LAUNCH_LOCAL_MKD(D) local_prune_mkd_templated<D><<<local_chunk_count, block, local_shared_bytes>>>( \
        ordered, valid_count, mkd_chunks.starts, mkd_chunks.ends, candidate_positions, candidate_counter)
    switch (dim) {
    case 2: if (use_mkd_partition) LAUNCH_LOCAL_MKD(2); else LAUNCH_LOCAL(2); break;
    case 3: if (use_mkd_partition) LAUNCH_LOCAL_MKD(3); else LAUNCH_LOCAL(3); break;
    case 4: if (use_mkd_partition) LAUNCH_LOCAL_MKD(4); else LAUNCH_LOCAL(4); break;
    case 5: if (use_mkd_partition) LAUNCH_LOCAL_MKD(5); else LAUNCH_LOCAL(5); break;
    case 6: if (use_mkd_partition) LAUNCH_LOCAL_MKD(6); else LAUNCH_LOCAL(6); break;
    case 7: if (use_mkd_partition) LAUNCH_LOCAL_MKD(7); else LAUNCH_LOCAL(7); break;
    case 8: if (use_mkd_partition) LAUNCH_LOCAL_MKD(8); else LAUNCH_LOCAL(8); break;
    case 9: if (use_mkd_partition) LAUNCH_LOCAL_MKD(9); else LAUNCH_LOCAL(9); break;
    case 10: if (use_mkd_partition) LAUNCH_LOCAL_MKD(10); else LAUNCH_LOCAL(10); break;
    case 11: if (use_mkd_partition) LAUNCH_LOCAL_MKD(11); else LAUNCH_LOCAL(11); break;
    case 12: if (use_mkd_partition) LAUNCH_LOCAL_MKD(12); else LAUNCH_LOCAL(12); break;
    case 13: if (use_mkd_partition) LAUNCH_LOCAL_MKD(13); else LAUNCH_LOCAL(13); break;
    case 14: if (use_mkd_partition) LAUNCH_LOCAL_MKD(14); else LAUNCH_LOCAL(14); break;
    case 15: if (use_mkd_partition) LAUNCH_LOCAL_MKD(15); else LAUNCH_LOCAL(15); break;
    case 16: if (use_mkd_partition) LAUNCH_LOCAL_MKD(16); else LAUNCH_LOCAL(16); break;
    }
#undef LAUNCH_LOCAL_MKD
#undef LAUNCH_LOCAL
    CHECK_CUDA_ERROR(cudaGetLastError());
    int candidate_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&candidate_count, candidate_counter, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(mkd_chunks.ends);
    cudaFree(mkd_chunks.starts);
    result.candidate_count = candidate_count;
    thrust::device_ptr<int> candidate_ptr(candidate_positions);
    thrust::sort(candidate_ptr, candidate_ptr + candidate_count);

    float* packed = nullptr;
    int* packed_indices = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&packed, static_cast<std::size_t>(candidate_count) * dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&packed_indices, static_cast<std::size_t>(candidate_count) * sizeof(int)));
    const dim3 candidate_grid((candidate_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    pack_candidates<<<candidate_grid, block>>>(ordered, valid_count, dim, candidate_positions,
                                               candidate_count, ordered_indices, packed, packed_indices);

    const int leaf_size = select_leaf_size(candidate_count, dim);
    const int fanout = 8;
    result.adaptive_leaf_size = leaf_size;
    float* leaf_mins = nullptr;
    float* group_mins = nullptr;
    float* super_mins = nullptr;
    const int leaf_count = (candidate_count + leaf_size - 1) / leaf_size;
    const int group_count = (leaf_count + fanout - 1) / fanout;
    const int super_count = (group_count + fanout - 1) / fanout;
    CHECK_CUDA_ERROR(cudaMalloc(&leaf_mins,
        static_cast<std::size_t>(leaf_count) * dim * sizeof(float)));
    compute_leaf_mins<<<(leaf_count + BLOCK_SIZE - 1) / BLOCK_SIZE, block>>>(
        packed, candidate_count, dim, leaf_size, leaf_mins);
    if (!use_flat_verification) {
        CHECK_CUDA_ERROR(cudaMalloc(&group_mins,
            static_cast<std::size_t>(group_count) * dim * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&super_mins,
            static_cast<std::size_t>(super_count) * dim * sizeof(float)));
        reduce_mins<<<(group_count + BLOCK_SIZE - 1) / BLOCK_SIZE, block>>>(
            leaf_mins, leaf_count, dim, fanout, group_mins);
        reduce_mins<<<(super_count + BLOCK_SIZE - 1) / BLOCK_SIZE, block>>>(
            group_mins, group_count, dim, fanout, super_mins);
    }

    int* skyline_indices = nullptr;
    int* skyline_count = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&skyline_indices, static_cast<std::size_t>(candidate_count) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&skyline_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(skyline_count, 0, sizeof(int)));
    unsigned long long* verification_triggers = nullptr;
    unsigned long long* false_triggers = nullptr;
    unsigned long long* exact_point_checks = nullptr;
    if (config.collect_telemetry) {
        CHECK_CUDA_ERROR(cudaMalloc(&verification_triggers, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMalloc(&false_triggers, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMalloc(&exact_point_checks, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMemset(verification_triggers, 0, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMemset(false_triggers, 0, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMemset(exact_point_checks, 0, sizeof(unsigned long long)));
    }
    const std::size_t hierarchy_bytes =
        static_cast<std::size_t>(leaf_count) * dim * sizeof(float) +
        (use_flat_verification ? 0 :
            (static_cast<std::size_t>(group_count) + static_cast<std::size_t>(super_count)) * dim * sizeof(float));
    const std::size_t final_stage_bytes = ordered_stage_bytes +
        static_cast<std::size_t>(candidate_count) * dim * sizeof(float) +
        static_cast<std::size_t>(candidate_count) * 2 * sizeof(int) +
        hierarchy_bytes + static_cast<std::size_t>(candidate_count) * sizeof(int) + sizeof(int) +
        (config.collect_telemetry ? 3 * sizeof(unsigned long long) : 0);
    peak_device_bytes = std::max(peak_device_bytes, final_stage_bytes);
    result.device_memory_mb = static_cast<double>(peak_device_bytes) / (1024.0 * 1024.0);
    CHECK_CUDA_ERROR(cudaEventRecord(phase_event));
#define LAUNCH_VERIFY(D) hierarchical_verify_templated<D><<<candidate_grid, block>>>( \
            packed, candidate_count, leaf_size, fanout, leaf_mins, group_mins, super_mins, \
            packed_indices, skyline_indices, skyline_count, verification_triggers, false_triggers, exact_point_checks)
#define LAUNCH_FLAT_VERIFY(D) flat_verify_templated<D><<<candidate_grid, block>>>( \
            packed, candidate_count, leaf_size, leaf_mins, packed_indices, skyline_indices, skyline_count, \
            verification_triggers, false_triggers, exact_point_checks)
    switch (dim) {
    case 2: if (use_flat_verification) LAUNCH_FLAT_VERIFY(2); else LAUNCH_VERIFY(2); break;
    case 3: if (use_flat_verification) LAUNCH_FLAT_VERIFY(3); else LAUNCH_VERIFY(3); break;
    case 4: if (use_flat_verification) LAUNCH_FLAT_VERIFY(4); else LAUNCH_VERIFY(4); break;
    case 5: if (use_flat_verification) LAUNCH_FLAT_VERIFY(5); else LAUNCH_VERIFY(5); break;
    case 6: if (use_flat_verification) LAUNCH_FLAT_VERIFY(6); else LAUNCH_VERIFY(6); break;
    case 7: if (use_flat_verification) LAUNCH_FLAT_VERIFY(7); else LAUNCH_VERIFY(7); break;
    case 8: if (use_flat_verification) LAUNCH_FLAT_VERIFY(8); else LAUNCH_VERIFY(8); break;
    case 9: if (use_flat_verification) LAUNCH_FLAT_VERIFY(9); else LAUNCH_VERIFY(9); break;
    case 10: if (use_flat_verification) LAUNCH_FLAT_VERIFY(10); else LAUNCH_VERIFY(10); break;
    case 11: if (use_flat_verification) LAUNCH_FLAT_VERIFY(11); else LAUNCH_VERIFY(11); break;
    case 12: if (use_flat_verification) LAUNCH_FLAT_VERIFY(12); else LAUNCH_VERIFY(12); break;
    case 13: if (use_flat_verification) LAUNCH_FLAT_VERIFY(13); else LAUNCH_VERIFY(13); break;
    case 14: if (use_flat_verification) LAUNCH_FLAT_VERIFY(14); else LAUNCH_VERIFY(14); break;
    case 15: if (use_flat_verification) LAUNCH_FLAT_VERIFY(15); else LAUNCH_VERIFY(15); break;
    case 16: if (use_flat_verification) LAUNCH_FLAT_VERIFY(16); else LAUNCH_VERIFY(16); break;
    }
#undef LAUNCH_FLAT_VERIFY
#undef LAUNCH_VERIFY
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
    float elapsed_ms = 0.0f;
    float preprocess_ms = 0.0f;
    float core_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&preprocess_ms, start_event, phase_event));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&core_ms, phase_event, stop_event));
    result.device_ms = elapsed_ms;
    result.preprocess_ms = preprocess_ms;
    result.core_ms = core_ms;
    if (config.collect_telemetry) {
        CHECK_CUDA_ERROR(cudaMemcpy(&result.verification_triggers, verification_triggers,
                                    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&result.false_triggers, false_triggers,
                                    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&result.exact_point_checks, exact_point_checks,
                                    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for (int index = 0; index < candidate_count; ++index) {
            result.theoretical_block_checks +=
                static_cast<unsigned long long>((index + leaf_size - 1) / leaf_size);
        }
    }

    int output_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&output_count, skyline_count, sizeof(int), cudaMemcpyDeviceToHost));
    result.skyline_indices.resize(static_cast<std::size_t>(output_count));
    CHECK_CUDA_ERROR(cudaMemcpy(result.skyline_indices.data(), skyline_indices,
                                static_cast<std::size_t>(output_count) * sizeof(int), cudaMemcpyDeviceToHost));
    std::sort(result.skyline_indices.begin(), result.skyline_indices.end());

    cudaFree(exact_point_checks);
    cudaFree(false_triggers);
    cudaFree(verification_triggers);
    cudaFree(skyline_count);
    cudaFree(skyline_indices);
    cudaFree(super_mins);
    cudaFree(group_mins);
    cudaFree(leaf_mins);
    cudaFree(packed_indices);
    cudaFree(packed);
    cudaFree(candidate_counter);
    cudaFree(candidate_positions);
    cudaFree(ordered_indices);
    cudaFree(ordered);
    cudaFree(pivot);
    cudaFree(sorted_indices);
    cudaFree(morton_keys);
    cudaFree(device_coords);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(phase_event);
    cudaEventDestroy(start_event);
    const auto wall_end = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    return result;
}

AlgorithmResult run_adaptive_mksky(const std::vector<MyDataPoint>& points,
                                   const AlgorithmConfig& config) {
    return run_mksky_experiment(points, config, false, false, false, false);
}

AlgorithmResult run_mkd_mksky(const std::vector<MyDataPoint>& points,
                              const AlgorithmConfig& config) {
    return run_mksky_experiment(points, config, true, false, false, false);
}

AlgorithmResult run_adaptive_mkd_mksky(const std::vector<MyDataPoint>& points,
                                       const AlgorithmConfig& config) {
    return run_mksky_experiment(points, config, true, true, false, false);
}

AlgorithmResult run_mkd_flat_mksky(const std::vector<MyDataPoint>& points,
                                   const AlgorithmConfig& config) {
    return run_mksky_experiment(points, config, true, false, true, false);
}

AlgorithmResult run_mkd_fixed_mksky(const std::vector<MyDataPoint>& points,
                                    const AlgorithmConfig& config) {
    return run_mksky_experiment(points, config, false, false, false, true);
}
