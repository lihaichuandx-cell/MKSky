#include "algorithm.h"

#include <cuda_fp16.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <vector>

namespace {

enum class PaperVariant {
    Full,
    GridMkdNoDirect,
    RoutingDisabled,
    MkdWithoutSkipping,
    FixedPartition,
    FixedWithoutSkipping,
    MortonSfs,
    SumOrder,
    SumOrderWithoutSkipping,
    GridMkd,
    GridMkdProjectionBound,
    GridMkdProjection3D,
    GridMkdProjectionScan,
    GridMkdProjectionCycle,
    GridMkdProjectionQuantizedCycle
};

constexpr int PROJECTION_PAIR_COUNT = 8;
constexpr int PROJECTION_CYCLE_PAIR_COUNT = MAX_DIM;
constexpr int PROJECTION_BIN_COUNT = 64;
constexpr int PROJECTION_BUILD_THRESHOLD = 65536;
constexpr int MICRO_CHUNK_SIZE = 8;
constexpr int MICRO_CODE_BITS = 8;
constexpr unsigned int MICRO_CODE_MAX = (1U << MICRO_CODE_BITS) - 1U;
constexpr int PACKED_MICRO_MIN_DIM = 10;
constexpr int MICRO_POINT_WORDS = 1;
constexpr unsigned int MICRO_POINT_GUARDS = 0x88888888U;
constexpr int MICRO_POINT_MASK_MIN_DIM = 10;
constexpr int SIGNATURE_MIN_DIM = 14;
constexpr int COMPATIBILITY_GROUP_SIZE = 8;
constexpr int COMPATIBILITY_MIN_CHUNKS = 2048;
constexpr std::size_t MAX_COMPATIBILITY_MASK_BYTES = 128ULL * 1024ULL * 1024ULL;
constexpr bool ENABLE_PAIR_SUMMARY = false;
constexpr int PROJECTION_3D_SIDE = 16;
constexpr int PROJECTION_3D_CELL_COUNT = PROJECTION_3D_SIDE * PROJECTION_3D_SIDE;

int greatest_common_divisor(int left, int right) {
    while (right != 0) {
        const int remainder = left % right;
        left = right;
        right = remainder;
    }
    return left < 0 ? -left : left;
}

int choose_sample_stride(int count) {
    if (count <= 1) return 1;
    int stride = static_cast<int>(2654435761ULL % static_cast<unsigned long long>(count));
    if (stride <= 0) stride = 1;
    while (greatest_common_divisor(stride, count) != 1) {
        ++stride;
        if (stride >= count) stride = 1;
    }
    return stride;
}
constexpr unsigned int PROJECTION_QUANTIZATION_MAX = 65535U;

struct ChunkLayout {
    int* starts = nullptr;
    int* ends = nullptr;
    void* storage = nullptr;
    int count = 0;
    int capacity = 0;
    std::size_t working_bytes = 0;
};

struct KeepMarkedPoint {
    __host__ __device__ bool operator()(int value) const { return value != 0; }
};

__global__ void reduce_pivot_blocks(const float* coordinates, int count, int dim,
                                    float* block_sums, int* block_indices) {
    __shared__ float sums[BLOCK_SIZE];
    __shared__ int indices[BLOCK_SIZE];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = FLT_MAX;
    int point_index = INT_MAX;
    if (index < count) {
        sum = 0.0f;
        for (int d = 0; d < dim; ++d) sum += coordinates[d * count + index];
        point_index = index;
    }
    sums[threadIdx.x] = sum;
    indices[threadIdx.x] = point_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_sum = sums[threadIdx.x + stride];
            const int other_index = indices[threadIdx.x + stride];
            if (other_sum < sums[threadIdx.x] ||
                (other_sum == sums[threadIdx.x] && other_index < indices[threadIdx.x])) {
                sums[threadIdx.x] = other_sum;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sums[0];
        block_indices[blockIdx.x] = indices[0];
    }
}

__global__ void finalize_pivot(const float* block_sums, const int* block_indices,
                               int block_count, int* pivot_index) {
    __shared__ float sums[BLOCK_SIZE];
    __shared__ int indices[BLOCK_SIZE];
    float best_sum = FLT_MAX;
    int best_index = INT_MAX;
    for (int i = threadIdx.x; i < block_count; i += blockDim.x) {
        const float sum = block_sums[i];
        const int index = block_indices[i];
        if (sum < best_sum || (sum == best_sum && index < best_index)) {
            best_sum = sum;
            best_index = index;
        }
    }
    sums[threadIdx.x] = best_sum;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_sum = sums[threadIdx.x + stride];
            const int other_index = indices[threadIdx.x + stride];
            if (other_sum < sums[threadIdx.x] ||
                (other_sum == sums[threadIdx.x] && other_index < indices[threadIdx.x])) {
                sums[threadIdx.x] = other_sum;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *pivot_index = indices[0];
}

__device__ __forceinline__ unsigned long long encode_morton(
    const float* point, const float* minimums, const float* maximums,
    int dim, int bits_per_dim) {
    unsigned long long code = 0;
    const unsigned long long mask = (1ULL << bits_per_dim) - 1ULL;
    for (int d = 0; d < dim; ++d) {
        const float range = maximums[d] - minimums[d];
        float normalized = range > EPS ? (point[d] - minimums[d]) / range : 0.0f;
        normalized = dev_max(0.0f, dev_min(1.0f, normalized));
        const unsigned long long quantized =
            static_cast<unsigned long long>(normalized * static_cast<float>(mask));
        for (int bit = 0; bit < bits_per_dim; ++bit) {
            if (quantized & (1ULL << bit)) {
                code |= 1ULL << (bit * dim + d);
            }
        }
    }
    return code == ULLONG_MAX ? ULLONG_MAX - 1ULL : code;
}

__global__ void mark_pivot_survivors(const float* coordinates, int count, int dim,
                                     const int* pivot_index_device,
                                     const int* prefilter_keep,
                                     int* survivor_flags) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (prefilter_keep && !prefilter_keep[index]) {
        survivor_flags[index] = 0;
        return;
    }
    const int pivot_index = *pivot_index_device;
    bool dominated = index != pivot_index;
    bool strict = false;
    for (int d = 0; d < dim; ++d) {
        const float coordinate = coordinates[d * count + index];
        const float pivot_coordinate = coordinates[d * count + pivot_index];
        if (pivot_coordinate > coordinate) dominated = false;
        strict = strict || pivot_coordinate < coordinate;
    }
    survivor_flags[index] = dominated && strict ? 0 : 1;
}

__global__ void estimate_sample_pivot(const float* coordinates, int count, int dim,
                                      int sample_count, int sample_stride,
                                      int sample_offset, int* pivot_index_device,
                                      int* survivor_count_device) {
    __shared__ float sums[BLOCK_SIZE];
    __shared__ int indices[BLOCK_SIZE];
    __shared__ int counts[BLOCK_SIZE];
    __shared__ int pivot_index;
    float best_sum = FLT_MAX;
    int best_index = 0;
    for (int sample = threadIdx.x; sample < sample_count; sample += BLOCK_SIZE) {
        const int index = static_cast<int>(
            (static_cast<long long>(sample) * sample_stride + sample_offset) % count);
        float sum = 0.0f;
        for (int d = 0; d < dim; ++d) sum += coordinates[d * count + index];
        if (sum < best_sum || (sum == best_sum && index < best_index)) {
            best_sum = sum;
            best_index = index;
        }
    }
    sums[threadIdx.x] = best_sum;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const float other_sum = sums[threadIdx.x + offset];
            const int other_index = indices[threadIdx.x + offset];
            if (other_sum < sums[threadIdx.x] ||
                (other_sum == sums[threadIdx.x] &&
                 other_index < indices[threadIdx.x])) {
                sums[threadIdx.x] = other_sum;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pivot_index = indices[0];
        *pivot_index_device = pivot_index;
    }
    __syncthreads();
    int survivor_count = 0;
    for (int sample = threadIdx.x; sample < sample_count; sample += BLOCK_SIZE) {
        const int index = static_cast<int>(
            (static_cast<long long>(sample) * sample_stride + sample_offset) % count);
        bool dominated = index != pivot_index;
        bool strict = false;
        for (int d = 0; d < dim; ++d) {
            const float value = coordinates[d * count + index];
            const float pivot_value = coordinates[d * count + pivot_index];
            if (pivot_value > value) dominated = false;
            strict = strict || pivot_value < value;
        }
        survivor_count += dominated && strict ? 0 : 1;
    }
    counts[threadIdx.x] = survivor_count;
    __syncthreads();
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) counts[threadIdx.x] += counts[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) *survivor_count_device = counts[0];
}

__global__ void assign_grid_cells_soa(
    const float* coordinates, int count, int dim, int side,
    const float* minimums, const float* maximums,
    unsigned int* cell_ids, int* occupancy) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int cell_id = 0;
    unsigned int stride = 1;
    for (int d = 0; d < dim; ++d) {
        const float range = maximums[d] - minimums[d];
        const float value = coordinates[d * count + index];
        const float normalized = range > EPS ? (value - minimums[d]) / range : 0.0f;
        int coordinate = static_cast<int>(normalized * side);
        coordinate = max(0, min(side - 1, coordinate));
        cell_id += static_cast<unsigned int>(coordinate) * stride;
        stride *= static_cast<unsigned int>(side);
    }
    cell_ids[index] = cell_id;
    occupancy[cell_id] = 1;
}

__global__ void assign_grid_cells_normalized_soa(
    const float* coordinates, int count, int dim, int side,
    unsigned int* cell_ids, int* occupancy) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int cell_id = 0;
    unsigned int stride = 1;
    for (int d = 0; d < dim; ++d) {
        const float value = coordinates[d * count + index];
        int coordinate = static_cast<int>(value * side);
        coordinate = max(0, min(side - 1, coordinate));
        cell_id += static_cast<unsigned int>(coordinate) * stride;
        stride *= static_cast<unsigned int>(side);
    }
    cell_ids[index] = cell_id;
    occupancy[cell_id] = 1;
}

__global__ void grid_prefix_scan_step(const int* input, int* output, int cell_count,
                                      int side, int stride, int offset) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cell_count) return;
    const int coordinate = (index / stride) % side;
    int value = input[index];
    if (coordinate >= offset) value += input[index - offset * stride];
    output[index] = value;
}

__global__ void mark_grid_candidates(const unsigned int* cell_ids,
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

int select_grid_side(int count, int dim) {
    const int maximum_rho = std::min(7, 22 / dim);
    const double target_cells = std::max(256.0, std::min(4194304.0, count * 2.0));
    int rho = 1;
    while (rho < maximum_rho && std::pow(2.0, (rho + 1) * dim) <= target_cells) ++rho;
    return 1 << rho;
}

__global__ void compact_survivor_indices(const int* survivor_flags,
                                          const int* survivor_offsets,
                                          int count, int* survivor_indices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || !survivor_flags[index]) return;
    survivor_indices[survivor_offsets[index]] = index;
}

__global__ void encode_compacted_survivors(const float* coordinates, int source_count,
                                            int dim, const int* survivor_indices,
                                            int survivor_count, const float* minimums,
                                            const float* maximums,
                                            unsigned long long* morton_keys,
                                            int bits_per_dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= survivor_count) return;
    const int source_index = survivor_indices[index];
    float point[MAX_DIM];
    for (int d = 0; d < dim; ++d) {
        point[d] = coordinates[d * source_count + source_index];
    }
    morton_keys[index] = encode_morton(point, minimums, maximums, dim, bits_per_dim);
}

__global__ void compute_compacted_coordinate_sums(
    const float* coordinates, int source_count, int dim,
    const int* survivor_indices, int survivor_count, float* sums) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= survivor_count) return;
    const int source_index = survivor_indices[index];
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) sum += coordinates[d * source_count + source_index];
    sums[index] = sum;
}

__global__ void compute_ordered_sum_keys(const float* ordered, int valid_count, int dim,
                                         unsigned long long* keys) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= valid_count) return;
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) sum += ordered[d * valid_count + index];
    keys[index] = static_cast<unsigned long long>(__float_as_uint(sum));
}

__global__ void reorder_paper_points(const float* source, int source_count, int dim,
                                     const int* sorted_indices, int valid_count,
                                     float* ordered, int* original_indices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= valid_count) return;
    const int source_index = sorted_indices[index];
    for (int d = 0; d < dim; ++d) {
        ordered[d * valid_count + index] = source[d * source_count + source_index];
    }
    original_indices[index] = source_index;
}

__global__ void split_paper_intervals(const unsigned long long* morton_keys,
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
    const unsigned long long difference = morton_keys[start] ^ morton_keys[end - 1];

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

__global__ void classify_mkd_frontier(
    const unsigned long long* morton_keys,
    const int* starts, const int* ends, int interval_count, int target_size,
    int* expansion_counts, int* split_positions) {
    const int interval = blockIdx.x * blockDim.x + threadIdx.x;
    if (interval >= interval_count) return;
    const int start = starts[interval];
    const int end = ends[interval];
    const int size = end - start;
    if (size <= target_size) {
        expansion_counts[interval] = 1;
        split_positions[interval] = end;
        return;
    }
    const unsigned long long difference =
        morton_keys[start] ^ morton_keys[end - 1];
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
    expansion_counts[interval] = 2;
    split_positions[interval] = split;
}

__global__ void scatter_mkd_frontier(
    const int* starts, const int* ends, int interval_count,
    const int* expansion_counts, const int* output_offsets,
    const int* split_positions, int* next_starts, int* next_ends) {
    const int interval = blockIdx.x * blockDim.x + threadIdx.x;
    if (interval >= interval_count) return;
    const int output = output_offsets[interval];
    const int start = starts[interval];
    const int end = ends[interval];
    if (expansion_counts[interval] == 1) {
        next_starts[output] = start;
        next_ends[output] = end;
        return;
    }
    const int split = split_positions[interval];
    next_starts[output] = start;
    next_ends[output] = split;
    next_starts[output + 1] = split;
    next_ends[output + 1] = end;
}

__global__ void build_mkd_depth_first(const unsigned long long* morton_keys,
                                      int valid_count, int target_size,
                                      int* leaf_starts, int* leaf_ends, int capacity,
                                      int* stack_starts, int* stack_ends,
                                      int* leaf_count_output, int* overflow_output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int stack_count = 1;
    int leaf_count = 0;
    int overflow = 0;
    stack_starts[0] = 0;
    stack_ends[0] = valid_count;
    while (stack_count > 0 && !overflow) {
        --stack_count;
        const int start = stack_starts[stack_count];
        const int end = stack_ends[stack_count];
        const int size = end - start;
        const unsigned long long difference = morton_keys[start] ^ morton_keys[end - 1];
        if (size <= target_size) {
            if (leaf_count >= capacity) {
                overflow = 1;
                break;
            }
            leaf_starts[leaf_count] = start;
            leaf_ends[leaf_count] = end;
            ++leaf_count;
            continue;
        }

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
        if (stack_count + 2 > 128) {
            overflow = 1;
            break;
        }
        // Push right first so the left interval is processed next and leaves stay sorted.
        stack_starts[stack_count] = split;
        stack_ends[stack_count] = end;
        ++stack_count;
        stack_starts[stack_count] = start;
        stack_ends[stack_count] = split;
        ++stack_count;
    }
    *leaf_count_output = leaf_count;
    *overflow_output = overflow;
}

__global__ void initialize_fixed_chunks(int valid_count, int chunk_size,
                                        int* starts, int* ends, int chunk_count) {
    const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk >= chunk_count) return;
    starts[chunk] = chunk * chunk_size;
    ends[chunk] = min(starts[chunk] + chunk_size, valid_count);
}

template <int DIM>
__global__ void local_skyline_by_chunk(const float* ordered, int valid_count,
                                       const int* chunk_starts, const int* chunk_ends,
                                       int* alive_flags) {
    extern __shared__ float tile[];
    const int start = chunk_starts[blockIdx.x];
    const int end = chunk_ends[blockIdx.x];
    const int size = end - start;

    for (int batch = 0; batch * blockDim.x < size; ++batch) {
        const int local = batch * blockDim.x + threadIdx.x;
        const int position = start + local;
        float point[DIM];
        bool alive = position < end;
        if (alive) {
#pragma unroll
            for (int d = 0; d < DIM; ++d) point[d] = ordered[d * valid_count + position];
        }

        for (int tile_start = start; tile_start < end; tile_start += blockDim.x) {
            const int tile_count = min(blockDim.x, end - tile_start);
            if (threadIdx.x < tile_count) {
                const int source = tile_start + threadIdx.x;
#pragma unroll
                for (int d = 0; d < DIM; ++d) {
                    tile[threadIdx.x * DIM + d] = ordered[d * valid_count + source];
                }
            }
            __syncthreads();
            if (alive) {
                for (int other = 0; other < tile_count; ++other) {
                    const int other_position = tile_start + other;
                    if (other_position == position) continue;
                    bool all_le = true;
                    bool any_lt = false;
#pragma unroll
                    for (int d = 0; d < DIM; ++d) {
                        const float value = tile[other * DIM + d];
                        if (value > point[d]) {
                            all_le = false;
                            break;
                        }
                        any_lt = any_lt || value < point[d];
                    }
                    if (all_le && any_lt) {
                        alive = false;
                        break;
                    }
                }
            }
            __syncthreads();
        }
        if (position < end) alive_flags[position] = alive ? 1 : 0;
    }
}

template <int DIM>
__global__ void local_skyline_append_by_chunk(
    const float* ordered, int valid_count,
    const int* chunk_starts, const int* chunk_ends,
    int* candidate_positions, int* candidate_chunk_ids,
    int* candidate_count, int* chunk_counts) {
    extern __shared__ float points[];
    __shared__ int block_survivors;
    if (threadIdx.x == 0) block_survivors = 0;
    const int start = chunk_starts[blockIdx.x];
    const int end = chunk_ends[blockIdx.x];
    const int size = end - start;
    for (int local = threadIdx.x; local < size; local += blockDim.x) {
#pragma unroll
        for (int d = 0; d < DIM; ++d) {
            points[local * DIM + d] = ordered[d * valid_count + start + local];
        }
    }
    __syncthreads();

    for (int local = threadIdx.x; local < size; local += blockDim.x) {
        float point[DIM];
#pragma unroll
        for (int d = 0; d < DIM; ++d) point[d] = points[local * DIM + d];
        bool alive = true;
        for (int other = 0; other < local; ++other) {
            bool all_le = true;
            bool any_lt = false;
#pragma unroll
            for (int d = 0; d < DIM; ++d) {
                const float value = points[other * DIM + d];
                if (value > point[d]) {
                    all_le = false;
                    break;
                }
                any_lt = any_lt || value < point[d];
            }
            if (all_le && any_lt) {
                alive = false;
                break;
            }
        }
        if (alive) {
            const int output = atomicAdd(candidate_count, 1);
            candidate_positions[output] = start + local;
            candidate_chunk_ids[output] = blockIdx.x;
            atomicAdd(&block_survivors, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) chunk_counts[blockIdx.x] = block_survivors;
}

template <int DIM>
__global__ void local_skyline_flags_by_chunk(
    const float* ordered, int valid_count,
    const int* chunk_starts, const int* chunk_ends,
    int* alive_flags) {
    extern __shared__ float points[];
    const int start = chunk_starts[blockIdx.x];
    const int end = chunk_ends[blockIdx.x];
    const int size = end - start;
    for (int local = threadIdx.x; local < size; local += blockDim.x) {
#pragma unroll
        for (int d = 0; d < DIM; ++d) {
            points[local * DIM + d] = ordered[d * valid_count + start + local];
        }
    }
    __syncthreads();

    for (int local = threadIdx.x; local < size; local += blockDim.x) {
        float point[DIM];
#pragma unroll
        for (int d = 0; d < DIM; ++d) point[d] = points[local * DIM + d];
        bool alive = true;
        for (int other = 0; other < size; ++other) {
            if (other == local) continue;
            bool all_le = true;
            bool any_lt = false;
#pragma unroll
            for (int d = 0; d < DIM; ++d) {
                const float value = points[other * DIM + d];
                if (value > point[d]) {
                    all_le = false;
                    break;
                }
                any_lt = any_lt || value < point[d];
            }
            if (all_le && any_lt) {
                alive = false;
                break;
            }
        }
        alive_flags[start + local] = alive ? 1 : 0;
    }
}

__global__ void pack_local_candidates(
    const float* ordered, int valid_count, int dim,
    const unsigned long long* morton_keys, const int* ordered_indices,
    const int* candidate_positions, int candidate_count,
    float* packed, unsigned long long* packed_keys, int* packed_indices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) return;
    const int position = candidate_positions[index];
    for (int d = 0; d < dim; ++d) {
        packed[index * dim + d] = ordered[d * valid_count + position];
    }
    packed_keys[index] = morton_keys[position];
    packed_indices[index] = ordered_indices[position];
}

__global__ void count_chunk_survivors(const int* alive_flags,
                                      const int* chunk_starts, const int* chunk_ends,
                                      int* chunk_counts) {
    __shared__ int sums[BLOCK_SIZE];
    const int start = chunk_starts[blockIdx.x];
    const int end = chunk_ends[blockIdx.x];
    int local_sum = 0;
    for (int position = start + threadIdx.x; position < end; position += blockDim.x) {
        local_sum += alive_flags[position];
    }
    sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) chunk_counts[blockIdx.x] = sums[0];
}

__global__ void scatter_chunk_survivors(const float* ordered, int valid_count, int dim,
                                        const unsigned long long* morton_keys,
                                        const int* ordered_indices,
                                        const int* alive_flags, const int* alive_prefix,
                                        const int* chunk_starts, const int* chunk_ends,
                                        float* packed, unsigned long long* packed_keys,
                                        int* packed_indices, int* packed_chunk_ids) {
    const int chunk = blockIdx.x;
    const int start = chunk_starts[chunk];
    const int end = chunk_ends[chunk];
    for (int position = start + threadIdx.x; position < end; position += blockDim.x) {
        if (!alive_flags[position]) continue;
        const int output = alive_prefix[position];
        for (int d = 0; d < dim; ++d) {
            packed[output * dim + d] = ordered[d * valid_count + position];
        }
        packed_keys[output] = morton_keys[position];
        packed_indices[output] = ordered_indices[position];
        packed_chunk_ids[output] = chunk;
    }
}

__global__ void pack_all_ordered_points(const float* ordered, int valid_count, int dim,
                                        const unsigned long long* morton_keys,
                                        const int* ordered_indices, float* packed,
                                        unsigned long long* packed_keys,
                                        int* packed_indices) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= valid_count) return;
    for (int d = 0; d < dim; ++d) {
        packed[index * dim + d] = ordered[d * valid_count + index];
    }
    packed_keys[index] = morton_keys[index];
    packed_indices[index] = ordered_indices[index];
}

__global__ void pack_unfiltered_chunks(
    const float* ordered, int valid_count, int dim,
    const unsigned long long* morton_keys, const int* ordered_indices,
    const int* chunk_starts, const int* chunk_ends, int* chunk_counts,
    float* packed, unsigned long long* packed_keys, int* packed_indices,
    int* packed_chunk_ids) {
    const int chunk = blockIdx.x;
    const int start = chunk_starts[chunk];
    const int end = chunk_ends[chunk];
    if (threadIdx.x == 0) chunk_counts[chunk] = end - start;
    for (int position = start + threadIdx.x; position < end;
         position += blockDim.x) {
        for (int d = 0; d < dim; ++d) {
            packed[position * dim + d] =
                ordered[d * valid_count + position];
        }
        packed_keys[position] = morton_keys[position];
        packed_indices[position] = ordered_indices[position];
        packed_chunk_ids[position] = chunk;
    }
}

__global__ void compute_chunk_min_corners(const float* packed, int dim,
                                          const int* chunk_offsets,
                                          const int* chunk_counts,
                                          float* min_corners) {
    const int chunk = blockIdx.x;
    const int start = chunk_offsets[chunk];
    const int count = chunk_counts[chunk];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float minimum = FLT_MAX;
        for (int i = 0; i < count; ++i) {
            minimum = fminf(minimum, packed[(start + i) * dim + d]);
        }
        min_corners[chunk * dim + d] = minimum;
    }
}

__global__ void compute_chunk_max_corners(const float* packed, int dim,
                                          const int* chunk_offsets,
                                          const int* chunk_counts,
                                          float* max_corners) {
    const int chunk = blockIdx.x;
    const int start = chunk_offsets[chunk];
    const int count = chunk_counts[chunk];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float maximum = -FLT_MAX;
        for (int i = 0; i < count; ++i) {
            maximum = fmaxf(maximum, packed[(start + i) * dim + d]);
        }
        max_corners[chunk * dim + d] = maximum;
    }
}

__global__ void compute_compatibility_group_min_corners(
    const float* chunk_min_corners, int chunk_count, int dim,
    float* group_min_corners) {
    const int group = blockIdx.x;
    const int group_count =
        (chunk_count + COMPATIBILITY_GROUP_SIZE - 1) /
        COMPATIBILITY_GROUP_SIZE;
    if (group >= group_count) return;
    const int begin = group * COMPATIBILITY_GROUP_SIZE;
    const int end = min(begin + COMPATIBILITY_GROUP_SIZE, chunk_count);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float minimum = FLT_MAX;
        for (int chunk = begin; chunk < end; ++chunk) {
            minimum = fminf(minimum, chunk_min_corners[chunk * dim + d]);
        }
        group_min_corners[group * dim + d] = minimum;
    }
}

__global__ void build_compatibility_group_masks(
    const float* group_min_corners, const float* target_max_corners,
    int chunk_count, int group_count, int words_per_row, int dim,
    unsigned int* masks) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = global_thread & 31;
    const int warp = global_thread >> 5;
    const int warp_count = chunk_count * words_per_row;
    if (warp >= warp_count) return;
    const int target = warp / words_per_row;
    const int word = warp - target * words_per_row;
    const int group = word * 32 + lane;
    bool compatible = group < group_count &&
        group * COMPATIBILITY_GROUP_SIZE < target;
    if (compatible) {
        for (int d = 0; d < dim; ++d) {
            if (group_min_corners[group * dim + d] >
                target_max_corners[target * dim + d]) {
                compatible = false;
                break;
            }
        }
    }
    const unsigned int mask = __ballot_sync(0xffffffffU, compatible);
    if (lane == 0) masks[target * words_per_row + word] = mask;
}

__global__ void compute_micro_chunk_counts(const int* chunk_counts, int chunk_count,
                                           int* micro_counts) {
    const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk < chunk_count) {
        micro_counts[chunk] =
            (chunk_counts[chunk] + MICRO_CHUNK_SIZE - 1) / MICRO_CHUNK_SIZE;
    }
}

__device__ __forceinline__ unsigned long long morton_to_dimension_signature(
    unsigned long long code, int dim, int bits_per_dim) {
    unsigned long long signature = 0;
    for (int d = 0; d < dim; ++d) {
        unsigned long long value = 0;
        for (int bit = 0; bit < bits_per_dim; ++bit) {
            value |= ((code >> (bit * dim + d)) & 1ULL) << bit;
        }
        signature |= value << (d * bits_per_dim);
    }
    return signature;
}

__global__ void build_dimension_signatures(unsigned long long* keys, int count,
                                           int dim, int bits_per_dim) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        keys[index] = morton_to_dimension_signature(
            keys[index], dim, bits_per_dim);
    }
}

__global__ void compute_micro_min_corners(const float* packed, int dim,
                                          const int* chunk_offsets,
                                          const int* chunk_counts,
                                          const int* micro_offsets,
                                          int chunk_count,
                                          __half* micro_min_corners) {
    const int chunk = blockIdx.x;
    if (chunk >= chunk_count) return;
    const int count = chunk_counts[chunk];
    const int micro_count = (count + MICRO_CHUNK_SIZE - 1) / MICRO_CHUNK_SIZE;
    for (int item = threadIdx.x; item < micro_count * dim; item += blockDim.x) {
        const int micro = item / dim;
        const int d = item - micro * dim;
        const int start = chunk_offsets[chunk] + micro * MICRO_CHUNK_SIZE;
        const int end = min(start + MICRO_CHUNK_SIZE,
                            chunk_offsets[chunk] + count);
        float minimum = FLT_MAX;
        for (int i = start; i < end; ++i) {
            minimum = fminf(minimum, packed[i * dim + d]);
        }
        micro_min_corners[(micro_offsets[chunk] + micro) * dim + d] =
            __float2half_rd(minimum);
    }
}

__device__ __forceinline__ unsigned int micro_quantize(
    float value, float minimum, float maximum) {
    const float range = maximum - minimum;
    float normalized = range > EPS ? (value - minimum) / range : 0.0f;
    normalized = fmaxf(0.0f, fminf(1.0f, normalized));
    return min(MICRO_CODE_MAX,
               static_cast<unsigned int>(normalized * MICRO_CODE_MAX));
}

__global__ void compute_micro_packed_corners(
    const float* packed, int dim, const int* chunk_offsets,
    const int* chunk_counts, const int* micro_offsets,
    const float* minimums, const float* maximums,
    int chunk_count, unsigned int* packed_corners) {
    const int chunk = blockIdx.x;
    if (chunk >= chunk_count) return;
    const int count = chunk_counts[chunk];
    const int micro_count = (count + MICRO_CHUNK_SIZE - 1) / MICRO_CHUNK_SIZE;
    const int words_per_corner = (dim + 3) / 4;
    for (int item = threadIdx.x; item < micro_count * words_per_corner;
         item += blockDim.x) {
        const int micro = item / words_per_corner;
        const int word = item - micro * words_per_corner;
        const int start = chunk_offsets[chunk] + micro * MICRO_CHUNK_SIZE;
        const int end = min(start + MICRO_CHUNK_SIZE,
                            chunk_offsets[chunk] + count);
        unsigned int packed_code = 0U;
        for (int lane = 0; lane < 4; ++lane) {
            const int d = word * 4 + lane;
            unsigned int code = 0U;
            if (d < dim) {
                float minimum = FLT_MAX;
                for (int i = start; i < end; ++i) {
                    minimum = fminf(minimum, packed[i * dim + d]);
                }
                code = micro_quantize(
                    minimum, minimums[d], maximums[d]);
            }
            packed_code |= code << (lane * 8);
        }
        packed_corners[
            (micro_offsets[chunk] + micro) * words_per_corner + word] =
            packed_code;
    }
}

__global__ void compute_micro_point_codes(
    const float* packed, int dim, const int* chunk_offsets,
    const int* chunk_counts, const int* micro_offsets,
    const float* minimums, const float* maximums,
    int chunk_count, unsigned int* point_codes) {
    const int chunk = blockIdx.x;
    if (chunk >= chunk_count) return;
    const int count = chunk_counts[chunk];
    const int micro_count = (count + MICRO_CHUNK_SIZE - 1) / MICRO_CHUNK_SIZE;
    for (int item = threadIdx.x;
         item < micro_count * dim * MICRO_POINT_WORDS;
         item += blockDim.x) {
        const int word = item % MICRO_POINT_WORDS;
        const int d = (item / MICRO_POINT_WORDS) % dim;
        const int micro = item / (dim * MICRO_POINT_WORDS);
        const int micro_start =
            chunk_offsets[chunk] + micro * MICRO_CHUNK_SIZE;
        const int chunk_end = chunk_offsets[chunk] + count;
        unsigned int packed_code = 0U;
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const int point_index = micro_start + lane;
            unsigned int code = MICRO_CODE_MAX;
            if (point_index < chunk_end) {
                code = micro_quantize(packed[point_index * dim + d],
                                      minimums[d], maximums[d]);
            }
            packed_code |= (code >> 5) << (lane * 4);
        }
        const int global_micro = micro_offsets[chunk] + micro;
        point_codes[(static_cast<std::size_t>(global_micro) * dim + d) *
                    MICRO_POINT_WORDS + word] = packed_code;
    }
}

template <int DIM>
__device__ __forceinline__ void pack_micro_target(
    const float* point, const float* minimums, const float* maximums,
    unsigned int* packed_target) {
    constexpr int WORDS = (DIM + 3) / 4;
#pragma unroll
    for (int word = 0; word < WORDS; ++word) {
        unsigned int target = 0U;
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            const int d = word * 4 + lane;
            const unsigned int code = d < DIM
                ? micro_quantize(point[d], minimums[d], maximums[d])
                : MICRO_CODE_MAX;
            target |= code << (lane * 8);
        }
        packed_target[word] = target;
    }
}

template <int DIM>
__device__ __forceinline__ bool packed_micro_corner_may_dominate(
    const unsigned int* packed_corner, const unsigned int* packed_target) {
    constexpr int WORDS = (DIM + 3) / 4;
#pragma unroll
    for (int word = 0; word < WORDS; ++word) {
        if (__vcmpleu4(packed_corner[word], packed_target[word]) !=
            0xffffffffU) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void atomic_min_float(float* address, float value) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    while (value < __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old == assumed) break;
    }
}

__device__ __forceinline__ int projection_bin(float value, float minimum,
                                               float maximum) {
    const float range = maximum - minimum;
    float normalized = range > EPS ? (value - minimum) / range : 0.0f;
    normalized = fmaxf(0.0f, fminf(1.0f, normalized));
    return min(PROJECTION_BIN_COUNT - 1,
               static_cast<int>(normalized * PROJECTION_BIN_COUNT));
}

__global__ void compute_chunk_projection_bounds(
    const float* packed, int dim, const int* chunk_offsets,
    const int* chunk_counts, const float* minimums, const float* maximums,
    int pair_count, bool cyclic_pairs, float* projection_bounds) {
    __shared__ float bin_minimums[
        PROJECTION_CYCLE_PAIR_COUNT * PROJECTION_BIN_COUNT];
    const int summary_count = pair_count * PROJECTION_BIN_COUNT;
    for (int i = threadIdx.x; i < summary_count; i += blockDim.x) {
        bin_minimums[i] = FLT_MAX;
    }
    __syncthreads();

    const int chunk = blockIdx.x;
    const int start = chunk_offsets[chunk];
    const int count = chunk_counts[chunk];
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float* point = packed + (start + i) * dim;
        for (int pair = 0; pair < pair_count; ++pair) {
            const int base_pair_count = dim / 2;
            const int extra_pair = pair - base_pair_count;
            const int x_dim = !cyclic_pairs || pair < base_pair_count
                ? pair * 2 : extra_pair * 2 + 1;
            const int y_dim = cyclic_pairs && pair >= base_pair_count
                ? (x_dim + 1) % dim : x_dim + 1;
            const int bin = projection_bin(
                point[x_dim], minimums[x_dim], maximums[x_dim]);
            atomic_min_float(&bin_minimums[pair * PROJECTION_BIN_COUNT + bin],
                             point[y_dim]);
        }
    }
    __syncthreads();

    if (threadIdx.x < pair_count) {
        const int pair = threadIdx.x;
        float prefix_minimum = FLT_MAX;
        for (int bin = 0; bin < PROJECTION_BIN_COUNT; ++bin) {
            prefix_minimum = fminf(
                prefix_minimum,
                bin_minimums[pair * PROJECTION_BIN_COUNT + bin]);
            projection_bounds[
                (chunk * pair_count + pair) * PROJECTION_BIN_COUNT + bin] =
                prefix_minimum;
        }
    }
}

__device__ __forceinline__ unsigned int projection_quantize_lower(
    float value, float minimum, float maximum);

__global__ void compute_chunk_projection_quantized_bounds(
    const float* packed, int dim, const int* chunk_offsets,
    const int* chunk_counts, const float* minimums, const float* maximums,
    int pair_count, unsigned short* projection_bounds) {
    __shared__ unsigned int bin_minimums[
        PROJECTION_CYCLE_PAIR_COUNT * PROJECTION_BIN_COUNT];
    const int summary_count = pair_count * PROJECTION_BIN_COUNT;
    for (int i = threadIdx.x; i < summary_count; i += blockDim.x) {
        bin_minimums[i] = PROJECTION_QUANTIZATION_MAX;
    }
    __syncthreads();

    const int chunk = blockIdx.x;
    const int start = chunk_offsets[chunk];
    const int count = chunk_counts[chunk];
    const int base_pair_count = dim / 2;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float* point = packed + (start + i) * dim;
        for (int pair = 0; pair < pair_count; ++pair) {
            const int extra_pair = pair - base_pair_count;
            const int x_dim = pair < base_pair_count
                ? pair * 2 : extra_pair * 2 + 1;
            const int y_dim = pair < base_pair_count
                ? x_dim + 1 : (x_dim + 1) % dim;
            const int bin = projection_bin(
                point[x_dim], minimums[x_dim], maximums[x_dim]);
            const unsigned int y_value = projection_quantize_lower(
                point[y_dim], minimums[y_dim], maximums[y_dim]);
            atomicMin(&bin_minimums[pair * PROJECTION_BIN_COUNT + bin], y_value);
        }
    }
    __syncthreads();

    if (threadIdx.x < pair_count) {
        const int pair = threadIdx.x;
        unsigned int prefix_minimum = PROJECTION_QUANTIZATION_MAX;
        for (int bin = 0; bin < PROJECTION_BIN_COUNT; ++bin) {
            prefix_minimum = min(
                prefix_minimum,
                bin_minimums[pair * PROJECTION_BIN_COUNT + bin]);
            projection_bounds[
                (chunk * pair_count + pair) * PROJECTION_BIN_COUNT + bin] =
                static_cast<unsigned short>(prefix_minimum);
        }
    }
}

__device__ __forceinline__ float normalized_coordinate(
    float value, float minimum, float maximum) {
    const float range = maximum - minimum;
    float normalized = range > EPS ? (value - minimum) / range : 0.0f;
    return fmaxf(0.0f, fminf(1.0f, normalized));
}

__device__ __forceinline__ int projection_3d_bin(
    float value, float minimum, float maximum) {
    const float normalized = normalized_coordinate(value, minimum, maximum);
    return min(PROJECTION_3D_SIDE - 1,
               static_cast<int>(normalized * PROJECTION_3D_SIDE));
}

__device__ __forceinline__ unsigned int projection_quantize_lower(
    float value, float minimum, float maximum) {
    const float normalized = normalized_coordinate(value, minimum, maximum);
    return min(PROJECTION_QUANTIZATION_MAX,
               static_cast<unsigned int>(
                   normalized * static_cast<float>(PROJECTION_QUANTIZATION_MAX)));
}

__global__ void compute_chunk_projection_3d_bounds(
    const float* packed, int dim, const int* chunk_offsets,
    const int* chunk_counts, const float* minimums, const float* maximums,
    int triple_count, unsigned short* projection_bounds) {
    __shared__ unsigned int cell_minimums[
        (MAX_DIM / 3) * PROJECTION_3D_CELL_COUNT];
    const int summary_count = triple_count * PROJECTION_3D_CELL_COUNT;
    for (int i = threadIdx.x; i < summary_count; i += blockDim.x) {
        cell_minimums[i] = PROJECTION_QUANTIZATION_MAX;
    }
    __syncthreads();

    const int chunk = blockIdx.x;
    const int start = chunk_offsets[chunk];
    const int count = chunk_counts[chunk];
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float* point = packed + (start + i) * dim;
        for (int triple = 0; triple < triple_count; ++triple) {
            const int x_dim = triple * 3;
            const int y_dim = x_dim + 1;
            const int z_dim = x_dim + 2;
            const int x_bin = projection_3d_bin(
                point[x_dim], minimums[x_dim], maximums[x_dim]);
            const int y_bin = projection_3d_bin(
                point[y_dim], minimums[y_dim], maximums[y_dim]);
            const unsigned int z_value = projection_quantize_lower(
                point[z_dim], minimums[z_dim], maximums[z_dim]);
            atomicMin(&cell_minimums[
                triple * PROJECTION_3D_CELL_COUNT +
                x_bin * PROJECTION_3D_SIDE + y_bin], z_value);
        }
    }
    __syncthreads();

    if (threadIdx.x < triple_count) {
        const int triple = threadIdx.x;
        unsigned int* cells =
            cell_minimums + triple * PROJECTION_3D_CELL_COUNT;
        for (int x = 0; x < PROJECTION_3D_SIDE; ++x) {
            for (int y = 1; y < PROJECTION_3D_SIDE; ++y) {
                const int index = x * PROJECTION_3D_SIDE + y;
                cells[index] = min(cells[index], cells[index - 1]);
            }
        }
        for (int x = 1; x < PROJECTION_3D_SIDE; ++x) {
            for (int y = 0; y < PROJECTION_3D_SIDE; ++y) {
                const int index = x * PROJECTION_3D_SIDE + y;
                cells[index] = min(
                    cells[index], cells[index - PROJECTION_3D_SIDE]);
            }
        }
        for (int cell = 0; cell < PROJECTION_3D_CELL_COUNT; ++cell) {
            projection_bounds[
                (chunk * triple_count + triple) *
                PROJECTION_3D_CELL_COUNT + cell] =
                static_cast<unsigned short>(cells[cell]);
        }
    }
}

template <int DIM>
__device__ __forceinline__ bool point_dominates(const float* other, const float* point) {
    bool any_lt = false;
#pragma unroll
    for (int d = 0; d < DIM; ++d) {
        if (other[d] > point[d]) return false;
        any_lt = any_lt || other[d] < point[d];
    }
    return any_lt;
}

template <int DIM>
__device__ __forceinline__ bool min_corner_may_dominate(const float* corner,
                                                        const float* point) {
    bool any_lt = false;
#pragma unroll
    for (int d = 0; d < DIM; ++d) {
        if (corner[d] > point[d]) return false;
        any_lt = any_lt || corner[d] < point[d];
    }
    return any_lt;
}

template <int DIM>
__device__ __forceinline__ bool quantized_corner_may_dominate(
    const __half* corner, const float* point) {
    bool any_lt = false;
#pragma unroll
    for (int d = 0; d < DIM; ++d) {
        const float value = __half2float(corner[d]);
        if (value > point[d]) return false;
        any_lt = any_lt || value < point[d];
    }
    return any_lt;
}

template <int DIM>
__device__ __forceinline__ bool signature_may_dominate(
    unsigned long long other, unsigned long long point, int bits_per_dim) {
    if (bits_per_dim <= 0) return true;
    const unsigned long long field_mask = (1ULL << bits_per_dim) - 1ULL;
    unsigned long long even_mask = 0;
    unsigned long long guard_mask = 0;
#pragma unroll
    for (int d = 0; d < DIM; d += 2) {
        even_mask |= field_mask << (d * bits_per_dim);
        guard_mask |= 1ULL << (d * bits_per_dim + bits_per_dim);
    }
    const unsigned long long even_comparisons =
        ((point & even_mask) | guard_mask) - (other & even_mask);
    if ((even_comparisons & guard_mask) != guard_mask) return false;
    const unsigned long long odd_point = (point >> bits_per_dim) & even_mask;
    const unsigned long long odd_other = (other >> bits_per_dim) & even_mask;
    const unsigned long long odd_comparisons =
        (odd_point | guard_mask) - odd_other;
    return (odd_comparisons & guard_mask) == guard_mask;
}

__device__ __forceinline__ int equal_key_upper_bound(
    const unsigned long long* keys, int count, int first,
    unsigned long long key) {
    int left = first;
    int right = count;
    while (left < right) {
        const int middle = left + (right - left) / 2;
        if (keys[middle] <= key) left = middle + 1;
        else right = middle;
    }
    return left;
}

template <int DIM, int PROJECTION_MODE, bool USE_POINT_MASK>
__global__ void verify_paper_chunks(const float* packed,
                                    const unsigned long long* packed_keys,
                                    const int* packed_indices, int candidate_count,
                                    const int* candidate_chunk_ids,
                                    const int* chunk_offsets, const int* chunk_counts,
                                    const float* min_corners,
                                    const float* projection_bounds,
                                    const float* projection_minimums,
                                    const float* projection_maximums,
                                    int projection_pair_count,
                                    const unsigned short* projection_3d_bounds,
                                    int projection_3d_triple_count,
                                    const int* micro_offsets,
                                    const int* micro_counts,
                                    const __half* micro_min_corners,
                                    const unsigned int* packed_micro_corners,
                                    const unsigned int* micro_point_codes,
                                    int bits_per_dim,
                                    bool check_equal_tail,
                                    bool check_own_prefix,
                                    int* skyline_indices, int* skyline_count,
                                    unsigned long long* verification_triggers,
                                    unsigned long long* projection_bound_skips,
                                    unsigned long long* projection_3d_skips,
                                    unsigned long long* false_triggers,
                                    unsigned long long* exact_point_checks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) return;
    float point[DIM];
#pragma unroll
    for (int d = 0; d < DIM; ++d) point[d] = packed[index * DIM + d];
    constexpr int PACKED_MICRO_WORDS = (DIM + 3) / 4;
    unsigned int packed_point_codes[PACKED_MICRO_WORDS];
    if (PROJECTION_MODE == 5 &&
        (packed_micro_corners || USE_POINT_MASK)) {
        pack_micro_target<DIM>(point, projection_minimums,
                               projection_maximums, packed_point_codes);
    }
    const int own_chunk = candidate_chunk_ids[index];
    bool alive = true;

    for (int chunk = 0; chunk < own_chunk && alive; ++chunk) {
        const int count = chunk_counts[chunk];
        if (count == 0) continue;
        if (!min_corner_may_dominate<DIM>(min_corners + chunk * DIM, point)) continue;
        if (PROJECTION_MODE == 1 || PROJECTION_MODE == 3 ||
            (PROJECTION_MODE == 5 && projection_bounds)) {
            bool projection_rejects = false;
#pragma unroll
            for (int pair = 0; pair < PROJECTION_CYCLE_PAIR_COUNT; ++pair) {
                if (pair >= projection_pair_count) break;
                const int base_pair_count = DIM / 2;
                const int extra_pair = pair - base_pair_count;
                const int x_dim = PROJECTION_MODE != 3 || pair < base_pair_count
                    ? pair * 2 : extra_pair * 2 + 1;
                const int y_dim = PROJECTION_MODE == 3 && pair >= base_pair_count
                    ? (x_dim + 1) % DIM : x_dim + 1;
                const int bin = projection_bin(
                    point[x_dim], projection_minimums[x_dim],
                    projection_maximums[x_dim]);
                const float minimum_y = projection_bounds[
                    (chunk * projection_pair_count + pair) *
                    PROJECTION_BIN_COUNT + bin];
                const float tolerance = EPS * fmaxf(1.0f, fabsf(point[y_dim]));
                if (minimum_y > point[y_dim] + tolerance) {
                    projection_rejects = true;
                    break;
                }
            }
            if (projection_rejects) {
                if (projection_bound_skips) atomicAdd(projection_bound_skips, 1ULL);
                continue;
            }
        }
        if (PROJECTION_MODE == 2) {
            bool projection_rejects = false;
#pragma unroll
            for (int triple = 0; triple < MAX_DIM / 3; ++triple) {
                if (triple >= projection_3d_triple_count) break;
                const int x_dim = triple * 3;
                const int y_dim = x_dim + 1;
                const int z_dim = x_dim + 2;
                const int x_bin = projection_3d_bin(
                    point[x_dim], projection_minimums[x_dim],
                    projection_maximums[x_dim]);
                const int y_bin = projection_3d_bin(
                    point[y_dim], projection_minimums[y_dim],
                    projection_maximums[y_dim]);
                const unsigned short minimum_z = projection_3d_bounds[
                    (chunk * projection_3d_triple_count + triple) *
                    PROJECTION_3D_CELL_COUNT +
                    x_bin * PROJECTION_3D_SIDE + y_bin];
                const unsigned int point_z = projection_quantize_lower(
                    point[z_dim], projection_minimums[z_dim],
                    projection_maximums[z_dim]);
                if (static_cast<unsigned int>(minimum_z) >
                    point_z) {
                    projection_rejects = true;
                    break;
                }
            }
            if (projection_rejects) {
                if (projection_3d_skips) atomicAdd(projection_3d_skips, 1ULL);
                continue;
            }
        }
        if (PROJECTION_MODE == 4) {
            bool projection_rejects = false;
            const int base_pair_count = DIM / 2;
#pragma unroll
            for (int pair = 0; pair < PROJECTION_CYCLE_PAIR_COUNT; ++pair) {
                if (pair >= projection_pair_count) break;
                const int extra_pair = pair - base_pair_count;
                const int x_dim = pair < base_pair_count
                    ? pair * 2 : extra_pair * 2 + 1;
                const int y_dim = pair < base_pair_count
                    ? x_dim + 1 : (x_dim + 1) % DIM;
                const int bin = projection_bin(
                    point[x_dim], projection_minimums[x_dim],
                    projection_maximums[x_dim]);
                const unsigned short minimum_y = projection_3d_bounds[
                    (chunk * projection_pair_count + pair) *
                    PROJECTION_BIN_COUNT + bin];
                const unsigned int point_y = projection_quantize_lower(
                    point[y_dim], projection_minimums[y_dim],
                    projection_maximums[y_dim]);
                if (static_cast<unsigned int>(minimum_y) > point_y) {
                    projection_rejects = true;
                    break;
                }
            }
            if (projection_rejects) {
                if (projection_bound_skips) atomicAdd(projection_bound_skips, 1ULL);
                continue;
            }
        }
        if (verification_triggers) atomicAdd(verification_triggers, 1ULL);
        bool dominated_in_chunk = false;
        const int start = chunk_offsets[chunk];
        if (PROJECTION_MODE == 5) {
            const int micro_count = micro_counts[chunk];
            const int micro_offset = micro_offsets[chunk];
            constexpr int WORDS_PER_MICRO = (DIM + 3) / 4;
            for (int micro = 0; micro < micro_count && alive; ++micro) {
                if (packed_micro_corners) {
                    if (!packed_micro_corner_may_dominate<DIM>(
                            packed_micro_corners +
                                (micro_offset + micro) * WORDS_PER_MICRO,
                            packed_point_codes)) {
                        continue;
                    }
                } else if (!quantized_corner_may_dominate<DIM>(
                               micro_min_corners +
                                   (micro_offset + micro) * DIM,
                               point)) {
                    continue;
                }
                unsigned int point_mask = MICRO_POINT_GUARDS;
                if (USE_POINT_MASK) {
#pragma unroll
                    for (int d = 0; d < DIM; ++d) {
                        const unsigned int target_code =
                            (packed_point_codes[d / 4] >> ((d % 4) * 8)) &
                            0xffU;
                        const unsigned int target_word =
                            (target_code >> 5) * 0x11111111U;
                        const std::size_t code_base =
                            (static_cast<std::size_t>(micro_offset + micro) *
                             DIM + d) * MICRO_POINT_WORDS;
                        const unsigned int lane_result =
                            ((target_word | MICRO_POINT_GUARDS) -
                             micro_point_codes[code_base]) &
                            MICRO_POINT_GUARDS;
                        point_mask &= lane_result;
                        if (point_mask == 0U) break;
                    }
                    if (point_mask == 0U) continue;
                }
                const int micro_start = start + micro * MICRO_CHUNK_SIZE;
                const int micro_end = min(micro_start + MICRO_CHUNK_SIZE,
                                          start + count);
                for (int i = micro_start; i < micro_end; ++i) {
                    const int local = i - micro_start;
                    if (USE_POINT_MASK &&
                        ((point_mask >> (local * 4)) & 0x8U) == 0U) {
                        continue;
                    }
                    if (!signature_may_dominate<DIM>(
                            packed_keys[i], packed_keys[index], bits_per_dim)) {
                        continue;
                    }
                    if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
                    if (point_dominates<DIM>(packed + i * DIM, point)) {
                        alive = false;
                        dominated_in_chunk = true;
                        break;
                    }
                }
            }
        } else {
            for (int i = 0; i < count; ++i) {
                if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
                if (point_dominates<DIM>(packed + (start + i) * DIM, point)) {
                    alive = false;
                    dominated_in_chunk = true;
                    break;
                }
            }
        }
        if (false_triggers && !dominated_in_chunk) atomicAdd(false_triggers, 1ULL);
    }

    // Endpoint-XOR chunks have already undergone Morton-ordered prefix filtering.
    // Equal-code tails are completed below; fixed-partition ablations still need
    // their own prefix check because they bypass the endpoint-XOR local pass.
    if (alive && check_own_prefix) {
        const int own_start = chunk_offsets[own_chunk];
        for (int other = own_start; other < index; ++other) {
            if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
            if (point_dominates<DIM>(packed + other * DIM, point)) {
                alive = false;
                break;
            }
        }
    }

    if (alive && check_equal_tail) {
        const unsigned long long key = packed_keys[index];
        int end = index + 1;
        if (PROJECTION_MODE == 5) {
            while (end < candidate_count && packed_keys[end] == key) ++end;
        } else {
            end = equal_key_upper_bound(
                packed_keys, candidate_count, index + 1, key);
        }
        for (int other = index + 1; other < end; ++other) {
            if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
            if (point_dominates<DIM>(packed + other * DIM, point)) {
                alive = false;
                break;
            }
        }
    }

    if (alive) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = packed_indices[index];
    }
}


template <int DIM, int PROJECTION_MODE, bool USE_POINT_MASK>
__global__ void verify_paper_chunks_compatible(const float* packed,
                                    const unsigned long long* packed_keys,
                                    const int* packed_indices, int candidate_count,
                                    const int* candidate_chunk_ids,
                                    const int* chunk_offsets, const int* chunk_counts,
                                    const float* min_corners,
                                    const float* projection_bounds,
                                    const float* projection_minimums,
                                    const float* projection_maximums,
                                    int projection_pair_count,
                                    const unsigned short* projection_3d_bounds,
                                    int projection_3d_triple_count,
                                    const int* micro_offsets,
                                    const int* micro_counts,
                                    const __half* micro_min_corners,
                                    const unsigned int* packed_micro_corners,
                                    const unsigned int* micro_point_codes,
                                    const unsigned int* compatibility_masks,
                                    int compatibility_words_per_row,
                                    int bits_per_dim,
                                    bool check_equal_tail,
                                    bool check_own_prefix,
                                    int* skyline_indices, int* skyline_count,
                                    unsigned long long* verification_triggers,
                                    unsigned long long* projection_bound_skips,
                                    unsigned long long* projection_3d_skips,
                                    unsigned long long* false_triggers,
                                    unsigned long long* exact_point_checks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) return;
    float point[DIM];
#pragma unroll
    for (int d = 0; d < DIM; ++d) point[d] = packed[index * DIM + d];
    constexpr int PACKED_MICRO_WORDS = (DIM + 3) / 4;
    unsigned int packed_point_codes[PACKED_MICRO_WORDS];
    if (PROJECTION_MODE == 5 &&
        (packed_micro_corners || USE_POINT_MASK)) {
        pack_micro_target<DIM>(point, projection_minimums,
                               projection_maximums, packed_point_codes);
    }
    const int own_chunk = candidate_chunk_ids[index];
    bool alive = true;

    int sequential_chunk = 0;
    int group_word = 0;
    int active_group_word = -1;
    unsigned int group_mask = 0;
    int group_chunk = 0;
    int group_chunk_end = 0;
    while (alive) {
        int chunk = -1;
        if (true) {
            while (group_chunk >= group_chunk_end) {
                while (group_mask == 0 &&
                       group_word < compatibility_words_per_row) {
                    active_group_word = group_word;
                    group_mask = compatibility_masks[
                        own_chunk * compatibility_words_per_row + group_word++];
                }
                if (group_mask == 0) break;
                const int bit = __ffs(static_cast<int>(group_mask)) - 1;
                group_mask &= group_mask - 1;
                const int group = active_group_word * 32 + bit;
                group_chunk = group * COMPATIBILITY_GROUP_SIZE;
                group_chunk_end = min(group_chunk + COMPATIBILITY_GROUP_SIZE,
                                      own_chunk);
            }
            if (group_chunk < group_chunk_end) chunk = group_chunk++;
        } else if (sequential_chunk < own_chunk) {
            chunk = sequential_chunk++;
        }
        if (chunk < 0) break;
        const int count = chunk_counts[chunk];
        if (count == 0) continue;
        if (!min_corner_may_dominate<DIM>(min_corners + chunk * DIM, point)) continue;
        if (PROJECTION_MODE == 1 || PROJECTION_MODE == 3 ||
            (PROJECTION_MODE == 5 && projection_bounds)) {
            bool projection_rejects = false;
#pragma unroll
            for (int pair = 0; pair < PROJECTION_CYCLE_PAIR_COUNT; ++pair) {
                if (pair >= projection_pair_count) break;
                const int base_pair_count = DIM / 2;
                const int extra_pair = pair - base_pair_count;
                const int x_dim = PROJECTION_MODE != 3 || pair < base_pair_count
                    ? pair * 2 : extra_pair * 2 + 1;
                const int y_dim = PROJECTION_MODE == 3 && pair >= base_pair_count
                    ? (x_dim + 1) % DIM : x_dim + 1;
                const int bin = projection_bin(
                    point[x_dim], projection_minimums[x_dim],
                    projection_maximums[x_dim]);
                const float minimum_y = projection_bounds[
                    (chunk * projection_pair_count + pair) *
                    PROJECTION_BIN_COUNT + bin];
                const float tolerance = EPS * fmaxf(1.0f, fabsf(point[y_dim]));
                if (minimum_y > point[y_dim] + tolerance) {
                    projection_rejects = true;
                    break;
                }
            }
            if (projection_rejects) {
                if (projection_bound_skips) atomicAdd(projection_bound_skips, 1ULL);
                continue;
            }
        }
        if (PROJECTION_MODE == 2) {
            bool projection_rejects = false;
#pragma unroll
            for (int triple = 0; triple < MAX_DIM / 3; ++triple) {
                if (triple >= projection_3d_triple_count) break;
                const int x_dim = triple * 3;
                const int y_dim = x_dim + 1;
                const int z_dim = x_dim + 2;
                const int x_bin = projection_3d_bin(
                    point[x_dim], projection_minimums[x_dim],
                    projection_maximums[x_dim]);
                const int y_bin = projection_3d_bin(
                    point[y_dim], projection_minimums[y_dim],
                    projection_maximums[y_dim]);
                const unsigned short minimum_z = projection_3d_bounds[
                    (chunk * projection_3d_triple_count + triple) *
                    PROJECTION_3D_CELL_COUNT +
                    x_bin * PROJECTION_3D_SIDE + y_bin];
                const unsigned int point_z = projection_quantize_lower(
                    point[z_dim], projection_minimums[z_dim],
                    projection_maximums[z_dim]);
                if (static_cast<unsigned int>(minimum_z) >
                    point_z) {
                    projection_rejects = true;
                    break;
                }
            }
            if (projection_rejects) {
                if (projection_3d_skips) atomicAdd(projection_3d_skips, 1ULL);
                continue;
            }
        }
        if (PROJECTION_MODE == 4) {
            bool projection_rejects = false;
            const int base_pair_count = DIM / 2;
#pragma unroll
            for (int pair = 0; pair < PROJECTION_CYCLE_PAIR_COUNT; ++pair) {
                if (pair >= projection_pair_count) break;
                const int extra_pair = pair - base_pair_count;
                const int x_dim = pair < base_pair_count
                    ? pair * 2 : extra_pair * 2 + 1;
                const int y_dim = pair < base_pair_count
                    ? x_dim + 1 : (x_dim + 1) % DIM;
                const int bin = projection_bin(
                    point[x_dim], projection_minimums[x_dim],
                    projection_maximums[x_dim]);
                const unsigned short minimum_y = projection_3d_bounds[
                    (chunk * projection_pair_count + pair) *
                    PROJECTION_BIN_COUNT + bin];
                const unsigned int point_y = projection_quantize_lower(
                    point[y_dim], projection_minimums[y_dim],
                    projection_maximums[y_dim]);
                if (static_cast<unsigned int>(minimum_y) > point_y) {
                    projection_rejects = true;
                    break;
                }
            }
            if (projection_rejects) {
                if (projection_bound_skips) atomicAdd(projection_bound_skips, 1ULL);
                continue;
            }
        }
        if (verification_triggers) atomicAdd(verification_triggers, 1ULL);
        bool dominated_in_chunk = false;
        const int start = chunk_offsets[chunk];
        if (PROJECTION_MODE == 5) {
            const int micro_count = micro_counts[chunk];
            const int micro_offset = micro_offsets[chunk];
            constexpr int WORDS_PER_MICRO = (DIM + 3) / 4;
            for (int micro = 0; micro < micro_count && alive; ++micro) {
                if (packed_micro_corners) {
                    if (!packed_micro_corner_may_dominate<DIM>(
                            packed_micro_corners +
                                (micro_offset + micro) * WORDS_PER_MICRO,
                            packed_point_codes)) {
                        continue;
                    }
                } else if (!quantized_corner_may_dominate<DIM>(
                               micro_min_corners +
                                   (micro_offset + micro) * DIM,
                               point)) {
                    continue;
                }
                unsigned int point_mask = MICRO_POINT_GUARDS;
                if (USE_POINT_MASK) {
#pragma unroll
                    for (int d = 0; d < DIM; ++d) {
                        const unsigned int target_code =
                            (packed_point_codes[d / 4] >> ((d % 4) * 8)) &
                            0xffU;
                        const unsigned int target_word =
                            (target_code >> 5) * 0x11111111U;
                        const std::size_t code_base =
                            (static_cast<std::size_t>(micro_offset + micro) *
                             DIM + d) * MICRO_POINT_WORDS;
                        const unsigned int lane_result =
                            ((target_word | MICRO_POINT_GUARDS) -
                             micro_point_codes[code_base]) &
                            MICRO_POINT_GUARDS;
                        point_mask &= lane_result;
                        if (point_mask == 0U) break;
                    }
                    if (point_mask == 0U) continue;
                }
                const int micro_start = start + micro * MICRO_CHUNK_SIZE;
                const int micro_end = min(micro_start + MICRO_CHUNK_SIZE,
                                          start + count);
                for (int i = micro_start; i < micro_end; ++i) {
                    const int local = i - micro_start;
                    if (USE_POINT_MASK &&
                        ((point_mask >> (local * 4)) & 0x8U) == 0U) {
                        continue;
                    }
                    if (!signature_may_dominate<DIM>(
                            packed_keys[i], packed_keys[index], bits_per_dim)) {
                        continue;
                    }
                    if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
                    if (point_dominates<DIM>(packed + i * DIM, point)) {
                        alive = false;
                        dominated_in_chunk = true;
                        break;
                    }
                }
            }
        } else {
            for (int i = 0; i < count; ++i) {
                if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
                if (point_dominates<DIM>(packed + (start + i) * DIM, point)) {
                    alive = false;
                    dominated_in_chunk = true;
                    break;
                }
            }
        }
        if (false_triggers && !dominated_in_chunk) atomicAdd(false_triggers, 1ULL);
    }

    // Endpoint-XOR chunks have already undergone Morton-ordered prefix filtering.
    // Equal-code tails are completed below; fixed-partition ablations still need
    // their own prefix check because they bypass the endpoint-XOR local pass.
    if (alive && check_own_prefix) {
        const int own_start = chunk_offsets[own_chunk];
        for (int other = own_start; other < index; ++other) {
            if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
            if (point_dominates<DIM>(packed + other * DIM, point)) {
                alive = false;
                break;
            }
        }
    }

    if (alive && check_equal_tail) {
        const unsigned long long key = packed_keys[index];
        int end = index + 1;
        if (PROJECTION_MODE == 5) {
            while (end < candidate_count && packed_keys[end] == key) ++end;
        } else {
            end = equal_key_upper_bound(
                packed_keys, candidate_count, index + 1, key);
        }
        for (int other = index + 1; other < end; ++other) {
            if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
            if (point_dominates<DIM>(packed + other * DIM, point)) {
                alive = false;
                break;
            }
        }
    }

    if (alive) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = packed_indices[index];
    }
}

template <int DIM>
__global__ void verify_morton_sfs(const float* packed,
                                  const unsigned long long* packed_keys,
                                  const int* packed_indices, int candidate_count,
                                  bool check_equal_tail,
                                  int* skyline_indices, int* skyline_count,
                                  unsigned long long* exact_point_checks) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) return;
    float point[DIM];
#pragma unroll
    for (int d = 0; d < DIM; ++d) point[d] = packed[index * DIM + d];
    bool alive = true;
    for (int other = 0; other < index; ++other) {
        if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
        if (point_dominates<DIM>(packed + other * DIM, point)) {
            alive = false;
            break;
        }
    }
    if (alive && check_equal_tail) {
        const unsigned long long key = packed_keys[index];
        const int end = equal_key_upper_bound(packed_keys, candidate_count, index + 1, key);
        for (int other = index + 1; other < end; ++other) {
            if (exact_point_checks) atomicAdd(exact_point_checks, 1ULL);
            if (point_dominates<DIM>(packed + other * DIM, point)) {
                alive = false;
                break;
            }
        }
    }
    if (alive) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = packed_indices[index];
    }
}

template <int DIM>
__global__ void verify_compacted_all_pairs(const float* coordinates,
                                           int source_count,
                                           const int* point_indices,
                                           int point_count,
                                           int* skyline_indices,
                                           int* skyline_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= point_count) return;
    const int source_index = point_indices[index];
    float point[DIM];
#pragma unroll
    for (int d = 0; d < DIM; ++d) {
        point[d] = coordinates[d * source_count + source_index];
    }
    bool alive = true;
    for (int other = 0; other < point_count && alive; ++other) {
        if (other == index) continue;
        const int other_source = point_indices[other];
        bool dominates = true;
        bool strict = false;
#pragma unroll
        for (int d = 0; d < DIM; ++d) {
            const float value = coordinates[d * source_count + other_source];
            if (value > point[d]) dominates = false;
            strict = strict || value < point[d];
        }
        alive = !(dominates && strict);
    }
    if (alive) {
        const int output = atomicAdd(skyline_count, 1);
        skyline_indices[output] = source_index;
    }
}

ChunkLayout build_paper_mkd_chunks_serial(
    const unsigned long long* morton_keys, int valid_count, int target_size) {
    ChunkLayout result;
    int capacity = std::min(valid_count, std::max(1024,
        16 * ((valid_count + target_size - 1) / target_size) + 64));
    for (;;) {
        int* storage = nullptr;
        const std::size_t storage_ints =
            static_cast<std::size_t>(capacity) * 2 + 258;
        CHECK_CUDA_ERROR(cudaMalloc(&storage, storage_ints * sizeof(int)));
        int* leaf_starts = storage;
        int* leaf_ends = leaf_starts + capacity;
        int* stack_starts = leaf_ends + capacity;
        int* stack_ends = stack_starts + 128;
        int* leaf_count_device = stack_ends + 128;
        int* overflow_device = leaf_count_device + 1;
        build_mkd_depth_first<<<1, 1>>>(
            morton_keys, valid_count, target_size, leaf_starts, leaf_ends,
            capacity, stack_starts, stack_ends,
            leaf_count_device, overflow_device);
        CHECK_CUDA_ERROR(cudaGetLastError());
        int host_status[2] = {};
        CHECK_CUDA_ERROR(cudaMemcpy(host_status, leaf_count_device,
                                    2 * sizeof(int), cudaMemcpyDeviceToHost));
        const int leaf_count = host_status[0];
        const bool overflow = host_status[1] != 0 || leaf_count > capacity;
        if (!overflow) {
            result.starts = leaf_starts;
            result.ends = leaf_ends;
            result.storage = storage;
            result.count = leaf_count;
            result.capacity = capacity;
            result.working_bytes = storage_ints * sizeof(int);
            return result;
        }
        cudaFree(storage);
        if (capacity == valid_count) {
            std::fprintf(stderr, "Paper MKD interval capacity exhausted\n");
            std::exit(EXIT_FAILURE);
        }
        capacity = std::min(valid_count, capacity * 4);
    }
}

ChunkLayout build_paper_mkd_chunks(const unsigned long long* morton_keys,
                                   int valid_count, int target_size) {
    constexpr int PARALLEL_FRONTIER_MIN_ESTIMATED_LEAVES = 256;
    const int estimated_leaves =
        (valid_count + target_size - 1) / target_size;
    if (estimated_leaves < PARALLEL_FRONTIER_MIN_ESTIMATED_LEAVES) {
        return build_paper_mkd_chunks_serial(
            morton_keys, valid_count, target_size);
    }
    ChunkLayout result;
    int capacity = std::min(valid_count, std::max(1024,
        4 * ((valid_count + target_size - 1) / target_size) + 64));
    for (;;) {
        int* storage = nullptr;
        const std::size_t storage_ints =
            static_cast<std::size_t>(capacity) * 7;
        CHECK_CUDA_ERROR(cudaMalloc(&storage, storage_ints * sizeof(int)));
        int* current_starts = storage;
        int* current_ends = current_starts + capacity;
        int* next_starts = current_ends + capacity;
        int* next_ends = next_starts + capacity;
        int* expansion_counts = next_ends + capacity;
        int* output_offsets = expansion_counts + capacity;
        int* split_positions = output_offsets + capacity;
        const int root_start = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(current_starts, &root_start, sizeof(int),
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(current_ends, &valid_count, sizeof(int),
                                    cudaMemcpyHostToDevice));
        int current_count = 1;
        bool overflow = false;
        for (;;) {
            classify_mkd_frontier<<<
                (current_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    morton_keys, current_starts, current_ends,
                    current_count, target_size,
                    expansion_counts, split_positions);
            CHECK_CUDA_ERROR(cudaGetLastError());
            thrust::device_ptr<int> expansion_ptr(expansion_counts);
            thrust::device_ptr<int> offset_ptr(output_offsets);
            thrust::exclusive_scan(expansion_ptr, expansion_ptr + current_count,
                                   offset_ptr);
            int last_expansion = 0;
            int last_offset = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&last_expansion,
                expansion_counts + current_count - 1, sizeof(int),
                cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(&last_offset,
                output_offsets + current_count - 1, sizeof(int),
                cudaMemcpyDeviceToHost));
            const int next_count = last_offset + last_expansion;
            if (next_count == current_count) break;
            if (next_count > capacity) {
                overflow = true;
                break;
            }
            scatter_mkd_frontier<<<
                (current_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    current_starts, current_ends, current_count,
                    expansion_counts, output_offsets, split_positions,
                    next_starts, next_ends);
            CHECK_CUDA_ERROR(cudaGetLastError());
            std::swap(current_starts, next_starts);
            std::swap(current_ends, next_ends);
            current_count = next_count;
        }
        if (!overflow) {
            result.starts = current_starts;
            result.ends = current_ends;
            result.storage = storage;
            result.count = current_count;
            result.capacity = capacity;
            result.working_bytes = storage_ints * sizeof(int);
            return result;
        }
        cudaFree(storage);
        if (capacity == valid_count) {
            std::fprintf(stderr, "Paper MKD interval capacity exhausted\n");
            std::exit(EXIT_FAILURE);
        }
        capacity = std::min(valid_count, capacity * 4);
    }
}

ChunkLayout build_fixed_chunks(int valid_count, int chunk_size) {
    ChunkLayout result;
    result.count = (valid_count + chunk_size - 1) / chunk_size;
    result.capacity = result.count;
    CHECK_CUDA_ERROR(cudaMalloc(&result.storage,
                                static_cast<std::size_t>(result.count) * 2 * sizeof(int)));
    result.starts = static_cast<int*>(result.storage);
    result.ends = result.starts + result.count;
    initialize_fixed_chunks<<<(result.count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        valid_count, chunk_size, result.starts, result.ends, result.count);
    CHECK_CUDA_ERROR(cudaGetLastError());
    result.working_bytes = static_cast<std::size_t>(result.count) * 2 * sizeof(int);
    return result;
}

template <typename KernelLauncher>
void launch_by_dimension(int dim, KernelLauncher launcher) {
    switch (dim) {
    case 3: launcher.template operator()<3>(); break;
    case 4: launcher.template operator()<4>(); break;
    case 5: launcher.template operator()<5>(); break;
    case 6: launcher.template operator()<6>(); break;
    case 7: launcher.template operator()<7>(); break;
    case 8: launcher.template operator()<8>(); break;
    case 9: launcher.template operator()<9>(); break;
    case 10: launcher.template operator()<10>(); break;
    case 11: launcher.template operator()<11>(); break;
    case 12: launcher.template operator()<12>(); break;
    case 13: launcher.template operator()<13>(); break;
    case 14: launcher.template operator()<14>(); break;
    case 15: launcher.template operator()<15>(); break;
    case 16: launcher.template operator()<16>(); break;
    default: std::fprintf(stderr, "Unsupported dimension: %d\n", dim); std::exit(EXIT_FAILURE);
    }
}

struct LocalPruneLauncher {
    const float* ordered;
    int valid_count;
    const int* starts;
    const int* ends;
    int chunk_count;
    int* alive;
    std::size_t shared_bytes;
    template <int DIM> void operator()() const {
        local_skyline_by_chunk<DIM><<<chunk_count, BLOCK_SIZE, shared_bytes>>>(
            ordered, valid_count, starts, ends, alive);
    }
};

struct LocalAppendLauncher {
    const float* ordered;
    int valid_count;
    const int* starts;
    const int* ends;
    int chunk_count;
    int* candidate_positions;
    int* candidate_chunk_ids;
    int* candidate_count;
    int* chunk_counts;
    std::size_t shared_bytes;

    template <int DIM>
    void operator()() const {
        local_skyline_append_by_chunk<DIM><<<chunk_count, BLOCK_SIZE, shared_bytes>>>(
            ordered, valid_count, starts, ends, candidate_positions, candidate_chunk_ids,
            candidate_count, chunk_counts);
    }
};

struct LocalFlagsLauncher {
    const float* ordered;
    int valid_count;
    const int* starts;
    const int* ends;
    int chunk_count;
    int* alive_flags;
    std::size_t shared_bytes;

    template <int DIM>
    void operator()() const {
        local_skyline_flags_by_chunk<DIM><<<chunk_count, BLOCK_SIZE, shared_bytes>>>(
            ordered, valid_count, starts, ends, alive_flags);
    }
};

struct ChunkVerifyLauncher {
    const float* packed;
    const unsigned long long* keys;
    const int* indices;
    int candidate_count;
    const int* chunk_ids;
    const int* offsets;
    const int* counts;
    const float* corners;
    const float* projection_bounds;
    const float* projection_minimums;
    const float* projection_maximums;
    int projection_pair_count;
    bool projection_cycle;
    bool projection_quantized_cycle;
    const unsigned short* projection_3d_bounds;
    int projection_3d_triple_count;
    const int* micro_offsets;
    const int* micro_counts;
    const __half* micro_min_corners;
    const unsigned int* packed_micro_corners;
    const unsigned int* micro_point_codes;
    const unsigned int* compatibility_masks;
    int compatibility_words_per_row;
    int bits_per_dim;
    bool check_equal_tail;
    bool check_own_prefix;
    int* skyline_indices;
    int* skyline_count;
    unsigned long long* triggers;
    unsigned long long* projection_bound_skips;
    unsigned long long* projection_3d_skips;
    unsigned long long* false_triggers;
    unsigned long long* checks;

    template <int DIM, int PROJECTION_MODE, bool USE_POINT_MASK>
    void launch_compatible() const {
        verify_paper_chunks_compatible<DIM, PROJECTION_MODE, USE_POINT_MASK><<<
            (candidate_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                packed, keys, indices, candidate_count, chunk_ids, offsets, counts,
                corners, projection_bounds, projection_minimums,
                projection_maximums, projection_pair_count, projection_3d_bounds,
                projection_3d_triple_count, micro_offsets, micro_counts,
                micro_min_corners, packed_micro_corners, micro_point_codes,
                compatibility_masks,
                compatibility_words_per_row, bits_per_dim, check_equal_tail,
                check_own_prefix, skyline_indices, skyline_count, triggers,
                projection_bound_skips, projection_3d_skips, false_triggers,
                checks);
    }

    template <int DIM, int PROJECTION_MODE, bool USE_POINT_MASK>
    void launch_legacy() const {
        verify_paper_chunks<DIM, PROJECTION_MODE, USE_POINT_MASK><<<
            (candidate_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                packed, keys, indices, candidate_count, chunk_ids, offsets, counts,
                corners, projection_bounds, projection_minimums,
                projection_maximums, projection_pair_count, projection_3d_bounds,
                projection_3d_triple_count, micro_offsets, micro_counts,
                micro_min_corners, packed_micro_corners, micro_point_codes,
                bits_per_dim,
                check_equal_tail,
                check_own_prefix, skyline_indices, skyline_count, triggers,
                projection_bound_skips, projection_3d_skips, false_triggers,
                checks);
    }

    template <int DIM, int PROJECTION_MODE, bool USE_POINT_MASK>
    void launch_selected_mode() const {
        if (compatibility_masks) {
            launch_compatible<DIM, PROJECTION_MODE, USE_POINT_MASK>();
        } else {
            launch_legacy<DIM, PROJECTION_MODE, USE_POINT_MASK>();
        }
    }

    template <int DIM, int PROJECTION_MODE>
    void launch_selected() const {
        if (micro_point_codes) {
            launch_selected_mode<DIM, PROJECTION_MODE, true>();
        } else {
            launch_selected_mode<DIM, PROJECTION_MODE, false>();
        }
    }

    template <int DIM> void operator()() const {
        const int projection_mode = (micro_min_corners || packed_micro_corners)
            ? 5
            : (projection_bounds ? (projection_cycle ? 3 : 1)
            : (projection_3d_bounds ? (projection_quantized_cycle ? 4 : 2) : 0));
        if (projection_mode == 1) {
            launch_selected<DIM, 1>();
        } else if (projection_mode == 2) {
            launch_selected<DIM, 2>();
        } else if (projection_mode == 3) {
            launch_selected<DIM, 3>();
        } else if (projection_mode == 4) {
            launch_selected<DIM, 4>();
        } else if (projection_mode == 5) {
            launch_selected<DIM, 5>();
        } else {
            launch_selected<DIM, 0>();
        }
    }
};

struct DirectVerifyLauncher {
    const float* packed;
    const unsigned long long* keys;
    const int* indices;
    int candidate_count;
    bool check_equal_tail;
    int* skyline_indices;
    int* skyline_count;
    unsigned long long* checks;
    template <int DIM> void operator()() const {
        verify_morton_sfs<DIM><<<(candidate_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            packed, keys, indices, candidate_count, check_equal_tail,
            skyline_indices, skyline_count, checks);
    }
};

struct CompactedDirectVerifyLauncher {
    const float* coordinates;
    int source_count;
    const int* point_indices;
    int point_count;
    int* skyline_indices;
    int* skyline_count;
    template <int DIM> void operator()() const {
        verify_compacted_all_pairs<DIM><<<
            (point_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            coordinates, source_count, point_indices, point_count,
            skyline_indices, skyline_count);
    }
};

AlgorithmResult run_paper_variant(const std::vector<MyDataPoint>& points,
                                  const AlgorithmConfig& config,
                                  PaperVariant variant) {
    AlgorithmResult result;
    if (variant == PaperVariant::Full) {
        result.name = "MKSky-paper";
        result.provenance = "Endpoint-XOR MKD local pruning with compacted survivor sorting";
    } else if (variant == PaperVariant::GridMkdNoDirect) {
        result.name = "MKSky-ablation-A";
        result.provenance = "Cumulative ablation A: disable exact small-candidate direct verification";
    } else if (variant == PaperVariant::RoutingDisabled) {
        result.name = "MKSky-ablation-B";
        result.provenance = "Cumulative ablation B: A plus disable low-dimensional occupied-cell pruning";
    } else if (variant == PaperVariant::MkdWithoutSkipping) {
        result.name = "MKSky-ablation-C";
        result.provenance = "Cumulative ablation C: B plus removal of Min-Corner chunk skipping";
    } else if (variant == PaperVariant::FixedPartition) {
        result.name = "MKSky-fixed-partition";
        result.provenance = "Equal-size chunks replacing endpoint-XOR MKD";
    } else if (variant == PaperVariant::FixedWithoutSkipping) {
        result.name = "MKSky-fixed-no-skipping";
        result.provenance = "Equal-size chunks without Min-Corner chunk skipping";
    } else if (variant == PaperVariant::MortonSfs) {
        result.name = "MKSky-morton-SFS";
        result.provenance = "Morton-ordered direct SFS diagnostic variant";
    } else if (variant == PaperVariant::SumOrder) {
        result.name = "MKSky-sum-order";
        result.provenance = "Dominance-preserving coordinate-sum order with fixed local chunks";
    } else if (variant == PaperVariant::SumOrderWithoutSkipping) {
        result.name = "MKSky-sum-order-no-skipping";
        result.provenance =
            "Coordinate-sum ordering with fixed local chunks and no chunk skipping";
    } else if (variant == PaperVariant::GridMkdProjectionBound) {
        result.name = "MKSky-prefix-local";
        result.provenance =
            "Exact Grid-MKD with dominance-monotone prefix-only local verification";
    } else if (variant == PaperVariant::GridMkdProjection3D) {
        result.name = "MKSky-projection-3d";
        result.provenance =
            "Grid-MKD with exact-safe quantized three-dimensional chunk summaries";
    } else if (variant == PaperVariant::GridMkdProjectionScan) {
        result.name = "MKSky-projection-scan";
        result.provenance =
            "Projection-bound Grid-MKD with scan-based stable candidate compaction";
    } else if (variant == PaperVariant::GridMkdProjectionCycle) {
        result.name = "MKSky-projection-cycle";
        result.provenance =
            "Grid-MKD with cyclic two-dimensional chunk summaries";
    } else if (variant == PaperVariant::GridMkdProjectionQuantizedCycle) {
        result.name = "MKSky-projection-qcycle";
        result.provenance =
            "Grid-MKD with conservative quantized cyclic chunk summaries";
    } else {
        result.name = "MKSky-grid-mkd";
        result.provenance =
            "Sampled workload gate with low-dimensional occupied-cell pruning, "
            "compacted Morton MKD, and exact small-candidate verification";
    }

    const auto wall_begin = std::chrono::high_resolution_clock::now();
    const int count = static_cast<int>(points.size());
    const int dim = config.dim;
    result.input_count = count;
    std::vector<float> host_coordinates(static_cast<std::size_t>(count) * dim);
    for (int d = 0; d < dim; ++d) {
        for (int i = 0; i < count; ++i) {
            host_coordinates[static_cast<std::size_t>(d) * count + i] = points[i].coords[d];
        }
    }

    float* coordinates = nullptr;
    float* pivot_block_sums = nullptr;
    int* pivot_block_indices = nullptr;
    int* pivot_index_device = nullptr;
    float* minimums = nullptr;
    float* maximums = nullptr;
    unsigned long long* morton_keys = nullptr;
    int* sorted_indices = nullptr;
    int* survivor_flags = nullptr;
    int* survivor_offsets = nullptr;
    float* ordered = nullptr;
    int* ordered_indices = nullptr;
    int* candidate_positions = nullptr;
    int* candidate_counter = nullptr;
    float* packed = nullptr;
    unsigned long long* packed_keys = nullptr;
    int* packed_indices = nullptr;
    int* packed_chunk_ids = nullptr;
    int* skyline_indices = nullptr;
    int* skyline_count = nullptr;
    float* sum_order_keys = nullptr;
    unsigned int* grid_cell_ids = nullptr;
    int* grid_occupancy_a = nullptr;
    int* grid_occupancy_b = nullptr;
    int* grid_keep = nullptr;
    const bool projection_variant =
        variant == PaperVariant::GridMkdProjectionBound ||
        variant == PaperVariant::GridMkdProjection3D ||
        variant == PaperVariant::GridMkdProjectionScan ||
        variant == PaperVariant::GridMkdProjectionCycle ||
        variant == PaperVariant::GridMkdProjectionQuantizedCycle;
    const bool optimize_memory = projection_variant;
    const bool reuse_coordinate_buffer = optimize_memory;
    // These integer buffers belong to disjoint pipeline phases. CUDA stream
    // ordering guarantees that the earlier reader has finished before reuse.
    const bool reuse_index_buffers = optimize_memory;
    const bool reuse_candidate_positions = reuse_index_buffers;
    const bool reuse_packed_indices = reuse_index_buffers;
    const bool reuse_skyline_indices = reuse_index_buffers;
    const bool reuse_packed_chunk_ids =
        reuse_index_buffers && variant != PaperVariant::GridMkdProjectionScan;
    const bool use_grid_prefilter = config.enable_grid_prefilter &&
                                    (variant == PaperVariant::GridMkd ||
                                     variant == PaperVariant::GridMkdProjectionBound ||
                                     variant == PaperVariant::GridMkdProjection3D ||
                                     variant == PaperVariant::GridMkdProjectionScan ||
                                     variant == PaperVariant::GridMkdProjectionCycle ||
                                     variant == PaperVariant::GridMkdProjectionQuantizedCycle ||
                                     variant == PaperVariant::GridMkdNoDirect);
    const bool allocate_grid_buffers =
        use_grid_prefilter && (!optimize_memory || dim <= 4);
    const int maximum_grid_side =
        allocate_grid_buffers ? select_grid_side(count, dim) : 0;
    int maximum_grid_cell_count = 0;
    if (allocate_grid_buffers) {
        maximum_grid_cell_count = 1;
        for (int d = 0; d < dim; ++d) maximum_grid_cell_count *= maximum_grid_side;
    }
    const int input_block_count = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA_ERROR(cudaMalloc(&coordinates, host_coordinates.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&pivot_block_sums, input_block_count * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&pivot_block_indices, input_block_count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&pivot_index_device, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&minimums, dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&maximums, dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&survivor_flags, count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&survivor_offsets, count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&morton_keys, count * sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMalloc(&sorted_indices, count * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&ordered,
                                static_cast<std::size_t>(count) * dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&ordered_indices, count * sizeof(int)));
    if (reuse_candidate_positions) {
        candidate_positions = survivor_offsets;
    } else {
        CHECK_CUDA_ERROR(cudaMalloc(&candidate_positions, count * sizeof(int)));
    }
    CHECK_CUDA_ERROR(cudaMalloc(&candidate_counter, sizeof(int)));
    if (reuse_coordinate_buffer) {
        packed = coordinates;
    } else {
        CHECK_CUDA_ERROR(cudaMalloc(&packed,
                                    static_cast<std::size_t>(count) * dim * sizeof(float)));
    }
    CHECK_CUDA_ERROR(cudaMalloc(&packed_keys, count * sizeof(unsigned long long)));
    if (reuse_packed_indices) {
        packed_indices = sorted_indices;
    } else {
        CHECK_CUDA_ERROR(cudaMalloc(&packed_indices, count * sizeof(int)));
    }
    if (reuse_packed_chunk_ids) {
        packed_chunk_ids = survivor_flags;
    } else {
        CHECK_CUDA_ERROR(cudaMalloc(&packed_chunk_ids, count * sizeof(int)));
    }
    if (reuse_skyline_indices) {
        skyline_indices = ordered_indices;
    } else {
        CHECK_CUDA_ERROR(cudaMalloc(&skyline_indices, count * sizeof(int)));
    }
    CHECK_CUDA_ERROR(cudaMalloc(&skyline_count, sizeof(int)));
    if (!optimize_memory) {
        CHECK_CUDA_ERROR(cudaMalloc(&sum_order_keys, count * sizeof(float)));
    }
    if (allocate_grid_buffers) {
        CHECK_CUDA_ERROR(cudaMalloc(&grid_cell_ids, count * sizeof(unsigned int)));
        CHECK_CUDA_ERROR(cudaMalloc(&grid_occupancy_a, maximum_grid_cell_count * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&grid_occupancy_b, maximum_grid_cell_count * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&grid_keep, count * sizeof(int)));
    }
    CHECK_CUDA_ERROR(cudaMemcpy(coordinates, host_coordinates.data(),
                                host_coordinates.size() * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start_event;
    cudaEvent_t phase_event;
    cudaEvent_t stop_event;
    cudaEvent_t pivot_event = nullptr;
    cudaEvent_t compact_event = nullptr;
    cudaEvent_t sort_event = nullptr;
    cudaEvent_t reorder_event = nullptr;
    cudaEvent_t mkd_event = nullptr;
    cudaEvent_t local_event = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&phase_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
    if (config.collect_telemetry) {
        CHECK_CUDA_ERROR(cudaEventCreate(&pivot_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&compact_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&sort_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&reorder_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&mkd_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&local_event));
    }
    CHECK_CUDA_ERROR(cudaEventRecord(start_event));

    if (config.input_normalized) {
        const std::vector<float> host_minimums(dim, 0.0f);
        const std::vector<float> host_maximums(dim, 1.0f);
        CHECK_CUDA_ERROR(cudaMemcpy(minimums, host_minimums.data(),
                                    dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(maximums, host_maximums.data(),
                                    dim * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        thrust::device_ptr<float> coordinate_ptr(coordinates);
        for (int d = 0; d < dim; ++d) {
            const auto pair = thrust::minmax_element(
                coordinate_ptr + static_cast<std::size_t>(d) * count,
                coordinate_ptr + static_cast<std::size_t>(d + 1) * count);
            CHECK_CUDA_ERROR(cudaMemcpy(minimums + d, thrust::raw_pointer_cast(pair.first),
                                        sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(maximums + d, thrust::raw_pointer_cast(pair.second),
                                        sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }

    const dim3 block(BLOCK_SIZE);
    const dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    bool use_grid_now = false;
    int valid_count = 0;
    if (!config.enable_candidate_reduction) {
        thrust::device_ptr<int> output_ptr(sorted_indices);
        thrust::sequence(output_ptr, output_ptr + count);
        valid_count = count;
        if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(pivot_event));
    } else {
        if (use_grid_prefilter && dim <= 4) {
            const int sample_count = std::min(count, config.mksky_sample_limit);
            const int sample_stride = choose_sample_stride(count);
            const int sample_offset = count > 1
                ? static_cast<int>(2246822519ULL %
                    static_cast<unsigned long long>(count))
                : 0;
            estimate_sample_pivot<<<1, BLOCK_SIZE>>>(
                coordinates, count, dim, sample_count, sample_stride, sample_offset,
                pivot_index_device, candidate_counter);
            int sample_survivor_count = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&sample_survivor_count, candidate_counter,
                                        sizeof(int), cudaMemcpyDeviceToHost));
            use_grid_now = static_cast<long long>(sample_survivor_count) *
                               config.mksky_survivor_ratio_divisor >
                           sample_count;
        }
        if (use_grid_now) {
            const int grid_side = select_grid_side(count, dim);
            int grid_cell_count = 1;
            for (int d = 0; d < dim; ++d) grid_cell_count *= grid_side;
            CHECK_CUDA_ERROR(cudaMemset(grid_occupancy_a, 0,
                                        grid_cell_count * sizeof(int)));
            if (config.input_normalized) {
                assign_grid_cells_normalized_soa<<<grid, block>>>(
                    coordinates, count, dim, grid_side,
                    grid_cell_ids, grid_occupancy_a);
            } else {
                assign_grid_cells_soa<<<grid, block>>>(
                    coordinates, count, dim, grid_side, minimums, maximums,
                    grid_cell_ids, grid_occupancy_a);
            }
            int* prefix_input = grid_occupancy_a;
            int* prefix_output = grid_occupancy_b;
            int stride = 1;
            const dim3 cell_grid((grid_cell_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
            for (int d = 0; d < dim; ++d) {
                for (int offset = 1; offset < grid_side; offset <<= 1) {
                    grid_prefix_scan_step<<<cell_grid, block>>>(
                        prefix_input, prefix_output, grid_cell_count,
                        grid_side, stride, offset);
                    std::swap(prefix_input, prefix_output);
                }
                stride *= grid_side;
            }
            mark_grid_candidates<<<grid, block>>>(
                grid_cell_ids, prefix_input, count, dim, grid_side, grid_keep);
        } else {
            reduce_pivot_blocks<<<grid, block>>>(coordinates, count, dim,
                                                 pivot_block_sums, pivot_block_indices);
            finalize_pivot<<<1, BLOCK_SIZE>>>(pivot_block_sums, pivot_block_indices,
                                              input_block_count, pivot_index_device);
            mark_pivot_survivors<<<grid, block>>>(coordinates, count, dim,
                                                  pivot_index_device, nullptr,
                                                  survivor_flags);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(pivot_event));
        int* active_survivor_flags = use_grid_now ? grid_keep : survivor_flags;
        thrust::device_ptr<int> survivor_offset_ptr(survivor_offsets);
        if (use_grid_now) {
            const auto index_begin = thrust::make_counting_iterator<int>(0);
            thrust::device_ptr<int> output_ptr(sorted_indices);
            thrust::device_ptr<int> flag_ptr(active_survivor_flags);
            valid_count = static_cast<int>(
                thrust::copy_if(index_begin, index_begin + count, flag_ptr,
                                output_ptr, KeepMarkedPoint()) - output_ptr);
        } else {
            thrust::device_ptr<int> survivor_flag_ptr(active_survivor_flags);
            thrust::exclusive_scan(survivor_flag_ptr, survivor_flag_ptr + count,
                                   survivor_offset_ptr);
            int last_offset = 0;
            int last_flag = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&last_offset, survivor_offsets + count - 1,
                                        sizeof(int), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaMemcpy(&last_flag, active_survivor_flags + count - 1,
                                        sizeof(int), cudaMemcpyDeviceToHost));
            valid_count = last_offset + last_flag;
            compact_survivor_indices<<<grid, block>>>(active_survivor_flags,
                                                       survivor_offsets, count,
                                                       sorted_indices);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    result.valid_count = valid_count;
    result.prefilter_route = !config.enable_candidate_reduction ? 2 : (use_grid_now ? 1 : 0);
    if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(compact_event));

    const bool use_compacted_direct = config.enable_small_candidate_direct &&
        (variant == PaperVariant::GridMkd ||
         variant == PaperVariant::GridMkdProjectionBound ||
         variant == PaperVariant::GridMkdProjection3D ||
         variant == PaperVariant::GridMkdProjectionScan ||
         variant == PaperVariant::GridMkdProjectionCycle ||
         variant == PaperVariant::GridMkdProjectionQuantizedCycle) &&
        dim <= 4 && valid_count <= config.mksky_direct_limit;
    if (use_compacted_direct) {
        result.candidate_count = valid_count;
        result.adaptive_local_chunk_size = 0;
        result.mkd_chunk_count = 0;
        CHECK_CUDA_ERROR(cudaMemset(skyline_count, 0, sizeof(int)));
        CHECK_CUDA_ERROR(cudaEventRecord(phase_event));
        launch_by_dimension(dim, CompactedDirectVerifyLauncher{
            coordinates, count, sorted_indices, valid_count,
            skyline_indices, skyline_count});
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaEventRecord(stop_event));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_event));
        float direct_device_ms = 0.0f;
        float direct_preprocess_ms = 0.0f;
        float direct_core_ms = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&direct_device_ms,
                                               start_event, stop_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&direct_preprocess_ms,
                                               start_event, phase_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&direct_core_ms,
                                               phase_event, stop_event));
        result.device_ms = direct_device_ms;
        result.preprocess_ms = direct_preprocess_ms;
        result.core_ms = direct_core_ms;
        int output_count = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(&output_count, skyline_count,
                                    sizeof(int), cudaMemcpyDeviceToHost));
        result.skyline_indices.resize(output_count);
        CHECK_CUDA_ERROR(cudaMemcpy(result.skyline_indices.data(), skyline_indices,
                                    output_count * sizeof(int), cudaMemcpyDeviceToHost));
        std::sort(result.skyline_indices.begin(), result.skyline_indices.end());
        const std::size_t approximate_bytes =
            static_cast<std::size_t>(count) *
                (dim * sizeof(float) + 8 * sizeof(int) +
                 3 * sizeof(unsigned long long) + 3 * dim * sizeof(float)) +
            static_cast<std::size_t>(maximum_grid_cell_count) * 2 * sizeof(int);
        result.device_memory_mb =
            static_cast<double>(approximate_bytes) / (1024.0 * 1024.0);

        cudaFree(skyline_count);
        if (!reuse_skyline_indices) cudaFree(skyline_indices);
        if (!reuse_packed_chunk_ids) cudaFree(packed_chunk_ids);
        if (!reuse_packed_indices) cudaFree(packed_indices);
        cudaFree(packed_keys);
        if (!reuse_coordinate_buffer) cudaFree(packed);
        cudaFree(ordered_indices);
        cudaFree(ordered);
        cudaFree(candidate_counter);
        if (!reuse_candidate_positions) cudaFree(candidate_positions);
        cudaFree(sorted_indices);
        cudaFree(morton_keys);
        cudaFree(sum_order_keys);
        cudaFree(grid_keep);
        cudaFree(grid_occupancy_b);
        cudaFree(grid_occupancy_a);
        cudaFree(grid_cell_ids);
        cudaFree(survivor_offsets);
        cudaFree(survivor_flags);
        cudaFree(maximums);
        cudaFree(minimums);
        cudaFree(pivot_index_device);
        cudaFree(pivot_block_indices);
        cudaFree(pivot_block_sums);
        cudaFree(coordinates);
        cudaEventDestroy(stop_event);
        cudaEventDestroy(phase_event);
        cudaEventDestroy(start_event);
        if (sort_event) cudaEventDestroy(sort_event);
        if (compact_event) cudaEventDestroy(compact_event);
        if (pivot_event) cudaEventDestroy(pivot_event);
        if (local_event) cudaEventDestroy(local_event);
        if (mkd_event) cudaEventDestroy(mkd_event);
        if (reorder_event) cudaEventDestroy(reorder_event);
        const auto wall_end = std::chrono::high_resolution_clock::now();
        result.wall_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
        return result;
    }

    const int bits_per_dim = std::min(20, 64 / dim);
    thrust::device_ptr<int> index_ptr(sorted_indices);
    const bool use_sum_order = variant == PaperVariant::SumOrder ||
                               variant == PaperVariant::SumOrderWithoutSkipping;
    if (use_sum_order) {
        compute_compacted_coordinate_sums<<<
            (valid_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            coordinates, count, dim, sorted_indices, valid_count, sum_order_keys);
        thrust::device_ptr<float> sum_key_ptr(sum_order_keys);
        thrust::sort_by_key(sum_key_ptr, sum_key_ptr + valid_count, index_ptr);
    } else {
        encode_compacted_survivors<<<
            (valid_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            coordinates, count, dim, sorted_indices, valid_count, minimums, maximums,
            morton_keys, bits_per_dim);
        CHECK_CUDA_ERROR(cudaGetLastError());
        thrust::device_ptr<unsigned long long> key_ptr(morton_keys);
        thrust::sort_by_key(key_ptr, key_ptr + valid_count, index_ptr);
    }
    if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(sort_event));
    reorder_paper_points<<<(valid_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        coordinates, count, dim, sorted_indices, valid_count, ordered, ordered_indices);
    CHECK_CUDA_ERROR(cudaGetLastError());
    if (use_sum_order) {
        compute_ordered_sum_keys<<<
            (valid_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            ordered, valid_count, dim, morton_keys);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(reorder_event));

    // Larger inputs use larger leaves to limit chunk metadata and cross-chunk work.
    int chunk_size = dim >= 10 ? 512 : 1024;
    if (dim >= 12) {
        chunk_size = count >= 16000000 ? 512 : (count >= 4000000 ? 256 : 128);
    }
    if (config.mksky_chunk_size_override > 0) {
        chunk_size = config.mksky_chunk_size_override;
    }
    result.adaptive_local_chunk_size = chunk_size;
    const bool use_chunk_layout = variant != PaperVariant::MortonSfs;
    const bool use_local_prefix_filtering =
        use_chunk_layout && config.enable_local_prefix_filtering;
    const bool use_chunk_skipping = config.enable_block_skipping &&
                                    use_local_prefix_filtering &&
                                    (variant == PaperVariant::Full ||
                                     variant == PaperVariant::GridMkdNoDirect ||
                                     variant == PaperVariant::RoutingDisabled ||
                                     variant == PaperVariant::FixedPartition ||
                                     variant == PaperVariant::SumOrder ||
                                     variant == PaperVariant::GridMkd ||
                                     variant == PaperVariant::GridMkdProjectionBound ||
                                     variant == PaperVariant::GridMkdProjection3D ||
                                     variant == PaperVariant::GridMkdProjectionScan ||
                                     variant == PaperVariant::GridMkdProjectionCycle ||
                                     variant == PaperVariant::GridMkdProjectionQuantizedCycle);
    const bool use_mkd = config.enable_mkd_partition &&
                         (variant == PaperVariant::Full ||
                          variant == PaperVariant::GridMkdNoDirect ||
                          variant == PaperVariant::RoutingDisabled ||
                          variant == PaperVariant::MkdWithoutSkipping ||
                          variant == PaperVariant::GridMkd ||
                          variant == PaperVariant::GridMkdProjectionBound ||
                          variant == PaperVariant::GridMkdProjection3D ||
                          variant == PaperVariant::GridMkdProjectionScan ||
                          variant == PaperVariant::GridMkdProjectionCycle ||
                          variant == PaperVariant::GridMkdProjectionQuantizedCycle);
    ChunkLayout chunks;
    int candidate_count = valid_count;
    int* chunk_counts = nullptr;
    int* chunk_offsets = nullptr;
    int* micro_counts = nullptr;
    int* micro_offsets = nullptr;
    float* min_corners = nullptr;
    __half* micro_min_corners = nullptr;
    unsigned int* packed_micro_corners = nullptr;
    unsigned int* micro_point_codes = nullptr;
    unsigned int* compatibility_masks = nullptr;
    int compatibility_words_per_row = 0;
    std::size_t compatibility_mask_bytes = 0;
    float* projection_bounds = nullptr;
    unsigned short* projection_3d_bounds = nullptr;
    const bool projection_quantized_cycle =
        variant == PaperVariant::GridMkdProjectionQuantizedCycle;
    const bool projection_cycle =
        variant == PaperVariant::GridMkdProjectionCycle ||
        projection_quantized_cycle;
    const int projection_pair_count = projection_cycle
        ? dim : std::min(PROJECTION_PAIR_COUNT, dim / 2);
    const int projection_3d_triple_count = dim / 3;
    bool use_projection_bound = false;
    bool use_projection_3d = false;
    bool use_projection_quantized_cycle = false;
    const bool use_scan_compaction =
        variant == PaperVariant::GridMkdProjectionScan;

    if (use_chunk_layout) {
        chunks = use_mkd
            ? build_paper_mkd_chunks(morton_keys, valid_count, chunk_size)
            : build_fixed_chunks(valid_count, chunk_size);
        result.mkd_chunk_count = use_mkd ? chunks.count : 0;
        if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(mkd_event));
        CHECK_CUDA_ERROR(cudaMalloc(&chunk_counts, chunks.count * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&chunk_offsets, chunks.count * sizeof(int)));
        const std::size_t local_shared_bytes =
            static_cast<std::size_t>(chunk_size) * dim * sizeof(float);
        if (!use_local_prefix_filtering) {
            candidate_count = valid_count;
            pack_unfiltered_chunks<<<chunks.count, BLOCK_SIZE>>>(
                ordered, valid_count, dim, morton_keys, ordered_indices,
                chunks.starts, chunks.ends, chunk_counts,
                packed, packed_keys, packed_indices, packed_chunk_ids);
            CHECK_CUDA_ERROR(cudaGetLastError());
            if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(local_event));
        } else {
            if (use_scan_compaction) {
                launch_by_dimension(dim, LocalFlagsLauncher{
                    ordered, valid_count, chunks.starts, chunks.ends, chunks.count,
                    survivor_flags, local_shared_bytes});
                CHECK_CUDA_ERROR(cudaGetLastError());
                if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(local_event));
                thrust::device_ptr<int> flag_ptr(survivor_flags);
                thrust::device_ptr<int> prefix_ptr(survivor_offsets);
                thrust::exclusive_scan(flag_ptr, flag_ptr + valid_count, prefix_ptr);
                int last_offset = 0;
                int last_flag = 0;
                CHECK_CUDA_ERROR(cudaMemcpy(&last_offset, survivor_offsets + valid_count - 1,
                                            sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA_ERROR(cudaMemcpy(&last_flag, survivor_flags + valid_count - 1,
                                            sizeof(int), cudaMemcpyDeviceToHost));
                candidate_count = last_offset + last_flag;
                count_chunk_survivors<<<chunks.count, BLOCK_SIZE>>>(
                    survivor_flags, chunks.starts, chunks.ends, chunk_counts);
            } else {
                CHECK_CUDA_ERROR(cudaMemset(candidate_counter, 0, sizeof(int)));
                launch_by_dimension(dim, LocalAppendLauncher{
                    ordered, valid_count, chunks.starts, chunks.ends, chunks.count,
                    candidate_positions, packed_chunk_ids, candidate_counter, chunk_counts,
                    local_shared_bytes});
                CHECK_CUDA_ERROR(cudaGetLastError());
                if (config.collect_telemetry) CHECK_CUDA_ERROR(cudaEventRecord(local_event));
                CHECK_CUDA_ERROR(cudaMemcpy(&candidate_count, candidate_counter,
                                            sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
        use_projection_bound = config.enable_aux_summaries &&
            ((ENABLE_PAIR_SUMMARY &&
              variant == PaperVariant::GridMkdProjectionBound) ||
             variant == PaperVariant::GridMkdProjectionScan ||
             variant == PaperVariant::GridMkdProjectionCycle) &&
            candidate_count >= config.mksky_aux_threshold;
        use_projection_3d = config.enable_aux_summaries &&
            variant == PaperVariant::GridMkdProjection3D &&
            candidate_count >= config.mksky_aux_threshold;
        use_projection_quantized_cycle = config.enable_aux_summaries &&
            projection_quantized_cycle &&
            candidate_count >= config.mksky_aux_threshold;
        thrust::device_ptr<int> count_ptr(chunk_counts);
        thrust::device_ptr<int> offset_ptr(chunk_offsets);
        thrust::exclusive_scan(count_ptr, count_ptr + chunks.count, offset_ptr);
        if (use_local_prefix_filtering) {
            if (use_scan_compaction) {
                scatter_chunk_survivors<<<chunks.count, BLOCK_SIZE>>>(
                    ordered, valid_count, dim, morton_keys, ordered_indices,
                    survivor_flags, survivor_offsets, chunks.starts, chunks.ends,
                    packed, packed_keys, packed_indices, packed_chunk_ids);
            } else {
                thrust::device_ptr<int> candidate_position_ptr(candidate_positions);
                thrust::device_ptr<int> candidate_chunk_ptr(packed_chunk_ids);
                thrust::sort_by_key(candidate_position_ptr,
                                    candidate_position_ptr + candidate_count,
                                    candidate_chunk_ptr);
                pack_local_candidates<<<
                    (candidate_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    ordered, valid_count, dim, morton_keys, ordered_indices,
                    candidate_positions, candidate_count,
                    packed, packed_keys, packed_indices);
            }
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        if (config.enable_aux_summaries &&
            variant == PaperVariant::GridMkdProjectionBound &&
            dim >= config.mksky_signature_min_dim &&
            candidate_count >= config.mksky_aux_threshold) {
            build_dimension_signatures<<<
                (candidate_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    packed_keys, candidate_count, dim, bits_per_dim);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
        if (use_chunk_skipping) {
            CHECK_CUDA_ERROR(cudaMalloc(&min_corners,
                static_cast<std::size_t>(chunks.count) * dim * sizeof(float)));
            compute_chunk_min_corners<<<chunks.count, 32>>>(
                packed, dim, chunk_offsets, chunk_counts, min_corners);
            CHECK_CUDA_ERROR(cudaGetLastError());
            if (config.enable_aux_summaries &&
                variant == PaperVariant::GridMkdProjectionBound &&
                candidate_count >= config.mksky_aux_threshold &&
                chunks.count >= config.mksky_compatibility_min_chunks) {
                const int compatibility_group_count =
                    (chunks.count + COMPATIBILITY_GROUP_SIZE - 1) /
                    COMPATIBILITY_GROUP_SIZE;
                compatibility_words_per_row =
                    (compatibility_group_count + 31) / 32;
                compatibility_mask_bytes =
                    static_cast<std::size_t>(chunks.count) *
                    compatibility_words_per_row * sizeof(unsigned int);
                if (compatibility_mask_bytes <= config.mksky_compatibility_max_bytes) {
                    float* compatibility_max_corners = nullptr;
                    float* compatibility_group_min_corners = nullptr;
                    CHECK_CUDA_ERROR(cudaMalloc(&compatibility_max_corners,
                        static_cast<std::size_t>(chunks.count) * dim * sizeof(float)));
                    CHECK_CUDA_ERROR(cudaMalloc(&compatibility_group_min_corners,
                        static_cast<std::size_t>(compatibility_group_count) *
                        dim * sizeof(float)));
                    CHECK_CUDA_ERROR(cudaMalloc(&compatibility_masks,
                        compatibility_mask_bytes));
                    compute_chunk_max_corners<<<chunks.count, 32>>>(
                        packed, dim, chunk_offsets, chunk_counts,
                        compatibility_max_corners);
                    compute_compatibility_group_min_corners<<<
                        compatibility_group_count, 32>>>(
                            min_corners, chunks.count, dim,
                            compatibility_group_min_corners);
                    const std::size_t compatibility_warps =
                        static_cast<std::size_t>(chunks.count) *
                        compatibility_words_per_row;
                    const std::size_t compatibility_threads =
                        compatibility_warps * 32ULL;
                    build_compatibility_group_masks<<<
                        static_cast<unsigned int>(
                            (compatibility_threads + BLOCK_SIZE - 1) / BLOCK_SIZE),
                        BLOCK_SIZE>>>(
                            compatibility_group_min_corners,
                            compatibility_max_corners, chunks.count,
                            compatibility_group_count,
                            compatibility_words_per_row, dim,
                            compatibility_masks);
                    CHECK_CUDA_ERROR(cudaGetLastError());
                    cudaFree(compatibility_group_min_corners);
                    cudaFree(compatibility_max_corners);
                } else {
                    compatibility_words_per_row = 0;
                    compatibility_mask_bytes = 0;
                }
            }
            if (use_projection_bound) {
                const std::size_t projection_value_count =
                    static_cast<std::size_t>(chunks.count) *
                    projection_pair_count * PROJECTION_BIN_COUNT;
                CHECK_CUDA_ERROR(cudaMalloc(
                    &projection_bounds, projection_value_count * sizeof(float)));
                compute_chunk_projection_bounds<<<chunks.count, 32>>>(
                    packed, dim, chunk_offsets, chunk_counts, minimums, maximums,
                    projection_pair_count, projection_cycle, projection_bounds);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }
            if (config.enable_aux_summaries &&
                variant == PaperVariant::GridMkdProjectionBound &&
                candidate_count >= config.mksky_aux_threshold) {
                CHECK_CUDA_ERROR(cudaMalloc(&micro_counts,
                    static_cast<std::size_t>(chunks.count) * sizeof(int)));
                CHECK_CUDA_ERROR(cudaMalloc(&micro_offsets,
                    static_cast<std::size_t>(chunks.count) * sizeof(int)));
                compute_micro_chunk_counts<<<
                    (chunks.count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                        chunk_counts, chunks.count, micro_counts);
                CHECK_CUDA_ERROR(cudaGetLastError());
                thrust::device_ptr<int> micro_count_ptr(micro_counts);
                thrust::device_ptr<int> micro_offset_ptr(micro_offsets);
                thrust::exclusive_scan(micro_count_ptr,
                                       micro_count_ptr + chunks.count,
                                       micro_offset_ptr);
                int last_micro_count = 0;
                int last_micro_offset = 0;
                CHECK_CUDA_ERROR(cudaMemcpy(
                    &last_micro_count, micro_counts + chunks.count - 1,
                    sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA_ERROR(cudaMemcpy(
                    &last_micro_offset, micro_offsets + chunks.count - 1,
                    sizeof(int), cudaMemcpyDeviceToHost));
                const int micro_total = last_micro_offset + last_micro_count;
                if (dim >= PACKED_MICRO_MIN_DIM) {
                    const int words_per_micro = (dim + 3) / 4;
                    CHECK_CUDA_ERROR(cudaMalloc(&packed_micro_corners,
                        static_cast<std::size_t>(micro_total) *
                            words_per_micro * sizeof(unsigned int)));
                    compute_micro_packed_corners<<<chunks.count, BLOCK_SIZE>>>(
                        packed, dim, chunk_offsets, chunk_counts, micro_offsets,
                        minimums, maximums, chunks.count,
                        packed_micro_corners);
                    CHECK_CUDA_ERROR(cudaGetLastError());
                } else {
                    CHECK_CUDA_ERROR(cudaMalloc(&micro_min_corners,
                        static_cast<std::size_t>(micro_total) * dim *
                            sizeof(__half)));
                    compute_micro_min_corners<<<chunks.count, BLOCK_SIZE>>>(
                        packed, dim, chunk_offsets, chunk_counts, micro_offsets,
                        chunks.count, micro_min_corners);
                    CHECK_CUDA_ERROR(cudaGetLastError());
                }
                if (dim >= MICRO_POINT_MASK_MIN_DIM &&
                    static_cast<long long>(candidate_count) * 4 >= count) {
                    CHECK_CUDA_ERROR(cudaMalloc(&micro_point_codes,
                        static_cast<std::size_t>(micro_total) * dim *
                            MICRO_POINT_WORDS * sizeof(unsigned int)));
                    compute_micro_point_codes<<<chunks.count, BLOCK_SIZE>>>(
                        packed, dim, chunk_offsets, chunk_counts, micro_offsets,
                        minimums, maximums, chunks.count, micro_point_codes);
                    CHECK_CUDA_ERROR(cudaGetLastError());
                }
            }
            if (use_projection_3d) {
                const std::size_t projection_value_count =
                    static_cast<std::size_t>(chunks.count) *
                    projection_3d_triple_count * PROJECTION_3D_CELL_COUNT;
                CHECK_CUDA_ERROR(cudaMalloc(
                    &projection_3d_bounds,
                    projection_value_count * sizeof(unsigned short)));
                compute_chunk_projection_3d_bounds<<<chunks.count, 128>>>(
                    packed, dim, chunk_offsets, chunk_counts, minimums, maximums,
                    projection_3d_triple_count, projection_3d_bounds);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }
            if (use_projection_quantized_cycle) {
                const std::size_t projection_value_count =
                    static_cast<std::size_t>(chunks.count) *
                    projection_pair_count * PROJECTION_BIN_COUNT;
                CHECK_CUDA_ERROR(cudaMalloc(
                    &projection_3d_bounds,
                    projection_value_count * sizeof(unsigned short)));
                compute_chunk_projection_quantized_bounds<<<chunks.count, 64>>>(
                    packed, dim, chunk_offsets, chunk_counts, minimums, maximums,
                    projection_pair_count, projection_3d_bounds);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }
        }
    } else {
        pack_all_ordered_points<<<(valid_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            ordered, valid_count, dim, morton_keys, ordered_indices,
            packed, packed_keys, packed_indices);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    result.candidate_count = candidate_count;

    CHECK_CUDA_ERROR(cudaMemset(skyline_count, 0, sizeof(int)));
    unsigned long long* verification_triggers = nullptr;
    unsigned long long* projection_bound_skips = nullptr;
    unsigned long long* projection_3d_skips = nullptr;
    unsigned long long* false_triggers = nullptr;
    unsigned long long* exact_point_checks = nullptr;
    if (config.collect_telemetry) {
        CHECK_CUDA_ERROR(cudaMalloc(&verification_triggers, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMalloc(&projection_bound_skips, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMalloc(&projection_3d_skips, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMalloc(&false_triggers, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMalloc(&exact_point_checks, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMemset(verification_triggers, 0, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMemset(projection_bound_skips, 0, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMemset(projection_3d_skips, 0, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMemset(false_triggers, 0, sizeof(unsigned long long)));
        CHECK_CUDA_ERROR(cudaMemset(exact_point_checks, 0, sizeof(unsigned long long)));
    }

    CHECK_CUDA_ERROR(cudaEventRecord(phase_event));
    if (use_chunk_skipping) {
        launch_by_dimension(dim, ChunkVerifyLauncher{
            packed, packed_keys, packed_indices, candidate_count,
            packed_chunk_ids, chunk_offsets, chunk_counts, min_corners,
            projection_bounds, minimums, maximums, projection_pair_count,
            projection_cycle,
            use_projection_quantized_cycle,
            projection_3d_bounds, projection_3d_triple_count,
            micro_offsets, micro_counts, micro_min_corners,
            packed_micro_corners, micro_point_codes,
            compatibility_masks, compatibility_words_per_row,
            (dim >= config.mksky_signature_min_dim &&
             candidate_count >= config.mksky_aux_threshold)
                ? bits_per_dim
                : 0,
            true, false, skyline_indices, skyline_count,
            verification_triggers, projection_bound_skips, projection_3d_skips,
            false_triggers, exact_point_checks});
    } else {
        launch_by_dimension(dim, DirectVerifyLauncher{
            packed, packed_keys, packed_indices, candidate_count, true,
            skyline_indices, skyline_count, exact_point_checks});
    }
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
        float pivot_ms = 0.0f;
        float compact_ms = 0.0f;
        float sort_ms = 0.0f;
        float local_ms = 0.0f;
        float reorder_ms = 0.0f;
        float mkd_ms = 0.0f;
        float local_prune_ms = 0.0f;
        float pack_ms = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&pivot_ms, start_event, pivot_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&compact_ms, pivot_event, compact_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&sort_ms, compact_event, sort_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&reorder_ms, sort_event, reorder_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&mkd_ms, reorder_event, mkd_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&local_prune_ms, mkd_event, local_event));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&pack_ms, local_event, phase_event));
        local_ms = reorder_ms + mkd_ms + local_prune_ms + pack_ms;
        std::fprintf(stderr,
            "MKSky stages: pivot=%.3f, compact=%.3f, sort=%.3f, reorder=%.3f, MKD=%.3f, local=%.3f, pack=%.3f ms\n",
            pivot_ms, compact_ms, sort_ms, reorder_ms, mkd_ms, local_prune_ms, pack_ms);
    }
    if (config.collect_telemetry) {
        CHECK_CUDA_ERROR(cudaMemcpy(&result.verification_triggers, verification_triggers,
                                    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&result.projection_bound_skips,
                                    projection_bound_skips,
                                    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&result.projection_3d_skips,
                                    projection_3d_skips,
                                    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&result.false_triggers, false_triggers,
                                    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&result.exact_point_checks, exact_point_checks,
                                    sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (use_chunk_skipping && candidate_count > 0) {
            std::vector<int> host_chunk_ids(candidate_count);
            CHECK_CUDA_ERROR(cudaMemcpy(host_chunk_ids.data(), packed_chunk_ids,
                                        static_cast<std::size_t>(candidate_count) * sizeof(int),
                                        cudaMemcpyDeviceToHost));
            for (int chunk_id : host_chunk_ids) {
                result.theoretical_block_checks += static_cast<unsigned long long>(chunk_id);
            }
        }
    }

    int output_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&output_count, skyline_count, sizeof(int), cudaMemcpyDeviceToHost));
    result.skyline_indices.resize(output_count);
    CHECK_CUDA_ERROR(cudaMemcpy(result.skyline_indices.data(), skyline_indices,
                                output_count * sizeof(int), cudaMemcpyDeviceToHost));
    std::sort(result.skyline_indices.begin(), result.skyline_indices.end());

    std::size_t approximate_bytes =
        static_cast<std::size_t>(count) * (dim * sizeof(float) + 2 * sizeof(int)) +
        static_cast<std::size_t>(input_block_count) * (sizeof(float) + sizeof(int)) +
        static_cast<std::size_t>(valid_count) *
            (dim * sizeof(float) + sizeof(unsigned long long) + 2 * sizeof(int)) +
        static_cast<std::size_t>(candidate_count) *
            (dim * sizeof(float) + sizeof(unsigned long long) + 3 * sizeof(int)) +
        chunks.working_bytes;
    if (reuse_coordinate_buffer) {
        approximate_bytes -=
            static_cast<std::size_t>(candidate_count) * dim * sizeof(float);
    }
    if (reuse_index_buffers) {
        approximate_bytes -= static_cast<std::size_t>(count) * 4 * sizeof(int);
    }
    if (projection_bounds) {
        approximate_bytes += static_cast<std::size_t>(chunks.count) *
            projection_pair_count * PROJECTION_BIN_COUNT * sizeof(float);
    }
    if (projection_3d_bounds) {
        const std::size_t summary_values = use_projection_quantized_cycle
            ? static_cast<std::size_t>(projection_pair_count) * PROJECTION_BIN_COUNT
            : static_cast<std::size_t>(projection_3d_triple_count) * PROJECTION_3D_CELL_COUNT;
        approximate_bytes += static_cast<std::size_t>(chunks.count) *
            summary_values * sizeof(unsigned short);
    }
    if (micro_min_corners || packed_micro_corners) {
        int last_micro_count = 0;
        int last_micro_offset = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(&last_micro_count,
            micro_counts + chunks.count - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(&last_micro_offset,
            micro_offsets + chunks.count - 1, sizeof(int), cudaMemcpyDeviceToHost));
        const int micro_total = last_micro_offset + last_micro_count;
        approximate_bytes += static_cast<std::size_t>(chunks.count) *
            2 * sizeof(int);
        if (micro_min_corners) {
            approximate_bytes += static_cast<std::size_t>(micro_total) *
                dim * sizeof(__half);
        }
        if (packed_micro_corners) {
            approximate_bytes += static_cast<std::size_t>(micro_total) *
                ((dim + 3) / 4) * sizeof(unsigned int);
        }
        if (micro_point_codes) {
            approximate_bytes += static_cast<std::size_t>(micro_total) * dim *
                MICRO_POINT_WORDS * sizeof(unsigned int);
        }
    }
    if (compatibility_masks) approximate_bytes += compatibility_mask_bytes;
    result.device_memory_mb = static_cast<double>(approximate_bytes) / (1024.0 * 1024.0);

    cudaFree(exact_point_checks);
    cudaFree(false_triggers);
    cudaFree(projection_3d_skips);
    cudaFree(projection_bound_skips);
    cudaFree(verification_triggers);
    cudaFree(skyline_count);
    if (!reuse_skyline_indices) cudaFree(skyline_indices);
    cudaFree(min_corners);
    cudaFree(compatibility_masks);
    cudaFree(micro_min_corners);
    cudaFree(packed_micro_corners);
    cudaFree(micro_point_codes);
    cudaFree(micro_offsets);
    cudaFree(micro_counts);
    cudaFree(projection_bounds);
    cudaFree(projection_3d_bounds);
    cudaFree(chunk_offsets);
    cudaFree(chunk_counts);
    if (!reuse_packed_chunk_ids) cudaFree(packed_chunk_ids);
    if (!reuse_packed_indices) cudaFree(packed_indices);
    cudaFree(packed_keys);
    if (!reuse_coordinate_buffer) cudaFree(packed);
    cudaFree(chunks.storage);
    cudaFree(ordered_indices);
    cudaFree(ordered);
    cudaFree(candidate_counter);
    if (!reuse_candidate_positions) cudaFree(candidate_positions);
    cudaFree(sorted_indices);
    cudaFree(morton_keys);
    cudaFree(sum_order_keys);
    cudaFree(grid_keep);
    cudaFree(grid_occupancy_b);
    cudaFree(grid_occupancy_a);
    cudaFree(grid_cell_ids);
    cudaFree(survivor_offsets);
    cudaFree(survivor_flags);
    cudaFree(maximums);
    cudaFree(minimums);
    cudaFree(pivot_index_device);
    cudaFree(pivot_block_indices);
    cudaFree(pivot_block_sums);
    cudaFree(coordinates);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(phase_event);
    cudaEventDestroy(start_event);
    if (sort_event) cudaEventDestroy(sort_event);
    if (compact_event) cudaEventDestroy(compact_event);
    if (pivot_event) cudaEventDestroy(pivot_event);
    if (local_event) cudaEventDestroy(local_event);
    if (mkd_event) cudaEventDestroy(mkd_event);
    if (reorder_event) cudaEventDestroy(reorder_event);
    const auto wall_end = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    return result;
}

}  // namespace

AlgorithmResult run_paper_mksky(const std::vector<MyDataPoint>& points,
                                const AlgorithmConfig& config) {
    return run_paper_variant(points, config, PaperVariant::Full);
}

AlgorithmResult run_paper_ablation_a(const std::vector<MyDataPoint>& points,
                                     const AlgorithmConfig& config) {
    AlgorithmResult result =
        run_paper_variant(points, config, PaperVariant::MkdWithoutSkipping);
    result.name = "MKSky-ablation-A";
    result.provenance = "Original ablation A: remove Min-Corner chunk skipping";
    return result;
}

AlgorithmResult run_paper_ablation_b(const std::vector<MyDataPoint>& points,
                                     const AlgorithmConfig& config) {
    AlgorithmResult result =
        run_paper_variant(points, config, PaperVariant::SumOrderWithoutSkipping);
    result.name = "MKSky-ablation-B";
    result.provenance =
        "Original cumulative ablation B: replace MKD ordering after removing chunk skipping";
    return result;
}

AlgorithmResult run_paper_ablation_c(const std::vector<MyDataPoint>& points,
                                     const AlgorithmConfig& config) {
    return run_paper_variant(points, config, PaperVariant::MkdWithoutSkipping);
}

AlgorithmResult run_sum_order_mksky(const std::vector<MyDataPoint>& points,
                                    const AlgorithmConfig& config) {
    return run_paper_variant(points, config, PaperVariant::SumOrder);
}

AlgorithmResult run_grid_mkd_mksky(const std::vector<MyDataPoint>& points,
                                   const AlgorithmConfig& config) {
    return run_paper_variant(points, config, PaperVariant::GridMkd);
}

AlgorithmResult run_projection_bound_mksky(const std::vector<MyDataPoint>& points,
                                           const AlgorithmConfig& config) {
    return run_paper_variant(points, config, PaperVariant::GridMkdProjectionBound);
}

AlgorithmResult run_projection_3d_mksky(const std::vector<MyDataPoint>& points,
                                        const AlgorithmConfig& config) {
    return run_paper_variant(points, config, PaperVariant::GridMkdProjection3D);
}

AlgorithmResult run_projection_scan_mksky(const std::vector<MyDataPoint>& points,
                                          const AlgorithmConfig& config) {
    return run_paper_variant(points, config, PaperVariant::GridMkdProjectionScan);
}

AlgorithmResult run_projection_cycle_mksky(const std::vector<MyDataPoint>& points,
                                           const AlgorithmConfig& config) {
    return run_paper_variant(points, config, PaperVariant::GridMkdProjectionCycle);
}

AlgorithmResult run_projection_qcycle_mksky(const std::vector<MyDataPoint>& points,
                                            const AlgorithmConfig& config) {
    return run_paper_variant(
        points, config, PaperVariant::GridMkdProjectionQuantizedCycle);
}
