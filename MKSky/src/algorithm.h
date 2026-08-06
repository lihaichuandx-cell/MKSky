#pragma once

#include "common.h"

#include <string>
#include <vector>

struct AlgorithmResult {
    std::string name;
    std::string provenance;
    std::vector<int> skyline_indices;
    double device_ms = 0.0;
    double wall_ms = 0.0;
    int input_count = 0;
    int candidate_count = 0;
    int adaptive_local_chunk_size = 0;
    int adaptive_leaf_size = 0;
    int mkd_chunk_count = 0;
    int valid_count = 0;
    // 0: pivot pruning, 1: grid pruning, 2: candidate reduction disabled.
    int prefilter_route = -1;
    double preprocess_ms = 0.0;
    double core_ms = 0.0;
    double device_memory_mb = 0.0;
    unsigned long long theoretical_block_checks = 0;
    unsigned long long verification_triggers = 0;
    unsigned long long projection_bound_skips = 0;
    unsigned long long projection_3d_skips = 0;
    unsigned long long false_triggers = 0;
    unsigned long long exact_point_checks = 0;
};

AlgorithmResult run_current_mksky(const std::vector<MyDataPoint>& points,
                                  const AlgorithmConfig& config);
AlgorithmResult run_adaptive_mksky(const std::vector<MyDataPoint>& points,
                                   const AlgorithmConfig& config);
AlgorithmResult run_mkd_mksky(const std::vector<MyDataPoint>& points,
                              const AlgorithmConfig& config);
AlgorithmResult run_adaptive_mkd_mksky(const std::vector<MyDataPoint>& points,
                                       const AlgorithmConfig& config);
AlgorithmResult run_mkd_flat_mksky(const std::vector<MyDataPoint>& points,
                                   const AlgorithmConfig& config);
AlgorithmResult run_mkd_fixed_mksky(const std::vector<MyDataPoint>& points,
                                    const AlgorithmConfig& config);
AlgorithmResult run_paper_mksky(const std::vector<MyDataPoint>& points,
                                const AlgorithmConfig& config);
AlgorithmResult run_paper_ablation_a(const std::vector<MyDataPoint>& points,
                                     const AlgorithmConfig& config);
AlgorithmResult run_paper_ablation_b(const std::vector<MyDataPoint>& points,
                                     const AlgorithmConfig& config);
AlgorithmResult run_paper_ablation_c(const std::vector<MyDataPoint>& points,
                                     const AlgorithmConfig& config);
AlgorithmResult run_paper_ablation_no_candidate_reduction(
    const std::vector<MyDataPoint>& points, const AlgorithmConfig& config);
AlgorithmResult run_paper_ablation_no_local_prefix(
    const std::vector<MyDataPoint>& points, const AlgorithmConfig& config);
AlgorithmResult run_sum_order_mksky(const std::vector<MyDataPoint>& points,
                                    const AlgorithmConfig& config);
AlgorithmResult run_grid_mkd_mksky(const std::vector<MyDataPoint>& points,
                                   const AlgorithmConfig& config);
AlgorithmResult run_projection_bound_mksky(const std::vector<MyDataPoint>& points,
                                           const AlgorithmConfig& config);
AlgorithmResult run_projection_3d_mksky(const std::vector<MyDataPoint>& points,
                                        const AlgorithmConfig& config);
AlgorithmResult run_projection_scan_mksky(const std::vector<MyDataPoint>& points,
                                          const AlgorithmConfig& config);
AlgorithmResult run_projection_cycle_mksky(const std::vector<MyDataPoint>& points,
                                           const AlgorithmConfig& config);
AlgorithmResult run_projection_qcycle_mksky(const std::vector<MyDataPoint>& points,
                                            const AlgorithmConfig& config);
AlgorithmResult run_skycell(const std::vector<MyDataPoint>& points,
                            const AlgorithmConfig& config);
AlgorithmResult run_skyalign(const std::vector<MyDataPoint>& points,
                             const AlgorithmConfig& config);
