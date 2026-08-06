#include "algorithm.h"
#include "current/mksky_current.h"

#include <algorithm>
#include <chrono>

AlgorithmResult run_current_mksky(const std::vector<MyDataPoint>& points,
                                  const AlgorithmConfig& config) {
    AlgorithmResult result;
    result.name = "MKSky-current";
    result.provenance = "unchanged algorithm body from jomyal overall-time project";
    result.input_count = static_cast<int>(points.size());

    std::vector<MyDataPoint> mutable_points = points;
    const auto wall_begin = std::chrono::high_resolution_clock::now();
    int count = 0;
    int candidate_count = -1;
    std::vector<MyDataPoint> skyline;
    std::tie(count, result.device_ms, skyline, result.device_memory_mb, candidate_count) =
        run_myal_algorithm(mutable_points, config);
    const auto wall_end = std::chrono::high_resolution_clock::now();
    result.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_begin).count();
    result.skyline_indices.reserve(skyline.size());
    for (const MyDataPoint& point : skyline) result.skyline_indices.push_back(point.original_idx);
    std::sort(result.skyline_indices.begin(), result.skyline_indices.end());
    result.candidate_count = candidate_count;
    return result;
}
