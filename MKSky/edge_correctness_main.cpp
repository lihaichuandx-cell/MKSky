#include "algorithm.h"
#include "reference.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

MyDataPoint point(std::initializer_list<float> values, int index) {
    MyDataPoint p{};
    int d = 0;
    for (float value : values) p.coords[d++] = value;
    p.original_idx = index;
    return p;
}

std::vector<MyDataPoint> all_equal() {
    std::vector<MyDataPoint> points;
    for (int i = 0; i < 2048; ++i) {
        points.push_back(point({0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}, i));
    }
    return points;
}

std::vector<MyDataPoint> duplicates_and_dominated() {
    std::vector<MyDataPoint> points;
    int index = 0;
    for (int i = 0; i < 768; ++i) {
        points.push_back(point({0.1f, 0.4f, 0.2f, 0.3f, 0.5f, 0.2f}, index++));
    }
    for (int i = 0; i < 768; ++i) {
        points.push_back(point({0.2f, 0.5f, 0.3f, 0.4f, 0.6f, 0.3f}, index++));
    }
    for (int i = 0; i < 512; ++i) {
        const float t = static_cast<float>(i) / 511.0f;
        points.push_back(point({0.15f + 0.5f * t, 0.65f - 0.5f * t,
                                0.25f, 0.25f, 0.45f, 0.45f}, index++));
    }
    return points;
}

std::vector<MyDataPoint> constant_dimension() {
    std::vector<MyDataPoint> points;
    for (int i = 0; i < 3072; ++i) {
        const float t = static_cast<float>(i) / 3071.0f;
        const float u = static_cast<float>((i * 97) % 3072) / 3071.0f;
        points.push_back(point({t, 1.0f - t, u, 0.25f,
                                0.2f + 0.6f * u, 0.8f - 0.6f * u}, i));
    }
    return points;
}

std::vector<MyDataPoint> quantization_boundaries() {
    std::vector<MyDataPoint> points;
    const float lo = std::nextafter(0.0f, 1.0f);
    const float hi = std::nextafter(1.0f, 0.0f);
    const float below = std::nextafter(0.5f, 0.0f);
    const float above = std::nextafter(0.5f, 1.0f);
    const float values[] = {0.0f, lo, below, 0.5f, above, hi, 1.0f};
    int index = 0;
    for (int a = 0; a < 7; ++a) {
        for (int b = 0; b < 7; ++b) {
            points.push_back(point({values[a], values[b], values[(a + b) % 7],
                                    values[(2 * a + b) % 7],
                                    values[(a + 2 * b) % 7],
                                    values[(3 * a + b) % 7]}, index++));
        }
    }
    return points;
}

std::vector<MyDataPoint> equal_morton_chain() {
    std::vector<MyDataPoint> points;
    for (int i = 0; i < 1024; ++i) {
        const float value = 0.5f + static_cast<float>(i) * 1.0e-7f;
        points.push_back(point({value, value, value, value, value, value}, i));
    }
    return points;
}

std::vector<MyDataPoint> low_dimensional_grid_path() {
    std::vector<MyDataPoint> points;
    for (int i = 0; i < 6000; ++i) {
        const float t = static_cast<float>(i) / 5999.0f;
        const float wobble = static_cast<float>((i * 37) % 211) / 210.0f;
        points.push_back(point({t, 1.0f - t, 0.25f + 0.5f * wobble}, i));
    }
    return points;
}

bool run_case(const std::string& name, int dim,
              const std::vector<MyDataPoint>& points) {
    AlgorithmConfig config;
    config.dim = dim;
    config.data_num = static_cast<int>(points.size());
    config.input_normalized = true;
    const std::vector<int> reference = cpu_reference_skyline(points, dim);
    const AlgorithmResult result = run_projection_bound_mksky(points, config);
    std::string detail;
    const bool pass = compare_index_sets(reference, result.skyline_indices, &detail);
    std::cout << name << ",n=" << points.size() << ",d=" << dim
              << ",cpu=" << reference.size()
              << ",gpu=" << result.skyline_indices.size()
              << "," << (pass ? "PASS" : "FAIL")
              << "," << detail << '\n';
    return pass;
}

}  // namespace

int main() {
    cudaDeviceProp device{};
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device, 0));
    std::cout << "GPU=" << device.name << '\n';
    bool pass = true;
    pass = run_case("all_equal", 6, all_equal()) && pass;
    pass = run_case("duplicates_and_dominated", 6, duplicates_and_dominated()) && pass;
    pass = run_case("constant_dimension", 6, constant_dimension()) && pass;
    pass = run_case("quantization_boundaries", 6, quantization_boundaries()) && pass;
    pass = run_case("equal_morton_chain", 6, equal_morton_chain()) && pass;
    pass = run_case("low_dimensional_grid_path", 3, low_dimensional_grid_path()) && pass;
    return pass ? 0 : 2;
}
