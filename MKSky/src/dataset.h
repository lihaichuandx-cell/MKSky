#pragma once

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

enum class Distribution {
    Independent = 0,
    Correlated = 1,
    Anticorrelated = 2,
};

const char* distribution_name(Distribution distribution);
bool parse_distribution(const std::string& text, Distribution* distribution);
std::vector<MyDataPoint> generate_dataset(int count, int dim,
                                          Distribution distribution,
                                          std::uint32_t seed,
                                          bool uniform_postprocessing = false);
std::vector<MyDataPoint> load_dataset_csv(const std::string& path, int dim);
