#pragma once

#include "common.h"

#include <string>
#include <vector>

std::vector<int> cpu_reference_skyline(const std::vector<MyDataPoint>& points, int dim);
bool compare_index_sets(const std::vector<int>& expected,
                        const std::vector<int>& actual,
                        std::string* details);

