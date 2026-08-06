#ifndef MYAL_ALGORITHM_H
#define MYAL_ALGORITHM_H

#include "common.h"
#include <tuple>
#include <vector>

std::tuple<int, double, std::vector<MyDataPoint>, double, int>
run_myal_algorithm(std::vector<MyDataPoint>& h_points, const AlgorithmConfig& config);

#endif // MYAL_ALGORITHM_H
