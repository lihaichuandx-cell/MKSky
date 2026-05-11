#pragma once
#include "common.h"
#include <tuple>
#include <vector>

std::tuple<int, double> run_hybrid_v2(std::vector<MyDataPoint>& h_points, const AlgorithmConfig& config);