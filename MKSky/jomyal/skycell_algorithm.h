#pragma once
#include "common.h"
#include <tuple>
#include <vector>

// 稀疏态满血 SkyCell 入口
std::tuple<int, double> run_skycell_algorithm(std::vector<MyDataPoint>& h_points, const AlgorithmConfig& config);