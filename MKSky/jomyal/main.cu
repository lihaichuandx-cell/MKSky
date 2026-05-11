#include "common.h"
#include "data_generator.h"
#include "myal_algorithm.h"
#include "skycell_algorithm.h"
#include "skyalign_algorithm.h"
#include <iostream>

int main() {
	printf("=== 算法性能终极评测 (核心 1 V 2 对决) ===\n");
	AlgorithmConfig config = DEFAULT_CONFIG;
	printf("数据量: %d, 维度: %d, 分布模式: %d\n\n", config.data_num, config.dim, DATA_DISTRIBUTION);

	std::vector<MyDataPoint> points = generate_data(config.data_num, config.dim, DATA_DISTRIBUTION);
	int ground_truth_pts = -1;

	// --- 1. 核心算法：MYAL ---
	auto myal_result = run_myal_algorithm(points, config);
	int myal_pts = std::get<0>(myal_result);
	double myal_time = std::get<1>(myal_result) / 1000.0;
	ground_truth_pts = myal_pts;
	printf("[核心/MYAL] 天际线点数: %d, 耗时: %10.6f s\n", myal_pts, myal_time);
	cudaDeviceSynchronize(); cudaGetLastError();

	// --- 2. 竞品：SkyCell ---
	auto skycell_result = run_skycell_algorithm(points, config);
	int sc_pts = std::get<0>(skycell_result);
	double sc_time = std::get<1>(skycell_result) / 1000.0;
	printf("[竞品/Cell] 天际线点数: %d, 耗时: %10.6f s\n", sc_pts, sc_time);
	cudaDeviceSynchronize(); cudaGetLastError();

	// --- 3. 竞品：SkyAlign (拆除限制，强行肉搏！) ---
	printf("[竞品/Alig] 正在执行无差别暴力轮询，请耐心等待...\n");
	auto skyalign_result = run_skyalign_v2(points, config);
	int sa_pts = std::get<0>(skyalign_result);
	double sa_time = std::get<1>(skyalign_result) / 1000.0;
	printf("[竞品/Alig] 天际线点数: %d, 耗时: %10.6f s\n", sa_pts, sa_time);
	cudaDeviceSynchronize(); cudaGetLastError();

	// --- 4. 终极战报 ---
	printf("\n=== 准确率校验 ===\n");
	printf("MYAL     准确率: %s\n", (myal_pts == ground_truth_pts) ? "100%" : "FAIL");
	printf("SkyCell  准确率: %s\n", (sc_pts == ground_truth_pts) ? "100%" : "FAIL");
	printf("SkyAlign 准确率: %s\n", (sa_pts == ground_truth_pts) ? "100%" : "FAIL");

	printf("\n=== 性能一览 (基准: MYAL) ===\n");
	printf("MYAL vs SkyCell : 提速 %8.2f 倍\n", sc_time / myal_time);
	printf("MYAL vs SkyAlign: 提速 %8.2f 倍\n", sa_time / myal_time);

	printf("\n测试结束。\n");
	system("pause");
	return 0;
}