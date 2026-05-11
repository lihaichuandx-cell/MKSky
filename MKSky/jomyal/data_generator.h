#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

#include "common.h"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>

#define SAFE_MAX_DIM 16 

struct GridKey {
	short k[SAFE_MAX_DIM];
	bool operator<(const GridKey& other) const {
		for (int i = 0; i < SAFE_MAX_DIM; ++i) {
			if (k[i] != other.k[i]) return k[i] < other.k[i];
		}
		return false;
	}
	bool operator==(const GridKey& other) const {
		for (int i = 0; i < SAFE_MAX_DIM; ++i) {
			if (k[i] != other.k[i]) return false;
		}
		return true;
	}
};

struct PointWithKey {
	GridKey key;
	MyDataPoint p;
};

static inline std::vector<MyDataPoint> generate_data(int n, int dim, int distribution) {
	std::vector<MyDataPoint> final_points;
	final_points.reserve(n);

	std::mt19937 generator(CONFIG_FIX_SEED ? 12345 : std::random_device{}());
	std::uniform_real_distribution<float> dis(0.0f, 1.0f);

	if (distribution == 2) {
		// =====================================================================
		// 👑 经典正常反相关 (Classic Normal Anti-correlated)
		// 逻辑：50% 纯反相关平面 + 50% 独立均匀分布
		// 特性：坐标之和严格 >= 0.5，杜绝秒杀 Bug 点；自带厚度，点数合理递增。
		// =====================================================================
		printf(">> 正在生成 %d 个经典正常反相关测试点...\n", n);

		for (int i = 0; i < n; ++i) {
			MyDataPoint p;
			float vec[SAFE_MAX_DIM];
			float sum = 0.0f;

			for (int d = 0; d < dim; d++) {
				vec[d] = dis(generator);
				sum += vec[d];
			}

			for (int d = 0; d < dim; d++) {
				float base = vec[d] / sum;
				float noise = dis(generator);

				float val = base * 0.5f + noise * 0.5f;

				if (val < 0.00001f) val = 0.00001f;
				if (val > 0.99999f) val = 0.99999f;
				p.coords[d] = val;
			}

			p.original_idx = i;
			final_points.push_back(p);

			if ((i + 1) % 1000000 == 0) {
				printf(".");
				fflush(stdout);
			}
		}
		printf("\n>> 成功生成 %zu 个测试点。\n", final_points.size());
		return final_points;
	}
	else {
		// =====================================================================
		// 👑 正相关 / 独立专属通道：保留万分位离散化去重 + 终极全局乱序
		// =====================================================================
		printf(">> 正在极速生成 %d 个测试点 (采用连续内存排序去重法)...\n", n);

		int batch_size = n * 1.05;
		std::vector<PointWithKey> buffer;
		buffer.reserve(batch_size);

		while (final_points.size() < n) {
			buffer.clear();
			for (int i = 0; i < batch_size; ++i) {
				PointWithKey item;
				float base = dis(generator);
				float temp_val[SAFE_MAX_DIM] = { 0 };

				if (distribution == 1) { // 正相关
					for (int d = 0; d < dim; ++d) temp_val[d] = 0.7f * base + 0.3f * dis(generator);
				}
				else { // 独立分布 (distribution == 0)
					for (int d = 0; d < dim; ++d) temp_val[d] = dis(generator);
				}

				for (int d = 0; d < SAFE_MAX_DIM; d++) item.key.k[d] = 0;

				for (int d = 0; d < dim; ++d) {
					float raw_val = temp_val[d];
					if (raw_val < 0.0001f) raw_val = 0.0001f;
					if (raw_val > 0.9999f) raw_val = 0.9999f;

					short discrete_val = static_cast<short>(roundf(raw_val * 10000.0f));
					item.key.k[d] = discrete_val;
					item.p.coords[d] = discrete_val / 10000.0f;
				}
				buffer.push_back(item);
			}

			// 这里的 sort 会导致数据变成按字典序排列 (小值全在前面)
			std::sort(buffer.begin(), buffer.end(), [](const PointWithKey& a, const PointWithKey& b) {
				return a.key < b.key;
				});

			if (!buffer.empty() && final_points.size() < n) {
				MyDataPoint p = buffer[0].p;
				final_points.push_back(p);
			}

			for (size_t i = 1; i < buffer.size() && final_points.size() < n; ++i) {
				if (!(buffer[i].key == buffer[i - 1].key)) {
					MyDataPoint p = buffer[i].p;
					final_points.push_back(p);
				}
			}
			batch_size = (n - final_points.size()) * 1.05 + 100;
		}

		// =====================================================================
		// 👑 全维度终极物理修复：彻底打乱字典序，逼出算法真实战力！
		// =====================================================================
		printf(">> 正在执行全局随机洗牌，消除数据物理排序偏差 (适用所有维度)...\n");
		std::shuffle(final_points.begin(), final_points.end(), generator);

		// 洗牌后重新编排物理索引，保证结果校对的正确性
		for (size_t i = 0; i < final_points.size(); ++i) {
			final_points[i].original_idx = i;
		}

		printf(">> 成功生成 %d 个绝对唯一的随机测试点。\n", n);
		return final_points;
	}
}

#endif // DATA_GENERATOR_H