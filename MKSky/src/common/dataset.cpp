#include "dataset.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

const char* distribution_name(Distribution distribution) {
    switch (distribution) {
    case Distribution::Independent: return "independent";
    case Distribution::Correlated: return "correlated";
    case Distribution::Anticorrelated: return "anticorrelated";
    }
    return "unknown";
}

bool parse_distribution(const std::string& text, Distribution* distribution) {
    if (text == "independent" || text == "ind" || text == "0") {
        *distribution = Distribution::Independent;
        return true;
    }
    if (text == "correlated" || text == "corr" || text == "1") {
        *distribution = Distribution::Correlated;
        return true;
    }
    if (text == "anticorrelated" || text == "anti" || text == "2") {
        *distribution = Distribution::Anticorrelated;
        return true;
    }
    return false;
}

static float quantize(float value) {
    value = std::max(0.0001f, std::min(0.9999f, value));
    return std::round(value * 10000.0f) / 10000.0f;
}

std::vector<MyDataPoint> generate_dataset(int count, int dim,
                                          Distribution distribution,
                                          std::uint32_t seed,
                                          bool uniform_postprocessing) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<MyDataPoint> points;
    points.reserve(static_cast<std::size_t>(count));

    if (distribution == Distribution::Anticorrelated && !uniform_postprocessing) {
        for (int i = 0; i < count; ++i) {
            MyDataPoint point{};
            float direction[MAX_DIM] = {};
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d) {
                direction[d] = uniform(generator);
                sum += direction[d];
            }
            for (int d = 0; d < dim; ++d) {
                const float surface = direction[d] / sum;
                point.coords[d] = std::max(0.00001f,
                    std::min(0.99999f, 0.5f * surface + 0.5f * uniform(generator)));
            }
            point.original_idx = i;
            points.push_back(point);
        }
        return points;
    }

    struct PointWithKey {
        std::array<unsigned short, MAX_DIM> key;
        MyDataPoint point;
    };
    int batch_size = static_cast<int>(count * 1.05) + 1;
    std::vector<PointWithKey> batch;
    batch.reserve(static_cast<std::size_t>(batch_size));
    while (static_cast<int>(points.size()) < count) {
        batch.clear();
        for (int i = 0; i < batch_size; ++i) {
            PointWithKey item{};
            float direction[MAX_DIM] = {};
            float direction_sum = 0.0f;
            if (distribution == Distribution::Anticorrelated) {
                for (int d = 0; d < dim; ++d) {
                    direction[d] = uniform(generator);
                    direction_sum += direction[d];
                }
            }
            const float base = distribution == Distribution::Anticorrelated
                ? 0.0f
                : uniform(generator);
            for (int d = 0; d < dim; ++d) {
                float value = uniform(generator);
                if (distribution == Distribution::Correlated) {
                    value = 0.7f * base + 0.3f * value;
                } else if (distribution == Distribution::Anticorrelated) {
                    value = 0.5f * (direction[d] / direction_sum) + 0.5f * value;
                }
                item.point.coords[d] = quantize(value);
                item.key[static_cast<std::size_t>(d)] =
                    static_cast<unsigned short>(std::lround(item.point.coords[d] * 10000.0f));
            }
            batch.push_back(item);
        }
        std::sort(batch.begin(), batch.end(), [](const PointWithKey& left,
                                                 const PointWithKey& right) {
            return left.key < right.key;
        });
        for (std::size_t i = 0; i < batch.size() && static_cast<int>(points.size()) < count; ++i) {
            if (i == 0 || batch[i].key != batch[i - 1].key) points.push_back(batch[i].point);
        }
        batch_size = static_cast<int>((count - points.size()) * 1.05) + 100;
    }
    std::shuffle(points.begin(), points.end(), generator);
    for (int i = 0; i < count; ++i) points[static_cast<std::size_t>(i)].original_idx = i;
    return points;
}

std::vector<MyDataPoint> load_dataset_csv(const std::string& path, int dim) {
    std::ifstream input(path.c_str());
    if (!input) throw std::runtime_error("cannot open input dataset: " + path);
    std::vector<MyDataPoint> points;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream stream(line);
        MyDataPoint point{};
        bool complete = true;
        for (int d = 0; d < dim; ++d) {
            if (!(stream >> point.coords[d])) {
                complete = false;
                break;
            }
        }
        if (!complete) {
            throw std::runtime_error("input dataset row has fewer than dim coordinates");
        }
        point.original_idx = static_cast<int>(points.size());
        points.push_back(point);
    }
    if (points.empty()) throw std::runtime_error("input dataset is empty");
    return points;
}
