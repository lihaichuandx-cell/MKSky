#include "reference.h"

#include <algorithm>
#include <sstream>

std::vector<int> cpu_reference_skyline(const std::vector<MyDataPoint>& points, int dim) {
    std::vector<int> order(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) order[i] = static_cast<int>(i);
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        double left_sum = 0.0;
        double right_sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            left_sum += points[static_cast<std::size_t>(left)].coords[d];
            right_sum += points[static_cast<std::size_t>(right)].coords[d];
        }
        return left_sum < right_sum;
    });

    std::vector<int> skyline;
    for (int index : order) {
        const MyDataPoint& point = points[static_cast<std::size_t>(index)];
        bool dominated = false;
        for (int skyline_index : skyline) {
            if (point.isDominatedBy(points[static_cast<std::size_t>(skyline_index)], dim)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) skyline.push_back(index);
    }
    std::sort(skyline.begin(), skyline.end());
    return skyline;
}

bool compare_index_sets(const std::vector<int>& expected,
                        const std::vector<int>& actual,
                        std::string* details) {
    std::vector<int> expected_sorted = expected;
    std::vector<int> actual_sorted = actual;
    std::sort(expected_sorted.begin(), expected_sorted.end());
    std::sort(actual_sorted.begin(), actual_sorted.end());
    expected_sorted.erase(std::unique(expected_sorted.begin(), expected_sorted.end()), expected_sorted.end());
    actual_sorted.erase(std::unique(actual_sorted.begin(), actual_sorted.end()), actual_sorted.end());
    if (expected_sorted == actual_sorted) {
        if (details) *details = "exact index match";
        return true;
    }

    std::vector<int> missing;
    std::vector<int> extra;
    std::set_difference(expected_sorted.begin(), expected_sorted.end(),
                        actual_sorted.begin(), actual_sorted.end(),
                        std::back_inserter(missing));
    std::set_difference(actual_sorted.begin(), actual_sorted.end(),
                        expected_sorted.begin(), expected_sorted.end(),
                        std::back_inserter(extra));
    if (details) {
        std::ostringstream stream;
        stream << "missing=" << missing.size() << ", extra=" << extra.size();
        if (!missing.empty()) stream << ", first_missing=" << missing.front();
        if (!extra.empty()) stream << ", first_extra=" << extra.front();
        *details = stream.str();
    }
    return false;
}

