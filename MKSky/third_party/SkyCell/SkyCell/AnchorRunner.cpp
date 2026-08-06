#include "DataSet3.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Options {
    int requested_count = 50000;
    int layer = 5;
    int seed = 3;
    int warmup = 1;
    int repeat = 3;
    std::string dataset_output;
    std::string result_output;
};

int parse_int(const char* value, const char* option) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error(std::string("invalid integer for ") + option);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            std::cout
                << "SkyCellAnchor options:\n"
                << "  --n <requested random point count>\n"
                << "  --layer <1..7>\n"
                << "  --seed <integer>\n"
                << "  --warmup <count>\n"
                << "  --repeat <count>\n"
                << "  --dataset-out <path>\n"
                << "  --result-out <path>\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing option value");
        const char* value = argv[++i];
        if (argument == "--n") options.requested_count = parse_int(value, "--n");
        else if (argument == "--layer") options.layer = parse_int(value, "--layer");
        else if (argument == "--seed") options.seed = parse_int(value, "--seed");
        else if (argument == "--warmup") options.warmup = parse_int(value, "--warmup");
        else if (argument == "--repeat") options.repeat = parse_int(value, "--repeat");
        else if (argument == "--dataset-out") options.dataset_output = value;
        else if (argument == "--result-out") options.result_output = value;
        else throw std::runtime_error("unknown option: " + argument);
    }
    if (options.requested_count <= 0 || options.layer < 1 || options.layer > 7 ||
        options.warmup < 0 || options.repeat <= 0) {
        throw std::runtime_error("invalid benchmark configuration");
    }
    return options;
}

using PointKey = std::tuple<int, int, int>;

std::set<PointKey> point_set(const std::vector<DataPoint3>& points) {
    std::set<PointKey> result;
    for (const DataPoint3& point : points) {
        result.insert(PointKey(point[0], point[1], point[2]));
    }
    return result;
}

void export_normalized_dataset(const std::vector<DataPoint3>& points,
                               const std::string& path) {
    if (path.empty()) return;
    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open dataset output");
    output << std::setprecision(9);
    for (const DataPoint3& point : points) {
        for (int d = 0; d < 3; ++d) {
            if (d != 0) output << ',';
            const double normalized =
                (static_cast<int>(point[d]) + 32768.0) / 65535.0;
            output << normalized;
        }
        output << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::srand(options.seed);
        DataSet3 dataset(options.requested_count, options.layer);
        export_normalized_dataset(*dataset.data_points, options.dataset_output);

        std::vector<DataPoint3> exact_input = *dataset.data_points;
        std::vector<DataPoint3> exact_skyline;
        dataset.skyline_points(exact_input, exact_skyline);
        const std::set<PointKey> exact_set = point_set(exact_skyline);

        for (int i = 0; i < options.warmup; ++i) {
            dataset.skyline_parallel();
        }

        double sum_ms = 0.0;
        double squared_sum_ms = 0.0;
        bool all_pass = true;
        std::size_t author_count = 0;
        for (int i = 0; i < options.repeat; ++i) {
            const std::vector<DataPoint3> result = dataset.skyline_parallel();
            author_count = result.size();
            all_pass = all_pass && point_set(result) == exact_set;
            const double elapsed_ms = (Timer::time1() + Timer::time2()) * 1000.0;
            sum_ms += elapsed_ms;
            squared_sum_ms += elapsed_ms * elapsed_ms;
        }
        const double average_ms = sum_ms / options.repeat;
        const double standard_deviation_ms = std::sqrt(std::max(
            0.0, squared_sum_ms / options.repeat - average_ms * average_ms));

        if (!options.result_output.empty()) {
            std::ofstream output(options.result_output.c_str(),
                                 std::ios::out | std::ios::trunc);
            if (!output) throw std::runtime_error("cannot open result output");
            output
                << "implementation,requested_n,actual_n,dim,layer,seed,warmup_runs,"
                   "measured_runs,skyline_count,algorithm_avg_ms,algorithm_std_ms,verification\n"
                << "SkyCell-author-public," << options.requested_count << ','
                << dataset.data_points->size() << ",3," << options.layer << ','
                << options.seed << ',' << options.warmup << ',' << options.repeat << ','
                << author_count << ',' << std::setprecision(9) << average_ms << ','
                << standard_deviation_ms << ','
                << (all_pass ? "PASS" : "FAIL") << '\n';
        }

        std::cout << "SkyCell public 3D source: actual_n="
                  << dataset.data_points->size()
                  << ", layer=" << options.layer
                  << ", skyline=" << author_count
                  << ", average=" << std::fixed << std::setprecision(3)
                  << average_ms << " +/- " << standard_deviation_ms << " ms, "
                  << (all_pass ? "PASS" : "FAIL") << '\n';
        return all_pass ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
