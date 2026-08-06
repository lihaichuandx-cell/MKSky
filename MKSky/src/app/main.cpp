#include "algorithm.h"
#include "dataset.h"
#include "reference.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ============================================================================
// USER TEST CONFIGURATION
// Edit only these three values for a normal benchmark run.
//
// TEST_DISTRIBUTION options:
//   Distribution::Independent     - independent dataset
//   Distribution::Correlated      - correlated dataset
//   Distribution::Anticorrelated  - anticorrelated dataset
// ============================================================================
const int TEST_DATA_COUNT = 10000000;
const int TEST_DIMENSION = 6;
const Distribution TEST_DISTRIBUTION = Distribution::Independent;

// Benchmark protocol: warm up once, then report the mean of three measured runs.
const int BENCHMARK_WARMUP_RUNS = 1;
const int BENCHMARK_MEASURED_RUNS = 3;

struct Options {
    int count = TEST_DATA_COUNT;
    int dim = TEST_DIMENSION;
    Distribution distribution = TEST_DISTRIBUTION;
    std::uint32_t seed = 12345;
    int repeat = BENCHMARK_MEASURED_RUNS;
    int warmup = BENCHMARK_WARMUP_RUNS;
    int verify_limit = 5000;
    int skycell_rho = 0;
    bool telemetry = false;
    bool uniform_postprocessing = false;
    int mksky_sample_limit = 4096;
    int mksky_survivor_ratio_divisor = 8;
    int mksky_direct_limit = 5000;
    int mksky_chunk_size = 0;
    int mksky_aux_threshold = 65536;
    int mksky_compatibility_min_chunks = 2048;
    int mksky_compatibility_max_mb = 128;
    int mksky_signature_min_dim = 14;
    std::string algorithms = "all";
    std::string csv_path = "results/benchmark.csv";
    std::string input_csv;
    std::string input_order = "generated";
};

void print_usage() {
    std::cout
        << "MKSkyBenchmark options:\n"
        << "  --n <count>                  number of points\n"
        << "  --dim <3..16>                dimensionality\n"
        << "  --distribution <ind|corr|anti>\n"
        << "  --seed <integer>\n"
        << "  --repeat <count>\n"
        << "  --warmup <count>\n"
        << "  --verify-limit <count>       CPU exact verification threshold\n"
        << "  --skycell-rho <integer>      0 selects automatically\n"
        << "  --telemetry <0|1>            collect diagnostic counters\n"
        << "  --dataset-policy <legacy|uniform4> uniform4 applies four-decimal quantization and deduplication to all distributions\n"
        << "  --input-csv <path>            load coordinates from a CSV file\n"
        << "  --input-order <generated|shuffled|sorted-d0|sorted-sum>\n"
        << "  --mksky-sample-limit <count> sensitivity control (default 4096)\n"
        << "  --mksky-survivor-divisor <n> use grid when survivors*n > sample count\n"
        << "  --mksky-direct-limit <count> exact low-dimensional fast-path limit\n"
        << "  --mksky-chunk-size <count>   0 selects the default dimension policy\n"
        << "  --mksky-aux-threshold <count> candidate threshold for auxiliary summaries\n"
        << "  --mksky-compat-min-chunks <count> block-group mask threshold\n"
        << "  --mksky-compat-max-mb <count> block-group mask memory cap\n"
        << "  --mksky-signature-min-dim <d> dimension threshold for coordinate fields\n"
        << "  --algorithms <all|comma-separated names>\n"
        << "       names: mksky,mkd,paper-mkd,mkd-legacy,grid-mkd,ablation-a,"
           "ablation-b,ablation-c,adaptive,adaptive-mkd,mkd-flat,mkd-fixed,"
           "ablation-no-prune,ablation-no-prefix,sum-order,optimized,projection-bound,"
           "projection-3d,projection-scan,projection-cycle,projection-qcycle,skycell,skyalign,cpu-sfs\n"
        << "  --csv <path>\n";
}

int parse_int(const char* value, const char* option) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error(std::string("invalid integer for ") + option + ": " + value);
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            print_usage();
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value after " + argument);
        const char* value = argv[++i];
        if (argument == "--n") options.count = parse_int(value, "--n");
        else if (argument == "--dim") options.dim = parse_int(value, "--dim");
        else if (argument == "--seed") options.seed = static_cast<std::uint32_t>(parse_int(value, "--seed"));
        else if (argument == "--repeat") options.repeat = parse_int(value, "--repeat");
        else if (argument == "--warmup") options.warmup = parse_int(value, "--warmup");
        else if (argument == "--verify-limit") options.verify_limit = parse_int(value, "--verify-limit");
        else if (argument == "--skycell-rho") options.skycell_rho = parse_int(value, "--skycell-rho");
        else if (argument == "--telemetry") options.telemetry = parse_int(value, "--telemetry") != 0;
        else if (argument == "--dataset-policy") {
            const std::string policy = value;
            if (policy == "legacy") options.uniform_postprocessing = false;
            else if (policy == "uniform4") options.uniform_postprocessing = true;
            else throw std::runtime_error("unknown dataset policy: " + policy);
        }
        else if (argument == "--mksky-sample-limit") options.mksky_sample_limit = parse_int(value, "--mksky-sample-limit");
        else if (argument == "--mksky-survivor-divisor") options.mksky_survivor_ratio_divisor = parse_int(value, "--mksky-survivor-divisor");
        else if (argument == "--mksky-direct-limit") options.mksky_direct_limit = parse_int(value, "--mksky-direct-limit");
        else if (argument == "--mksky-chunk-size") options.mksky_chunk_size = parse_int(value, "--mksky-chunk-size");
        else if (argument == "--mksky-aux-threshold") options.mksky_aux_threshold = parse_int(value, "--mksky-aux-threshold");
        else if (argument == "--mksky-compat-min-chunks") options.mksky_compatibility_min_chunks = parse_int(value, "--mksky-compat-min-chunks");
        else if (argument == "--mksky-compat-max-mb") options.mksky_compatibility_max_mb = parse_int(value, "--mksky-compat-max-mb");
        else if (argument == "--mksky-signature-min-dim") options.mksky_signature_min_dim = parse_int(value, "--mksky-signature-min-dim");
        else if (argument == "--algorithms") options.algorithms = value;
        else if (argument == "--csv") options.csv_path = value;
        else if (argument == "--input-csv") options.input_csv = value;
        else if (argument == "--input-order") options.input_order = value;
        else if (argument == "--distribution") {
            if (!parse_distribution(value, &options.distribution)) {
                throw std::runtime_error(std::string("unknown distribution: ") + value);
            }
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.count <= 0) throw std::runtime_error("--n must be positive");
    if (options.dim < 3 || options.dim > MAX_DIM) throw std::runtime_error("--dim must be in [3,16]");
    if (options.repeat <= 0 || options.warmup < 0) throw std::runtime_error("repeat/warmup values are invalid");
    if (options.mksky_sample_limit <= 0 || options.mksky_survivor_ratio_divisor <= 0 ||
        options.mksky_direct_limit < 0 || options.mksky_chunk_size < 0 ||
        options.mksky_aux_threshold < 0 || options.mksky_compatibility_min_chunks < 0 ||
        options.mksky_compatibility_max_mb < 0 || options.mksky_signature_min_dim < 0) {
        throw std::runtime_error("MKSky sensitivity controls contain an invalid negative or zero value");
    }
    if (options.input_order != "generated" && options.input_order != "shuffled" &&
        options.input_order != "sorted-d0" && options.input_order != "sorted-sum") {
        throw std::runtime_error("unknown input order: " + options.input_order);
    }
    return options;
}

void apply_input_order(std::vector<MyDataPoint>* points, int dim,
                       const std::string& input_order, std::uint32_t seed) {
    if (input_order == "generated") return;
    if (input_order == "shuffled") {
        std::mt19937 generator(seed ^ 0x9e3779b9U);
        std::shuffle(points->begin(), points->end(), generator);
    } else if (input_order == "sorted-d0") {
        std::sort(points->begin(), points->end(),
            [](const MyDataPoint& left, const MyDataPoint& right) {
                return left.coords[0] < right.coords[0];
            });
    } else {
        std::sort(points->begin(), points->end(),
            [dim](const MyDataPoint& left, const MyDataPoint& right) {
                float left_sum = 0.0f;
                float right_sum = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    left_sum += left.coords[d];
                    right_sum += right.coords[d];
                }
                return left_sum < right_sum;
            });
    }
    for (std::size_t i = 0; i < points->size(); ++i) {
        (*points)[i].original_idx = static_cast<int>(i);
    }
}

std::vector<std::string> split_algorithms(const std::string& value) {
    if (value == "all") return {"mksky", "skycell", "skyalign"};
    std::vector<std::string> names;
    std::stringstream stream(value);
    std::string name;
    while (std::getline(stream, name, ',')) {
        if (!name.empty()) names.push_back(name);
    }
    return names;
}

using Runner = std::function<AlgorithmResult(const std::vector<MyDataPoint>&, const AlgorithmConfig&)>;

Runner find_runner(const std::string& name) {
    if (name == "mksky") return run_projection_bound_mksky;
    if (name == "adaptive") return run_adaptive_mksky;
    if (name == "mkd") return run_grid_mkd_mksky;
    if (name == "paper-mkd") return run_paper_mksky;
    if (name == "mkd-legacy") return run_mkd_mksky;
    if (name == "ablation-a") return run_paper_ablation_a;
    if (name == "ablation-b") return run_paper_ablation_b;
    if (name == "ablation-c") return run_paper_ablation_c;
    if (name == "ablation-no-prune") return run_paper_ablation_no_candidate_reduction;
    if (name == "ablation-no-prefix") return run_paper_ablation_no_local_prefix;
    if (name == "sum-order") return run_sum_order_mksky;
    if (name == "grid-mkd") return run_grid_mkd_mksky;
    if (name == "optimized") return run_projection_bound_mksky;
    if (name == "projection-bound") return run_projection_bound_mksky;
    if (name == "projection-3d") return run_projection_3d_mksky;
    if (name == "projection-scan") return run_projection_scan_mksky;
    if (name == "projection-cycle") return run_projection_cycle_mksky;
    if (name == "projection-qcycle") return run_projection_qcycle_mksky;
    if (name == "adaptive-mkd") return run_adaptive_mkd_mksky;
    if (name == "mkd-flat") return run_mkd_flat_mksky;
    if (name == "mkd-fixed") return run_mkd_fixed_mksky;
    if (name == "skycell") return run_skycell;
    if (name == "skyalign") return run_skyalign;
    if (name == "cpu-sfs") {
        return [](const std::vector<MyDataPoint>& points, const AlgorithmConfig& config) {
            AlgorithmResult result;
            result.name = "CPU-SFS";
            result.provenance = "serial sort-filter-skyline baseline";
            result.input_count = static_cast<int>(points.size());
            const auto begin = std::chrono::high_resolution_clock::now();
            result.skyline_indices = cpu_reference_skyline(points, config.dim);
            const auto end = std::chrono::high_resolution_clock::now();
            result.wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
            result.device_ms = result.wall_ms;
            result.core_ms = result.wall_ms;
            return result;
        };
    }
    throw std::runtime_error("unknown algorithm name: " + name);
}

std::string csv_escape(const std::string& value) {
    std::string escaped = "\"";
    for (char character : value) {
        if (character == '\"') escaped += '\"';
        escaped += character;
    }
    escaped += "\"";
    return escaped;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        cudaDeviceProp device{};
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&device, 0));
        std::cout << "GPU: " << device.name << ", compute capability "
                  << device.major << '.' << device.minor << '\n';
        std::vector<MyDataPoint> points = options.input_csv.empty()
            ? generate_dataset(options.count, options.dim, options.distribution,
                               options.seed, options.uniform_postprocessing)
            : load_dataset_csv(options.input_csv, options.dim);
        apply_input_order(&points, options.dim, options.input_order, options.seed);
        options.count = static_cast<int>(points.size());
        const std::string distribution_label =
            options.input_csv.empty() ? distribution_name(options.distribution) : "external";
        std::cout << "Dataset: n=" << options.count << ", d=" << options.dim
                  << ", distribution=" << distribution_label
                  << ", seed=" << options.seed
                  << ", order=" << options.input_order
                  << ", postprocessing="
                  << (options.uniform_postprocessing ? "uniform4" : "legacy")
                  << "\n\n";
        std::vector<int> reference;
        const bool has_cpu_reference = options.count <= options.verify_limit;
        if (has_cpu_reference) {
            std::cout << "Computing exact CPU reference...\n";
            reference = cpu_reference_skyline(points, options.dim);
            std::cout << "CPU skyline count: " << reference.size() << "\n\n";
        }

        AlgorithmConfig config;
        config.dim = options.dim;
        config.data_num = options.count;
        config.skycell_rho = options.skycell_rho;
        config.collect_telemetry = options.telemetry;
        config.input_normalized = true;
        config.mksky_sample_limit = options.mksky_sample_limit;
        config.mksky_survivor_ratio_divisor = options.mksky_survivor_ratio_divisor;
        config.mksky_direct_limit = options.mksky_direct_limit;
        config.mksky_chunk_size_override = options.mksky_chunk_size;
        config.mksky_aux_threshold = options.mksky_aux_threshold;
        config.mksky_compatibility_min_chunks = options.mksky_compatibility_min_chunks;
        config.mksky_compatibility_max_bytes =
            static_cast<unsigned long long>(options.mksky_compatibility_max_mb) * 1024ULL * 1024ULL;
        config.mksky_signature_min_dim = options.mksky_signature_min_dim;
        const std::vector<std::string> algorithm_names = split_algorithms(options.algorithms);

        std::ofstream csv(options.csv_path.c_str(), std::ios::out | std::ios::trunc);
        if (!csv) throw std::runtime_error("cannot open CSV output: " + options.csv_path);
        csv << "gpu,n,dim,distribution,seed,input_order,warmup_runs,measured_runs,algorithm,provenance,"
               "skyline_count,candidate_count,adaptive_local_chunk_size,adaptive_leaf_size,"
               "mkd_chunk_count,prefilter_route,"
               "valid_count,preprocess_avg_ms,core_avg_ms,device_memory_mb,"
               "theoretical_block_checks,verification_triggers,projection_bound_skips,projection_3d_skips,false_triggers,exact_point_checks,"
               "algorithm_avg_ms,algorithm_std_ms,end_to_end_avg_ms,end_to_end_std_ms,verification\n";

        std::vector<int> cross_reference;
        bool cross_reference_initialized = false;
        for (const std::string& algorithm_name : algorithm_names) {
            const Runner runner = find_runner(algorithm_name);
            for (int warmup = 0; warmup < options.warmup; ++warmup) {
                std::cout << "Warmup " << algorithm_name << " " << (warmup + 1)
                          << '/' << options.warmup << "..." << std::flush;
                runner(points, config);
                std::cout << " done\n";
            }

            double algorithm_ms_sum = 0.0;
            double algorithm_ms_squared_sum = 0.0;
            double end_to_end_ms_sum = 0.0;
            double end_to_end_ms_squared_sum = 0.0;
            double preprocess_ms_sum = 0.0;
            double core_ms_sum = 0.0;
            AlgorithmResult representative;
            std::string summary_verification;
            for (int repetition = 0; repetition < options.repeat; ++repetition) {
                std::cout << "Measure " << algorithm_name << " " << (repetition + 1)
                          << '/' << options.repeat << "..." << std::flush;
                AlgorithmResult result = runner(points, config);
                std::string verification;
                bool correct = true;
                if (has_cpu_reference) {
                    correct = compare_index_sets(reference, result.skyline_indices, &verification);
                    verification = std::string("independent CPU exact reference: ") + verification;
                } else if (!cross_reference_initialized) {
                    cross_reference = result.skyline_indices;
                    cross_reference_initialized = true;
                    verification =
                        "first GPU result retained for agreement checks; not independent ground truth";
                } else {
                    correct = compare_index_sets(cross_reference, result.skyline_indices, &verification);
                    verification = std::string("GPU agreement against first algorithm: ") + verification;
                }
                std::cout << ' ' << (correct ? "PASS" : "FAIL")
                          << " (" << verification << ")\n";
                if (!correct) return 2;

                if (repetition == 0) representative = result;
                algorithm_ms_sum += result.device_ms;
                algorithm_ms_squared_sum += result.device_ms * result.device_ms;
                end_to_end_ms_sum += result.wall_ms;
                end_to_end_ms_squared_sum += result.wall_ms * result.wall_ms;
                preprocess_ms_sum += result.preprocess_ms;
                core_ms_sum += result.core_ms;
                summary_verification = verification;
            }

            const double algorithm_avg_ms = algorithm_ms_sum / options.repeat;
            const double end_to_end_avg_ms = end_to_end_ms_sum / options.repeat;
            const double algorithm_std_ms = std::sqrt(std::max(
                0.0, algorithm_ms_squared_sum / options.repeat -
                         algorithm_avg_ms * algorithm_avg_ms));
            const double end_to_end_std_ms = std::sqrt(std::max(
                0.0, end_to_end_ms_squared_sum / options.repeat -
                         end_to_end_avg_ms * end_to_end_avg_ms));
            const double preprocess_avg_ms = preprocess_ms_sum / options.repeat;
            const double core_avg_ms = core_ms_sum / options.repeat;
            std::cout << "Average " << algorithm_name
                      << ": count=" << representative.skyline_indices.size()
                      << ", algorithm=" << std::fixed << std::setprecision(3)
                      << algorithm_avg_ms << " ms"
                      << " +/- " << algorithm_std_ms << " ms"
                      << ", end-to-end=" << end_to_end_avg_ms
                      << " +/- " << end_to_end_std_ms << " ms"
                      << " (" << options.repeat << " measured runs)\n\n";

            csv << csv_escape(device.name) << ',' << options.count << ',' << options.dim << ','
                << distribution_label << ',' << options.seed << ','
                << csv_escape(options.input_order) << ','
                << options.warmup << ',' << options.repeat << ','
                << csv_escape(representative.name) << ','
                << csv_escape(representative.provenance) << ','
                << representative.skyline_indices.size() << ','
                << representative.candidate_count << ','
                << representative.adaptive_local_chunk_size << ','
                << representative.adaptive_leaf_size << ','
                << representative.mkd_chunk_count << ','
                << representative.prefilter_route << ','
                << representative.valid_count << ','
                << preprocess_avg_ms << ',' << core_avg_ms << ','
                << representative.device_memory_mb << ','
                << representative.theoretical_block_checks << ','
                << representative.verification_triggers << ','
                << representative.projection_bound_skips << ','
                << representative.projection_3d_skips << ','
                << representative.false_triggers << ','
                << representative.exact_point_checks << ','
                << std::setprecision(9) << algorithm_avg_ms << ',' << algorithm_std_ms << ','
                << end_to_end_avg_ms << ',' << end_to_end_std_ms << ','
                << csv_escape("PASS: " + summary_verification) << '\n';
            csv.flush();
        }
        std::cout << "\nCSV written to " << options.csv_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        print_usage();
        return 1;
    }
}
