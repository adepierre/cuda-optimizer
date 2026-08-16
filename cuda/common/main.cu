#include "impl.cu"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>

static constexpr uint64_t validate_seed = 42;
static constexpr float validate_atol = 1e-4f;
static constexpr float validate_rtol = 1e-4f;

static void print_usage(const char* prog)
{
    std::cerr << "Usage: " << prog << " --mode <validate|bench>" << std::endl;
    std::cerr << "  validate [--seed S] <output_file>  -- compare or write baseline for seed S (default: 0)" << std::endl;
    std::cerr << "  bench [--warmup N] [--timed N] [--seed S]  -- benchmark (default: --warmup 5 --timed 10)" << std::endl;
    std::cerr << "         --warmup 0 --timed 3 for ncu profiling" << std::endl;
    std::exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
    // Parse arguments
    std::string mode;
    int warmupIters = 5;
    int timedIters = 10;
    uint64_t bench_seed = 0;
    std::string filepath;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--mode" && i + 1 < argc)
        {
            mode = argv[++i];
        }
        else if (arg == "--warmup" && i + 1 < argc)
        {
            warmupIters = std::stoi(argv[++i]);
        }
        else if (arg == "--timed" && i + 1 < argc)
        {
            timedIters = std::stoi(argv[++i]);
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            bench_seed = std::stoull(argv[++i]);
        }
        else if (mode == "validate" && filepath.empty())
        {
            filepath = argv[i];
        }
        else
        {
            print_usage(argv[0]);
        }
    }

    if (mode.empty())
    {
        print_usage(argv[0]);
    }

    if (mode != "validate" && mode != "bench")
    {
        print_usage(argv[0]);
    }

    if (mode == "validate" && filepath.empty())
    {
        print_usage(argv[0]);
    }

    // Setup
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Sample runtime hyperparameters (input/output sizes) from the seed in both modes
    SampleRuntimeVals(bench_seed);

    Inputs inputs;
    Outputs outputs;

    // Setup: let implementation allocate buffers, get input permutations
    InputsPermutations in_perm = Setup(inputs, outputs);

    // Generate data after setup so implementation can't precompute
    InitializeData(inputs, outputs, validate_seed);

    // Apply input permutations
    PermuteInputs(inputs, in_perm);

    if (mode == "validate")
    {
        // Reset inputs for the selected seed, same as one timed benchmark run
        CUDA_CHECK(cudaStreamSynchronize(stream));
        InitializeData(inputs, outputs, bench_seed);
        PermuteInputs(inputs, in_perm);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        RunPipeline(inputs, outputs, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        OutputsPermutations out_perm = Teardown();
        PermuteOutputs(outputs, out_perm);

        if (std::filesystem::exists(filepath))
        {
            std::ifstream f(filepath, std::ios::binary);
            std::vector<uint8_t> buf(std::istreambuf_iterator<char>(f), {});
            Outputs baseline = DeserializeOutputs(buf);

            float maxError = CompareOutputs(outputs, baseline, validate_atol, validate_rtol);
            bool passed = maxError >= 0.0f;
            float absError = std::abs(maxError);
            if (passed)
            {
                std::cout << "PASS (max_error=" << absError << ")" << std::endl;
                return 0;
            }
            else
            {
                std::cerr << "FAIL: outputs differ from baseline (max_error=" << absError << ")" << std::endl;
                return 1;
            }
        }
        else
        {
            std::vector<uint8_t> buf = SerializeOutputs(outputs);
            std::ofstream f(filepath, std::ios::binary);
            f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
            std::cout << "Baseline written to " << filepath << std::endl;
            return 0;
        }
    }
    else if (mode == "bench")
    {
        // Warmup runs
        for (int i = 0; i < warmupIters; ++i)
        {
            RunPipeline(inputs, outputs, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Timed runs
        std::vector<double> timings;
        timings.reserve(timedIters);

        for (int i = 0; i < timedIters; ++i)
        {
            // Reset inputs outside timing to avoid measuring data transfer
            CUDA_CHECK(cudaStreamSynchronize(stream));
            InitializeData(inputs, outputs, validate_seed + i + 1);
            PermuteInputs(inputs, in_perm);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start, stream));
            RunPipeline(inputs, outputs, stream);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            timings.push_back(ms);

            Teardown();

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        std::sort(timings.begin(), timings.end());
        double median = timings[timedIters / 2];
        double avg = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
        double min_t = timings.front();
        double max_t = timings.back();

        std::cout << "TIMING_AVG:" << avg << std::endl;
        std::cout << "TIMING_MEDIAN:" << median << std::endl;
        std::cout << "TIMING_MIN:" << min_t << std::endl;
        std::cout << "TIMING_MAX:" << max_t << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}
