#pragma once

#include "interface_base.hpp"
#include <random>

// Hyper parameters, constexpr for compile time constants
static constexpr size_t batch_size   = 1048576;
static           size_t input_size   = 12;
static constexpr size_t hidden_size1 = 16;
static constexpr size_t hidden_size2 = 16;
static constexpr size_t hidden_size3 = 32;
static           size_t output_size  = 48;

// Any arbitrary constraints on non constexpr hyper parameters
static const bool _constraints = [] {
    if (input_size % 4 != 0) { throw std::runtime_error("input_size must be multiple of 4"); }
    if (input_size > 32) { throw std::runtime_error("input_size must be <= 32"); }
    if (output_size > 128) { throw std::runtime_error("output_size must be <= 128"); }
    return true;
}();

// Set non constexpr hyper parameters
void SampleRuntimeVals(uint64_t seed)
{
    std::mt19937 rng(static_cast<uint32_t>(seed));
    // input_size: multiple of 4, in [4, 32]
    static constexpr size_t in_step = 4;
    static constexpr size_t in_max = 32;
    input_size = in_step * (1 + (rng() % (in_max / in_step)));
    // output_size: in [1, 128]
    output_size = 1 + (rng() % 128);
}

struct Inputs
{
    CudaVector<float> data{batch_size * input_size};
    CudaVector<float> W1{input_size * hidden_size1};
    CudaVector<float> b1{hidden_size1};
    CudaVector<float> W2{hidden_size1 * hidden_size2};
    CudaVector<float> b2{hidden_size2};
    CudaVector<float> W3{hidden_size2 * hidden_size3};
    CudaVector<float> b3{hidden_size3};
    CudaVector<float> W4{hidden_size3 * output_size};
    CudaVector<float> b4{output_size};
};

// Each vector can be used to store a permutation of the indices for
// the corresponding input vector. Identity if empty.
struct InputsPermutations
{
    std::vector<size_t> data;
    std::vector<size_t> W1;
    std::vector<size_t> W2;
    std::vector<size_t> W3;
    std::vector<size_t> W4;
};

void PermuteInputs(Inputs& i, const InputsPermutations& p)
{
    PermuteVector<float>(i.data, p.data);
    PermuteVector<float>(i.W1, p.W1);
    PermuteVector<float>(i.W2, p.W2);
    PermuteVector<float>(i.W3, p.W3);
    PermuteVector<float>(i.W4, p.W4);
}

struct Outputs
{
    CudaVector<float> predictions{batch_size * output_size};
};

// Each vector can be used to store a permutation of the indices for
// the corresponding output vector. Identity if empty
struct OutputsPermutations
{
    std::vector<size_t> predictions;
};

void PermuteOutputs(Outputs& o, const OutputsPermutations& p)
{
    PermuteVector<float>(o.predictions, p.predictions);
}

void InitializeData(Inputs& inputs, Outputs& outputs, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    inputs.data.fill([&dist, &rng]() { return dist(rng); });
    inputs.W1.fill([&dist, &rng]() { return dist(rng); });
    inputs.b1.fill([&dist, &rng]() { return dist(rng); });
    inputs.W2.fill([&dist, &rng]() { return dist(rng); });
    inputs.b2.fill([&dist, &rng]() { return dist(rng); });
    inputs.W3.fill([&dist, &rng]() { return dist(rng); });
    inputs.b3.fill([&dist, &rng]() { return dist(rng); });
    inputs.W4.fill([&dist, &rng]() { return dist(rng); });
    inputs.b4.fill([&dist, &rng]() { return dist(rng); });
}

std::vector<uint8_t> SerializeOutputs(const Outputs& outputs)
{
    std::vector<uint8_t> buf;
    const std::vector<uint8_t> serialized_outputs = SerializeVector(outputs.predictions);
    buf.insert(buf.end(), serialized_outputs.begin(), serialized_outputs.end());
    return buf;
}

Outputs DeserializeOutputs(const std::vector<uint8_t>& buf)
{
    Outputs outputs;
    size_t pos = 0;
    DeserializeVector(buf, pos, outputs.predictions);
    return outputs;
}

float CompareOutputs(const Outputs& a, const Outputs& b, float atol, float rtol)
{
    return CompareVector(a.predictions, b.predictions, atol, rtol);
}
