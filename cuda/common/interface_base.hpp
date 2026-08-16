#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

// Wrapper around a raw cuda pointer. Always initialized to 0
template <typename T>
class CudaVector
{
public:
    explicit CudaVector() : ptr(nullptr), _size(0) {}

    explicit CudaVector(size_t count) : ptr(nullptr), _size(count)
    {
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    explicit CudaVector(const std::vector<T>& v) : ptr(nullptr), _size(v.size())
    {
        CUDA_CHECK(cudaMalloc(&ptr, _size * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(ptr, v.data(), _size * sizeof(T), cudaMemcpyHostToDevice));
    }

    explicit CudaVector(const size_t size, std::function<T()> gen) : ptr(nullptr), _size(size)
    {
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
        std::vector<T> tmp(size);
        std::generate(tmp.begin(), tmp.end(), gen);
        CUDA_CHECK(cudaMemcpy(ptr, tmp.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void fill(const std::vector<T>& data)
    {
        if (data.size() != _size)
        {
            throw std::runtime_error("Trying to fill CudaVector with wrongly sized data");
        }
        CUDA_CHECK(cudaMemcpy(ptr, data.data(), _size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void fill(std::function<T()> gen)
    {
        std::vector<T> tmp(_size);
        std::generate(tmp.begin(), tmp.end(), gen);
        CUDA_CHECK(cudaMemcpy(ptr, tmp.data(), _size * sizeof(T), cudaMemcpyHostToDevice));
    }

    size_t size() const
    {
        return _size;
    }

    CudaVector(const CudaVector&) = delete;
    CudaVector& operator=(const CudaVector&) = delete;

    CudaVector(CudaVector&& other) noexcept : ptr(other.ptr), _size(other._size)
    {
        other.ptr = nullptr;
    }

    CudaVector& operator=(CudaVector&& other) noexcept
    {
        if (this != &other)
        {
            if (ptr != nullptr)
            {
                CUDA_CHECK(cudaFree(ptr));
            }
            ptr = other.ptr;
            _size = other._size;
            other.ptr = nullptr;
            other._size = 0;
        }
        return *this;
    }

    ~CudaVector()
    {
        if (ptr != nullptr)
        {
            CUDA_CHECK(cudaFree(ptr));
        }
    }

    std::vector<T> ToCPU() const
    {
        std::vector<T> output(_size);
        CUDA_CHECK(cudaMemcpy(output.data(), ptr, _size * sizeof(T), cudaMemcpyDeviceToHost));
        return output;
    }

    operator T* () { return ptr; }
    operator const T* () const { return ptr; }

    T& operator*() { return *ptr; }
    const T& operator*() const { return *ptr; }

    T* operator->() { return ptr; }
    const T* operator->() const { return ptr; }

private:
    T* ptr;
    size_t _size;
};

template <typename T>
void PermuteVector(CudaVector<T>& v, const std::vector<size_t>& indices)
{
    if (indices.size() == 0)
    {
        return;
    }

    const size_t N = v.size();

    if (indices.size() !=N)
    {
        throw std::runtime_error("Invalid permutation");
    }

    // Perform the permutation on the CPU
    const std::vector<T> cpu_data = v.ToCPU();
    std::vector<T> permuted(N);
    for (size_t i = 0; i < N; ++i)
    {
        const size_t index = indices[i];
        if (index >= N)
        {
            throw std::runtime_error("Invalid index in permutation");
        }
        permuted[i] = cpu_data[index];
    }
    v.fill(permuted);
}

template <typename T>
std::vector<uint8_t> SerializeVector(const CudaVector<T>& v)
{
    std::vector<uint8_t> buf;
    std::vector<T> cpu = v.ToCPU();
    size_t n = cpu.size();
    buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&n), reinterpret_cast<const uint8_t*>(&n) + sizeof(n));
    buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(cpu.data()), reinterpret_cast<const uint8_t*>(cpu.data()) + n * sizeof(T));
    return buf;
}

template <typename T>
void DeserializeVector(const std::vector<uint8_t>& buf, size_t& pos, CudaVector<T>& v)
{
    size_t n;
    memcpy(&n, buf.data() + pos, sizeof(n));
    pos += sizeof(n);

    std::vector<T> cpu(n);
    memcpy(cpu.data(), buf.data() + pos, n * sizeof(T));
    pos += n * sizeof(T);

    if (v.size() != n)
    {
        v = CudaVector<T>(cpu);
    }
    else
    {
        v.fill(cpu);
    }
}

template <typename T>
float CompareVector(const CudaVector<T>& a, const CudaVector<T>& b, float atol, float rtol)
{
    if (a.size() != b.size())
    {
        return -1.0f;
    }

    std::vector<T> ca = a.ToCPU();
    std::vector<T> cb = b.ToCPU();

    float maxDiff = 0.0f;
    bool allWithin = true;

    for (size_t i = 0; i < ca.size(); ++i)
    {
        float diff = std::abs(ca[i] - cb[i]);
        if (diff > maxDiff)
        {
            maxDiff = diff;
        }
        float tol = atol + rtol * std::abs(cb[i]);
        if (diff > tol)
        {
            allWithin = false;
        }
    }
    return allWithin ? maxDiff : -maxDiff;
}

// Problem specific, implemented in ``interface_specific.hpp``
void SampleRuntimeVals(uint64_t seed);
struct Inputs;
struct InputsPermutations;
void PermuteInputs(Inputs& i, const InputsPermutations& p);
struct Outputs;
struct OutputsPermutations;
void PermuteOutputs(Outputs& i, const OutputsPermutations& p);
void InitializeData(Inputs& inputs, Outputs& outputs, uint64_t seed);

// Problem specific, implemented in ``impl.cu``
InputsPermutations Setup(const Inputs& inputs, Outputs& outputs);
void RunPipeline(const Inputs& inputs, Outputs& outputs, cudaStream_t stream);
OutputsPermutations Teardown();
