#include "interface_base.hpp"
#include "interface_specific.hpp"

namespace
{
    CudaVector<float> h1{batch_size * hidden_size1};
    CudaVector<float> h2{batch_size * hidden_size2};
    CudaVector<float> h3{batch_size * hidden_size3};
}

__global__ void matmul_kernel(const float* input, const float* W, float* output, int batch, int K, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch && col < N)
    {
        float sum = 0.f;
        for (int k = 0; k < K; ++k)
        {
            sum += input[row * K + k] * W[k * N + col];
        }

        output[row * N + col] = sum;
    }
}

__global__ void add_bias_kernel(float* data, const float* b, int batch, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch && col < N)
    {
        data[row * N + col] += b[col];
    }
}

__global__ void relu_kernel(float* data, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
    {
        data[idx] = fmaxf(data[idx], 0.f);
    }
}

InputsPermutations Setup(const Inputs& inputs, Outputs& outputs)
{
    InputsPermutations perm;

    perm.data.resize(inputs.data.size());
    for (size_t i = 0; i < inputs.data.size(); ++i) { perm.data[i] = i; }

    perm.W1.resize(inputs.W1.size());
    for (size_t i = 0; i < inputs.W1.size(); ++i) { perm.W1[i] = i; }

    perm.W2.resize(inputs.W2.size());
    for (size_t i = 0; i < inputs.W2.size(); ++i) { perm.W2[i] = i; }

    perm.W3.resize(inputs.W3.size());
    for (size_t i = 0; i < inputs.W3.size(); ++i) { perm.W3[i] = i; }

    perm.W4.resize(inputs.W4.size());
    for (size_t i = 0; i < inputs.W4.size(); ++i) { perm.W4[i] = i; }

    return perm;
}

void RunPipeline(const Inputs& inputs, Outputs& outputs, cudaStream_t stream)
{
    dim3 block2d(16, 16);
    dim3 block1d(256);

    // Layer 1: data -> h1
    matmul_kernel<<<dim3((batch_size + block2d.x - 1) / block2d.x, (hidden_size1 + block2d.y - 1) / block2d.y), block2d, 0, stream>>>(
        inputs.data, inputs.W1, h1, batch_size, input_size, hidden_size1);
    add_bias_kernel<<<dim3((batch_size + block2d.x - 1) / block2d.x, (hidden_size1 + block2d.y - 1) / block2d.y), block2d, 0, stream>>>(
        h1, inputs.b1, batch_size, hidden_size1);
    relu_kernel<<<(batch_size * hidden_size1 + block1d.x - 1) / block1d.x, block1d, 0, stream>>>(
        h1, batch_size * hidden_size1);

    // Layer 2: h1 -> h2
    matmul_kernel<<<dim3((batch_size + block2d.x - 1) / block2d.x, (hidden_size2 + block2d.y - 1) / block2d.y), block2d, 0, stream>>>(
        h1, inputs.W2, h2, batch_size, hidden_size1, hidden_size2);
    add_bias_kernel<<<dim3((batch_size + block2d.x - 1) / block2d.x, (hidden_size2 + block2d.y - 1) / block2d.y), block2d, 0, stream>>>(
        h2, inputs.b2, batch_size, hidden_size2);
    relu_kernel<<<(batch_size * hidden_size2 + block1d.x - 1) / block1d.x, block1d, 0, stream>>>(
        h2, batch_size * hidden_size2);

    // Layer 3: h2 -> h3
    matmul_kernel<<<dim3((batch_size + block2d.x - 1) / block2d.x, (hidden_size3 + block2d.y - 1) / block2d.y), block2d, 0, stream>>>(
        h2, inputs.W3, h3, batch_size, hidden_size2, hidden_size3);
    add_bias_kernel<<<dim3((batch_size + block2d.x - 1) / block2d.x, (hidden_size3 + block2d.y - 1) / block2d.y), block2d, 0, stream>>>(
        h3, inputs.b3, batch_size, hidden_size3);
    relu_kernel<<<(batch_size * hidden_size3 + block1d.x - 1) / block1d.x, block1d, 0, stream>>>(
        h3, batch_size * hidden_size3);

    // Layer 4: h3 -> predictions (no ReLU)
    matmul_kernel<<<dim3((batch_size + block2d.x - 1) / block2d.x, (output_size + block2d.y - 1) / block2d.y), block2d, 0, stream>>>(
        h3, inputs.W4, outputs.predictions, batch_size, hidden_size3, output_size);
    add_bias_kernel<<<dim3((batch_size + block2d.x - 1) / block2d.x, (output_size + block2d.y - 1) / block2d.y), block2d, 0, stream>>>(
        outputs.predictions, inputs.b4, batch_size, output_size);
}

OutputsPermutations Teardown()
{
    OutputsPermutations perm;

    perm.predictions.resize(batch_size * output_size);
    for (size_t i = 0; i < perm.predictions.size(); ++i) { perm.predictions[i] = i; }

    return perm;
}
