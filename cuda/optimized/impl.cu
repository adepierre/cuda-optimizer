#include "interface_base.hpp"
#include "interface_specific.hpp"

namespace {
    CudaVector<float> h3{batch_size * hidden_size3};
}

static_assert(hidden_size1 == hidden_size2, "Fused kernel requires hidden_size1 == hidden_size2");

// Fused Layer 1 + Layer 2 + Layer 3 with grid-stride loop.
// h3 uses interleaved layout: h3_new[row][2c] = h3_old[row][c], h3_new[row][2c+1] = h3_old[row][c+16]
// This enables float2 stores, halving h3 L1/TEX store wavefronts.
template <int K>
__global__ void __launch_bounds__(512)
fused_l1_l2_l3(const float* __restrict__ data,
               const float* __restrict__ W1,
               const float* __restrict__ b1,
               const float* __restrict__ W2,
               const float* __restrict__ b2,
               const float* __restrict__ W3,
               const float* __restrict__ b3,
               float* __restrict__ h3_out,
               int batch)
{
    constexpr int N1 = (int)hidden_size1;  // 16
    constexpr int N2 = (int)hidden_size2;  // 16
    constexpr int N3 = (int)hidden_size3;  // 32
    constexpr int QUADS = 32;
    constexpr int ROWS = QUADS * 4;        // 128
    constexpr int N3H = N3 / 2;            // 16

    constexpr int S1 = ((K / 4) % 2 == 1) ? K : K + 4;
    constexpr int S2 = 20;
    constexpr int S3 = 18;

    __shared__ float sW1_T[N1 * S1];
    __shared__ float sW2_T[N2 * S2];
    __shared__ float sW3_T[N3 * S3];
    __shared__ float sH1[ROWS * N1];
    __shared__ float sH2[ROWS * N2];

    const float4* sW1_4 = reinterpret_cast<const float4*>(sW1_T);
    const float4* sW2_4 = reinterpret_cast<const float4*>(sW2_T);

    for (int i = threadIdx.x; i < K * N1; i += 512) {
        int k = i / N1;
        int col = i % N1;
        sW1_T[col * S1 + k] = W1[i];
    }
    for (int i = threadIdx.x; i < N2 * N2; i += 512) {
        int k = i / N2;
        int col = i % N2;
        sW2_T[col * S2 + k] = W2[i];
    }
    for (int i = threadIdx.x; i < N2 * N3; i += 512) {
        int k = i / N3;
        int col = i % N3;
        sW3_T[col * S3 + k] = W3[i];
    }
    __syncthreads();

    const int col = threadIdx.x % N1;
    const int row_quad = threadIdx.x / N1;
    const int row_a = row_quad * 4;
    const int row_b = row_quad * 4 + 1;
    const int row_c = row_quad * 4 + 2;
    const int row_d = row_quad * 4 + 3;
    const int col3a = col;
    const int col3b = col + N3H;

    const float b1_val  = b1[col];
    const float b2_val  = b2[col];
    const float b3a_val = b3[col3a];
    const float b3b_val = b3[col3b];

    const int stride_rows = gridDim.x * ROWS;

    for (int row_base = blockIdx.x * ROWS; row_base < batch; row_base += stride_rows) {
        // Phase 1: compute h1 for 4 rows
        {
            const float4* in4_a = reinterpret_cast<const float4*>(data + (size_t)(row_base + row_a) * K);
            const float4* in4_b = reinterpret_cast<const float4*>(data + (size_t)(row_base + row_b) * K);
            const float4* in4_c = reinterpret_cast<const float4*>(data + (size_t)(row_base + row_c) * K);
            const float4* in4_d = reinterpret_cast<const float4*>(data + (size_t)(row_base + row_d) * K);
            float sa = 0.f, sb = 0.f, sc = 0.f, sd = 0.f;
            #pragma unroll
            for (int k4 = 0; k4 < K / 4; k4++) {
                float4 ia = in4_a[k4];
                float4 ib = in4_b[k4];
                float4 ic = in4_c[k4];
                float4 id = in4_d[k4];
                float4 wv = sW1_4[col * (S1 / 4) + k4];
                sa = fmaf(ia.x, wv.x, sa); sa = fmaf(ia.y, wv.y, sa);
                sa = fmaf(ia.z, wv.z, sa); sa = fmaf(ia.w, wv.w, sa);
                sb = fmaf(ib.x, wv.x, sb); sb = fmaf(ib.y, wv.y, sb);
                sb = fmaf(ib.z, wv.z, sb); sb = fmaf(ib.w, wv.w, sb);
                sc = fmaf(ic.x, wv.x, sc); sc = fmaf(ic.y, wv.y, sc);
                sc = fmaf(ic.z, wv.z, sc); sc = fmaf(ic.w, wv.w, sc);
                sd = fmaf(id.x, wv.x, sd); sd = fmaf(id.y, wv.y, sd);
                sd = fmaf(id.z, wv.z, sd); sd = fmaf(id.w, wv.w, sd);
            }
            sa += b1_val; sa = fmaxf(sa, 0.f);
            sb += b1_val; sb = fmaxf(sb, 0.f);
            sc += b1_val; sc = fmaxf(sc, 0.f);
            sd += b1_val; sd = fmaxf(sd, 0.f);
            sH1[row_a * N1 + col] = sa;
            sH1[row_b * N1 + col] = sb;
            sH1[row_c * N1 + col] = sc;
            sH1[row_d * N1 + col] = sd;
        }
        __syncwarp(0xFFFFFFFFu);

        // Phase 2: compute h2 for 4 rows
        {
            const float4* h1_4a = reinterpret_cast<const float4*>(&sH1[row_a * N1]);
            const float4* h1_4b = reinterpret_cast<const float4*>(&sH1[row_b * N1]);
            const float4* h1_4c = reinterpret_cast<const float4*>(&sH1[row_c * N1]);
            const float4* h1_4d = reinterpret_cast<const float4*>(&sH1[row_d * N1]);
            float sa = 0.f, sb = 0.f, sc = 0.f, sd = 0.f;
            #pragma unroll
            for (int k4 = 0; k4 < N2 / 4; k4++) {
                float4 ha = h1_4a[k4];
                float4 hb = h1_4b[k4];
                float4 hc = h1_4c[k4];
                float4 hd = h1_4d[k4];
                float4 wv = sW2_4[col * (S2 / 4) + k4];
                sa = fmaf(ha.x, wv.x, sa); sa = fmaf(ha.y, wv.y, sa);
                sa = fmaf(ha.z, wv.z, sa); sa = fmaf(ha.w, wv.w, sa);
                sb = fmaf(hb.x, wv.x, sb); sb = fmaf(hb.y, wv.y, sb);
                sb = fmaf(hb.z, wv.z, sb); sb = fmaf(hb.w, wv.w, sb);
                sc = fmaf(hc.x, wv.x, sc); sc = fmaf(hc.y, wv.y, sc);
                sc = fmaf(hc.z, wv.z, sc); sc = fmaf(hc.w, wv.w, sc);
                sd = fmaf(hd.x, wv.x, sd); sd = fmaf(hd.y, wv.y, sd);
                sd = fmaf(hd.z, wv.z, sd); sd = fmaf(hd.w, wv.w, sd);
            }
            sa += b2_val; sa = fmaxf(sa, 0.f);
            sb += b2_val; sb = fmaxf(sb, 0.f);
            sc += b2_val; sc = fmaxf(sc, 0.f);
            sd += b2_val; sd = fmaxf(sd, 0.f);
            sH2[row_a * N2 + col] = sa;
            sH2[row_b * N2 + col] = sb;
            sH2[row_c * N2 + col] = sc;
            sH2[row_d * N2 + col] = sd;
        }
        __syncwarp(0xFFFFFFFFu);

        // Phase 3: compute h3 for 4 rows x 2 cols (W3 via float2, zero bank conflict)
        {
            const float4* h2_4a = reinterpret_cast<const float4*>(&sH2[row_a * N2]);
            const float4* h2_4b = reinterpret_cast<const float4*>(&sH2[row_b * N2]);
            const float4* h2_4c = reinterpret_cast<const float4*>(&sH2[row_c * N2]);
            const float4* h2_4d = reinterpret_cast<const float4*>(&sH2[row_d * N2]);
            float saa = 0.f, sab = 0.f, sba = 0.f, sbb = 0.f;
            float sca = 0.f, scb = 0.f, sda = 0.f, sdb = 0.f;
            #pragma unroll
            for (int k4 = 0; k4 < N2 / 4; k4++) {
                float4 ha = h2_4a[k4];
                float4 hb = h2_4b[k4];
                float4 hc = h2_4c[k4];
                float4 hd = h2_4d[k4];
                const int kb = k4 * 4;
                float2 wa_lo = *(const float2*)&sW3_T[col3a * S3 + kb];
                float2 wa_hi = *(const float2*)&sW3_T[col3a * S3 + kb + 2];
                float2 wb_lo = *(const float2*)&sW3_T[col3b * S3 + kb];
                float2 wb_hi = *(const float2*)&sW3_T[col3b * S3 + kb + 2];
                saa = fmaf(ha.x, wa_lo.x, saa); saa = fmaf(ha.y, wa_lo.y, saa);
                saa = fmaf(ha.z, wa_hi.x, saa); saa = fmaf(ha.w, wa_hi.y, saa);
                sab = fmaf(ha.x, wb_lo.x, sab); sab = fmaf(ha.y, wb_lo.y, sab);
                sab = fmaf(ha.z, wb_hi.x, sab); sab = fmaf(ha.w, wb_hi.y, sab);
                sba = fmaf(hb.x, wa_lo.x, sba); sba = fmaf(hb.y, wa_lo.y, sba);
                sba = fmaf(hb.z, wa_hi.x, sba); sba = fmaf(hb.w, wa_hi.y, sba);
                sbb = fmaf(hb.x, wb_lo.x, sbb); sbb = fmaf(hb.y, wb_lo.y, sbb);
                sbb = fmaf(hb.z, wb_hi.x, sbb); sbb = fmaf(hb.w, wb_hi.y, sbb);
                sca = fmaf(hc.x, wa_lo.x, sca); sca = fmaf(hc.y, wa_lo.y, sca);
                sca = fmaf(hc.z, wa_hi.x, sca); sca = fmaf(hc.w, wa_hi.y, sca);
                scb = fmaf(hc.x, wb_lo.x, scb); scb = fmaf(hc.y, wb_lo.y, scb);
                scb = fmaf(hc.z, wb_hi.x, scb); scb = fmaf(hc.w, wb_hi.y, scb);
                sda = fmaf(hd.x, wa_lo.x, sda); sda = fmaf(hd.y, wa_lo.y, sda);
                sda = fmaf(hd.z, wa_hi.x, sda); sda = fmaf(hd.w, wa_hi.y, sda);
                sdb = fmaf(hd.x, wb_lo.x, sdb); sdb = fmaf(hd.y, wb_lo.y, sdb);
                sdb = fmaf(hd.z, wb_hi.x, sdb); sdb = fmaf(hd.w, wb_hi.y, sdb);
            }
            saa += b3a_val; saa = fmaxf(saa, 0.f);
            sab += b3b_val; sab = fmaxf(sab, 0.f);
            sba += b3a_val; sba = fmaxf(sba, 0.f);
            sbb += b3b_val; sbb = fmaxf(sbb, 0.f);
            sca += b3a_val; sca = fmaxf(sca, 0.f);
            scb += b3b_val; scb = fmaxf(scb, 0.f);
            sda += b3a_val; sda = fmaxf(sda, 0.f);
            sdb += b3b_val; sdb = fmaxf(sdb, 0.f);

            // Interleaved h3 layout: h3_new[row][2c]=h3_old[row][c], h3_new[row][2c+1]=h3_old[row][c+16]
            // Enables float2 stores, halving L1/TEX store wavefronts (16→8 per warp per iteration)
            *(float2*)&h3_out[(size_t)(row_base + row_a) * N3 + 2 * col3a] = make_float2(saa, sab);
            *(float2*)&h3_out[(size_t)(row_base + row_b) * N3 + 2 * col3a] = make_float2(sba, sbb);
            *(float2*)&h3_out[(size_t)(row_base + row_c) * N3 + 2 * col3a] = make_float2(sca, scb);
            *(float2*)&h3_out[(size_t)(row_base + row_d) * N3 + 2 * col3a] = make_float2(sda, sdb);
        }
    }
}

// Layer 4: cp.async double-buffered prefetch, W in registers
template <int K>
__global__ void __launch_bounds__(64, 18)
matmul_bias_cpasync(const float* __restrict__ input,
                    const float* __restrict__ W,
                    const float* __restrict__ b,
                    float* __restrict__ output,
                    int batch, int N)
{
    constexpr int K4 = K / 4;
    int col = threadIdx.y * 32 + threadIdx.x;
    bool active = (col < N);

    float w[K];
    if (active) {
        #pragma unroll
        for (int k = 0; k < K; k++)
            w[k] = W[k * N + col];
    }
    float bias = active ? b[col] : 0.f;

    __shared__ float sbuf[2][K];

    if (threadIdx.x < K4) {
        unsigned saddr = __cvta_generic_to_shared(&sbuf[0][threadIdx.x * 4]);
        const float* gptr = input + (size_t)blockIdx.x * K + threadIdx.x * 4;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                     :: "r"(saddr), "l"(gptr) : "memory");
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    int buf = 0;
    for (int row = blockIdx.x; row < batch; row += gridDim.x) {
        int next_row = row + gridDim.x;

        if (next_row < batch) {
            if (threadIdx.x < K4) {
                unsigned saddr = __cvta_generic_to_shared(&sbuf[1 - buf][threadIdx.x * 4]);
                const float* gptr = input + (size_t)next_row * K + threadIdx.x * 4;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                             :: "r"(saddr), "l"(gptr) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;");

        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        if (active) {
            float sum = 0.f;
            #pragma unroll
            for (int k4 = 0; k4 < K4; k4++) {
                float4 iv = *(const float4*)&sbuf[buf][k4 * 4];
                sum = fmaf(iv.x, w[k4 * 4 + 0], sum);
                sum = fmaf(iv.y, w[k4 * 4 + 1], sum);
                sum = fmaf(iv.z, w[k4 * 4 + 2], sum);
                sum = fmaf(iv.w, w[k4 * 4 + 3], sum);
            }
            sum += bias;
            output[row * N + col] = sum;
        }

        __syncthreads();
        buf = 1 - buf;
    }
}

// Fallback: Layer 4 with W in registers, grid-stride (32 < N <= 64)
template <int K>
__global__ void __launch_bounds__(64, 14)
matmul_bias_regW64(const float* __restrict__ input,
                   const float* __restrict__ W,
                   const float* __restrict__ b,
                   float* __restrict__ output,
                   int batch, int N)
{
    int col = threadIdx.y * 32 + threadIdx.x;
    if (col >= N) return;

    float w[K];
    #pragma unroll
    for (int k = 0; k < K; k++) {
        w[k] = W[k * N + col];
    }

    float bias = b[col];

    for (int row = blockIdx.x; row < batch; row += gridDim.x) {
        const float4* in4 = reinterpret_cast<const float4*>(input + row * K);
        float sum = 0.f;
        #pragma unroll
        for (int k4 = 0; k4 < K / 4; k4++) {
            float4 iv = in4[k4];
            sum = fmaf(iv.x, w[k4 * 4 + 0], sum);
            sum = fmaf(iv.y, w[k4 * 4 + 1], sum);
            sum = fmaf(iv.z, w[k4 * 4 + 2], sum);
            sum = fmaf(iv.w, w[k4 * 4 + 3], sum);
        }
        sum += bias;
        output[row * N + col] = sum;
    }
}

// Fallback kernels for edge cases
template <int K, int N, bool APPLY_RELU>
__global__ void matmul_bias_sw(const float* __restrict__ input,
                               const float* __restrict__ W,
                               const float* __restrict__ b,
                               float* __restrict__ output,
                               int batch)
{
    __shared__ float sW[K * N];
    for (int i = threadIdx.x; i < K * N; i += blockDim.x) {
        sW[i] = W[i];
    }
    __syncthreads();

    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * N;
    if (linear >= total) return;

    int col = linear % N;
    int row = linear / N;

    const float4* in_row4 = reinterpret_cast<const float4*>(input + row * K);

    float sum = 0.f;
    #pragma unroll
    for (int k4 = 0; k4 < K / 4; ++k4) {
        float4 iv = in_row4[k4];
        sum = fmaf(iv.x, sW[(k4 * 4 + 0) * N + col], sum);
        sum = fmaf(iv.y, sW[(k4 * 4 + 1) * N + col], sum);
        sum = fmaf(iv.z, sW[(k4 * 4 + 2) * N + col], sum);
        sum = fmaf(iv.w, sW[(k4 * 4 + 3) * N + col], sum);
    }
    sum += b[col];
    if constexpr (APPLY_RELU) sum = fmaxf(sum, 0.f);
    output[linear] = sum;
}

template <int K>
__global__ void matmul_bias_f4(const float* __restrict__ input,
                               const float* __restrict__ W,
                               const float* __restrict__ b,
                               float* __restrict__ output,
                               int batch, int N)
{
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * N;
    if (linear >= total) return;

    int col = linear % N;
    int row = linear / N;

    const float4* in_row4 = reinterpret_cast<const float4*>(input + row * K);

    float sum = 0.f;
    #pragma unroll
    for (int k4 = 0; k4 < K / 4; k4++) {
        float4 iv = in_row4[k4];
        sum = fmaf(iv.x, W[(k4 * 4 + 0) * N + col], sum);
        sum = fmaf(iv.y, W[(k4 * 4 + 1) * N + col], sum);
        sum = fmaf(iv.z, W[(k4 * 4 + 2) * N + col], sum);
        sum = fmaf(iv.w, W[(k4 * 4 + 3) * N + col], sum);
    }
    sum += b[col];
    output[linear] = sum;
}

template <int K>
__global__ void __launch_bounds__(32)
matmul_bias_regW(const float* __restrict__ input,
                 const float* __restrict__ W,
                 const float* __restrict__ b,
                 float* __restrict__ output,
                 int batch, int N)
{
    int col = blockIdx.y * 32 + threadIdx.x;
    if (col >= N) return;

    float w[K];
    #pragma unroll
    for (int k = 0; k < K; k++) {
        w[k] = W[k * N + col];
    }

    float bias = b[col];

    for (int row = blockIdx.x; row < batch; row += gridDim.x) {
        const float4* in4 = reinterpret_cast<const float4*>(input + row * K);
        float sum = 0.f;
        #pragma unroll
        for (int k4 = 0; k4 < K / 4; k4++) {
            float4 iv = in4[k4];
            sum = fmaf(iv.x, w[k4 * 4 + 0], sum);
            sum = fmaf(iv.y, w[k4 * 4 + 1], sum);
            sum = fmaf(iv.z, w[k4 * 4 + 2], sum);
            sum = fmaf(iv.w, w[k4 * 4 + 3], sum);
        }
        sum += bias;
        output[row * N + col] = sum;
    }
}

InputsPermutations Setup(const Inputs& inputs, Outputs& outputs)
{
    InputsPermutations perm;

    // Permute W4 rows to match interleaved h3 layout.
    // h3_new[row][2c] = h3_old[row][c], h3_new[row][2c+1] = h3_old[row][c+16]
    // perm.W4[j * N4 + col] = orig_k(j) * N4 + col where:
    //   orig_k(j) = j/2 if j even (maps to h3_old position j/2)
    //   orig_k(j) = (j-1)/2 + 16 if j odd (maps to h3_old position (j-1)/2+16)
    const size_t N3 = hidden_size3;
    const size_t N3H = N3 / 2;
    const size_t N4 = output_size;

    perm.W4.resize(N3 * N4);
    for (size_t j = 0; j < N3; j++) {
        size_t orig_k;
        if (j % 2 == 0) {
            orig_k = j / 2;
        } else {
            orig_k = (j / 2) + N3H;
        }
        for (size_t col = 0; col < N4; col++) {
            perm.W4[j * N4 + col] = orig_k * N4 + col;
        }
    }

    return perm;
}

void RunPipeline(const Inputs& inputs, Outputs& outputs, cudaStream_t stream)
{
    const int batch = (int)batch_size;
    const int eighth = batch / 8;
    const int N = (int)output_size;

    for (int c = 0; c < 8; c++) {
        const float* data_ptr = inputs.data + (size_t)c * eighth * input_size;
        float* h3_ptr = h3 + (size_t)c * eighth * hidden_size3;
        float* pred_ptr = outputs.predictions + (size_t)c * eighth * N;

        // Fused Layer 1 + Layer 2 + Layer 3: data -> h3 (interleaved layout)
        {
            constexpr int ROWS = 128;
            int grid = eighth / ROWS;
            int max_grid = 256;
            if (grid > max_grid) grid = max_grid;
            switch ((int)input_size) {
                case 4:  fused_l1_l2_l3<4><<<grid, 512, 0, stream>>>(data_ptr, inputs.W1, inputs.b1, inputs.W2, inputs.b2, inputs.W3, inputs.b3, h3_ptr, eighth); break;
                case 8:  fused_l1_l2_l3<8><<<grid, 512, 0, stream>>>(data_ptr, inputs.W1, inputs.b1, inputs.W2, inputs.b2, inputs.W3, inputs.b3, h3_ptr, eighth); break;
                case 12: fused_l1_l2_l3<12><<<grid, 512, 0, stream>>>(data_ptr, inputs.W1, inputs.b1, inputs.W2, inputs.b2, inputs.W3, inputs.b3, h3_ptr, eighth); break;
                case 16: fused_l1_l2_l3<16><<<grid, 512, 0, stream>>>(data_ptr, inputs.W1, inputs.b1, inputs.W2, inputs.b2, inputs.W3, inputs.b3, h3_ptr, eighth); break;
                case 20: fused_l1_l2_l3<20><<<grid, 512, 0, stream>>>(data_ptr, inputs.W1, inputs.b1, inputs.W2, inputs.b2, inputs.W3, inputs.b3, h3_ptr, eighth); break;
                case 24: fused_l1_l2_l3<24><<<grid, 512, 0, stream>>>(data_ptr, inputs.W1, inputs.b1, inputs.W2, inputs.b2, inputs.W3, inputs.b3, h3_ptr, eighth); break;
                case 28: fused_l1_l2_l3<28><<<grid, 512, 0, stream>>>(data_ptr, inputs.W1, inputs.b1, inputs.W2, inputs.b2, inputs.W3, inputs.b3, h3_ptr, eighth); break;
                case 32: fused_l1_l2_l3<32><<<grid, 512, 0, stream>>>(data_ptr, inputs.W1, inputs.b1, inputs.W2, inputs.b2, inputs.W3, inputs.b3, h3_ptr, eighth); break;
            }
        }

        // Layer 4: h3 (interleaved) -> predictions (W4 is pre-permuted in Setup)
        {
            constexpr int K4 = (int)hidden_size3;
            if (N > 32 && N <= 64) {
                dim3 block(32, 2);
                int max_blocks = 128 * 18;
                int gx = (max_blocks < eighth) ? max_blocks : eighth;
                matmul_bias_cpasync<K4><<<gx, block, 0, stream>>>(h3_ptr, inputs.W4, inputs.b4, pred_ptr, eighth, N);
            } else if (N >= 32) {
                int cols_y = (N + 31) / 32;
                int max_blocks = 128 * 24;
                int gx = (max_blocks / cols_y < eighth) ? (max_blocks / cols_y) : eighth;
                dim3 grid(gx, cols_y);
                matmul_bias_regW<K4><<<grid, 32, 0, stream>>>(h3_ptr, inputs.W4, inputs.b4, pred_ptr, eighth, N);
            } else {
                constexpr int BLOCK = 256;
                int total = eighth * N;
                int grid = (total + BLOCK - 1) / BLOCK;
                matmul_bias_f4<K4><<<grid, BLOCK, 0, stream>>>(h3_ptr, inputs.W4, inputs.b4, pred_ptr, eighth, N);
            }
        }
    }
}

OutputsPermutations Teardown()
{
    return OutputsPermutations{};
}