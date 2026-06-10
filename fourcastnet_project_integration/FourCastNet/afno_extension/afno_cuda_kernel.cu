#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

template <typename scalar_t>
__device__ scalar_t relu_device(scalar_t x) {
    return x > scalar_t(0) ? x : scalar_t(0);
}

template <typename scalar_t>
__device__ scalar_t softshrink_device(scalar_t x, scalar_t lambd) {
    if (x > lambd) {
        return x - lambd;
    } else if (x < -lambd) {
        return x + lambd;
    } else {
        return scalar_t(0);
    }
}

template <typename scalar_t>
__global__ void identity_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int64_t numel
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        y[idx] = x[idx];
    }
}

torch::Tensor identity_forward_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);

    int64_t numel = x.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "identity_forward_cuda", ([&] {
        identity_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            numel
        );
    }));

    return y;
}

/*
    o1_real / o1_imag shape:
        [B, H, W_freq, num_blocks, Kmid]

    x_real / x_imag shape:
        [B, H, W_freq, num_blocks, K]

    w1 shape:
        [2, num_blocks, K, Kmid]

    b1 shape:
        [2, num_blocks, Kmid]
*/
template <typename scalar_t>
__global__ void compute_o1_kernel(
    const scalar_t* __restrict__ x_real,
    const scalar_t* __restrict__ x_imag,
    const scalar_t* __restrict__ w1,
    const scalar_t* __restrict__ b1,
    scalar_t* __restrict__ o1_real,
    scalar_t* __restrict__ o1_imag,
    int B,
    int H,
    int W_freq,
    int NB,
    int K,
    int Kmid,
    int h_start,
    int h_count,
    int w_count
) {
    int64_t total = (int64_t)B * h_count * w_count * NB * Kmid;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    int kmid = idx % Kmid;
    int64_t t = idx / Kmid;

    int nb = t % NB;
    t /= NB;

    int wf_local = t % w_count;
    t /= w_count;

    int h_local = t % h_count;
    t /= h_count;

    int b = t;

    int h = h_start + h_local;
    int wf = wf_local;

    scalar_t acc_real = b1[(0 * NB + nb) * Kmid + kmid];
    scalar_t acc_imag = b1[(1 * NB + nb) * Kmid + kmid];

    for (int kin = 0; kin < K; ++kin) {
        int64_t x_idx = (((((int64_t)b * H + h) * W_freq + wf) * NB + nb) * K + kin);

        scalar_t xr = x_real[x_idx];
        scalar_t xi = x_imag[x_idx];

        int64_t w1_r_idx = ((((int64_t)0 * NB + nb) * K + kin) * Kmid + kmid);
        int64_t w1_i_idx = ((((int64_t)1 * NB + nb) * K + kin) * Kmid + kmid);

        scalar_t wr = w1[w1_r_idx];
        scalar_t wi = w1[w1_i_idx];

        acc_real += xr * wr - xi * wi;
        acc_imag += xi * wr + xr * wi;
    }

    int64_t o1_idx = (((((int64_t)b * H + h) * W_freq + wf) * NB + nb) * Kmid + kmid);

    o1_real[o1_idx] = relu_device(acc_real);
    o1_imag[o1_idx] = relu_device(acc_imag);
}

/*
    out_real / out_imag shape:
        [B, H, W_freq, num_blocks, K]

    w2 shape:
        [2, num_blocks, Kmid, K]

    b2 shape:
        [2, num_blocks, K]
*/
template <typename scalar_t>
__global__ void compute_o2_kernel(
    const scalar_t* __restrict__ o1_real,
    const scalar_t* __restrict__ o1_imag,
    const scalar_t* __restrict__ w2,
    const scalar_t* __restrict__ b2,
    scalar_t* __restrict__ out_real,
    scalar_t* __restrict__ out_imag,
    int B,
    int H,
    int W_freq,
    int NB,
    int K,
    int Kmid,
    int h_start,
    int h_count,
    int w_count,
    scalar_t sparsity_threshold
) {
    int64_t total = (int64_t)B * h_count * w_count * NB * K;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    int kout = idx % K;
    int64_t t = idx / K;

    int nb = t % NB;
    t /= NB;

    int wf_local = t % w_count;
    t /= w_count;

    int h_local = t % h_count;
    t /= h_count;

    int b = t;

    int h = h_start + h_local;
    int wf = wf_local;

    scalar_t acc_real = b2[(0 * NB + nb) * K + kout];
    scalar_t acc_imag = b2[(1 * NB + nb) * K + kout];

    for (int kmid = 0; kmid < Kmid; ++kmid) {
        int64_t o1_idx = (((((int64_t)b * H + h) * W_freq + wf) * NB + nb) * Kmid + kmid);

        scalar_t r1 = o1_real[o1_idx];
        scalar_t i1 = o1_imag[o1_idx];

        int64_t w2_r_idx = ((((int64_t)0 * NB + nb) * Kmid + kmid) * K + kout);
        int64_t w2_i_idx = ((((int64_t)1 * NB + nb) * Kmid + kmid) * K + kout);

        scalar_t wr = w2[w2_r_idx];
        scalar_t wi = w2[w2_i_idx];

        acc_real += r1 * wr - i1 * wi;
        acc_imag += i1 * wr + r1 * wi;
    }

    acc_real = softshrink_device(acc_real, sparsity_threshold);
    acc_imag = softshrink_device(acc_imag, sparsity_threshold);

    int64_t out_idx = (((((int64_t)b * H + h) * W_freq + wf) * NB + nb) * K + kout);

    out_real[out_idx] = acc_real;
    out_imag[out_idx] = acc_imag;
}

std::vector<torch::Tensor> frequency_mlp_forward_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    double sparsity_threshold
) {
    int B = x_real.size(0);
    int H = x_real.size(1);
    int W_freq = x_real.size(2);
    int NB = x_real.size(3);
    int K = x_real.size(4);

    int Kmid = w1.size(3);

    auto out_real = torch::zeros_like(x_real);
    auto out_imag = torch::zeros_like(x_imag);

    auto o1_real = torch::zeros(
        {B, H, W_freq, NB, Kmid},
        x_real.options()
    );

    auto o1_imag = torch::zeros(
        {B, H, W_freq, NB, Kmid},
        x_real.options()
    );

    int total_modes = H / 2 + 1;
    int kept_modes = total_modes;  // current FourCastNet uses hard_thresholding_fraction = 1.0

    int h_start = total_modes - kept_modes;
    int h_end = total_modes + kept_modes;

    if (h_start < 0) h_start = 0;
    if (h_end > H) h_end = H;

    int h_count = h_end - h_start;

    int w_count = kept_modes;
    if (w_count > W_freq) w_count = W_freq;

    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(x_real.scalar_type(), "frequency_mlp_forward_cuda", ([&] {
        int64_t total_o1 = (int64_t)B * h_count * w_count * NB * Kmid;
        int blocks_o1 = (total_o1 + threads - 1) / threads;

        compute_o1_kernel<scalar_t><<<blocks_o1, threads>>>(
            x_real.data_ptr<scalar_t>(),
            x_imag.data_ptr<scalar_t>(),
            w1.data_ptr<scalar_t>(),
            b1.data_ptr<scalar_t>(),
            o1_real.data_ptr<scalar_t>(),
            o1_imag.data_ptr<scalar_t>(),
            B,
            H,
            W_freq,
            NB,
            K,
            Kmid,
            h_start,
            h_count,
            w_count
        );

        int64_t total_o2 = (int64_t)B * h_count * w_count * NB * K;
        int blocks_o2 = (total_o2 + threads - 1) / threads;

        compute_o2_kernel<scalar_t><<<blocks_o2, threads>>>(
            o1_real.data_ptr<scalar_t>(),
            o1_imag.data_ptr<scalar_t>(),
            w2.data_ptr<scalar_t>(),
            b2.data_ptr<scalar_t>(),
            out_real.data_ptr<scalar_t>(),
            out_imag.data_ptr<scalar_t>(),
            B,
            H,
            W_freq,
            NB,
            K,
            Kmid,
            h_start,
            h_count,
            w_count,
            static_cast<scalar_t>(sparsity_threshold)
        );
    }));

    return {out_real, out_imag};
}