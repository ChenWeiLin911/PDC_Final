#include <torch/extension.h>
#include <vector>

torch::Tensor identity_forward_cuda(torch::Tensor x);

std::vector<torch::Tensor> frequency_mlp_forward_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    double sparsity_threshold
);

torch::Tensor identity_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    return identity_forward_cuda(x);
}

std::vector<torch::Tensor> frequency_mlp_forward(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2,
    double sparsity_threshold
) {
    TORCH_CHECK(x_real.is_cuda(), "x_real must be a CUDA tensor");
    TORCH_CHECK(x_imag.is_cuda(), "x_imag must be a CUDA tensor");
    TORCH_CHECK(w1.is_cuda(), "w1 must be a CUDA tensor");
    TORCH_CHECK(b1.is_cuda(), "b1 must be a CUDA tensor");
    TORCH_CHECK(w2.is_cuda(), "w2 must be a CUDA tensor");
    TORCH_CHECK(b2.is_cuda(), "b2 must be a CUDA tensor");

    TORCH_CHECK(x_real.is_contiguous(), "x_real must be contiguous");
    TORCH_CHECK(x_imag.is_contiguous(), "x_imag must be contiguous");
    TORCH_CHECK(w1.is_contiguous(), "w1 must be contiguous");
    TORCH_CHECK(b1.is_contiguous(), "b1 must be contiguous");
    TORCH_CHECK(w2.is_contiguous(), "w2 must be contiguous");
    TORCH_CHECK(b2.is_contiguous(), "b2 must be contiguous");

    return frequency_mlp_forward_cuda(
        x_real,
        x_imag,
        w1,
        b1,
        w2,
        b2,
        sparsity_threshold
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("identity_forward", &identity_forward, "AFNO identity forward CUDA");
    m.def("frequency_mlp_forward", &frequency_mlp_forward, "AFNO frequency MLP forward CUDA");
}