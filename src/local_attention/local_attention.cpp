
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/DeviceGuard.h>

#include <cmath>
#include <vector>
#include <torch/torch.h>


// CUDA forward declarations
at::Tensor local_attention_forward_cuda(
        at::Tensor input,
        at::Tensor weight,
        size_t kernel_size);
std::vector<at::Tensor> local_attention_backward_cuda(
        at::Tensor d_output,
        at::Tensor x,
        at::Tensor weight);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_LOG(rep, x, log) TORCH_CHECK(rep, #x log) 

at::Tensor local_attention_forward_cuda_wrapper(
        at::Tensor input,
        at::Tensor weight) 
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_LOG(input.sizes().size()==4, input, " must be a 4D tensor");
    CHECK_LOG(weight.sizes().size()==4, weight, " must be a 4D tensor");
    CHECK_LOG(weight.size(2)==weight.size(3), weight, " must be a square kernel");
    CHECK_LOG(weight.size(2) < 10 && weight.size(2) % 2 == 1, weight, "kernel size must be 1, 3, 5, 7 or 9");
    int kernel_size = weight.size(2);
    CHECK_LOG(input.size(2) % kernel_size == 0, input, " width should be divided by the kernel size.");
    CHECK_LOG(input.size(3) % kernel_size == 0, input, " height should be divided by the kernel size.");

    return local_attention_forward_cuda(input, weight, kernel_size);
}

std::vector<at::Tensor> local_attention_backward_cuda_wrapper(
        at::Tensor d_output,
        at::Tensor x,
        at::Tensor weight)
{
    CHECK_INPUT(d_output);
    CHECK_INPUT(x);
    CHECK_INPUT(weight);
    return local_attention_backward_cuda(d_output, x, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("local_attention_forward_cuda", &local_attention_forward_cuda_wrapper,
        "local_attention_forward (CUDA)");
  m.def("local_attention_backward_cuda", &local_attention_backward_cuda_wrapper,
        "local_attention_backward (CUDA)");
}