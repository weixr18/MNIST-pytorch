#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

namespace {
template <typename scalar_t>
__global__ void local_attention_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    size_t batch_size,
    size_t channel_num,
    size_t input_height,
    size_t input_width,
    size_t chunk_size,
    size_t chunk_num_x,
    size_t chunk_num_y,
    size_t kernel_size) 
{
    // input shape: [b, c, h, w]
    // block number: batch_size * channel * chunk_num_x * chunk_num_y
    const int b = blockIdx.x / (channel_num  * chunk_num_x * chunk_num_y);
    const int c = (blockIdx.x / (chunk_num_x * chunk_num_y)) % channel_num;
    const int chunk_id_x = (blockIdx.x / chunk_num_y) % chunk_num_x;
    const int chunk_id_y = blockIdx.x % chunk_num_y;

    const int idx_x = chunk_id_x * chunk_size + threadIdx.x; // logic x
    const int idx_y = chunk_id_y * chunk_size + threadIdx.y; // logic y
    const int patch_id_x = threadIdx.x / kernel_size;
    const int patch_id_y = threadIdx.y / kernel_size;
    const int px = threadIdx.x % kernel_size;
    const int py = threadIdx.y % kernel_size;

    const int patch_logic_offset_x = idx_x - px; // = chunk_id_x*chunk_size+patch_id_x*kernel_size
    const int patch_logic_offset_y = idx_y - py; // = chunk_id_y*chunk_size+patch_id_y*kernel_size

    // boundary check IMPORTANT!!!!!!!!!!!
    bool is_in_boundary = (threadIdx.x < chunk_size && idx_x < input_height) 
        && (threadIdx.y < chunk_size && idx_y < input_width);
    if( is_in_boundary ){
        scalar_t tA=0., tB=0., tC=0., tD=0.;
        for(int i = 0; i < kernel_size; i++){
            tA += weight[0][c][px][i]*input[b][c][patch_logic_offset_x+i][patch_logic_offset_y+py];
            tB += weight[1][c][i][py]*input[b][c][patch_logic_offset_x+px][patch_logic_offset_y+i];
            tC += weight[2][c][px][i]*input[b][c][patch_logic_offset_x+py][patch_logic_offset_y+i];
            tD += weight[3][c][i][py]*input[b][c][patch_logic_offset_x+i][patch_logic_offset_y+px];
        }
        scalar_t tmp = (tA+tB+tC+tD);
        output[b][c][idx_x][idx_y] = tmp*weight[4][c][px][py];
    }

    
    // 对每个element都加N, 看到input不是constant的, 我们直接对传入的4维张量做修改.
    // int num_threads = blockDim.x * gridDim.x;
    // while(idx < batch_size*channel*input_height*input_width) {
    //     input[idx] = input[idx] + N;
    //     idx += num_threads;
    // }
}
}

at::Tensor local_attention_forward_cuda(
        at::Tensor input,
        at::Tensor weight,
        size_t kernel_size) {
    const auto batch_size = input.size(0);
    const auto channel = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);

    const auto chunk_size = (BLOCK_SIZE / kernel_size) * kernel_size; // equal or less than BLOCK_SIZE
    const auto chunk_num_x = input_height / chunk_size + 1;
    const auto chunk_num_y = input_width / chunk_size + 1;
    const dim3 threads (BLOCK_SIZE, BLOCK_SIZE);    // actuallu only (chunk_size, chunk_size) is used.
    const auto block_num = min(batch_size * channel * chunk_num_x * chunk_num_y, 65536);
    printf("batch size:%lld, channel:%lld, input_height:%lld, input_width:%lld, " \
            "chunk_size:%lld, chunk_num_x:%lld, chunk_num_y:%lld, block_num:%lld\n", 
            batch_size, channel, input_height, input_width, 
            chunk_size, chunk_num_x, chunk_num_y, block_num);
    auto output = torch::zeros_like(input);

    // Attention!! AT_DISPATCH_FLOATING_TYPES 's 2nd patameter must be same as this function's name!!!
    AT_DISPATCH_FLOATING_TYPES(input.type(), "local_attention_forward_cuda", ([&] {
      local_attention_forward_cuda_kernel<scalar_t><<<block_num, threads>>>(
        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size,
        channel,
        input_height,
        input_width,
        chunk_size, 
        chunk_num_x, 
        chunk_num_y,
        kernel_size);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in local_attentnion_cuda: %s\n", cudaGetErrorString(err));
    return output;
}

namespace {
template <typename scalar_t>
__global__ void local_attention_backward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_output,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_x,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_weight,
    size_t batch_size,
    size_t channel_num,
    size_t input_height,
    size_t input_width,
    size_t chunk_size,
    size_t chunk_num_x,
    size_t chunk_num_y,
    size_t kernel_size) 
{
    // input shape: [b, c, h, w]
    // block number: batch_size * channel * chunk_num_x * chunk_num_y
//     const int b = blockIdx.x / (channel_num  * chunk_num_x * chunk_num_y);
//     const int c = (blockIdx.x / (chunk_num_x * chunk_num_y)) % channel_num;
//     const int chunk_id_x = (blockIdx.x / chunk_num_y) % chunk_num_x;
//     const int chunk_id_y = blockIdx.x % chunk_num_y;

//     const int idx_x = chunk_id_x * chunk_size + threadIdx.x; // logic x
//     const int idx_y = chunk_id_y * chunk_size + threadIdx.y; // logic y
//     const int patch_id_x = threadIdx.x / kernel_size;
//     const int patch_id_y = threadIdx.y / kernel_size;
//     const int px = threadIdx.x % kernel_size;
//     const int py = threadIdx.y % kernel_size;

//     const int patch_logic_offset_x = idx_x - px; // = chunk_id_x*chunk_size+patch_id_x*kernel_size
//     const int patch_logic_offset_y = idx_y - py; // = chunk_id_y*chunk_size+patch_id_y*kernel_size

//     // boundary check IMPORTANT!!!!!!!!!!!
//     bool is_in_boundary = (threadIdx.x < chunk_size && idx_x < input_height) 
//         && (threadIdx.y < chunk_size && idx_y < input_width);
//     if( is_in_boundary ){
//         scalar_t tA=0., tB=0., tC=0., tD=0.;
//         for(int i = 0; i < kernel_size; i++){
//             tA += weight[0][c][px][i]*input[b][c][patch_logic_offset_x+i][patch_logic_offset_y+py];
//             tB += weight[1][c][i][py]*input[b][c][patch_logic_offset_x+px][patch_logic_offset_y+i];
//             tC += weight[2][c][px][i]*input[b][c][patch_logic_offset_x+py][patch_logic_offset_y+i];
//             tD += weight[3][c][i][py]*input[b][c][patch_logic_offset_x+i][patch_logic_offset_y+px];
//         }
//         scalar_t tmp = (tA+tB+tC+tD);
//         output[b][c][idx_x][idx_y] = tmp*weight[4][c][px][py];
    // }
}
}


std::vector<at::Tensor> local_attention_backward_cuda(
        at::Tensor d_output,
        at::Tensor x,
        at::Tensor weight){
    const auto batch_size = x.size(0);
    const auto channel = x.size(1);
    const auto input_height = x.size(2);
    const auto input_width = x.size(3);
    const auto kernel_size = weight.size(2);

    auto d_x = torch::zeros_like(x);
    auto d_weight = torch::zeros_like(weight);

    // Attention!! AT_DISPATCH_FLOATING_TYPES 's 2nd patameter must be same as this function's name!!!
    AT_DISPATCH_FLOATING_TYPES(input.type(), "local_attention_forward_cuda", ([&] {
      local_attention_backward_cuda_kernel<scalar_t><<<block_num, threads>>>(
        d_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in local_attentnion_cuda: %s\n", cudaGetErrorString(err));
    return {d_x, d_weight};
}