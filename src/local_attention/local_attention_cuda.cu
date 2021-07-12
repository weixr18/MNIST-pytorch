#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define GRAD_MAX 1e4
#define GRAD_MIN -1e4


namespace {
template <typename scalar_t>
__global__ void local_attention_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    size_t batch_size,size_t channel_num,size_t input_height,size_t input_width,
    size_t chunk_size,size_t chunk_num_x,size_t chunk_num_y,size_t kernel_size) 
{
    // input shape: [b, c, h, w]
    // block number: batch_size * channel * chunk_num_x * chunk_num_y
    const int b = blockIdx.x / (channel_num  * chunk_num_x * chunk_num_y);
    const int c = (blockIdx.x / (chunk_num_x * chunk_num_y)) % channel_num;
    const int chunk_id_x = (blockIdx.x / chunk_num_y) % chunk_num_x;
    const int chunk_id_y = blockIdx.x % chunk_num_y;
    const int idx_x = chunk_id_x * chunk_size + threadIdx.x; // logic x
    const int idx_y = chunk_id_y * chunk_size + threadIdx.y; // logic y
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
}

template <typename scalar_t>
__global__ void pointwise_div_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    size_t batch_size,size_t channel_num,size_t chunk_num_x,size_t chunk_num_y,
    size_t kernel_per_line,size_t kernel_size,size_t input_height,size_t input_width,size_t chunk_size
    ) 
{
    // input shape: [b, c, h, w], weight shape: [5, c, k, k]
    // block number: batch_size * channel * chunk_num_x * chunk_num_y
    const int b = blockIdx.x / (channel_num  * chunk_num_x * chunk_num_y);
    const int c = (blockIdx.x / (chunk_num_x * chunk_num_y)) % channel_num;
    const int chunk_id_x = (blockIdx.x / chunk_num_y) % chunk_num_x;
    const int chunk_id_y = blockIdx.x % chunk_num_y;
    const int idx_x = chunk_id_x * chunk_size + threadIdx.x; // logic x
    const int idx_y = chunk_id_y * chunk_size + threadIdx.y; // logic y
    const int px = threadIdx.x % kernel_size;
    const int py = threadIdx.y % kernel_size;

    // boundary check IMPORTANT!!!!!!!!!!!
    bool is_in_boundary = (threadIdx.x < chunk_size && idx_x < input_height) 
        && (threadIdx.y < chunk_size && idx_y < input_width);
    if( is_in_boundary ){
        output[b][c][idx_x][idx_y] = input[b][c][idx_x][idx_y]/(weight[4][c][px][py]+1e-6);
    }
}

// in each kernel C = AB
template <typename scalar_t>
__global__ void per_kernel_matmul(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> C,
    size_t batch_size,size_t channel_num,size_t chunk_num_x,size_t chunk_num_y,
    size_t kernel_per_line,size_t kernel_size,size_t input_height,size_t input_width,
    size_t chunk_size,bool transpose_A = false,bool transpose_B = false
    ) 
{
    // input shape: [b, c, h, w], weight shape: [5, c, k, k]
    // block number: batch_size * channel * chunk_num_x * chunk_num_y
    const int b = blockIdx.x / (channel_num  * chunk_num_x * chunk_num_y);
    const int c = (blockIdx.x / (chunk_num_x * chunk_num_y)) % channel_num;
    const int chunk_id_x = (blockIdx.x / chunk_num_y) % chunk_num_x;
    const int chunk_id_y = blockIdx.x % chunk_num_y;
    const int idx_x = chunk_id_x * chunk_size + threadIdx.x; // logic x
    const int idx_y = chunk_id_y * chunk_size + threadIdx.y; // logic y
    const int px = threadIdx.x % kernel_size;
    const int py = threadIdx.y % kernel_size;
    const int patch_logic_offset_x = idx_x - px; // = chunk_id_x*chunk_size+patch_id_x*kernel_size
    const int patch_logic_offset_y = idx_y - py; // = chunk_id_y*chunk_size+patch_id_y*kernel_size

    // boundary check IMPORTANT!!!!!!!!!!!
    bool is_in_boundary = (threadIdx.x < chunk_size && idx_x < input_height) 
        && (threadIdx.y < chunk_size && idx_y < input_width);
    if( is_in_boundary ){
        scalar_t sum = 0.;
        // C = A^T B^T
        if(transpose_A && transpose_B){
            for(int i = 0; i < kernel_size; i++){
                sum += A[b][c][patch_logic_offset_x+i][patch_logic_offset_y+px] \
                    * B[b][c][patch_logic_offset_x+py][patch_logic_offset_y+i];
            }
        }
        // C = A^T B
        else if((transpose_A && !transpose_B)){
            for(int i = 0; i < kernel_size; i++){
                sum += A[b][c][patch_logic_offset_x+i][patch_logic_offset_y+px] \
                    * B[b][c][patch_logic_offset_x+i][idx_y];
            }
        }
        // C = A B^T
        else if (!transpose_A && transpose_B){
            for(int i = 0; i < kernel_size; i++){
                sum += A[b][c][idx_x][patch_logic_offset_y+i] \
                    * B[b][c][patch_logic_offset_x+py][patch_logic_offset_y+i];
            }
        }
        // C = A B
        else{
            for(int i = 0; i < kernel_size; i++){
                sum += A[b][c][idx_x][patch_logic_offset_y+i] * B[b][c][patch_logic_offset_x+i][idx_y];
            }
        }
        C[b][c][idx_x][idx_y] = sum;
    }
}

extern __shared__ char shared_arr[];

template <typename scalar_t>
__global__ void add_all_kernels(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> src,
    size_t batch_size,
    size_t channel_num,
    size_t kernel_size,
    size_t kernel_per_h, 
    size_t kernel_per_w
    )
    {
        // 1D block
        size_t tid = threadIdx.x;
        size_t bid = blockIdx.x;

        size_t thread_per_block = blockDim.x;
        size_t using_threads = kernel_per_h*kernel_per_w;
        size_t b = bid / (kernel_size * kernel_size * channel_num);
        size_t c = (bid / kernel_size / kernel_size) % channel_num;
        size_t px = (bid / kernel_size) % kernel_size;
        size_t py = bid % kernel_size;

        // in batch summary
        scalar_t* arr = (scalar_t*)shared_arr;
        arr[tid] = 0;
        for(int i = tid; i < using_threads; i += thread_per_block){
            size_t kx = i / kernel_per_w;
            size_t ky = i % kernel_per_w;
            arr[tid] += src[b][c][kx*kernel_size+px][ky*kernel_size+py];
        }
        __syncthreads();

        //merge sum
        for (int stride = 1; stride < using_threads; stride = stride << 1)        
        {
            if (tid % (stride << 1) == 0 && tid + stride < using_threads)
            {
                arr[tid] += arr[tid + stride];
            }
            __syncthreads();
        }
        if(tid == 0){
            src[b][c][px][py] = arr[0];
        }
    }

template <typename scalar_t>
__global__ void add_all_batches(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> src,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> target,
    size_t batch_size,
    size_t channel_num,
    size_t kernel_size,
    size_t target_index
    )
    {
        // 1D block
        size_t tid = threadIdx.x;
        size_t bid = blockIdx.x;

        // size_t thread_per_block = blockDim.x;
        size_t using_threads = batch_size;
        size_t b = tid;
        size_t c = bid / kernel_size / kernel_size;
        size_t px = (bid / kernel_size) % kernel_size;
        size_t py = bid % kernel_size;

        // across batch summary
        scalar_t* arr = (scalar_t*)shared_arr;
        arr[tid] = src[b][c][px][py];
        __syncthreads();

        //merge sum
        for (int stride = 1; stride < using_threads; stride = stride << 1)        
        {
            if (tid % (stride << 1) == 0 && tid + stride< using_threads)
            {
                arr[tid] += arr[tid + stride];
            }
            __syncthreads();
        }
        if(tid == 0){
            target[target_index][c][px][py] = arr[0] / (scalar_t)batch_size;
        }
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
    const auto block_num = min((size_t)batch_size * channel * chunk_num_x * chunk_num_y, (size_t)65536);
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
        batch_size,channel,input_height,input_width,
        chunk_size, chunk_num_x, chunk_num_y,kernel_size);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in local_attentnion_cuda: %s\n", cudaGetErrorString(err));
    return output;
}

void add_among_kernel_and_batch(
    at::Tensor src,
    at::Tensor target,
    size_t batch_size,
    size_t channel_num,
    size_t kernel_size,
    size_t kernel_per_h, 
    size_t kernel_per_w,
    size_t target_index
){
    size_t block_num_1 = kernel_size * kernel_size * channel_num * batch_size;
    size_t thread_num_1 = min((size_t)kernel_per_h * kernel_per_w, (size_t)1024);
    AT_DISPATCH_FLOATING_TYPES(src.type(), "add_among_kernel_and_batch", ([&] {
      add_all_kernels<scalar_t><<<block_num_1, thread_num_1, sizeof(scalar_t)*thread_num_1>>>(
        src.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size, channel_num, kernel_size, 
        kernel_per_h, kernel_per_w
        );
    }));
    cudaDeviceSynchronize();

    size_t block_num_2 = kernel_size * kernel_size * channel_num;
    size_t thread_num_2 = batch_size;
    AT_DISPATCH_FLOATING_TYPES(src.type(), "add_among_kernel_and_batch", ([&] {
      add_all_batches<scalar_t><<<block_num_2, thread_num_2, sizeof(scalar_t)*thread_num_2>>>(
        src.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        target.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size, channel_num, kernel_size, target_index);
    }));
    cudaDeviceSynchronize();
}

std::vector<at::Tensor> local_attention_backward_cuda(
        at::Tensor d_output,
        at::Tensor x,
        at::Tensor weight){

    auto d_x = torch::zeros_like(x);
    auto d_weight = torch::zeros_like(weight);

    // chunk/block in grid
    // index: [b, c, chunk_id_x, chunk_id_y]
    // shape: [batch_size, channel_num, chunk_num_x, chunk_num_y] 
    // thread in chunk 
    // index: [kernel_id_x, kernel_id_y, px, py]
    // shape: [kernel_per_line, kernel_per_line, kernel_size, kernel_size]

    // input shapes
    const auto batch_size = x.size(0);
    const auto channel_num = x.size(1);
    const auto input_height = x.size(2);
    const auto input_width = x.size(3);
    const auto kernel_size = weight.size(2);

    // model shapes
    const auto chunk_size = (BLOCK_SIZE / kernel_size) * kernel_size; // equal or less than BLOCK_SIZE
    const auto chunk_num_x = (input_height-1) / chunk_size + 1;
    const auto chunk_num_y = (input_width-1) / chunk_size + 1;
    const auto kernel_per_h = input_height / kernel_size;
    const auto kernel_per_w = input_width / kernel_size;
    const auto kernels_per_graph = kernel_per_h * kernel_per_w;
    const auto kernel_per_line = chunk_size / kernel_size;
    const auto kernels_per_chunk = kernel_per_line * kernel_per_line;
    const auto kernel_num = (input_height / kernel_size) * (input_width / kernel_size);

    // dispatch parameters
    const dim3 thread_num (chunk_size, chunk_size);    // actuallu only (chunk_size, chunk_size) is used.
    const auto block_num = min((size_t)batch_size * channel_num * chunk_num_x * chunk_num_y, (size_t)65536);
    printf( ">>> input_height: %lld, input_width: %lld, kernel_size: %lld, chunk_size: %lld, \n"\
            ">>> batch size: %lld, channel_num: %lld, chunk_num_x: %lld, chunk_num_y: %lld,\n" \
            ">>> kernel_per_h: %lld, kernel_per_w: %lld, kernels_per_chunk: %lld,\n"\
            ">>> kernel_per_line: %lld, kernels_per_chunk: %lld, kernel_num: %lld, block_num: %lld\n", 
            input_height, input_width, kernel_size, chunk_size, 
            batch_size, channel_num, chunk_num_x, chunk_num_y, 
            kernel_per_h, kernel_per_w, kernels_per_graph, 
            kernel_per_line, kernels_per_chunk, kernel_num, block_num);

    // 1. calc dL/dT
    auto d_T = torch::zeros_like(x);
    AT_DISPATCH_FLOATING_TYPES(x.type(), "local_attention_backward_cuda", ([&] {
      pointwise_div_kernel<scalar_t><<<block_num, thread_num>>>(
        d_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size, channel_num, chunk_num_x, chunk_num_y,
        kernel_per_line, kernel_size, input_height, input_width, chunk_size
        );
    }));
    cudaDeviceSynchronize();

    // 2. calc dL/dA, dL/dB, dL/dC, dL/dD (matmul & merge sum)

    // dL/dA
    at::Tensor tmp_T = torch::zeros_like(x);
    AT_DISPATCH_FLOATING_TYPES(x.type(), "local_attention_backward_cuda", ([&] {
      per_kernel_matmul<scalar_t><<<block_num, thread_num>>>(
        d_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tmp_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size, channel_num, chunk_num_x, chunk_num_y,
        kernel_per_line, kernel_size, input_height, input_width, chunk_size,
        false,
        true);
    }));
    cudaDeviceSynchronize();
    add_among_kernel_and_batch(
        tmp_T, d_weight, batch_size, channel_num, 
        kernel_size, kernel_per_h, kernel_per_w, 0
    );

    // dL/dB
    tmp_T.zero_();
    AT_DISPATCH_FLOATING_TYPES(x.type(), "local_attention_backward_cuda", ([&] {
      per_kernel_matmul<scalar_t><<<block_num, thread_num>>>(
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tmp_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size, channel_num, chunk_num_x, chunk_num_y,
        kernel_per_line, kernel_size, input_height, input_width, chunk_size,
        true,
        false);
    }));
    cudaDeviceSynchronize();
    add_among_kernel_and_batch(
        tmp_T, d_weight, batch_size, channel_num, 
        kernel_size, kernel_per_h, kernel_per_w, 1
    );

    // dL/dC
    tmp_T.zero_();
    AT_DISPATCH_FLOATING_TYPES(x.type(), "local_attention_backward_cuda", ([&] {
      per_kernel_matmul<scalar_t><<<block_num, thread_num>>>(
        d_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tmp_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size, channel_num, chunk_num_x, chunk_num_y,
        kernel_per_line, kernel_size, input_height, input_width, chunk_size,
        false,
        false);
    }));
    cudaDeviceSynchronize();
    add_among_kernel_and_batch(
        tmp_T, d_weight, batch_size, channel_num, 
        kernel_size, kernel_per_h, kernel_per_w, 2  // should be 2
    );

    // dL/dD
    tmp_T.zero_();
    AT_DISPATCH_FLOATING_TYPES(x.type(), "local_attention_backward_cuda", ([&] {
      per_kernel_matmul<scalar_t><<<block_num, thread_num>>>(
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tmp_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size, channel_num, chunk_num_x, chunk_num_y,
        kernel_per_line, kernel_size, input_height, input_width, chunk_size,
        false,
        false);
    }));
    cudaDeviceSynchronize();
    add_among_kernel_and_batch(
        tmp_T, d_weight, batch_size, channel_num, 
        kernel_size, kernel_per_h, kernel_per_w, 3 // should be 3
    );

    // 3. calc dL/dU (call forward)
    tmp_T.zero_();
    at::Tensor tmp_weight = torch::zeros_like(weight);
    tmp_weight.copy_(weight);
    tmp_weight[4] = torch::ones_like(weight[4]);
    AT_DISPATCH_FLOATING_TYPES(x.type(), "local_attention_backward_cuda", ([&] {
      local_attention_forward_cuda_kernel<scalar_t><<<block_num, thread_num>>>(
        x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tmp_weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tmp_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size,channel_num,input_height,input_width,
        chunk_size, chunk_num_x, chunk_num_y,kernel_size);
    }));
    cudaDeviceSynchronize();
    add_among_kernel_and_batch(
        tmp_T, d_weight, batch_size, channel_num, 
        kernel_size, kernel_per_h, kernel_per_w, 4
    );

    // 4. calc dL/dX (matmul & merge sum)
    tmp_weight[0] = weight[0].transpose(1, 2);
    tmp_weight[1] = weight[1].transpose(1, 2);
    tmp_weight[2] = weight[3];
    tmp_weight[3] = weight[2];
    tmp_weight[4] = torch::ones_like(weight[4]);
    AT_DISPATCH_FLOATING_TYPES(x.type(), "local_attention_backward_cuda", ([&] {
      local_attention_forward_cuda_kernel<scalar_t><<<block_num, thread_num>>>(
        d_T.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        tmp_weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        batch_size,channel_num,input_height,input_width,
        chunk_size, chunk_num_x, chunk_num_y,kernel_size);
    }));
    cudaDeviceSynchronize();

    // gradient clamp
    d_x = torch::clamp(d_x, GRAD_MIN, GRAD_MAX); 
    d_weight = torch::clamp(d_weight, GRAD_MIN, GRAD_MAX); 
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
            printf("Error in local_attentnion_cuda: %s\n", cudaGetErrorString(err));
    return {d_x, d_weight};
}