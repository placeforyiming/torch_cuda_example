#include <stdexcept>
#include <sstream>

#include "forward.h"
#include "auxiliary.h"
#include "config.h"

__forceinline__  __device__ void mat_multiply_slice(
    const int in_features,
    const float*  input_slice_ptr,
    const float*  weights_slice_ptr,
    float * output_each_ptr)
{
    for (int i = 0; i < in_features; i++){
        output_each_ptr[0] = output_each_ptr[0] + input_slice_ptr[i]*weights_slice_ptr[i];
    }
}

__global__ void forwardCUDA(
    const int batch_size,
    const int in_features, const int out_features,
    const float*  input_tensor,
    const float*  trainable_weights,
    float*  output)
{
    // iters along each_batch x out_features
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= batch_size*out_features) return;
    int batch_idx = n/out_features;
    int out_feat_idx = n%out_features;
    const float* input_tensor_slice = input_tensor + batch_idx * in_features;
    const float* weight_slice = trainable_weights + out_feat_idx * in_features;
    float* output_each = output + batch_idx * out_features + out_feat_idx;
    mat_multiply_slice(in_features, input_tensor_slice, weight_slice, output_each);
    return;
}


void FORWARD::validate_input_channels(int input_channels, int max_channels) {
    if (input_channels > max_channels) {
        std::ostringstream oss;
        oss << "Only support input and output channels not larger than " << max_channels;
        throw std::invalid_argument(oss.str());
    }
}


void FORWARD::forward(
    const int batch_size,
    const int in_features, const int out_features,
    const float*  input_tensor,
    const float*  trainable_weights,
    float*  output) {
    
    // not nessesary, just to show the config.h and how to use other c++ functions 
    validate_input_channels(out_features, MAX_BLOCK_SIZE);
    validate_input_channels(in_features, MAX_BLOCK_SIZE);
    
    int n_grid = 0;
    int n_thread = 0;
    if (out_features%32 == 0) {
        n_grid = batch_size;
        n_thread = out_features;
    } else {
        n_thread = (out_features/32+1) * 32;
        n_grid = out_features*batch_size/n_thread+1;
    } 
	forwardCUDA <<<batch_size, n_thread>>> (
        batch_size,in_features, out_features, input_tensor, trainable_weights, output
	);

}

