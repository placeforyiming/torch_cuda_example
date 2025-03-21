#include "backward.h"
#include "auxiliary.h"
#include "config.h"


__forceinline__  __device__ void update_grad_slice(
    const int in_features,
    const float*  input_tensor_slice,
    const float*  weight_slice,
    const float*  output_grad_slice,
    float * input_tensor_grad_each,
    float * trainable_weights_grad_each)
{
    for (int i = 0; i < in_features; i++){
        atomicAdd(&input_tensor_grad_each[i], weight_slice[i]*output_grad_slice[0]);
        atomicAdd(&trainable_weights_grad_each[i], input_tensor_slice[i] * output_grad_slice[0]);
    }
    return;
}



__global__ void backwardCUDA(
    const int batch_size,
    const int in_features,
    const int out_features,
    const float* input_tensor,
    const float* output_grads,
    const float* trainable_weights,
    float* input_tensor_grad,
    float* trainable_weights_grad)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= batch_size*out_features) return;
    int batch_idx = n/out_features;
    int out_feat_idx = n%out_features;

    const float* input_tensor_slice = input_tensor + batch_idx * in_features;
    const float* weight_slice = trainable_weights + out_feat_idx * in_features;
    const float* output_grad_slice = output_grads + batch_idx * out_features + out_feat_idx;
    float* input_tensor_grad_each = input_tensor_grad + batch_idx * in_features;
    float* trainable_weights_grad_each = trainable_weights_grad + out_feat_idx * in_features;
    update_grad_slice(in_features, input_tensor_slice, weight_slice, output_grad_slice,
        input_tensor_grad_each, trainable_weights_grad_each);
    return;
}


__global__ void collectGradients(
    float* keys_unsorted,
    int* values_unsorted,
    float* trainable_weights_grad)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    keys_unsorted[n] = abs(trainable_weights_grad[n]);
    values_unsorted[n] = n;
    return;
}

void BACKWARD::backward(
    const int batch_size,
    const int in_features,
    const int out_features,
    const float* input_tensor,
    const float* output_grad,
    const float* trainable_weights,
    float* keys_unsorted,
    int* values_unsorted,
    float* input_tensor_grad,
    float* trainable_weights_grad) {

    int n_grid = 0;
    int n_thread = 0;
    if (out_features%32 == 0) {
        n_grid = batch_size;
        n_thread = out_features;
    } else {
        n_thread = (out_features/32+1) * 32;
        n_grid = out_features*batch_size/n_thread+1;
    }
	backwardCUDA <<<n_grid, n_thread>>> (
        batch_size, in_features, out_features,
        input_tensor, output_grad, trainable_weights, input_tensor_grad,
        trainable_weights_grad
	);

    collectGradients <<<out_features, in_features>>> (
        keys_unsorted, values_unsorted, trainable_weights_grad
    );

    return;
}

__global__ void assigningTopK(
    const int topk_num,
    const int in_features,
    float* keys,
    int* values,
    float* topk_grads,
    int* pos_1st_dim_of_topk_grads_int,
    int* pos_2nd_dim_of_topk_grads_int)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < topk_num) {
        topk_grads[n] = keys[n];
        int idx = values[n];
        pos_1st_dim_of_topk_grads_int[n] = idx/in_features;
        pos_2nd_dim_of_topk_grads_int[n] = idx - pos_1st_dim_of_topk_grads_int[n]*in_features;
    }
    return;
}

void BACKWARD::assigning_topk(
    const int topk_num,
    const int in_features,
    const int out_features,
    float* keys,
    int* values,
    float* topk_grads,
    int* pos_1st_dim_of_topk_grads_int,
    int* pos_2nd_dim_of_topk_grads_int) {

    assigningTopK <<<out_features, in_features>>> (
        topk_num, in_features, 
        keys, values, topk_grads, 
        pos_1st_dim_of_topk_grads_int, pos_2nd_dim_of_topk_grads_int
    );

    return;
}