#include <torch/extension.h>
#include "torch_custom_nn_linear_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


CustomOpsNNLinear::ReOrderedByKeyState CustomOpsNNLinear::ReOrderedByKeyState::fromChunk(char*& chunk, size_t P)
{
    // the memory has been scheduled with starting from chunk, there are P element in total
	ReOrderedByKeyState binning;
	obtain(chunk, binning.values, P, 128);
	obtain(chunk, binning.values_unsorted, P, 128);
	obtain(chunk, binning.keys, P, 128);
	obtain(chunk, binning.keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.size,
		binning.keys_unsorted, binning.keys,
		binning.values_unsorted, binning.values, P);
	obtain(chunk, binning.list_sorting_space, binning.size, 128);
	return binning;
}


void CustomOpsNNLinear::CustomNNLinear::forward(
    const int batch_size,
    const int in_features, const int out_features,
    std::function<char* (size_t)> grad_pos_buffer_Func,
    const float* input_tensor,
    const float* trainable_weights,
    float* output,
    bool debug)
{
    // schedule memory for key-value orderer
    int n_params = in_features*out_features;
    size_t chunk_size = required<ReOrderedByKeyState>(n_params);
	char* chunkptr = grad_pos_buffer_Func(chunk_size);
	ReOrderedByKeyState key_value_state = ReOrderedByKeyState::fromChunk(chunkptr, n_params);

    // define the cuda forward code 
    CHECK_CUDA(FORWARD::forward(
        batch_size, in_features, out_features, 
        input_tensor,
        trainable_weights,
        output), debug);
    return ;
}

void CustomOpsNNLinear::CustomNNLinear::backward(
    const int batch_size,
    const int in_features,
    const int out_features,
    char* grad_pos_buffer,
    const float* input_tensor,
    const float* output_grads,
    const float* trainable_weights,
    float* input_tensor_grad,
    float* trainable_weights_grad,
    const int topk_grad_buffer,
    float* topk_grads,
    int* pos_1st_dim_of_topk_grads_int,
    int* pos_2st_dim_of_topk_grads_int,
    bool debug)
{
    // get memory for key-value orderer
    int n_params = in_features*out_features;
    ReOrderedByKeyState key_value_state = ReOrderedByKeyState::fromChunk(grad_pos_buffer, n_params);

    // define the cuda backward code
    CHECK_CUDA(BACKWARD::backward(
        batch_size, in_features, out_features,
        input_tensor,
        output_grads,
        trainable_weights,
        key_value_state.keys_unsorted,
        key_value_state.values_unsorted,
        input_tensor_grad,
        trainable_weights_grad), debug);

    // cub::DeviceRadixSort::SortPairs is orderd from small to large
	CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(
		key_value_state.list_sorting_space,
		key_value_state.size,
		key_value_state.keys_unsorted, key_value_state.keys,
		key_value_state.values_unsorted, key_value_state.values,
		in_features*out_features, 0, 32), debug); // the key is float, so start and end bit is 0 and 32

    CHECK_CUDA(BACKWARD::assigning_topk(
        topk_grad_buffer, in_features, out_features,
        key_value_state.keys,
        key_value_state.values,
        topk_grads,
        pos_1st_dim_of_topk_grads_int,
        pos_2st_dim_of_topk_grads_int), debug);

    return;
}
