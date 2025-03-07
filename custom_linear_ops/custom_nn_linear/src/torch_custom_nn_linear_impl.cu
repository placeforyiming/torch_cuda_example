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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"


CustomOpsNNLinear::InclusiveSumState CustomOpsNNLinear::InclusiveSumState::fromChunk(char*& chunk, size_t P)
{
    // the memory has been scheduled with starting from chunk, there are P element in total
	InclusiveSumState geom;
	obtain(chunk, geom.list, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.list, geom.list, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.list_inclusive_sumed, P, 128);
	return geom;
}


CustomOpsNNLinear::ReOrderedByKeyState CustomOpsNNLinear::ReOrderedByKeyState::fromChunk(char*& chunk, size_t P)
{
    // the memory has been scheduled with starting from chunk, there are P element in total
	ReOrderedByKeyState binning;
	obtain(chunk, binning.list, P, 128);
	obtain(chunk, binning.list_unsorted, P, 128);
	obtain(chunk, binning.keys, P, 128);
	obtain(chunk, binning.keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.size,
		binning.keys_unsorted, binning.keys,
		binning.list_unsorted, binning.list, P);
	obtain(chunk, binning.list_sorting_space, binning.size, 128);
	return binning;
}


void CustomOpsNNLinear::CustomNNLinear::forward(
    std::function<char* (size_t)> grad_order_buffer,
    std::function<char* (size_t)> pos_1st_dim_buffer,
    std::function<char* (size_t)> pos_2st_dim_buffer,
    const float* input_tensor,
    const float* trainable_weights,
    const float* output,
    bool debug)
{
    // define the cuda forward code 
    return ;
}

void CustomOpsNNLinear::CustomNNLinear::backward(
    const int in_features,
    const int out_features,
    char* grad_order_buffer,
    char* pos_1st_dim_buffer_int,
    char* pos_2st_dim_buffer_int,
    const float* input_tensor,
    const float* trainable_weights,
    const float* input_tensor_grad,
    const float* trainable_weights_grad,
    bool debug)
{
    // define the cuda backward code
    return;
}