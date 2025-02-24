#include <torch/extension.h>
#include "torch_template_impl.h"
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


TorchOpsTemplate::InclusiveSumState TorchOpsTemplate::InclusiveSumState::fromChunk(char*& chunk, size_t P)
{
    // the memory has been scheduled with starting from chunk, there are P element in total
	InclusiveSumState geom;
	obtain(chunk, geom.list, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.list, geom.list, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.list_inclusive_sumed, P, 128);
	return geom;
}


TorchOpsTemplate::ReOrderedByKeyState TorchOpsTemplate::ReOrderedByKeyState::fromChunk(char*& chunk, size_t P)
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


int TorchOpsTemplate::TorchTemplate::forward(
    std::function<char* (size_t)> sumBuffer,
    std::function<char* (size_t)> reorderBuffer,
    const float* input_1,
    const int* input_2_int,
    const float* input_3,
    const float* input_4,
    int H,
    float* out_1,
    int* out_2,
    bool debug)
{
    // define the cuda forward code 
    return 0;
}

void TorchOpsTemplate::TorchTemplate::backward(
    const int P,
    char* sumBuffer,
    char* reorderBuffer,
    const float* input_1,
    const int* input_2,
    int* temp_1,
    const float* temp_2,
    float* output_grad_1,
    float* output_grad_2,
    bool debug)
{
    // define the cuda backward code
    return;
}