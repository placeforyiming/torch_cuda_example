#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "src/config.h"
#include "src/torch_template.h"
#include <fstream>
#include <string>
#include <functional>


// dynamic schedule cuda memory, equal to cudaMalloc, the memory will be released like smart pointer  
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TorchOpsTemplateCUDA(
    const torch::Tensor& input_1,
    const torch::Tensor& input_2_int,
    const torch::Tensor& input_3,
    const torch::Tensor& input_4,
    const int args_1, int args_2)
{
    // get input types
    const int BATCH_SIZE = input_1.size(0);
    auto float_opts = input_1.options().dtype(torch::kFloat32);
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);

    // init output value  
    int output_1;

    // schedule cuda memory for output tensors
    torch::Tensor output_2 = torch::full({BATCH_SIZE, args_1}, 0.0, float_opts);
    torch::Tensor output_3 = torch::full({BATCH_SIZE, NUM_CHANNELS}, 0.0, input_1.options().dtype(torch::kInt32));

    // init the dynamic memory address, prepare the resize func, later, the resize func will dynamic schedule desired memory 
    torch::Tensor output_4 = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> output_4_Func = resizeFunctional(output_4);
    torch::Tensor output_5 = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> output_5_Func = resizeFunctional(output_5);

    output_1 = TorchOpsTemplate::TorchTemplate::forward(
		output_4_Func,
		output_5_Func,
		input_1.contiguous().data_ptr<float>(),
		input_2_int.contiguous().data_ptr<int>(),
		input_3.contiguous().data_ptr<float>(),
        input_4.contiguous().data_ptr<float>(),
		args_2,
		output_2.contiguous().data_ptr<float>(),
		output_3.contiguous().data_ptr<int>()
    );

    return std::make_tuple(output_1, output_2, output_3, output_4, output_5);
}
	


std::tuple<torch::Tensor, torch::Tensor>
TorchOpsTemplateBackwardCUDA(
   const torch::Tensor& input_1,
   const torch::Tensor& input_2_int,
   const torch::Tensor& input_3,
   const torch::Tensor& input_4,
   const int args_1)
{

    const int BATCH_SIZE = input_1.size(0);

	torch::Tensor output_grad_1 = torch::zeros({BATCH_SIZE, 3}, input_1.options());
	torch::Tensor output_grad_4 = torch::zeros({BATCH_SIZE, NUM_CHANNELS}, input_1.options());

    torch::Tensor temp_1 = torch::full({BATCH_SIZE * NUM_CHANNELS}, -1, input_1.options().dtype(torch::kInt32));
    torch::Tensor temp_2 = torch::zeros({BATCH_SIZE, 3}, input_1.options());

    TorchOpsTemplate::TorchTemplate::backward(
		args_1,
		reinterpret_cast<char*>(input_3.contiguous().data_ptr()),
		reinterpret_cast<char*>(input_4.contiguous().data_ptr()),
        input_1.contiguous().data_ptr<float>(),
		input_2_int.contiguous().data_ptr<int>(),
        temp_1.contiguous().data_ptr<int>(),
        temp_2.contiguous().data_ptr<float>(),
		output_grad_1.contiguous().data_ptr<float>(),
		output_grad_4.contiguous().data_ptr<float>()
    );

	return std::make_tuple(output_grad_1, output_grad_4);
}
