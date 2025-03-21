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
#include "src/torch_custom_nn_linear.h"
#include <fstream>
#include <string>
#include <functional>
#include <stdexcept>


// dynamic schedule cuda memory, equal to cudaMalloc, the memory will be released like smart pointer  
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<torch::Tensor, torch::Tensor>
CustomNNLinearCUDA(
    const torch::Tensor& input_tensor,
    const torch::Tensor& trainable_weights,
    const int in_features, const int out_features)
{
    // get input types
    const int batch_size = input_tensor.size(0);
    auto float_opts = input_tensor.options().dtype(torch::kFloat32);
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);

    // schedule cuda memory for output tensors
    torch::Tensor output = torch::full({batch_size, out_features}, 0.0, float_opts);
    
    // init the dynamic memory address, prepare the resize func, later, the resize func will dynamic schedule desired memory 
    torch::Tensor grad_pos_buffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> grad_pos_buffer_Func = resizeFunctional(grad_pos_buffer);

    CustomOpsNNLinear::CustomNNLinear::forward(
        batch_size,
        in_features, out_features,
		grad_pos_buffer_Func,
		input_tensor.contiguous().data_ptr<float>(),
        trainable_weights.contiguous().data_ptr<float>(),
		output.contiguous().data_ptr<float>()
    );

    return std::make_tuple(output, grad_pos_buffer);
}
	


std::tuple<torch::Tensor, torch::Tensor>
CustomNNLinearBackwardCUDA(
    const torch::Tensor& input_tensor,
    const torch::Tensor& output_grads,
    const torch::Tensor& trainable_weights,
    const torch::Tensor& topk_grads,
    const torch::Tensor& pos_1st_dim_of_topk_grads_int,
    const torch::Tensor& pos_2nd_dim_of_topk_grads_int,
    const int in_features, const int out_features, const int topk_grad_buffer,
    const torch::Tensor& grad_pos_buffer)
{
    const int batch_size = input_tensor.size(0);
    if (in_features != trainable_weights.size(1)) {
        throw std::runtime_error("trainable_weights shape[0] is not compatible");
    }
    if (out_features != trainable_weights.size(0)) {
        throw std::runtime_error("trainable_weights shape[1] is not compatible");
    }
    if (topk_grad_buffer != topk_grads.size(0)) {
        throw std::runtime_error("topk_grad_buffer is not compatible");
    }

	torch::Tensor input_tensor_grad = torch::zeros({batch_size, in_features}, input_tensor.options());
	torch::Tensor trainable_weights_grad = torch::zeros({out_features, in_features}, input_tensor.options());
    CustomOpsNNLinear::CustomNNLinear::backward(
		batch_size, in_features, out_features,
		reinterpret_cast<char*>(grad_pos_buffer.contiguous().data_ptr()),
        input_tensor.contiguous().data_ptr<float>(),
        output_grads.contiguous().data_ptr<float>(),
        trainable_weights.contiguous().data_ptr<float>(),
		input_tensor_grad.contiguous().data_ptr<float>(),
		trainable_weights_grad.contiguous().data_ptr<float>(),
        topk_grad_buffer,
        topk_grads.contiguous().data_ptr<float>(),
        pos_1st_dim_of_topk_grads_int.contiguous().data_ptr<int>(),
        pos_2nd_dim_of_topk_grads_int.contiguous().data_ptr<int>()
    );
    
	return std::make_tuple(input_tensor_grad, trainable_weights_grad);
}
