 #pragma once
 #include <torch/extension.h>
 #include <cstdio>
 #include <tuple>
 #include <string>
     


std::tuple<torch::Tensor, torch::Tensor>
CustomNNLinearCUDA(
    const torch::Tensor& input_tensor,
    const torch::Tensor& trainable_weights,
    const int in_features, const int out_features);
 
std::tuple<torch::Tensor, torch::Tensor>
CustomNNLinearBackwardCUDA(
    const torch::Tensor& input_tensor,
    const torch::Tensor& output_grads,
    const torch::Tensor& trainable_weights,
    const torch::Tensor& topk_grads,
    const torch::Tensor& pos_1st_dim_of_topk_grads_int,
    const torch::Tensor& pos_2nd_dim_of_topk_grads_int,
    const int in_features, const int out_features, const int topk_grad_buffer,
    const torch::Tensor& grad_pos_buffer);
 