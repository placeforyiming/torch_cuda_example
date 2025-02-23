 #pragma once
 #include <torch/extension.h>
 #include <cstdio>
 #include <tuple>
 #include <string>
     
 std::tuple<int, torch::Tensor, torch::Tensor>
 TorchOpsTemplateCUDA(
     const torch::Tensor& input_1,
     const torch::Tensor& input_2,
     const torch::Tensor& input_3,
     const int args_1, int args_2);
 
 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 TorchOpsTemplateBackwardCUDA(
    const torch::Tensor& input_1,
    const torch::Tensor& input_2,
    const torch::Tensor& input_3,
    const torch::Tensor& input_4,
    const int args_1, int args_2, int args_3, int args_4);
 
