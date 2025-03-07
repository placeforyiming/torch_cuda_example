#ifndef CUDA_TORCH_TEMPLATE_H_INCLUDED
#define CUDA_TORCH_TEMPLATE_H_INCLUDED

#include <vector>
#include <functional>

namespace CustomOpsNNLinear
{
	class CustomNNLinear
	{
	public:
		static void forward(
			std::function<char* (size_t)> grad_order_buffer,
			std::function<char* (size_t)> pos_1st_dim_buffer,
			std::function<char* (size_t)> pos_2st_dim_buffer,
			const float* input_tensor,
			const float* trainable_weights,
            const float* output,
			bool debug = false);

		static void backward(
			const int in_features,
			const int out_features,
			char* grad_order_buffer,
			char* pos_1st_dim_buffer_int,
			char* pos_2st_dim_buffer_int,
            const float* input_tensor,
			const float* trainable_weights,
			const float* input_tensor_grad,
			const float* trainable_weights_grad,
			bool debug = false);
	};
};

#endif