#ifndef CUDA_CUSTOM_NN_LINEAR_H_INCLUDED
#define CUDA_CUSTOM_NN_LINEAR_H_INCLUDED

#include <vector>
#include <functional>

namespace CustomOpsNNLinear
{
	class CustomNNLinear
	{
	public:
		static void forward(
			const int batch_size,
			const int in_features, const int out_features,
			std::function<char* (size_t)> grad_pos_buffer_Func,
			const float* input_tensor,
			const float* trainable_weights,
            float* output,
			bool debug = false);

		static void backward(
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
			bool debug = false);
	};
};

#endif