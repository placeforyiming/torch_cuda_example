#ifndef CUDA_CUSTOM_NN_LINEAR_BACKWARD_H_INCLUDED
#define CUDA_CUSTOM_NN_LINEAR_BACKWARD_H_INCLUDED

#include <cuda.h>

namespace BACKWARD
{

	void backward(
		const int batch_size,
		const int in_features,
		const int out_features,
		const float* input_tensor,
		const float* output_grad,
		const float* trainable_weights,
		float* keys_unsorted,
		int* values_unsorted,
		float* input_tensor_grad,
		float* trainable_weights_grad);

	void assigning_topk(
		const int topk_num,
		const int in_features,
		const int out_features,
        float* keys,
        int* values,
		float* topk_grads,
		int* pos_1st_dim_of_topk_grads_int,
		int* pos_2nd_dim_of_topk_grads_int);
}

#endif