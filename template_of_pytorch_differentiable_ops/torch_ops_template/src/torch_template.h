#ifndef CUDA_TORCH_TEMPLATE_H_INCLUDED
#define CUDA_TORCH_TEMPLATE_H_INCLUDED

#include <vector>
#include <functional>

namespace TorchOpsTemplate
{
	class TorchTemplate
	{
	public:
		static int forward(
			std::function<char* (size_t)> sumBuffer,
			std::function<char* (size_t)> reorderBuffer,
			const float* input_1,
			const int* input_2_int,
			const float* input_3,
            const float* input_4,
			int H,
			float* out_1,
			int* out_2,
			bool debug = false);

		static void backward(
			const int P,
			char* sumBuffer,
			char* reorderBuffer,
            const float* input_1,
			const int* input_2,
			int* temp_1,
			const float* temp_2,
			float* output_grad_1,
			float* output_grad_2,
			bool debug = false);
	};
};

#endif