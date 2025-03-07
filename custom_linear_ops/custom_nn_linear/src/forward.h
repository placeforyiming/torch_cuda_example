#ifndef CUDA_CUSTOM_NN_LINEAR_FORWARD_H_INCLUDED
#define CUDA_CUSTOM_NN_LINEAR_FORWARD_H_INCLUDED

namespace FORWARD
{
    void validate_input_channels(int input_channels, int max_channels);
    
	void forward(
        const int batch_size,
        const int in_features, const int out_features,
        const float*  input_tensor,
        const float*  trainable_weights,
        float*  output);
}

#endif