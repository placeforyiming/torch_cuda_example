# develop cuda as python ops via torch 
1. In the folder Dockerfile, one can prepare the develop env by following the readme.

2. In the folder simple_cuda_ops, three different ways are presented to wrap cuda ops as python functions. All the interface data types are tensors defined in pytorch.

3. Besides the cuda ops, if the differentiable pytorch ops are further needed, the folder torch_ops_template gives an extendable template.

4. custom_linear_ops gives an runnable example of a linear ops with an extra function to check the top-k gradients of the network parameter.