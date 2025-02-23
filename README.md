# develop cuda as python ops via torch 
1. In the folder Dockerfile, one can prepare the develop env by following the readme.

2. In the folder wrap_simple_cuda_via_torch, three different ways are presented to wrap cuda ops as python functions. All the interface data types are tensors defined in pytorch.

3. Besides the cuda ops, if the differentiable pytorch ops are further needed, the folder template_of_pytorch_differentiable_ops gives an extendable template.