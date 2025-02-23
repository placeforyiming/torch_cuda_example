
import torch
from . import _C
import torch.nn as nn
import torch.nn.functional as F



class _TorchOpsTemplate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_1,
        input_2,
        input_3,
        args_1,
        args_2
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            input_1,
            input_2,
            input_3,
            args_1,
            args_2
        )
        # Invoke C++/CUDA rasterizer
        output_1, output_2, output_3, output_4, output_5 = _C.TorchOpsTemplateCUDA(*args) # todo
        
        # Keep relevant tensors for backward
        ctx.saved_args = output_1

        ctx.save_for_backward(
            output_2, 
            output_3, 
            output_4, 
            output_5
        )
        return output_1, output_3

    @staticmethod # todo
    def backward(ctx, output_1_grad, output_3_grad):

        # Restore necessary values from context
        saved_args = ctx.saved_args

        output_2, output_3, output_4, output_5 = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            output_2, 
            output_3, 
            output_4, 
            output_5,
            saved_args)

        # Compute gradients for relevant tensors by invoking backward method
        output_1_grad, output_3_grad, output_4_grad = _C.TorchOpsTemplateBackwardCUDA(*args)

        # the same order and store the grads as in args
        grads = (
            output_1_grad,
            None,
            output_3_grad,
            output_4_grad,
            None
        )

        return grads



class TorchOpsTemplate(nn.Module):
    def __init__(self, args_1, args_2):
        super().__init__()
        self.args_1 = args_1
        self.args_2 = args_2
        self.register_buffer('buffer_param', torch.tensor([0, 0], dtype=torch.float))
        self.trainable_param = nn.Parameter(torch.tensor([0, 0], dtype=torch.float))
        
    def forward(self, input_1, input_2): 

        # Invoke C++/CUDA rasterization routine
        output_1, output_3 = _TorchOpsTemplate.apply(
            input_1,
            input_2,
            self.trainable_param,
            self.args_1,
            self.args_2
        )

        return output_1, output_3 
