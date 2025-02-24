
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
        input_4,
        args_1,
        args_2
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            input_1,
            input_2,
            input_3,
            input_4,
            args_1,
            args_2
        )
        # Invoke C++/CUDA rasterizer
        output_1, output_2, output_3, output_4, output_5 = _C.torch_ops_template(*args) # todo
        
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
    def backward(ctx, output_1_grad, output_4_grad):

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
        output_1_grad, output_4_grad = _C.torch_ops_template_backward(*args)

        # the same order and store the grads as in args
        grads = (
            output_1_grad,              # grad for output_2 in args
            None,                       # grad for output_3 in args
            None,                       # grad for output_4 in args
            output_4_grad,              # grad for output_5 in args
            None                        # grad for saved_args in args
        )

        return grads



class TorchOpsTemplate(nn.Module):
    def __init__(self, args_1, args_2):
        super().__init__()
        self.args_1 = args_1
        self.args_2 = args_2
        self.register_buffer('buffer_param', torch.tensor([0, 0], dtype=torch.int))
        self.trainable_param = nn.Parameter(torch.tensor([0, 0], dtype=torch.float))
        
    def forward(self, input_1, input_2): 
        print ("forward once")
        # Invoke C++/CUDA rasterization routine
        output_1, output_3 = _TorchOpsTemplate.apply(
            input_1,
            self.buffer_param,
            self.trainable_param,
            input_2,
            self.args_1,
            self.args_2
        )

        return output_1, output_3 
