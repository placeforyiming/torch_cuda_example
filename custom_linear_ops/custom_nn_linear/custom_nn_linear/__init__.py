
import torch
from . import _C
import torch.nn as nn
import torch.nn.functional as F



class _CustomNNLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor, # tensor::float
        trainable_weights, # tensor::float
        topk_grads, # tensor::float
        pos_1st_dim_of_topk_grads, # tensor::int
        pos_2st_dim_of_topk_grads, # tensor::int
        weights_of_topk_grads, # tensor::float
        in_features, # int
        out_features, # int
        topk_grad_buffer # int
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            input_tensor,
            trainable_weights,
            # topk_grads,
            # pos_1st_dim_of_topk_grads,
            # pos_2st_dim_of_topk_grads,
            # weights_of_topk_grads,
            in_features,
            out_features,
            # topk_grad_buffer
        )
        # Invoke C++/CUDA rasterizer
        output, grad_order_buffer, pos_1st_dim_buffer, pos_2st_dim_buffer = _C.custom_nn_linear(*args) # todo
        
        # Keep relevant tensors for backward
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.topk_grad_buffer = topk_grad_buffer

        ctx.save_for_backward(
            input_tensor,
            trainable_weights,
            topk_grads,
            pos_1st_dim_of_topk_grads,
            pos_2st_dim_of_topk_grads,
            weights_of_topk_grads,
            grad_order_buffer,
            pos_1st_dim_buffer,
            pos_2st_dim_buffer
        )
        return output

    @staticmethod # todo
    def backward(ctx, output_grad):

        # Restore necessary values from context
        in_features = ctx.in_features
        out_features = ctx.out_features
        topk_grad_buffer = ctx.topk_grad_buffer

        input_tensor, \
        trainable_weights, \
        topk_grads, \
        pos_1st_dim_of_topk_grads, \
        pos_2st_dim_of_topk_grads, \
        weights_of_topk_grads, \
        grad_order_buffer, \
        pos_1st_dim_buffer, \
        pos_2st_dim_buffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            input_tensor,
            trainable_weights,
            topk_grads,
            pos_1st_dim_of_topk_grads,
            pos_2st_dim_of_topk_grads,
            weights_of_topk_grads,
            in_features,
            out_features,
            topk_grad_buffer,
            grad_order_buffer,
            pos_1st_dim_buffer,
            pos_2st_dim_buffer)

        # Compute gradients for relevant tensors by invoking backward method
        input_tensor_grad, trainable_weights_grad = _C.custom_nn_linear_backward(*args)

        # the same order and store the grads as in args
        grads = (
            input_tensor_grad,             
            trainable_weights_grad,
            None,              
            None,           
            None,              
            None,     
            None,              
            None,        
            None,
            None,              
            None,        
            None                     
        )

        return grads



class CustomNNLinear(nn.Module):
    
    def __init__(self, args_1, args_2, args_3):
        super().__init__()
        self.in_features = args_1 # num of in_features
        self.out_features = args_2 # num of out_features
        self.trainable_weights = nn.Parameter(torch.zeros(self.in_features, self.out_features))

        assert (args_1*args_2 >= args_3)
        # save and print the top-k grads, args_3
        self.topk_grad_buffer = args_3
        self.register_buffer('topk_grads', torch.zeros(self.topk_grad_buffer, dtype=torch.float))
        self.register_buffer('pos_1st_dim_of_topk_grads', torch.zeros(self.topk_grad_buffer, dtype=torch.int))
        self.register_buffer('pos_2st_dim_of_topk_grads', torch.zeros(self.topk_grad_buffer, dtype=torch.int))
        self.register_buffer('weights_of_topk_grads', torch.zeros(self.topk_grad_buffer, dtype=torch.float))

    def set_weights(self, new_weights_np):
        new_weights_tensor = torch.from_numpy(new_weights_np).to(
            device=self.trainable_weights.device,
            dtype=self.trainable_weights.dtype)
        self.trainable_weights.data = new_weights_tensor
        return

    def forward(self, input): 
        print ("forward once")
        # Invoke C++/CUDA rasterization routine
        output = _CustomNNLinear.apply(
            input,
            self.trainable_weights,
            self.topk_grads,
            self.pos_1st_dim_of_topk_grads,
            self.pos_2st_dim_of_topk_grads,
            self.weights_of_topk_grads,
            self.in_features,
            self.out_features,
            self.topk_grad_buffer
        )

        return output, self.topk_grads, self.weights_of_topk_grads, self.pos_1st_dim_of_topk_grads, self.pos_2st_dim_of_topk_grads
