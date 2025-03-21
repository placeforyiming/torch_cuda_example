import torch
from torch_ops_template import TorchOpsTemplate

args_1 = 10
args_2 = 100 

torch_op_example = TorchOpsTemplate(args_1, args_2)

input_1 = torch.ones(8, 128, 128, 3)
input_2 = torch.ones(8, 3, 3, 1)

result = torch_op_example(input_1, input_2)