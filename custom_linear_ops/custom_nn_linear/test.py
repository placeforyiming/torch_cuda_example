import torch
import numpy as np
from torch import nn
from custom_nn_linear import CustomNNLinear

batch_size = 2
in_feats = 4
out_feats = 6

input_tensor = torch.randn(batch_size, in_feats, requires_grad=True)
np_weight = np.random.randn(out_feats, in_feats)

linear_layer = nn.Linear(in_features=in_feats, out_features=out_feats, bias=False)
linear_layer.weight.data = torch.from_numpy(np_weight).float()

top_k_grads = 5
custom_nn_linear_layer = CustomNNLinear(in_feats, out_feats, top_k_grads)
custom_nn_linear_layer.set_weights(np_weight)
custom_nn_linear_layer(input_tensor)