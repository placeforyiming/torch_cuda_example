import time
import torch
import numpy as np
from torch import nn
from custom_nn_linear import CustomNNLinear

batch_size = 2
in_feats = 4
out_feats = 6

input_tensor = torch.randn(batch_size, in_feats, requires_grad=True).cuda()
input_tensor_custom = input_tensor.clone().cuda()
np_weight = np.random.randn(out_feats, in_feats)
print ("supposed result:")
print (np.dot(input_tensor.cpu().detach().numpy(),np_weight.transpose()))

print ("============== torch nn linear ============")
linear_layer = nn.Linear(in_features=in_feats, out_features=out_feats, bias=False)
linear_layer.weight.data = torch.from_numpy(np_weight).float().cuda()
print ("torch nn linear result:")
t_start = time.time()
result = linear_layer(input_tensor)
t_end = time.time()
print (result)
print ("torch time cost:", t_end-t_start)
print ("torch input grads:")
input_tensor.retain_grad()
y = torch.ones_like(result)
loss = torch.abs(y-result).sum()
loss.backward()
print (input_tensor.grad)
print ("torch weights grads:")
print (linear_layer.weight.grad)

print ("============== custom nn linear ============")
top_k_grads = 5
custom_nn_linear_layer = CustomNNLinear(in_feats, out_feats, top_k_grads)
custom_nn_linear_layer.to("cuda")
custom_nn_linear_layer.set_weights(np_weight)
t_start = time.time()
result = custom_nn_linear_layer(input_tensor_custom)
t_end = time.time()
output, topk_grads, pos_1st_dim_of_topk_grads, pos_2st_dim_of_topk_grads = result
print ("custom nn linear result:")
print (output)
print ("custom nn linear time cost:", t_end-t_start)
print ("custom nn linear input grads:")
input_tensor_custom.retain_grad()
y = torch.ones_like(output)
loss = torch.abs(y-output).sum()
loss.backward()
print (input_tensor_custom.grad)
print ("custom nn linear weights grads:")
print (custom_nn_linear_layer.trainable_weights.grad)

print ("============== check the top k grads ============")
print (topk_grads)
for i in range(top_k_grads):
    idx_1 = pos_1st_dim_of_topk_grads[i]
    idx_2 = pos_2st_dim_of_topk_grads[i]
    print (custom_nn_linear_layer.trainable_weights.grad[idx_1, idx_2])
