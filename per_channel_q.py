import torch

from helper import linear_q_symmetric, get_q_scale_symmetric, linear_dequantization, linear_q_with_scale_and_zero_point
from helper import plot_quantization_errors, eval_quantization

test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)


output_dim = test_tensor.shape[0]
# print(output_dim)
scale = torch.zeros(output_dim)
# print(scale)

dim = 0
for index in range(output_dim):
    sub_tensor = test_tensor.select(dim, index)
    # print(sub_tensor)
    scale[index] = get_q_scale_symmetric(sub_tensor)

print('scale:',scale)        # ([_,_,_])
# print(scale.shape)        # [3]

scale_shape = [1] * test_tensor.dim()
print(scale_shape)  # [1,1]

scale_shape[dim] = -1       # [-1,1]
scale = scale.view(scale_shape)     # [3] -> [3,1]
print(scale)
print(scale.shape)      # [3,1]


'''
m = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
s = torch.tensor([1,5,10])
print('m:\n ',m)
print('s:\n ',s)        # [3]
s1 = s.view(1, 3)
print(s1.shape)
print(s1)
s2 = s.view(1, -1)
print(s2.shape)
print(s2)

scale = torch.tensor([[1], [5], [10]])
print(scale.shape)
print(m / scale)
'''


quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale=scale, zero_point=0)
print(test_tensor)
print(quantized_tensor)

def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
    output_dim = r_tensor.shape[dim]
    # store the scales
    scale = torch.zeros(output_dim)

    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype=dtype)

    # reshape the scale
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_q_with_scale_and_zero_point(
        r_tensor, scale=scale, zero_point=0, dtype=dtype)

    return quantized_tensor, scale

quantized_tensor_0, scale_0 = linear_q_symmetric_per_channel(test_tensor, dim=0)
quantized_tensor_1, scale_1 = linear_q_symmetric_per_channel(test_tensor, dim=1)

dequantized_tensor_0 = linear_dequantization(quantized_tensor_0, scale=scale_0, zero_point=0)
eval_quantization(test_tensor, dequantized_tensor_0)
plot_quantization_errors(test_tensor, quantized_tensor_0, dequantized_tensor_0)

dequantized_tensor_1 = linear_dequantization(quantized_tensor_1, scale=scale_1, zero_point=0)
eval_quantization(test_tensor, dequantized_tensor_1)
plot_quantization_errors(test_tensor, quantized_tensor_1, dequantized_tensor_1)