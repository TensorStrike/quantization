import torch

from helper import linear_q_symmetric, get_q_scale_symmetric, linear_dequantization
from helper import plot_quantization_errors, eval_quantization


test_tensor=torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5,  -184],
     [0,     684.6, 245.5]]
)

quantized_tensor, scale = linear_q_symmetric(test_tensor)
dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)
plot_quantization_errors(test_tensor, quantized_tensor,dequantized_tensor)
print(f"""Quantization Error : \
{eval_quantization(test_tensor, dequantized_tensor)}""")

