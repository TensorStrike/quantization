import torch
from helper import *


def get_q_scale_symmetric(tensor, dtype=torch.int8):
    '''
    :returns scale for symmetric quantization
    '''
    r_max = abs(torch.max(tensor).item())
    q_max = torch.iinfo(dtype).max

    return r_max/q_max


def linear_q_symetric(tensor, dtype=torch.int8):
    '''
    symmetric linear quantization
    '''
    scale = get_q_scale_symmetric(tensor)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, 0)
    return quantized_tensor, scale


test_tensor = torch.randn((4,4))

quantized_tensor, scale = linear_q_symetric(test_tensor)
dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)

plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)

eval_quantization(test_tensor,dequantized_tensor)