import torch
from helper import plot_quantization_errors, linear_dequantization

def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    '''
    quantize a tensor with linear quantization
    linear quantization formula r=s(q-z)
    '''
    scaled_and_shifted_tensor = tensor/scale + zero_point           # r/s+z
    rounded_tensor = torch.round(scaled_and_shifted_tensor)         # round()
    q_min = torch.iinfo(dtype).min          # output min and max values that can be stored with dtype
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max)           # limits eles of the tensor to fall within range (min,max)

    return q_tensor

def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    '''
    computes scale and zero point of a tensor
    '''
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    r_min = torch.min(tensor).item()
    r_max = torch.max(tensor).item()

    # scale = (r_min - r_max)/(q_min - q_max)
    s = (r_min - r_max) / (q_min - q_max)

    # zero point = int(round(q_min - r_min/s))
    z = int(round(q_min - r_min/s))

    # edge cases
    if z < q_min:
        z = q_min
    if z > q_max:
        z = q_max

    return s, z


def linear_quantization(tensor, dtype=torch.int8):
    '''
    put everything together
    :returns quantized tensor, scale, zero point
    '''
    s, z = get_q_scale_and_zero_point(tensor)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, s, z)
    return quantized_tensor, s, z


def eval_quantization(original_tensor, dequantized_tensor, verbose=True):
    error = dequantized_tensor - original_tensor
    if verbose:
        print('error: \n', error)
        print('mean squared error: \n', error.square().mean())
    return error, error.square().mean()


test_tensor = torch.tensor(
    [[191.6, -13.5, 728.6],
     [92.14, 295.5, -184],
     [0, 684.6, 245.5]]
)

scale = 3.5
zero_point = -70

quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale, zero_point)

print('quantized tensor:\n',quantized_tensor)

# dequantization with random scale and zero point
dequantized_tensor = scale * (quantized_tensor.float() - zero_point)
print('dequantized tensor:\n', dequantized_tensor)

# plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)

eval_quantization(test_tensor, dequantized_tensor)

# s, z = get_q_scale_and_zero_point(test_tensor)
# print('computed scale = {} and zero point = {}: '.format(s, z))
# new_quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, s, z)
# print('new dequantized tensor: \n', new_quantized_tensor)

quantized_tensor, s, z = linear_quantization(test_tensor)
dequantized_tensor = linear_dequantization(quantized_tensor, s, z)

plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)
eval_quantization(test_tensor, dequantized_tensor)


r_tensor = torch.randn((4,4))
print('original_tensor:\n', r_tensor)
quantized_tensor, s, z = linear_quantization(r_tensor)
dequantized_tensor = linear_dequantization(quantized_tensor, s, z)
plot_quantization_errors(r_tensor, quantized_tensor, dequantized_tensor)
eval_quantization(r_tensor, dequantized_tensor)