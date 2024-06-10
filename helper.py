import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

def plot_matrix(tensor, ax, title, vmin=0, vmax=1, cmap=None):
    """
    Plot a heatmap of tensors using seaborn
    """
    sns.heatmap(tensor.cpu().numpy(), ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=".2f", cbar=False)
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def plot_quantization_errors(original_tensor, quantized_tensor, dequantized_tensor, dtype = torch.int8, n_bits = 8):
    """
    A method that plots 4 matrices, the original tensor, the quantized tensor
    the de-quantized tensor and the error tensor.
    """
    # Get a figure of 4 plots
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    # Plot the first matrix
    plot_matrix(original_tensor, axes[0], 'Original Tensor', cmap=ListedColormap(['white']))

    # Get the quantization range and plot the quantized tensor
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    plot_matrix(quantized_tensor, axes[1], f'{n_bits}-bit Linear Quantized Tensor', vmin=q_min, vmax=q_max, cmap='coolwarm')

    # Plot the de-quantized tensors
    plot_matrix(dequantized_tensor, axes[2], 'Dequantized Tensor', cmap='coolwarm')

    # Get the quantization errors
    q_error_tensor = abs(original_tensor - dequantized_tensor)
    plot_matrix(q_error_tensor, axes[3], 'Quantization Error Tensor', cmap=ListedColormap(['white']))

    fig.tight_layout()
    plt.show()


def linear_dequantization(quantized_tensor, scale, zero_point):
    # r = s (q - z)
    return scale * (quantized_tensor - zero_point)


def eval_quantization(original_tensor, dequantized_tensor, verbose=True):
    error = dequantized_tensor - original_tensor
    if verbose:
        print('error: \n', error)
        print('mean squared error: \n', error.square().mean())
    return error, error.square().mean()


def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype=torch.int8):
    '''
    quantize a tensor with linear quantization
    linear quantization formula r=s(q-z)
    '''
    scaled_and_shifted_tensor = tensor/scale + zero_point           # r/s+z
    rounded_tensor = torch.round(scaled_and_shifted_tensor)         # round()
    q_min = torch.iinfo(dtype).min          # output min and max values of tensor
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)           # limits eles of the tensor to fall within range (min,max)

    return q_tensor

def get_q_scale_symmetric(tensor, dtype=torch.int8):
    '''
    :returns scale for symmetric quantization
    '''
    r_max = abs(torch.max(tensor).item())
    q_max = torch.iinfo(dtype).max

    return r_max/q_max

def linear_q_symmetric(tensor, dtype=torch.int8):
    '''
    symmetric linear quantization
    '''
    scale = get_q_scale_symmetric(tensor)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale, 0, dtype=dtype)
    return quantized_tensor, scale

def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
    output_dim = r_tensor.shape[dim]
    scale = torch.zeros(output_dim)
    for index in range(output_dim):
        sub_tensor = r_tensor.select(dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor)

    # reshape tensor [1,1] -> [3.1]
    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)

    quantized_tensor = linear_q_with_scale_and_zero_point(r_tensor, scale=scale, zero_point=0, dtype=dtype)

    return quantized_tensor, scale


def linear_q_symmetric_per_group(tensor, group_size, dtype=torch.int8):
    t_shape = tensor.shape
    assert t_shape[1] % group_size == 0
    assert tensor.dim() == 2

    tensor = tensor.view(-1, group_size)

    quantized_tensor, scale = linear_q_symmetric_per_channel(tensor, dim=0, dtype=dtype)

    quantized_tensor = quantized_tensor.view(t_shape)

    return quantized_tensor, scale


def linear_dequantization_per_group(quantized_tensor, scale, group_size):
    q_shape = quantized_tensor.shape
    quantized_tensor = quantized_tensor.view(-1, group_size)

    dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)

    dequantized_tensor = dequantized_tensor.view(q_shape)

    return dequantized_tensor
