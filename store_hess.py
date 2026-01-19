import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import math

def to_numpy(tensor):
    """
    Convert a PyTorch tensor to a NumPy array on CPU.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        np.ndarray: The tensor converted to a NumPy array.
    """
    return tensor.cpu().detach().numpy()



def plot_comparison(matrix1, matrix2, title1, title2, filename):
    """
    Plot two matrices side-by-side for comparison and save the figure.

    Args:
        matrix1 (torch.Tensor): The first matrix.
        matrix2 (torch.Tensor): The second matrix.
        title1 (str): Title for the first subplot.
        title2 (str): Title for the second subplot.
        filename (str): Path to save the resulting image.
    """
    vmin = np.percentile(to_numpy(matrix1), 1)
    vmax = np.percentile(to_numpy(matrix1), 99)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot
    axs[0].imshow(to_numpy(matrix1), cmap='coolwarm', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[0].set_title(title1)
    axs[0].axis("off")
    # Add a rectangle patch as an outline to the first subplot:
    rect0 = patches.Rectangle((0, 0), 1, 1, transform=axs[0].transAxes,
                            fill=False, edgecolor='black', linewidth=2)
    axs[0].add_patch(rect0)

    # Second subplot
    axs[1].imshow(to_numpy(matrix2), cmap='coolwarm', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    axs[1].set_title(title2)
    axs[1].axis("off")
    # Add a rectangle patch as an outline to the second subplot:
    rect1 = patches.Rectangle((0, 0), 1, 1, transform=axs[1].transAxes,
                            fill=False, edgecolor='black', linewidth=2)
    axs[1].add_patch(rect1)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
hess_dict = torch.load('./hessian.pt')
hess_num_dict = torch.load('./hess_num.pt')


new_hess_dict = {}
for name in hess_dict.keys():
    scale = hess_dict[name] 
    scale2 = scale * (1/math.sqrt(hess_num_dict[name]))
    new_hess_dict[name] = scale2.float()

example1 = new_hess_dict['model.layers.0.self_attn.k_proj']
example = hess_dict['model.layers.0.self_attn.k_proj']

breakpoint()
plot_comparison(
    example,
    example.T,
    "Hessian Matrix 1",
    "Hessian Matrix 2",
    "./hessian_comparison.png"
)
breakpoint()