import torch

class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.

    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.

    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cuda'):

        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[
            None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[
            None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor.
        """

        if not self.inplace:
            tensor = tensor.clone()
        
        return (tensor - self.mean) / self.std


def denorm(x, device,add_bg):
    # get from [-1,1] to [0,1]
    if add_bg:
        norm = Normalize(mean=(-1, -1, -1), std=(2, 2, 2), device=device)
    else:
        norm = Normalize(mean=(-1, -1, -1, 0), std=(2, 2, 2, 1), device=device)
    return norm(x)

