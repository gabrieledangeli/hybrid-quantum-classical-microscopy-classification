import torch
import numpy as np

def np2torch(nparray):
    """Convert numpy array to torch tensor
       For array with more than 3 channels, it is better to use an input array in the format BxHxWxC

       Args:
           numpy array (ndarray) BxHxWxC
       Returns:
           torch tensor (tensor) BxCxHxW"""

    tensor = torch.Tensor(nparray)

    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        height, width, channels = tensor.shape
        if channels <= 3:  # Single image with more channels (HxWxC)
            return tensor.permute(2, 0, 1)

    if tensor.ndim == 4:  # More images with more channels (BxHxWxC)
        return tensor.permute(0, 3, 1, 2)

    return tensor

def torch2np(torchtensor):
    """Convert torch tensor to numpy array 
       For tensor with more than 3 channels or batch, it is better to use an input tensor in the format BxCxHxW

       Args:
           torch tensor (tensor) BxCxHxW
       Returns:
           numpy array (ndarray) BxHxWxC"""

    ndarray = torchtensor.detach().cpu().numpy().astype(np.float32)

    if ndarray.ndim == 3:  # Single image with more channels (CxHxW)
        channels, height, width = ndarray.shape
        if channels <= 3:
            return ndarray.transpose(1, 2, 0)

    if ndarray.ndim == 4:  # More images with more channels (BxCxHxW)
        return ndarray.transpose(0, 2, 3, 1)

    return ndarray

def get_name(obj):
    return obj.__name__ if hasattr(obj, '__name__') else type(obj).__name__

