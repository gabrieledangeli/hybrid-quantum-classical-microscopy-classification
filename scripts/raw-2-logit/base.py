import torch
import numpy as np

def np2torch(nparray):
    """Convert numpy array to torch tensor
       For array with more than 3 channels, it is better to use an input array in the format BxHxWxC

       Args:
           numpy array (ndarray) BxHxWxC or HxWxC
       Returns:
           torch tensor (tensor) BxCxHxW or CxHxW"""

    # Use from_numpy to avoid copy if possible
    if isinstance(nparray, np.ndarray):
        tensor = torch.from_numpy(nparray)
    else:
        tensor = torch.as_tensor(nparray)
    
    # Ensure float
    if not tensor.is_floating_point():
        tensor = tensor.float()

    if tensor.ndim == 2:
        return tensor
        
    if tensor.ndim == 3:
        # Check if likely HWC based on channel size
        # If last dim is 1, 3, or 4, assume HWC
        d0, d1, d2 = tensor.shape
        if d2 <= 4 and d0 > 4: 
             # HWC -> CHW
             return tensor.permute(2, 0, 1)
        # Else assume CHW or unknown
        return tensor

    if tensor.ndim == 4:  # More images with more channels (BxHxWxC)
        # Check if likely BHWC
        b, d1, d2, d3 = tensor.shape
        if d3 <= 4 and d1 > 4:
            # BHWC -> BCHW
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

