import torch

def adjust_sign(w):
    """
    adjust sign of w
    """
    w = ((w+1)*63.5).clamp(0, 127).round().byte()
    w = torch.where(w>0, 70, 0)
    w = w/63.5-1
    w = w.permute(0, 1, 3, 2)

    return w

