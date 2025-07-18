import torch.nn.functional as F

def mse_loss_mean(pred, target):
    """
    Calculate mean squared error loss

    params
    ------
    pred: torch.tensor #prediction
    target: torch.tensor #target

    return
    ------
    loss: torch.tensor #mean squared error loss
    """
    return F.mse_loss(pred, target, reduction='none').mean(dim=-1)

def zero_one_loss_mean(pred, target):
    """
    Calculate zero one loss

    params
    ------
    pred: torch.tensor #prediction
    target: torch.tensor #target

    return
    ------
    loss: torch.tensor #zero one loss
    """
    return (pred != target).float().mean(dim=-1)