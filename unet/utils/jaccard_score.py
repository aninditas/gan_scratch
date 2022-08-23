import torch
from torch import Tensor


def jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Jaccard coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Jaccard: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = inter

        return (inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        jaccard = 0
        for i in range(input.shape[0]):
            jaccard += jaccard_coeff(input[i, ...], target[i, ...])
        return jaccard / input.shape[0]


def multiclass_jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of jaccard coefficient for all classes
    assert input.size() == target.size()
    jaccard = 0
    for channel in range(input.shape[1]):
        jaccard += jaccard_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return jaccard / input.shape[1]


def jaccard_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # jaccard loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_jaccard_coeff if multiclass else jaccard_coeff
    return 1 - fn(input, target, reduce_batch_first=True)