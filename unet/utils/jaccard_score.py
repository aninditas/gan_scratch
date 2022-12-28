import torch
from torch import Tensor
import sklearn.metrics
from torchmetrics import JaccardIndex


def jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Jaccard coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Jaccard: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    # compute and average metric for each batch element
    jaccard = 0
    jc = JaccardIndex(num_classes=2, average='macro').to(device)
    for i in range(input.shape[0]):
        # jaccard += jaccard_coeff(input[i, ...], target[i, ...])
        # jaccard += sklearn.metrics.jaccard_score(input[i, ...], target[i, ...], average='macro')
        jaccard += jc(input[i, ...].type(torch.int64), target[i, ...].type(torch.int64))
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
