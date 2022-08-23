import torch
import torch.nn.functional as F
from tqdm import tqdm

from unet.utils.dice_score import multiclass_dice_coeff, dice_coeff
from unet.utils.jaccard_score import jaccard_coeff, multiclass_jaccard_coeff


def evaluate(net, dataloader, device, metrics, purpose):
    net.eval()
    num_val_batches = len(dataloader)
    score={}
    score['jaccard']=0
    score['dice']=0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False, disable=True):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        mask_true = torch.where(mask_true > 1, 1, 0) if purpose=='segmentation' else mask_true
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        if purpose=='segmentation':
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        elif purpose=='classification':
            mask_true = F.one_hot(mask_true, net.n_classes).float()

        with torch.no_grad():
            # predict the mask
            # if image.shape!=torch.Size([1, 3, 240, 320]):
            #     print(image.shape)
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the score
                if metrics=='jaccard':
                    score['jaccard'] += jaccard_coeff(mask_pred, mask_true, reduce_batch_first=False)
                elif metrics=='dice':
                    score['dice'] += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                if purpose=='segmentation':
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                elif purpose=='classification':
                    mask_pred = F.one_hot(mask_pred[0].argmax(dim=1), net.n_classes).float()
                # compute the score, ignoring background
                if metrics=='jaccard':
                    score['jaccard'] += multiclass_jaccard_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                elif metrics=='dice':
                    score['dice'] += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return score[metrics]
    return score[metrics] / num_val_batches
