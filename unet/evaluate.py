import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from unet.utils.dice_score import multiclass_dice_coeff, dice_coeff
from unet.utils.jaccard_score import jaccard_coeff, multiclass_jaccard_coeff
import matplotlib.pyplot as plt

def evaluate(net, dataloader, device, metrics, purpose, dataset_name=None, export_detail=False, export_path=None):
    net.eval()
    num_val_batches = len(dataloader)
    score={}
    score['jaccard']=0
    score['dice']=0
    count = 0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False, disable=True):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        if purpose =='segmentation':
            if dataset_name == 'lid': #TODO check dataset
                mask_true = torch.where(mask_true > 1, 1, 0)
            else:
                mask_true = torch.where(mask_true >= 1, 1, 0)
        assert image.size()[-2:] == torch.Size([240, 320])
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        if purpose=='segmentation':
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        elif purpose=='classification':
            if export_detail: logging.info(mask_true)
            mask_true = F.one_hot(mask_true, net.n_classes).float()

        with torch.no_grad():
            # predict the mask
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
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1),net.n_classes)
                    if export_detail:
                        export_path = export_path+'_'+net.arch
                        try:
                            os.mkdir(export_path)
                        except:
                            temp = os.listdir(export_path)
                            for t in temp:
                                try:
                                    os.remove(os.path.join(export_path, t))
                                except: pass
                        logging.info(mask_pred)
                        for i, mt, mp in zip(image, mask_true.argmax(dim=1), mask_pred.argmax(dim=1)):
                            temp = mt-mp
                            status = 'FP' if temp<0 else 'FN' if temp>0 else 'TN' if temp==0 and mt==0 else 'TP'
                            plt.imsave(os.path.join(export_path,'{}_{}.jpg'.format(str(count),status)),i.permute(1,2,0).cpu().numpy())
                            count+=1
                    # mask_pred = F.one_hot(mask_pred, net.n_classes).float()
                # compute the score, ignoring background
                if metrics=='jaccard':
                    score['jaccard'] += multiclass_jaccard_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                elif metrics=='dice':
                    score['dice'] += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    # net.train_seg_cls()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return score[metrics]
    return score[metrics] / num_val_batches
