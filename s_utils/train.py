import logging
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from s_utils.utils import create_data_loader
from unet.evaluate import evaluate
from s_models.unet_model import UNet
from s_models.cls_models import Cls_model

def train_seg_cls(scale, epochs, batch_size, fold, purpose, image_scratch, label_scratch=None, image_normal=None, metrics=None,
                  new_w=320, new_h=240, checkpoint=None, dataset_name='lid', ori_numbers=None, learning_rate=None, export_path=None,
                  last_scratch_segments=None, output_unet_prediction=None, visualize_unet=False, arch='vgg',load=False, amp=False, classes=2):
    logging.basicConfig(filename='log_'+dataset_name+'_'+purpose, level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if purpose=='segmentation':
        net = UNet(n_channels=3, n_classes=classes, bilinear=False)
    elif purpose=='classification':
        net=Cls_model(classes,arch=arch)

    if load:
        net.load_state_dict(torch.load(load, map_location=device))
        logging.info(f'Model loaded from {load}')

    net.to(device=device)
    try:
        train_loader, val_loader, test_loader = create_data_loader(purpose, image_scratch, image_normal, label_scratch,
                                                                   scale, new_w, new_h, last_scratch_segments,
                                                                   ori_numbers, batch_size)
        logging.info(f'''Starting training:
                Image scratch:   {image_scratch}
                Image normal:    {image_normal}
                Label:           {label_scratch}
                Checkpoints:     {checkpoint}
                Epochs:          {epochs}
                Batch size:      {batch_size}
                Learning rate:   {learning_rate}
                Training size:   {len(train_loader) * batch_size}
                Val      size:   {len(val_loader) * batch_size}
                Test size:       {len(test_loader) * batch_size}
                Device:          {device.type}
                Images scaling:  {scale}
                Mixed Precision: {amp}
                Fold:            {fold}
                Metrics:         {metrics}
                Purpose:         {purpose}
            ''')
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5,
                                                         factor=.8)  # goal: maximize score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss()
        global_step = 0

        # 5. Begin training
        for epoch in range(1, epochs + 1):
            net.train()
            epoch_loss = 0
            with tqdm(total=len(train_loader) * batch_size, desc=f'Epoch {epoch}/{epochs}', unit='img',
                      disable=True) as pbar:
                for batch in train_loader:
                    images = batch['image']
                    true_masks = batch['mask']
                    if purpose == 'segmentation':
                        if dataset_name == 'lid':  # TODO check dataset
                            true_masks = torch.where(true_masks > 1, 1, 0)
                        else:
                            true_masks = torch.where(true_masks >= 1, 1, 0)
                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    assert images.size()[-2:] == torch.Size([240, 320])

                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)
                        if metrics == 'jaccard':
                            from torchmetrics import JaccardIndex
                            jaccard = JaccardIndex(num_classes=2).to(device)
                            score = jaccard(masks_pred, true_masks)
                            loss = criterion(masks_pred, true_masks) + 1 - score

                        elif metrics == 'dice':
                            from torchmetrics import Dice
                            dice = Dice(average='micro').to(device)
                            score = dice(torch.argmax(masks_pred, dim=1), true_masks)
                            loss = criterion(masks_pred, F.one_hot(true_masks, net.n_classes).float()) + 1 - score

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    division_step = (len(train_loader) * batch_size // (5 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            val_score = evaluate(net, val_loader, device, metrics, purpose=purpose,
                                                 dataset_name=dataset_name)
                            scheduler.step(val_score)
                            logging.info('Validation {} score: {}'.format(metrics, val_score))

            Path(checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(checkpoint, 'checkpoint_fold.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

        test_score = evaluate(net, test_loader, device, metrics, purpose=purpose, dataset_name=dataset_name,
                              export_detail=False, export_path=export_path)
        logging.info('Testing {} score: {}'.format(metrics, test_score))

        if visualize_unet:
            split = 1 if purpose == 'segmentation' else 2
            os.system('cmd /c "python unet/predict.py --model {} --input {} --output {} --new_w 320 --new_h 240'.format(
                os.path.join(checkpoint, 'checkpoint_fold.pth'),
                'data/{}_1_output_scratch_sliced/split_{}_fold_{}/JPEGImages'.format(dataset_name, split, fold),
                output_unet_prediction))

        print('=========== val results fold {}==========='.format(fold))
        print(val_score)
        print('=========== test results fold {}=========='.format(fold))
        print(test_score)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
