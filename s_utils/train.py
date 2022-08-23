import argparse
import logging
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
import copy
import numpy as np
from s_utils.data_loading import BasicDataset, ClassifierDataset
from unet.evaluate import evaluate
from unet.unet_.unet_model import UNet
from unet.utils.dice_score import dice_loss
from unet.utils.jaccard_score import jaccard_loss
from s_utils.vgg_model import VGG


def train_net(net,
              device,
              image_scratch,
              image_normal,
              label_scratch,
              checkpoint,
              ori_numbers,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              new_w=None,
              new_h=None,
              amp: bool = False,
              fold: int = 5,
              metrics: str = 'jaccard',
              purpose: str = 'segmentation'):

    net_master = copy.deepcopy(net)
    # 1. Create dataset
    if purpose=='segmentation':
        dataset = BasicDataset(image_scratch, label_scratch, img_scale, new_w=new_w, new_h=new_h)
    elif purpose=='classification':
        dataset = ClassifierDataset(image_scratch, image_normal, img_scale, new_w=new_w, new_h=new_h )
    # val_scores_folds = np.zeros(fold)
    # kfold = KFold(n_splits=fold, shuffle=True, random_state=42)
    test_ids = np.arange(ori_numbers)
    val_ids = np.arange(ori_numbers, int(ori_numbers + ori_numbers / 2))
    last = len(dataset)
    if purpose=='classification':
        test_ids = np.concatenate((test_ids,np.arange(len(dataset)-ori_numbers,len(dataset))))
        val_ids = np.concatenate((val_ids,np.arange(len(dataset)-int(ori_numbers*1.5),len(dataset)-ori_numbers)))
        last=len(dataset)-int(ori_numbers*1.5)
    train_ids = np.arange(int(ori_numbers+ori_numbers/2),last)
    # for f, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    # for f, (train_ids, val_ids) in enumerate(_train_ids, _val_ids):
    #     # Print
    #     print('--------------------------------')
    #     print(f'FOLD {f}')
    #     print('--------------------------------')
    net = copy.deepcopy(net_master)
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    test_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, sampler=test_subsampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

    # # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    n_test = len(test_ids)
    n_train = len(train_ids)
    n_val = len(val_ids)
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    #
    # # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Image scratch:   {image_scratch}
        Image normal:    {image_normal}
        Label:           {label_scratch}
        Checkpoints:     {checkpoint}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Val      size:   {n_val}
        Test size:       {n_test}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Fold:            {fold}
        Metrics:         {metrics}
        Purpose:         {purpose}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=.8)  # goal: maximize score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                true_masks = torch.where(true_masks > 1, 1, 0) if purpose == 'segmentation' else true_masks

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    if metrics=='jaccard':
                        loss = criterion(masks_pred, true_masks) \
                               + jaccard_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                    elif metrics=='dice':
                        loss = criterion(masks_pred[0], true_masks) \
                               + dice_loss(F.softmax(masks_pred[0], dim=1).float(),
                                              F.one_hot(true_masks, net.n_classes).float(),
                                              multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device, metrics, purpose=purpose)
                        scheduler.step(val_score)

                        logging.info('Validation {} score: {}'.format(metrics,val_score))
                        masks_pred_ = masks_pred if purpose=='segmentation' else masks_pred[0]
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation '+metrics: val_score,
                            # 'images': wandb.Image(images[0].cpu()),
                            # 'masks': {
                            #     'true': wandb.Image(true_masks[0].float().cpu()) ,
                            #     'pred': wandb.Image(masks_pred_.argmax(dim=1)[0].float().cpu()),
                            # },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(checkpoint).mkdir(parents=True, exist_ok=True)
            # torch.save(net.state_dict(), os.path.join(checkpoint,'checkpoint_epoch_{}_fold_0.pth'.format(epoch)))
            torch.save(net.state_dict(), os.path.join(checkpoint,'checkpoint_fold.pth'))
            logging.info(f'Checkpoint {epoch} saved!')
    test_score = evaluate(net, test_loader, device, metrics, purpose=purpose)

    print('=========== val results fold {}==========='.format(fold))
    print(val_score)
    print('=========== test results fold {}=========='.format(fold))
    print(test_score)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--image_scratch', nargs='+', default='data/1_input_scratch/JPEGImages', help='Path to scratch image files')
    parser.add_argument('--image_normal', type=str, default='data/1_input_normal/JPEGImages', help='Path to normal image files for classification')
    parser.add_argument('--label_scratch', nargs='+', default='data/1_input_scratch/SegmentationClass', help='Path to label files')
    parser.add_argument('--checkpoint', type=str, default='data/14_output_checkpoint_unet', help='Path to checkpoint files')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--new_w', type=int, default=None, help='New image size')
    parser.add_argument('--new_h', type=int, default=None, help='New image size')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--fold', type=int, default=5, help='Number of fold')
    parser.add_argument('--metrics', type=str, default='jaccard', help='CNN metrics: jaccard for segmentation OR dice for classification')
    parser.add_argument('--purpose', type=str, default='segmentation', help='segmentation OR classification')
    parser.add_argument('--ori_numbers', type=int, help='number of original images')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.purpose=='segmentation':
        net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.purpose=='classification':
        net = VGG(output_dim=2)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  image_scratch=args.image_scratch,
                  image_normal=args.image_normal,
                  label_scratch=args.label_scratch,
                  checkpoint=args.checkpoint,
                  ori_numbers=args.ori_numbers,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  new_w=args.new_w,
                  new_h=args.new_h,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  fold=args.fold,
                  metrics=args.metrics,
                  purpose=args.purpose
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
