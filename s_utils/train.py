import argparse
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
# from unet.utils.dice_score import dice_loss
# from unet.utils.jaccard_score import jaccard_loss
from s_models.cls_models import Cls_model


# def train_net(net,
#               device,
#               image_scratch,
#               image_normal,
#               label_scratch,
#               checkpoint,
#               ori_numbers,
#               epochs: int = 5,
#               batch_size: int = 1,
#               learning_rate: float = 1e-5,
#               val_percent: float = 0.1,
#               save_checkpoint: bool = True,
#               img_scale: float = 0.5,
#               new_w=None,
#               new_h=None,
#               amp: bool = False,
#               fold: int = 5,
#               metrics: str = 'jaccard',
#               purpose: str = 'segmentation',
#               dataset_name=None,
#               last_scratch_segments=None,
#               output_unet_prediction=None,
#               visualize_unet=False,
#               export_path=None
#               ):
#
#     # net_master = copy.deepcopy(net)
#     # 1. Create dataset
#     # if purpose=='segmentation':
#     #     dataset = SegmentationDataset(image_scratch, label_scratch, img_scale, new_w=new_w, new_h=new_h, last_scratch_segments=last_scratch_segments)
#     # elif purpose=='classification':
#     #     dataset = ClassifierDataset(image_scratch, image_normal, img_scale, new_w=new_w, new_h=new_h )
#     # # val_scores_folds = np.zeros(fold)
#     # # kfold = KFold(n_splits=fold, shuffle=True, random_state=42)
#     # n_test = int(ori_numbers/5)
#     # n_val = int(ori_numbers/5)
#     # n_train = len(dataset)-n_val-n_test
#     # test_ids = np.arange(n_test)
#     # val_ids = np.arange(n_test, n_test+n_val)
#     # last = len(dataset)
#     # if purpose=='classification':
#     #     test_ids = np.concatenate((test_ids,np.arange(len(dataset)-n_test,len(dataset))))
#     #     val_ids = np.concatenate((val_ids,np.arange(len(dataset)-n_test-n_val,len(dataset)-n_test)))
#     #     last=len(dataset)-n_test-n_val
#     # train_ids = np.arange(n_test+n_val,last)
#     #
#     # # Sample elements randomly from a given list of ids, no replacement.
#     # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#     # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
#     # val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
#     #
#     # # Define data loaders for training and testing data in this fold
#     # train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
#     # test_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, sampler=test_subsampler)
#     # val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
#
#     train_loader, val_loader, test_loader = create_data_loader(purpose,image_scratch,image_normal,label_scratch,
#                        img_scale, new_w, new_h, last_scratch_segments, ori_numbers, batch_size)
#
#     # (Initialize logging)
#     # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must',allow_val_change=True)
#     # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
#     #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
#     #                               amp=amp, allow_val_change=True))
#
#     logging.info(f'''Starting training:
#         Image scratch:   {image_scratch}
#         Image normal:    {image_normal}
#         Label:           {label_scratch}
#         Checkpoints:     {checkpoint}
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {learning_rate}
#         Training size:   {len(train_loader)*batch_size}
#         Val      size:   {len(val_loader)*batch_size}
#         Test size:       {len(test_loader)*batch_size}
#         Checkpoints:     {save_checkpoint}
#         Device:          {device.type}
#         Images scaling:  {img_scale}
#         Mixed Precision: {amp}
#         Fold:            {fold}
#         Metrics:         {metrics}
#         Purpose:         {purpose}
#     ''')
#
#     # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
#     optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=.8)  # goal: maximize score
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
#     criterion = nn.CrossEntropyLoss()
#     global_step = 0
#
#     # 5. Begin training
#     for epoch in range(1, epochs+1):
#         net.train()
#         epoch_loss = 0
#         with tqdm(total=len(train_loader)*batch_size, desc=f'Epoch {epoch}/{epochs}', unit='img', disable=True) as pbar:
#             for batch in train_loader:
#                 images = batch['image']
#                 true_masks = batch['mask']
#                 if purpose == 'segmentation':
#                     if dataset_name == 'lid': #TODO check dataset
#                         true_masks = torch.where(true_masks > 1, 1, 0)
#                     else:
#                         true_masks = torch.where(true_masks >= 1, 1, 0)
#                 images = images.to(device=device, dtype=torch.float32)
#                 true_masks = true_masks.to(device=device, dtype=torch.long)
#
#                 assert images.size()[-2:] == torch.Size([240, 320])
#
#                 with torch.cuda.amp.autocast(enabled=amp):
#                     masks_pred = net(images)
#                     if metrics=='jaccard':
#                         # assert F.softmax(masks_pred, dim=1).float().shape == F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float().shape
#                         # loss = criterion(masks_pred, true_masks) + jaccard_loss(F.softmax(masks_pred, dim=1).float(),
#                         #                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
#                         #                    multiclass=True)
#                         from torchmetrics import JaccardIndex
#                         jaccard = JaccardIndex(num_classes=2).to(device)
#                         score = jaccard(masks_pred,true_masks)
#                         loss = criterion(masks_pred, true_masks) + 1 - score
#
#                     elif metrics=='dice':
#                         # loss = criterion(masks_pred[0], true_masks) + dice_loss(F.softmax(masks_pred[0], dim=1).float(),
#                         #                       F.one_hot(true_masks, net.n_classes).float(),
#                         #                       multiclass=True)
#                         from torchmetrics import Dice
#                         dice = Dice(average='micro').to(device)
#                         score = dice(torch.argmax(masks_pred,dim=1),true_masks)
#                         loss = criterion(masks_pred,F.one_hot(true_masks, net.n_classes).float())+1-score
#                         # score = dice(masks_pred[0], true_masks)
#                         # loss = criterion(masks_pred[0], true_masks) + 1 - score
#
#
#                 optimizer.zero_grad(set_to_none=True)
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()
#
#                 pbar.update(images.shape[0])
#                 global_step += 1
#                 epoch_loss += loss.item()
#                 # experiment.log({
#                 #     'train loss': loss.item(),
#                 #     'step': global_step,
#                 #     'epoch': epoch
#                 # })
#                 pbar.set_postfix(**{'loss (batch)': loss.item()})
#
#                 # Evaluation round
#                 division_step = (len(train_loader)*batch_size // (5 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                         # histograms = {}
#                         # for tag, value in net.named_parameters():
#                         #     tag = tag.replace('/', '.')
#                         #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                         #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
#
#                         val_score = evaluate(net, val_loader, device, metrics, purpose=purpose,dataset_name=dataset_name)
#                         scheduler.step(val_score)
#
#                         logging.info('Validation {} score: {}'.format(metrics,val_score))
#                         # experiment.log({
#                         #     'learning rate': optimizer.param_groups[0]['lr'],
#                         #     'validation '+metrics: val_score,
#                         #     'step': global_step,
#                         #     'epoch': epoch,
#                         #     # **histograms
#                         # })
#
#         if save_checkpoint:
#             Path(checkpoint).mkdir(parents=True, exist_ok=True)
#             torch.save(net.state_dict(), os.path.join(checkpoint,'checkpoint_fold.pth'))
#             logging.info(f'Checkpoint {epoch} saved!')
#
#     test_score = evaluate(net, test_loader, device, metrics, purpose=purpose,dataset_name=dataset_name, export_detail=False, export_path=export_path)
#     logging.info('Testing {} score: {}'.format(metrics, test_score))
#
#     if visualize_unet:
#         split = 1 if purpose=='segmentation' else 2
#         os.system('cmd /c "python unet/predict.py --model {} --input {} --output {} --new_w 320 --new_h 240'.format(
#             os.path.join(checkpoint, 'checkpoint_fold.pth'),
#             'data/{}_1_output_scratch_sliced/split_{}_fold_{}/JPEGImages'.format(dataset_name,split,fold), output_unet_prediction))
#
#     print('=========== val results fold {}==========='.format(fold))
#     print(val_score)
#     print('=========== test results fold {}=========='.format(fold))
#     print(test_score)


# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
#     parser.add_argument('--image_scratch', nargs='+', default='data/1_input_scratch/JPEGImages', help='Path to scratch image files')
#     parser.add_argument('--image_normal', nargs='+', default='data/1_input_normal/JPEGImages', help='Path to normal image files for classification')
#     parser.add_argument('--label_scratch', nargs='+', default='data/1_input_scratch/SegmentationClass', help='Path to label files')
#     parser.add_argument('--checkpoint', type=str, default='data/14_output_checkpoint_unet', help='Path to checkpoint files')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
#     parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
#     parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
#     parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('--new_w', type=int, default=None, help='New image size')
#     parser.add_argument('--new_h', type=int, default=None, help='New image size')
#     parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
#     parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
#     parser.add_argument('--fold', type=int, default=5, help='Number of fold')
#     parser.add_argument('--metrics', type=str, default='jaccard', help='CNN metrics: jaccard for segmentation OR dice for classification')
#     parser.add_argument('--purpose', type=str, default='segmentation', help='segmentation OR classification')
#     parser.add_argument('--last_scratch_segments', type=int, help='last_scratch_segments')
#     parser.add_argument('--ori_numbers', type=int, help='number of original images')
#     parser.add_argument('--dataset_name', type=str, help='dataset name, ex: magTile, lid')
#     parser.add_argument('--output_trained_unet', type=str, help='output_trained_unet')
#     parser.add_argument('--output_unet_prediction', type=str, help='output_trained_unet')
#     parser.add_argument('--visualize_unet', type=bool, help='visualize_unet')
#     parser.add_argument('--export_path', type=str, help='vgg results image export')
#
#     return parser.parse_args()

# class Train_args(object):
#     def __init__(self,scale=0.5, epochs=5, batch_size=1, fold=5, purpose='segmentation',
#                  image_scratch=['data/1_input_scratch/JPEGImages'],
#                  label_scratch=['data/1_input_scratch/SegmentationClass'],
#                  image_normal=['data/1_input_normal/JPEGImages'],
#                  metrics='jaccard', new_w=320, new_h=240, checkpoint='data/14_output_checkpoint_unet',
#                  dataset_name='lid', ori_numbers=0, learning_rate=1e-5, arch='vgg',
#                  export_path=None,last_scratch_segments=None,output_unet_prediction=None, visualize_unet=None):
#         self.load=False #, help='Load model from a .pth file')
#         self.validation=10.0 #, help='Percent of the data that is used as validation (0-100)')
#         self.amp=False #, help='Use mixed precision')
#         self.bilinear=False #, help='Use bilinear upsampling')
#         self.classes=2 #, help='Number of classes')
#         self.scale = scale
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.fold = fold
#         self.purpose = purpose
#         self.image_scratch = image_scratch
#         self.label_scratch = label_scratch
#         self.image_normal = image_normal
#         self.metrics = metrics
#         self.new_w = new_w
#         self.new_h = new_h
#         self.checkpoint = checkpoint
#         self.dataset_name = dataset_name
#         self.ori_numbers = ori_numbers
#         self.learning_rate = learning_rate
#         self.export_path = export_path
#         self.last_scratch_segments = last_scratch_segments
#         self.output_unet_prediction = output_unet_prediction
#         self.visualize_unet = visualize_unet
#         self.arch=arch

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
