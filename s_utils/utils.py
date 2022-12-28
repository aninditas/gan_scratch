import copy
import os
import shutil
import random
from itertools import product
from s_utils.data_loading import SegmentationDataset, ClassifierDataset
import torch
import cv2
import numpy as np
import tqdm
from s_utils.cut_basic_units import cut_basic_units


def slice_dataset(light_dark_datasplit_path, output_dataslice, input_scratch, output_scratch_sliced, input_normal,
                  output_normal_sliced, dataset_names, folds, data_split, dataset_name=None):
    data_length_scratch = []
    for t in [output_dataslice, output_normal_sliced]:
        try:
            os.mkdir(os.path.join(t, 'indices'))
        except FileExistsError:
            pass

    for dn in dataset_names:
        try:
            file_list_scratch = os.listdir(os.path.join(light_dark_datasplit_path, 'data_' + str(dn), 'train_B'))
        except FileNotFoundError:
            file_list_scratch = os.listdir(os.path.join(input_scratch, 'JPEGImages'))
        data_length_scratch.append(len(file_list_scratch))
        for f in range(folds):
            random.seed(f)
            temp_scratch = copy.deepcopy(file_list_scratch)
            random.shuffle(temp_scratch)

            try:
                [os.mkdir(os.path.join(output_dataslice, 'split_{}_fold_{}_data_{}'.format(str(ds), str(f), str(dn))))
                 for ds in range(data_split)]
            except FileExistsError:
                pass

            try:
                [os.mkdir(os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)))) for ds in
                 range(data_split)]
            except FileExistsError:
                pass

            for t in ['train_A', 'train_B']:
                try:
                    [os.mkdir(
                        os.path.join(output_dataslice, 'split_{}_fold_{}_data_{}'.format(str(ds), str(f), str(dn)), t))
                        for ds in range(data_split)]
                except FileExistsError:
                    pass

            for t in ['JPEGImages', 'SegmentationClass']:
                try:
                    [os.mkdir(
                        os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f), str(dn)), t)) for
                        ds in range(data_split)]
                except FileExistsError:
                    pass

            for ds in range(data_split):
                file_slice_scratch = temp_scratch[int(ds * (len(temp_scratch) / data_split)):int(
                    ((ds + 1) * (len(temp_scratch) / data_split)))]
                for fs in file_slice_scratch:
                    try:
                        shutil.copy(os.path.join(light_dark_datasplit_path, 'data_' + str(dn), 'train_A', fs),
                                    os.path.join(output_dataslice,
                                                 'split_{}_fold_{}_data_{}'.format(str(ds), str(f), str(dn)),
                                                 'train_A'))

                        shutil.copy(
                            os.path.join(light_dark_datasplit_path, 'data_' + str(dn), 'train_B', fs[:-4] + '.jpg'),
                            os.path.join(output_dataslice, 'split_{}_fold_{}_data_{}'.format(str(ds), str(f), str(dn)),
                                         'train_B'))
                    except (FileExistsError, FileNotFoundError):
                        pass
                    if dataset_name == 'lid':  # TODO check dataset
                        shutil.copy(os.path.join(input_scratch, 'SegmentationClass', fs[:-4] + '.npy'),
                                    os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)),
                                                 'SegmentationClass'))
                        shutil.copy(os.path.join(input_scratch, 'JPEGImages', fs[:-4] + '.jpg'),
                                    os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)),
                                                 'JPEGImages'))
                    elif dataset_name == 'magTile':
                        shutil.copy(os.path.join(input_scratch, 'SegmentationClass', fs[:-4] + '.png'),
                                    os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)),
                                                 'SegmentationClass'))
                        shutil.copy(os.path.join(input_scratch, 'JPEGImages', fs[:-4] + '.jpg'),
                                    os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)),
                                                 'JPEGImages'))
                    elif dataset_name in ['concrete', 'conc2', 'asphalt']:
                        try:
                            shutil.copy(os.path.join(input_scratch, 'SegmentationClass', fs[:-4] + '.jpg'),
                                        os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)),
                                                     'SegmentationClass'))
                            shutil.copy(os.path.join(input_scratch, 'JPEGImages', fs[:-4] + '.jpg'),
                                        os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)),
                                                     'JPEGImages'))
                        except (FileExistsError, FileNotFoundError):
                            continue

            try:
                np.savetxt(os.path.join(output_dataslice, 'indices', 'data_' + str(dn) + '_fold_' + str(f) + '.csv'),
                           temp_scratch, fmt='% s')
            except (FileExistsError, FileNotFoundError):
                pass

    if input_normal is not None:
        file_list_normal = os.listdir(os.path.join(input_normal, 'JPEGImages'))
        for f in range(folds):
            random.seed(f)
            temp_normal = copy.deepcopy(file_list_normal)
            random.shuffle(temp_normal)

            try:
                [os.mkdir(os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)))) for ds in
                 range(data_split - 1)]
            except (FileExistsError, FileNotFoundError):
                pass

            for t in ['JPEGImages', 'SegmentationClass']:
                try:
                    [os.mkdir(os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)), t)) for ds
                     in range(data_split - 1)]
                except (FileExistsError, FileNotFoundError):
                    pass

            data_length_normal = [int(len(file_list_normal) / 2), int(len(file_list_normal) / 2)]
            for idx, dl in enumerate(data_length_normal):
                file_slice_normal = temp_normal[int(idx * data_length_normal[idx - 1]):int(
                    (idx * data_length_normal[idx - 1]) + dl)]
                for fs in file_slice_normal:
                    if dataset_name == 'lid':  # TODO check dataset
                        shutil.copy(os.path.join(input_normal, 'SegmentationClass', fs[:-4] + '.npy'),
                                    os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(idx), str(f)),
                                                 'SegmentationClass'))
                    if dataset_name == 'magTile':
                        shutil.copy(os.path.join(input_normal, 'SegmentationClass', fs[:-4] + '.png'),
                                    os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(idx), str(f)),
                                                 'SegmentationClass'))
                    if dataset_name in ['concrete', 'conc2', 'asphalt']:
                        try:
                            shutil.copy(os.path.join(input_normal, 'SegmentationClass', fs[:-4] + '.jpg'),
                                        os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(idx), str(f)),
                                                     'SegmentationClass'))
                        except (FileExistsError, FileNotFoundError):
                            continue
                    shutil.copy(os.path.join(input_normal, 'JPEGImages', fs),
                                os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(idx), str(f)),
                                             'JPEGImages'))

                np.savetxt(
                    os.path.join(output_normal_sliced, 'indices', 'data_' + str(dn) + '_fold_' + str(f) + '.csv'),
                    temp_scratch, fmt='% s')

    try:
        np.savetxt(os.path.join(output_dataslice, 'indices', 'data_length.csv'),
                   (np.array(data_length_scratch) / data_split).astype(np.uint), fmt='% i')
    except (FileExistsError, FileNotFoundError):
        pass


def slice_dataset_training(output_scratch_basic_unit_training, subdataset_numbers, scratch_basic_units,
                           output_light_dark_sliced,
                           image_size, scratch_segments, unit_single_color, unit_multi_colors, f, dataset_name=None):
    for i in subdataset_numbers:
        try:
            os.mkdir(os.path.join(output_scratch_basic_unit_training,
                                  'split_{}_fold_{}_data_{}'.format(str(0), str(f), str(i))))
        except FileExistsError:
            pass
        for b in scratch_basic_units:
            try:
                os.mkdir(os.path.join(output_scratch_basic_unit_training,
                                      'split_{}_fold_{}_data_{}'.format(str(0), str(f), str(i)), b))
            except FileExistsError:
                pass
    for i, j in product(subdataset_numbers, ['train_A']):  # ,'test_A']):
        cut_image = True if j == 'train_A' else False
        cut_basic_units(
            os.path.join(output_light_dark_sliced, 'split_{}_fold_{}_data_{}'.format(str(0), str(f), str(i)), j),
            os.path.join(output_scratch_basic_unit_training, 'split_{}_fold_{}_data_{}'.format(str(0), str(f), str(i))),
            j, image_size, scratch_segments, scratch_basic_units, unit_single_color, unit_multi_colors,
            cut_image=cut_image, dataset_name=dataset_name)


def augment_image(ori_path, aug_path, num_of_aug, image_size, dataset_name):
    ori_img_list = os.listdir(os.path.join(ori_path, 'JPEGImages'))
    try:
        os.mkdir(aug_path)
    except FileExistsError:
        pass

    for f in ['JPEGImages', 'SegmentationClass']:
        try:
            os.mkdir(os.path.join(aug_path, f))
        except FileExistsError:
            pass

    for idx in tqdm.tqdm(range(num_of_aug)):
        selected = random.choice(ori_img_list)
        image = cv2.imread(os.path.join(ori_path, 'JPEGImages', selected))
        image = cv2.resize(image, image_size)
        if dataset_name == 'lid':  # TODO check dataset
            label = np.load(os.path.join(ori_path, 'SegmentationClass', selected[:-3] + 'npy'))
            label = np.resize(label, tuple(reversed(image_size)))
        elif dataset_name in ['concrete', 'conc2', 'asphalt']:
            label = cv2.imread(os.path.join(ori_path, 'SegmentationClass', selected))
            label = cv2.resize(label, image_size)
        elif dataset_name == 'magTile':
            label = cv2.imread(os.path.join(ori_path, 'SegmentationClass', selected[:-3] + 'png'))
            label = cv2.resize(label, image_size)
        image = cv2.flip(image, -1)
        label = cv2.flip(label, -1)
        cv2.imwrite(os.path.join(aug_path, 'JPEGImages', str(idx) + '.jpg'), image)
        cv2.imwrite(os.path.join(aug_path, 'SegmentationClass', str(idx) + '.jpg'), label)


def create_data_loader(purpose, image_scratch, image_normal, label_scratch,
                       img_scale, new_w, new_h, last_scratch_segments, ori_numbers, batch_size):
    if purpose == 'segmentation':
        dataset = SegmentationDataset(image_scratch, label_scratch, img_scale, new_w=new_w, new_h=new_h,
                                      last_scratch_segments=last_scratch_segments)
    elif purpose == 'classification':
        dataset = ClassifierDataset(image_scratch, image_normal, img_scale, new_w=new_w, new_h=new_h)
    n_test = int(ori_numbers / 5)
    n_val = int(ori_numbers / 5)
    n_train = len(dataset) - n_val - n_test
    test_ids = np.arange(n_test)
    val_ids = np.arange(n_test, n_test + n_val)
    last = len(dataset)
    if purpose == 'classification':
        test_ids = np.concatenate((test_ids, np.arange(len(dataset) - n_test, len(dataset))))
        val_ids = np.concatenate((val_ids, np.arange(len(dataset) - n_test - n_val, len(dataset) - n_test)))
        last = len(dataset) - n_test - n_val
    train_ids = np.arange(n_test + n_val, last)

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)

    return train_loader, val_loader, test_loader


def calculate_diversity(image_scratch, new_w, new_h, batch_size=2):
    from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
    dataset = ClassifierDataset(image_scratch, new_w=new_w, new_h=new_h)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    msssim_scores = []
    for batch in data_loader:
        images = batch['image']
        if len(images) < 2: continue
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        msssim_scores.append(ms_ssim(torch.unsqueeze(images[0], 0), torch.unsqueeze(images[1], 0)))
    return torch.nanmean(torch.stack(msssim_scores))
