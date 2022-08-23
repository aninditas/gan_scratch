import pickle
import random
import gc
import cv2
import imutils
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil

def load_image(info_dict_ns, image_size):
    """Load, resize, and extract lid area of the image
    :param info_dict_ns: info_dict for normal or scratch image
    :return: info_dict normal pr scratch with updated data
    """
    info_dict_ns['image'] = cv2.resize(
        cv2.imread(os.path.join(info_dict_ns['path'][:-18], 'JPEGImages', info_dict_ns['idx_file_random'][:-4]+'.jpg')),
        dsize=image_size,interpolation=cv2.INTER_NEAREST)
    info_dict_ns['npy'] = cv2.resize(
        np.array(np.load(os.path.join(info_dict_ns['path'], info_dict_ns['idx_file_random'])), dtype='uint8'), dsize=image_size,
        interpolation=cv2.INTER_NEAREST)
    info_dict_ns['lid'] = np.array(
        [1 if ii in info_dict_ns['idx_segment'] else 0 for i in info_dict_ns['npy'] for ii in i]).reshape(
        tuple(reversed(image_size))).squeeze()
    return info_dict_ns


def extract_scratch(info_dict, image_size, scratch_segment, scratch_new_segment):
    """Take the scratch area only from the lid_scratch image

    :param info_dict: info_dict
    :returns info_dict: info_dict with updated scratch data
    """

    # match the pixel value of the lid_scratch with the idx_segment (segment numbers that represent scratch)
    info_dict['scratch']['npy'] = np.array(
        [ii + (scratch_new_segment[0] - scratch_segment[0]) if ii in info_dict['scratch']['idx_segment'] else 0 for i in
         info_dict['lid_scratch']['npy'] for ii in
         i]).reshape(tuple(reversed(image_size))).squeeze()

    # 1. Dilate the scratch to connect slightly disconnected scratch, such as diagonally connected pixel
    # 2. Find how many scratches in one image, save the count, label, stats, and centroid
    kernel = np.ones((3, 3), np.uint8)
    (info_dict['scratch']['count'], info_dict['scratch']['label'], info_dict['scratch']['stats'],
     info_dict['scratch']['centroids']) = cv2.connectedComponentsWithStatsWithAlgorithm(
        cv2.dilate(info_dict['scratch']['npy'].astype("uint8"), kernel, iterations=1), 8, cv2.CV_32S,
        cv2.CCL_DEFAULT)

    # save each scratch in different variable for individual augmentation
    for sc in range(info_dict['scratch']['count']):
        if sc == 0: continue
        temp = np.clip(info_dict['scratch']['label']*(1*(info_dict['scratch']['label']==sc)),0,1)
        info_dict['scratch'][str(sc)] = temp*info_dict['scratch']['npy']

    return info_dict


def augment_scratch(info_dict_sc, factor_shift, factor_rotate, image_size, scratch_new_segments):
    """Augment scratch area.
    :param info_dict_sc: info_dict for scratch
    :param image_size: tuple with image size (width, height)
    :returns info_dict_sc: info_dict scratch with updated data
    """

    for sc in range(info_dict_sc['count']):
        if sc == 0: continue
        shift_horizontal = round(random.uniform(-image_size[0] * factor_shift, image_size[0] * factor_shift))
        shift_vertical = round(random.uniform(-image_size[1] * factor_shift, image_size[1] * factor_shift))
        rotate = round(random.uniform(0, 180 * factor_rotate))
        flip = round(random.uniform(-1,1))

        # rotate and scale scratch based on rotate and scale
        (h, w) = info_dict_sc[str(sc)].astype(np.uint8).shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), rotate, 1.0)
        info_dict_sc[str(sc) + '_aug'] = cv2.warpAffine(info_dict_sc[str(sc)].astype(np.uint8), M, (w, h),
                                                        flags=cv2.INTER_NEAREST)

        # shift scratch based on shift horizontal and vertical
        info_dict_sc[str(sc)+'_aug'] = imutils.translate(info_dict_sc[str(sc)+'_aug'].astype(np.uint8), shift_horizontal,
                                                      shift_vertical)

        # flip scratch based on flip variable (negative for vertical & horizontal, 0 for vertical, positive for horizontal)
        info_dict_sc[str(sc) + '_aug'] = cv2.flip(info_dict_sc[str(sc)+'_aug'].astype(np.uint8), flip)


        for ii in scratch_new_segments:
            temp = np.clip(info_dict_sc[str(sc) + '_aug'] * (1 * (info_dict_sc[str(sc) + '_aug'] == ii)), 0, 1)
            kernel = np.ones((5, 5), np.uint8)
            temp = cv2.morphologyEx(temp.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            temp = temp * ii
            info_dict_sc[str(sc) + '_aug'] = np.maximum(temp, info_dict_sc[str(sc) + '_aug'])

    return info_dict_sc


def add_scratch_to_lid(info_dict, scratch_new_segments):
    """Add scratch image to the lid image.

    Steps:
    1. Move the lid_scratch to the same position as lid_normal
    2. Add scratch segments that have been augmented

    :param info_dict: info_dict
    :returns info_dict: info_dict for with updated scratch data
    """

    # erode the edge of the lid to not letting the scratches get too close to the edge
    kernel = np.ones((10, 10), np.uint8)
    eroded_lid = cv2.erode(info_dict['lid_normal']['lid'].astype('uint8'), kernel, iterations=1)

    info_dict['scratch']['image_aug_stack_n'] = info_dict['lid_normal']['npy']
    for sc in range(info_dict['scratch']['count']):
        if sc == 0: continue

        # remove scratch area that is close to the lid's edge
        temp = eroded_lid * info_dict['scratch'][str(sc)+'_aug']
        if np.count_nonzero(np.any([temp == i for i in scratch_new_segments])) == 0:
            continue

        # add augmented scratch to lid_scratch based on lid_normal
        info_dict['scratch']['image_aug_stack_n'] = np.max([info_dict['scratch']['image_aug_stack_n'], temp], axis=0)

    return info_dict


def generate_scratch_segments(number_of_generated_images, input_normal_segment, input_scratch_segment,
                              lid_normal_segment, lid_scratch_segment, scratch_segments, output_normal_random_image,
                              image_size, factor_shift, factor_rotate, scratch_new_segments,
                              output_scratch_before_after, output_scratch_segment_image, output_scratch_segment_npy,
                              flatten_bg_lid=False, bg=[], lid=[]):
    info_dict_list = []
    count_img = 0
    while count_img<number_of_generated_images:
        info_dict = {}
        info_dict['lid_normal'] = {}
        info_dict['lid_scratch'] = {}
        info_dict['scratch'] = {}

        info_dict['lid_normal']['path'] = input_normal_segment
        info_dict['lid_scratch']['path'] = input_scratch_segment

        info_dict['lid_normal']['idx_segment'] = lid_normal_segment
        info_dict['lid_scratch']['idx_segment'] = lid_scratch_segment
        info_dict['scratch']['idx_segment'] = scratch_segments

        # randomly select and load images from the dataset
        for i in ['lid_normal', 'lid_scratch']:
            info_dict[i]['idx_file_random'] = np.random.choice(os.listdir(info_dict[i]['path']), replace=False)
            info_dict[i] = load_image(info_dict_ns=info_dict[i], image_size=image_size)

        cv2.imwrite(os.path.join(output_normal_random_image, str(count_img) + '.jpg'), info_dict['lid_normal']['image'])

        # extract the scratch area
        info_dict = extract_scratch(info_dict, image_size, scratch_segments, scratch_new_segments)

        # augment the scratch
        info_dict['scratch'] = augment_scratch(info_dict['scratch'], factor_shift, factor_rotate, image_size, scratch_new_segments)

        # put augmented scratch on the lid
        info_dict = add_scratch_to_lid(info_dict, scratch_new_segments)

        # if the new scratch image does not have any scratch, repeat the image generation
        if np.count_nonzero(np.any([info_dict['scratch']['image_aug_stack_n'] == i for i in scratch_new_segments])) == 0:
            continue

        if flatten_bg_lid:
            for bg_ in bg:
                temp = np.where(info_dict['scratch']['image_aug_stack_n']==bg_)
                info_dict['scratch']['image_aug_stack_n'][temp]=0
            for lid_ in lid:
                temp = np.where(info_dict['scratch']['image_aug_stack_n'] == lid_)
                info_dict['scratch']['image_aug_stack_n'][temp] = 1

        norm = mpl.colors.Normalize(vmin=0, vmax=scratch_new_segments[-1]+1)
        cmap = cm.nipy_spectral # change the cmap here if you want to change the color. Options available at https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig, ax = plt.subplots(2, 2, figsize=(30, 30))
        ax[0, 0].imshow(m.to_rgba(info_dict['lid_normal']['npy']))
        ax[0, 1].imshow(m.to_rgba(info_dict['lid_scratch']['npy']))
        ax[1, 0].imshow(m.to_rgba(info_dict['scratch']['npy']))
        ax[1, 1].imshow(m.to_rgba(info_dict['scratch']['image_aug_stack_n']))
        plt.savefig(os.path.join(output_scratch_before_after, str(count_img) + '.jpg'))
        plt.close('all')
        plt.close(fig)
        gc.collect()

        cv2.imwrite(os.path.join(output_scratch_segment_image, str(count_img) + '.jpg'),
                    m.to_rgba(info_dict['scratch']['image_aug_stack_n']) * 255)
        np.save(os.path.join(output_scratch_segment_npy, str(count_img) + '.npy'),
                info_dict['scratch']['image_aug_stack_n'])

        info_dict_list.append(info_dict)
        count_img+=1


def split_light_dark_dataset(dataset_names, dataset_information, input_before_datasplit,
                             input_file_name_datasplit, output_light_dark_datasplit):
    if not os.path.isdir(os.path.join(output_light_dark_datasplit, 'data_1', 'train_A')):
        for i in dataset_names:
            os.mkdir(os.path.join(output_light_dark_datasplit, 'data_' + str(i)))
            os.mkdir(os.path.join(output_light_dark_datasplit, 'data_' + str(i), 'train_A'))
            os.mkdir(os.path.join(output_light_dark_datasplit, 'data_' + str(i), 'train_B'))
            os.mkdir(os.path.join(output_light_dark_datasplit, 'data_' + str(i), 'test_A'))
    for a, b in zip(dataset_names, dataset_information):
        file_names = pd.DataFrame(pd.read_csv(os.path.join(input_file_name_datasplit, b+'.csv'))).to_numpy()[:,1]
        for fn in file_names:
            try:
                source_A = os.path.join(input_before_datasplit, 'train_A', fn[:-4] + '_All.png')
                source_B = os.path.join(input_before_datasplit, 'train_B', fn[:-4] + '_All.jpg')
                dest_A = os.path.join(output_light_dark_datasplit, 'data_'+str(a), 'train_A', fn[:-4] + '_All.png')
                dest_B = os.path.join(output_light_dark_datasplit, 'data_'+str(a), 'train_B', fn[:-4] + '_All.jpg')
                shutil.copyfile(source_A, dest_A)
                shutil.copyfile(source_B, dest_B)
            except:
                source_A = os.path.join(input_before_datasplit, 'test_A', fn[:-4] + '_All.png')
                dest_A = os.path.join(output_light_dark_datasplit, 'data_' + str(a), 'test_A', fn[:-4] + '_All.png')
                shutil.copyfile(source_A, dest_A)

