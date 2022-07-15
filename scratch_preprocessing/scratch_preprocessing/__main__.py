import gc
import os
import pickle
import random

import cv2
import imutils
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def load_image(info_dict_ns):
    """Load, resize, and extract lid area of the image
    :param info_dict_ns: info_dict for normal or scratch image
    :return: info_dict normal pr scratch with updated data
    """
    info_dict_ns['image'] = cv2.resize(
        np.array(np.load(info_dict_ns['path'] + info_dict_ns['idx_file_random']), dtype='uint8'), dsize=IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR)
    info_dict_ns['lid'] = np.array(
        [1 if ii in info_dict_ns['idx_segment'] else 0 for i in info_dict_ns['image'] for ii in i]).reshape(
        tuple(reversed(IMAGE_SIZE))).squeeze()
    return info_dict_ns


def extract_scratch(info_dict):
    """Take the scratch area only from the lid_scratch image

    :param info_dict: info_dict
    :returns info_dict: info_dict with updated scratch data
    """

    # match the pixel value of the lid_scratch with the idx_segment (segment numbers that represent scratch)
    info_dict['scratch']['image'] = np.array(
        [ii+(SCRATCH_NEW_SEGMENT[0]-SCRATCH_SEGMENT[0]) if ii in info_dict['scratch']['idx_segment'] else 0 for i in info_dict['lid_scratch']['image'] for ii in
         i]).reshape(tuple(reversed(IMAGE_SIZE))).squeeze()

    # 1. Dilate the scratch to connect slightly disconnected scratch, such as diagonally connected pixel
    # 2. Find how many scratches in one image, save the count, label, stats, and centroid
    kernel = np.ones((3, 3), np.uint8)
    (info_dict['scratch']['count'], info_dict['scratch']['label'], info_dict['scratch']['stats'],
     info_dict['scratch']['centroids']) = cv2.connectedComponentsWithStatsWithAlgorithm(
        cv2.dilate(info_dict['scratch']['image'].astype("uint8"), kernel, iterations=1), 8, cv2.CV_32S,
        cv2.CCL_DEFAULT)

    # save each scratch in different variable for individual augmentation
    for sc in range(info_dict['scratch']['count']):
        if sc == 0: continue
        temp = np.clip(info_dict['scratch']['label'] * (1 * (info_dict['scratch']['label'] == sc)), 0, 1)
        info_dict['scratch'][str(sc)] = temp * info_dict['scratch']['image']

    return info_dict


def augment_scratch(info_dict_sc, factor_shift, factor_rotate):
    """Augment scratch area.
    :param info_dict_sc: info_dict for scratch
    :param image_size: tuple with image size (width, height)
    :returns info_dict_sc: info_dict scratch with updated data
    """

    for sc in range(info_dict_sc['count']):
        if sc == 0: continue
        shift_horizontal = round(random.uniform(-IMAGE_SIZE[0] * factor_shift, IMAGE_SIZE[0] * factor_shift))
        shift_vertical = round(random.uniform(-IMAGE_SIZE[1] * factor_shift, IMAGE_SIZE[1] * factor_shift))
        rotate = round(random.uniform(0, 180 * factor_rotate))
        flip = round(random.uniform(-1, 1))

        # rotate and scale scratch based on rotate and scale
        (h, w) = info_dict_sc[str(sc)].astype(np.uint8).shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), rotate, 1.0)
        info_dict_sc[str(sc) + '_aug'] = cv2.warpAffine(info_dict_sc[str(sc)].astype(np.uint8), M, (w, h),
                                                        flags=cv2.INTER_NEAREST)

        # shift scratch based on shift horizontal and vertical
        info_dict_sc[str(sc) + '_aug'] = imutils.translate(info_dict_sc[str(sc) + '_aug'].astype(np.uint8),
                                                           shift_horizontal,
                                                           shift_vertical)

        # flip scratch based on flip variable (negative for vertical & horizontal, 0 for vertical, positive for horizontal)
        info_dict_sc[str(sc) + '_aug'] = cv2.flip(info_dict_sc[str(sc) + '_aug'].astype(np.uint8), flip)

        for ii in SCRATCH_NEW_SEGMENT:
            temp = np.clip(info_dict_sc[str(sc) + '_aug'] * (1 * (info_dict_sc[str(sc) + '_aug'] == ii)), 0, 1)
            kernel = np.ones((5, 5), np.uint8)
            temp = cv2.morphologyEx(temp.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            temp = temp * ii
            info_dict_sc[str(sc) + '_aug'] = np.maximum(temp, info_dict_sc[str(sc) + '_aug'])

    return info_dict_sc


def add_scratch_to_lid(info_dict):
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

    info_dict['scratch']['image_aug_stack_n'] = info_dict['lid_normal']['image']
    for sc in range(info_dict['scratch']['count']):
        if sc == 0: continue

        # remove scratch area that is close to the lid's edge
        temp = eroded_lid * info_dict['scratch'][str(sc) + '_aug']
        if np.count_nonzero(np.any([temp == i for i in SCRATCH_NEW_SEGMENT])) == 0:
        # if np.count_nonzero((temp == 2) | (temp == 3)) == 0:
        #     temp = temp * 0
            continue

        # add augmented scratch to lid_scratch based on lid_normal
        info_dict['scratch']['image_aug_stack_n'] = np.max([info_dict['scratch']['image_aug_stack_n'],temp], axis=0)

    return info_dict


def main():
    info_dict_list = []
    count_img = 0
    while count_img < NUMBER_OF_GENERATED_IMAGES:
        info_dict = {}
        info_dict['lid_normal'] = {}
        info_dict['lid_scratch'] = {}
        info_dict['scratch'] = {}

        info_dict['lid_normal']['path'] = INPUT_NORMAL_SEGMENT
        info_dict['lid_scratch']['path'] = INPUT_SCRATCH_SEGMENT

        info_dict['lid_normal']['idx_segment'] = LID_NORMAL_SEGMENT
        info_dict['lid_scratch']['idx_segment'] = LID_SCRATCH_SEGMENT
        info_dict['scratch']['idx_segment'] = SCRATCH_SEGMENT

        # randomly select and load images from the dataset
        for i in ['lid_normal', 'lid_scratch']:
            info_dict[i]['idx_file_random'] = np.random.choice(os.listdir(info_dict[i]['path']), replace=False)
            info_dict[i] = load_image(info_dict_ns=info_dict[i])

        # extract the scratch area
        info_dict = extract_scratch(info_dict)

        # augment the scratch
        info_dict['scratch'] = augment_scratch(info_dict['scratch'], FACTOR_SHIFT, FACTOR_ROTATE)

        # put augmented scratch on the lid
        info_dict = add_scratch_to_lid(info_dict)

        # if the new scratch image does not have any scratch, repeat the image generation
        if np.count_nonzero(np.any([info_dict['scratch']['image_aug_stack_n'] == i for i in SCRATCH_NEW_SEGMENT])) == 0:
            continue

        # export the scratch images
        norm = mpl.colors.Normalize(vmin=0, vmax=SCRATCH_NEW_SEGMENT[-1]+1)
        cmap = cm.nipy_spectral  # change the cmap here if you want to change the color. Options available at https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig, ax = plt.subplots(2, 2, figsize=(30, 30))
        ax[0,0].imshow(m.to_rgba(info_dict['lid_normal']['image']))
        ax[0,1].imshow(m.to_rgba(info_dict['lid_scratch']['image']))
        ax[1,0].imshow(m.to_rgba(info_dict['scratch']['image']))
        ax[1,1].imshow(m.to_rgba(info_dict['scratch']['image_aug_stack_n']))
        plt.savefig(EXPORT_PATH + str(count_img) + '.jpg')
        plt.close('all')
        plt.close(fig)
        gc.collect()

        info_dict_list.append(info_dict)
        count_img += 1

    with open(EXPORT_PATH + '/info_dict.pickle', 'wb') as f:
        pickle.dump(info_dict_list, f)


if __name__ == '__main__':
    global IMAGE_SIZE, NUMBER_OF_GENERATED_IMAGES, FACTOR_SHIFT, FACTOR_ROTATE, EXPORT_PATH, INPUT_NORMAL_SEGMENT, INPUT_SCRATCH_SEGMENT, LID_NORMAL_SEGMENT, LID_SCRATCH_SEGMENT, SCRATCH_SEGMENT, SCRATCH_NEW_SEGMENT
    IMAGE_SIZE = (320, 240)
    NUMBER_OF_GENERATED_IMAGES = 10
    FACTOR_SHIFT = 0.3
    FACTOR_ROTATE = 1
    EXPORT_PATH = 'datasets/scratch_segment_aug/'
    INPUT_NORMAL_SEGMENT = 'datasets/scratch/new_scratch_out/SegmentationClass/'
    INPUT_SCRATCH_SEGMENT = 'datasets/scratch/new_scratch_out/SegmentationClass/'
    LID_NORMAL_SEGMENT = [1,2,3,4] #[3, 4]  # the lid's segment number of normal images
    LID_SCRATCH_SEGMENT = [1, 2, 3, 4]  # the lid and scratch's segment number of scratch images
    SCRATCH_SEGMENT = [2, 3, 4]  # the scratch's segment number of scratch images
    SCRATCH_NEW_SEGMENT = [5,6,7] # the new scratch's segment number of scratch images (for scratch on scratch images)

    main()
