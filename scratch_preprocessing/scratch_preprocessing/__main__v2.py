import os
import pickle
import random

import cv2
import imutils
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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
        [ii if ii in info_dict['scratch']['idx_segment'] else 0 for i in info_dict['lid_scratch']['image'] for ii in
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
        temp = np.clip(info_dict['scratch']['label']*(1*(info_dict['scratch']['label']==sc)),0,1)
        info_dict['scratch'][str(sc)] = temp*info_dict['scratch']['image']

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
        flip = round(random.uniform(-1,1))

        # rotate and scale scratch based on rotate and scale
        info_dict_sc[str(sc)+'_aug'] = imutils.rotate(info_dict_sc[str(sc)].astype(np.uint8), angle=rotate)

        # shift scratch based on shift horizontal and vertical
        info_dict_sc[str(sc)+'_aug'] = imutils.translate(info_dict_sc[str(sc)+'_aug'].astype(np.uint8), shift_horizontal,
                                                      shift_vertical)

        # flip scratch based on flip variable (negative for vertical & horizontal, 0 for vertical, positive for horizontal)
        info_dict_sc[str(sc) + '_aug'] = cv2.flip(info_dict_sc[str(sc)+'_aug'].astype(np.uint8), flip)

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

    info_dict['scratch']['image_aug_stack_n'] = info_dict['lid_normal']['lid']
    for sc in range(info_dict['scratch']['count']):
        if sc == 0: continue

        # remove scratch area that is close to the lid's edge
        temp = eroded_lid * info_dict['scratch'][str(sc)+'_aug']
        if np.count_nonzero((temp==2)|(temp==3)) == 0 :
            temp = temp*0
            continue

        # add augmented scratch to lid_scratch based on lid_normal
        info_dict['scratch']['image_aug_stack_n'] = np.max([info_dict['scratch']['image_aug_stack_n'], temp], axis=0)

    return info_dict


def main():
    info_dict_list = []
    count_img = 0
    while count_img<NUMBER_OF_GENERATED_IMAGES:
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

        normal_use = cv2.imread(r"scratch_preprocessing/datasets/normal/JPEG/" + info_dict['lid_normal']['idx_file_random'][:-4] + ".jpg"
                                , cv2.IMREAD_COLOR)
        cv2.imwrite(EXPORT_PATH + '/normal_img/' + str(count_img) + '.jpg', normal_use)

        # extract the scratch area
        info_dict = extract_scratch(info_dict)

        # augment the scratch
        info_dict['scratch'] = augment_scratch(info_dict['scratch'], FACTOR_SHIFT, FACTOR_ROTATE)

        # put augmented scratch on the lid
        info_dict = add_scratch_to_lid(info_dict)

        if np.count_nonzero((info_dict['scratch']['image_aug_stack_n'] == 2) |
                            (info_dict['scratch']['image_aug_stack_n'] == 3) |
                            (info_dict['scratch']['image_aug_stack_n'] == 4)) == 0:
            continue

        norm = mpl.colors.Normalize(vmin=0, vmax=4)
        cmap = cm.nipy_spectral # change the cmap here if you want to change the color. Options available at https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig, ax = plt.subplots(1,2, figsize=(30,30))
        ax[0].imshow(m.to_rgba(info_dict['lid_scratch']['image']))
        ax[1].imshow(m.to_rgba(info_dict['scratch']['image_aug_stack_n']))

        cv2.imwrite(EXPORT_PATH + '/segmentation_img/' + str(count_img) + '.png',
                    info_dict['scratch']['image_aug_stack_n'] * 100)
        np.save(EXPORT_PATH + '/segmentation_file/' + str(count_img) + '.npy', info_dict['scratch']['image_aug_stack_n'])

        plt.savefig(EXPORT_PATH+str(count_img)+'.jpg')
        plt.close('all')
        plt.close(fig)
        info_dict_list.append(info_dict)
        count_img+=1

    with open(EXPORT_PATH + '/info_dict.pickle', 'wb') as f:
        pickle.dump(info_dict_list, f)


def cut_basic_unit():
    npy_path = os.path.join(EXPORT_PATH, "segmentation_file")
    save_path = os.path.join(EXPORT_PATH, "basic_unit")
    name_list = os.listdir(npy_path)
    if not os.path.isdir(save_path + '\\' + "straight"):
        os.mkdir(save_path + '\\straight')
        os.mkdir(save_path + '\\curve')
        os.mkdir(save_path + '\\end')
        os.mkdir(save_path + '\\all')
    for name in name_list:
        npy_file = np.load(npy_path + "\\" + name)
        straight = np.zeros(shape=(240, 320, 3))
        curve = np.zeros(shape=(240, 320, 3))
        end = np.zeros(shape=(240, 320, 3))
        LID = np.zeros(shape=(240, 320, 3))
        for i in range(240):
            for j in range(320):
                if npy_file[i, j] == 2:
                    straight[i, j] = [5, 4, 120]
                    LID[i, j] = [128, 0, 0]
                elif npy_file[i, j] == 3:
                    curve[i, j] = [5, 4, 120]
                    LID[i, j] = [0, 128, 128]
                elif npy_file[i, j] == 4:
                    end[i, j] = [5, 4, 120]
                    LID[i, j] = [0, 128, 0]
                elif npy_file[i, j] == 1:
                    LID[i, j] = [0, 0, 128]
        cv2.imwrite(save_path + "\\straight\\" + name[:-4]+".png", straight)
        cv2.imwrite(save_path + "\\curve\\" + name[:-4]+".png", curve)
        cv2.imwrite(save_path + "\\end\\" + name[:-4]+".png", end)
        cv2.imwrite(save_path + "\\all\\" + name[:-4]+".png", LID)


def make_boundary():
    path = os.path.join(EXPORT_PATH, "segmentation_file")
    name_list = os.listdir(path)
    for name in name_list:
        npy = np.load(os.path.join(path, name))
        scr = npy.copy()
        for i in range(240):
            for j in range(320):
                if npy[i, j] < 2:
                    scr[i, j] = 0
                else:
                    scr[i, j] = 1
        bnd = np.zeros(shape=(240, 320))
        for i in range(240):
            for j in range(320):
                if scr[i, j] == 1:
                    if scr[i - 1, j] == 0:
                        bnd[i, j] = 1
                    elif scr[i + 1, j] == 0:
                        bnd[i, j] = 1
                    elif scr[i, j + 1] == 0:
                        bnd[i, j] = 1
                    elif scr[i, j - 1] == 0:
                        bnd[i, j] = 1
        np.save(os.path.join('scratch_preprocessing/datasets/bnd_mask', name), bnd)


def combine_scratch():
    gen_path = "scratch_preprocessing/datasets/pix2pixHD"
    npy_path = os.path.join(EXPORT_PATH, "segmentation_file")

    name_list = os.listdir(npy_path)
    gen_list = os.listdir(os.path.join(gen_path, "curve"))

    for num in range(len(gen_list)):
        npy = np.load(os.path.join(npy_path, gen_list[num][:-22] + ".npy"))
        curve_img = cv2.imread(os.path.join(gen_path, "curve", gen_list[num]), cv2.IMREAD_COLOR)
        straight_img = cv2.imread(os.path.join(gen_path, "straight", gen_list[num]), cv2.IMREAD_COLOR)
        end_img = cv2.imread(os.path.join(gen_path, "end", gen_list[num]), cv2.IMREAD_COLOR)

        result = np.zeros(shape=(240, 320, 3))

        for i in range(240):
            for j in range(320):
                if npy[i][j] == 2:
                    result[i][j] = straight_img[i][j]
                elif npy[i][j] == 3:
                    result[i][j] = curve_img[i][j]
                elif npy[i][j] == 4:
                    result[i][j] = end_img[i][j]
        # result = result / 255
        cv2.imwrite("scratch_preprocessing/datasets/combine/" + str(gen_list[num][:-22]) + ".jpg", result)


def combine_LID():
    GEN_SCRATCH_PATH = "scratch_preprocessing/datasets/combine"
    LID_PATH = os.path.join(EXPORT_PATH, "normal_img")
    NPY_PATH = os.path.join(EXPORT_PATH, "segmentation_file")
    BND_PATH = "scratch_preprocessing/datasets/bnd_mask"

    name_list = os.listdir(GEN_SCRATCH_PATH)

    for name in name_list:
        npy = np.load(os.path.join(NPY_PATH, name[:-4] + ".npy"))
        lid = cv2.imread(os.path.join(LID_PATH, name), cv2.IMREAD_COLOR)
        gen = cv2.imread(os.path.join(GEN_SCRATCH_PATH, name), cv2.IMREAD_COLOR)
        bnd = np.load(os.path.join(BND_PATH, name[:-4] + ".npy"))

        FIN = lid.copy()

        for i in range(240):
            for j in range(320):
                if npy[i][j] > 1 and gen[i, j, 0] != 0:
                    FIN[i][j] = gen[i][j]
                if bnd[i, j] == 1 and gen[i, j, 0] < 65:
                    FIN[i, j] = lid[i, j]
        cv2.imwrite("scratch_preprocessing/datasets/final/" + name, FIN)


if __name__ == '__main__':
    global IMAGE_SIZE, NUMBER_OF_GENERATED_IMAGES, FACTOR_SHIFT, FACTOR_ROTATE, EXPORT_PATH, INPUT_NORMAL_SEGMENT, INPUT_SCRATCH_SEGMENT, LID_NORMAL_SEGMENT, LID_SCRATCH_SEGMENT, SCRATCH_SEGMENT
    IMAGE_SIZE = (320, 240)
    NUMBER_OF_GENERATED_IMAGES = 10
    FACTOR_SHIFT = 0.3
    FACTOR_ROTATE = 1
    EXPORT_PATH = 'scratch_preprocessing/datasets/synthetic/'
    INPUT_NORMAL_SEGMENT = 'scratch_preprocessing/datasets/normal/SegmentationClass/'
    INPUT_SCRATCH_SEGMENT = 'scratch_preprocessing/datasets/scratch/new_scratch_out/SegmentationClass/' #'dataset/new_scratch/'
    LID_NORMAL_SEGMENT = [3, 4] # the lid's segment number of normal images
    LID_SCRATCH_SEGMENT = [1, 2, 3, 4] # the lid and scratch's segment number of scratch images
    SCRATCH_SEGMENT = [2, 3, 4] # the scratch's segment number of scratch images

    if not os.path.isdir(os.path.join(EXPORT_PATH, "normal_img")):
        os.mkdir(os.path.join(EXPORT_PATH, "normal_img"))
        os.mkdir(os.path.join(EXPORT_PATH, "segmentation_file"))
        os.mkdir(os.path.join(EXPORT_PATH, "segmentation_img"))
        os.mkdir(os.path.join(EXPORT_PATH, "basic_unit"))

    main()
    cut_basic_unit()
    # make_boundary() # unused
    # feed basic unit to Pix2PixHD then do the following function

    combine_scratch()
    combine_LID()