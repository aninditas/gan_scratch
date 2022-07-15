import os
import pickle
import random
import gc
import cv2
import imutils
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pix2pixHD import train
import os
from scratch_preprocessing.scratch_preprocessing import generate_scratch_segments, cut_basic_unit, combine_scratch, combine_LID

# def load_image(info_dict_ns):
#     """Load, resize, and extract lid area of the image
#     :param info_dict_ns: info_dict for normal or scratch image
#     :return: info_dict normal pr scratch with updated data
#     """
#     info_dict_ns['image'] = cv2.resize(
#         cv2.imread(info_dict_ns['path'][:-18]+ 'JPEGImages/'+ info_dict_ns['idx_file_random'][:-4]+'.jpg'),
#         dsize=IMAGE_SIZE,interpolation=cv2.INTER_NEAREST)
#     info_dict_ns['npy'] = cv2.resize(
#         np.array(np.load(info_dict_ns['path'] + info_dict_ns['idx_file_random']), dtype='uint8'), dsize=IMAGE_SIZE,
#         interpolation=cv2.INTER_NEAREST)
#     info_dict_ns['lid'] = np.array(
#         [1 if ii in info_dict_ns['idx_segment'] else 0 for i in info_dict_ns['npy'] for ii in i]).reshape(
#         tuple(reversed(IMAGE_SIZE))).squeeze()
#     return info_dict_ns
#
#
# def extract_scratch(info_dict):
#     """Take the scratch area only from the lid_scratch image
#
#     :param info_dict: info_dict
#     :returns info_dict: info_dict with updated scratch data
#     """
#
#     # match the pixel value of the lid_scratch with the idx_segment (segment numbers that represent scratch)
#     info_dict['scratch']['npy'] = np.array(
#         [ii if ii in info_dict['scratch']['idx_segment'] else 0 for i in info_dict['lid_scratch']['npy'] for ii in
#          i]).reshape(tuple(reversed(IMAGE_SIZE))).squeeze()
#
#     # 1. Dilate the scratch to connect slightly disconnected scratch, such as diagonally connected pixel
#     # 2. Find how many scratches in one image, save the count, label, stats, and centroid
#     kernel = np.ones((3, 3), np.uint8)
#     (info_dict['scratch']['count'], info_dict['scratch']['label'], info_dict['scratch']['stats'],
#      info_dict['scratch']['centroids']) = cv2.connectedComponentsWithStatsWithAlgorithm(
#         cv2.dilate(info_dict['scratch']['npy'].astype("uint8"), kernel, iterations=1), 8, cv2.CV_32S,
#         cv2.CCL_DEFAULT)
#
#     # save each scratch in different variable for individual augmentation
#     for sc in range(info_dict['scratch']['count']):
#         if sc == 0: continue
#         temp = np.clip(info_dict['scratch']['label']*(1*(info_dict['scratch']['label']==sc)),0,1)
#         info_dict['scratch'][str(sc)] = temp*info_dict['scratch']['npy']
#
#     return info_dict
#
#
# def augment_scratch(info_dict_sc, factor_shift, factor_rotate):
#     """Augment scratch area.
#     :param info_dict_sc: info_dict for scratch
#     :param image_size: tuple with image size (width, height)
#     :returns info_dict_sc: info_dict scratch with updated data
#     """
#
#     for sc in range(info_dict_sc['count']):
#         if sc == 0: continue
#         shift_horizontal = round(random.uniform(-IMAGE_SIZE[0] * factor_shift, IMAGE_SIZE[0] * factor_shift))
#         shift_vertical = round(random.uniform(-IMAGE_SIZE[1] * factor_shift, IMAGE_SIZE[1] * factor_shift))
#         rotate = round(random.uniform(0, 180 * factor_rotate))
#         flip = round(random.uniform(-1,1))
#
#         # rotate and scale scratch based on rotate and scale
#         info_dict_sc[str(sc)+'_aug'] = imutils.rotate(info_dict_sc[str(sc)].astype(np.uint8), angle=rotate)
#
#         # shift scratch based on shift horizontal and vertical
#         info_dict_sc[str(sc)+'_aug'] = imutils.translate(info_dict_sc[str(sc)+'_aug'].astype(np.uint8), shift_horizontal,
#                                                       shift_vertical)
#
#         # flip scratch based on flip variable (negative for vertical & horizontal, 0 for vertical, positive for horizontal)
#         info_dict_sc[str(sc) + '_aug'] = cv2.flip(info_dict_sc[str(sc)+'_aug'].astype(np.uint8), flip)
#
#     return info_dict_sc
#
#
# def add_scratch_to_lid(info_dict):
#     """Add scratch image to the lid image.
#
#     Steps:
#     1. Move the lid_scratch to the same position as lid_normal
#     2. Add scratch segments that have been augmented
#
#     :param info_dict: info_dict
#     :returns info_dict: info_dict for with updated scratch data
#     """
#
#     # erode the edge of the lid to not letting the scratches get too close to the edge
#     kernel = np.ones((10, 10), np.uint8)
#     eroded_lid = cv2.erode(info_dict['lid_normal']['lid'].astype('uint8'), kernel, iterations=1)
#
#     info_dict['scratch']['image_aug_stack_n'] = info_dict['lid_normal']['lid']
#     for sc in range(info_dict['scratch']['count']):
#         if sc == 0: continue
#
#         # remove scratch area that is close to the lid's edge
#         temp = eroded_lid * info_dict['scratch'][str(sc)+'_aug']
#         if np.count_nonzero((temp==2)|(temp==3)) == 0 :
#             temp = temp*0
#             continue
#
#         # add augmented scratch to lid_scratch based on lid_normal
#         info_dict['scratch']['image_aug_stack_n'] = np.max([info_dict['scratch']['image_aug_stack_n'], temp], axis=0)
#
#     return info_dict
#
#
# def generate_scratch_segments():
#     print("STEP 1: GENERATING "+str(NUMBER_OF_GENERATED_IMAGES)+" NEW SCRATCH SEGMENTS")
#     info_dict_list = []
#     count_img = 0
#     while count_img<NUMBER_OF_GENERATED_IMAGES:
#         info_dict = {}
#         info_dict['lid_normal'] = {}
#         info_dict['lid_scratch'] = {}
#         info_dict['scratch'] = {}
#
#         info_dict['lid_normal']['path'] = INPUT_NORMAL_SEGMENT
#         info_dict['lid_scratch']['path'] = INPUT_SCRATCH_SEGMENT
#
#         info_dict['lid_normal']['idx_segment'] = LID_NORMAL_SEGMENT
#         info_dict['lid_scratch']['idx_segment'] = LID_SCRATCH_SEGMENT
#         info_dict['scratch']['idx_segment'] = SCRATCH_SEGMENT
#
#         # randomly select and load images from the dataset
#         for i in ['lid_normal', 'lid_scratch']:
#             info_dict[i]['idx_file_random'] = np.random.choice(os.listdir(info_dict[i]['path']), replace=False)
#             info_dict[i] = load_image(info_dict_ns=info_dict[i])
#
#         cv2.imwrite(os.path.join(OUTPUT_NORMAL_RANDOM_IMAGE, str(count_img) + '.jpg'), info_dict['lid_normal']['image'])
#
#         # extract the scratch area
#         info_dict = extract_scratch(info_dict)
#
#         # augment the scratch
#         info_dict['scratch'] = augment_scratch(info_dict['scratch'], FACTOR_SHIFT, FACTOR_ROTATE)
#
#         # put augmented scratch on the lid
#         info_dict = add_scratch_to_lid(info_dict)
#
#         if np.count_nonzero((info_dict['scratch']['image_aug_stack_n'] == 2) |
#                             (info_dict['scratch']['image_aug_stack_n'] == 3) |
#                             (info_dict['scratch']['image_aug_stack_n'] == 4)) == 0:
#             continue
#
#         norm = mpl.colors.Normalize(vmin=0, vmax=4)
#         cmap = cm.nipy_spectral # change the cmap here if you want to change the color. Options available at https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
#         m = cm.ScalarMappable(norm=norm, cmap=cmap)
#         fig, ax = plt.subplots(2, 2, figsize=(30, 30))
#         ax[0, 0].imshow(m.to_rgba(info_dict['lid_normal']['npy']))
#         ax[0, 1].imshow(m.to_rgba(info_dict['lid_scratch']['npy']))
#         ax[1, 0].imshow(m.to_rgba(info_dict['scratch']['npy']))
#         ax[1, 1].imshow(m.to_rgba(info_dict['scratch']['image_aug_stack_n']))
#         plt.savefig(os.path.join(OUTPUT_SCRATCH_BEFORE_AFTER, str(count_img) + '.jpg'))
#         plt.close('all')
#         plt.close(fig)
#         gc.collect()
#
#         cv2.imwrite(os.path.join(OUTPUT_SCRATCH_SEGMENT_IMAGE, str(count_img) + '.jpg'),
#                     m.to_rgba(info_dict['scratch']['image_aug_stack_n'])*255)
#         np.save(os.path.join(OUTPUT_SCRATCH_SEGMENT_NPY, str(count_img) + '.npy'),
#                 info_dict['scratch']['image_aug_stack_n'])
#
#         info_dict_list.append(info_dict)
#         count_img+=1
#
#     # with open(EXPORT_PATH + '/info_dict.pickle', 'wb') as f:
#     #     pickle.dump(info_dict_list, f)
#
#
# def cut_basic_unit():
#     print('STEP 2: CUT INTO BASIC UNITS')
#     name_list = os.listdir(OUTPUT_SCRATCH_SEGMENT_NPY)
#     if not os.path.isdir(OUTPUT_SCRATCH_BASIC_UNIT + '\\' + "straight\\test_A"):
#         os.mkdir(OUTPUT_SCRATCH_BASIC_UNIT + '\\straight\\test_A')
#         os.mkdir(OUTPUT_SCRATCH_BASIC_UNIT + '\\curve\\test_A')
#         os.mkdir(OUTPUT_SCRATCH_BASIC_UNIT + '\\end\\test_A')
#         os.mkdir(OUTPUT_SCRATCH_BASIC_UNIT + '\\all\\test_A')
#     for name in name_list:
#         npy_file = np.load(OUTPUT_SCRATCH_SEGMENT_NPY + "\\" + name)
#         straight = np.zeros(shape=tuple(reversed(IMAGE_SIZE))+(3,))
#         curve = np.zeros(shape=tuple(reversed(IMAGE_SIZE))+(3,))
#         end = np.zeros(shape=tuple(reversed(IMAGE_SIZE))+(3,))
#         LID = np.zeros(shape=tuple(reversed(IMAGE_SIZE))+(3,))
#         for i in range(IMAGE_SIZE[1]):
#             for j in range(IMAGE_SIZE[0]):
#                 if npy_file[i, j] == 2:
#                     straight[i, j] = [5, 4, 120]
#                     LID[i, j] = [128, 0, 0]
#                 elif npy_file[i, j] == 3:
#                     curve[i, j] = [5, 4, 120]
#                     LID[i, j] = [0, 128, 128]
#                 elif npy_file[i, j] == 4:
#                     end[i, j] = [5, 4, 120]
#                     LID[i, j] = [0, 128, 0]
#                 elif npy_file[i, j] == 1:
#                     LID[i, j] = [0, 0, 128]
#         cv2.imwrite(OUTPUT_SCRATCH_BASIC_UNIT + "\\straight\\test_A\\" + name[:-4] + ".png", straight)
#         cv2.imwrite(OUTPUT_SCRATCH_BASIC_UNIT + "\\curve\\test_A\\" + name[:-4] + ".png", curve)
#         cv2.imwrite(OUTPUT_SCRATCH_BASIC_UNIT + "\\end\\test_A\\" + name[:-4] + ".png", end)
#         cv2.imwrite(OUTPUT_SCRATCH_BASIC_UNIT + "\\all\\test_A\\" + name[:-4] + ".png", LID)
#
#
# def combine_scratch():
#     print('STEP 5: COMBINE BASIC UNITS')
#     gen_list = os.listdir(os.path.join(OUTPUT_PIX2PIX_INFERENCE, "curve/test_latest/images/"))
#
#     for num in range(len(gen_list)):
#         npy = np.load(os.path.join(OUTPUT_SCRATCH_SEGMENT_NPY, gen_list[num][:-22] + ".npy"))
#         curve_img = cv2.imread(os.path.join(OUTPUT_PIX2PIX_INFERENCE, "curve/test_latest/images/", gen_list[num]), cv2.IMREAD_COLOR)
#         straight_img = cv2.imread(os.path.join(OUTPUT_PIX2PIX_INFERENCE, "straight/test_latest/images/", gen_list[num]), cv2.IMREAD_COLOR)
#         end_img = cv2.imread(os.path.join(OUTPUT_PIX2PIX_INFERENCE, "end/test_latest/images/", gen_list[num]), cv2.IMREAD_COLOR)
#
#         result = np.zeros(shape=tuple(reversed(IMAGE_SIZE))+(3,))
#
#         for i in range(IMAGE_SIZE[1]):
#             for j in range(IMAGE_SIZE[0]):
#                 if npy[i][j] == 2:
#                     result[i][j] = straight_img[i][j]
#                 elif npy[i][j] == 3:
#                     result[i][j] = curve_img[i][j]
#                 elif npy[i][j] == 4:
#                     result[i][j] = end_img[i][j]
#         # result = result / 255
#         cv2.imwrite(os.path.join(OUTPUT_SCRATCH_COMBINED, str(gen_list[num][:-22]) + ".jpg"), result)
#
#
# def combine_LID():
#     print('STEP 6: COMBINE WITH LIDS')
#     name_list = os.listdir(OUTPUT_SCRATCH_COMBINED)
#
#     for name in name_list:
#         npy = np.load(os.path.join(OUTPUT_SCRATCH_SEGMENT_NPY, name[:-4] + ".npy"))
#         lid = cv2.imread(os.path.join(OUTPUT_NORMAL_RANDOM_IMAGE, name), cv2.IMREAD_COLOR)
#         gen = cv2.imread(os.path.join(OUTPUT_SCRATCH_COMBINED, name), cv2.IMREAD_COLOR)
#         # bnd = np.load(os.path.join(BND_PATH, name[:-4] + ".npy"))
#
#         FIN = lid.copy()
#
#         for i in range(IMAGE_SIZE[1]):
#             for j in range(IMAGE_SIZE[0]):
#                 if npy[i][j] > 1 and gen[i, j, 0] != 0:
#                     FIN[i][j] = gen[i][j]
#                 # if bnd[i, j] == 1 and gen[i, j, 0] < 65:
#                 #     FIN[i, j] = lid[i, j]
#         cv2.imwrite(os.path.join(OUTPUT_SCRATCH_LID_COMBINED, name), FIN)


if __name__ == '__main__':
    # global IMAGE_SIZE, NUMBER_OF_GENERATED_IMAGES, FACTOR_SHIFT, FACTOR_ROTATE, \
    #     EXPORT_PATH, INPUT_NORMAL_SEGMENT, INPUT_SCRATCH_SEGMENT, \
    #     OUTPUT_SCRATCH_BEFORE_AFTER, OUTPUT_NORMAL_RANDOM_IMAGE, OUTPUT_SCRATCH_SEGMENT_IMAGE, OUTPUT_SCRATCH_SEGMENT_NPY, \
    #     OUTPUT_SCRATCH_BASIC_UNIT, OUTPUT_PIX2PIX_INFERENCE, OUTPUT_SCRATCH_COMBINED, OUTPUT_SCRATCH_LID_COMBINED,\
    #     LID_NORMAL_SEGMENT, LID_SCRATCH_SEGMENT, SCRATCH_SEGMENT

    image_size = (320, 240)
    number_of_generated_images = 10
    factor_shift = 0.3
    factor_rotate = 1

    # EXPORT_PATH = 'scratch_preprocessing/datasets/synthetic/'
    input_normal_segment = 'data/1_input_normal/SegmentationClass/'
    input_scratch_segment = 'data/1_input_scratch/SegmentationClass/'
    output_scratch_before_after = 'data/2_output_scratch_before_after'
    output_normal_random_image = 'data/2_output_normal_random_selection'
    output_scratch_segment_image = 'data/2_output_scratch_segment_image'
    output_scratch_segment_npy = 'data/2_output_scratch_segment_npy'
    output_scratch_basic_unit = 'data/3_output_scratch_basic_unit'
    output_pix2pix_inference = 'data/5_pix2pix_inference'
    output_scratch_combined = 'data/6_output_scratch_combined/'
    output_scratch_lid_combined = 'data/7_output_scratch_lid_combined'

    lid_normal_segments = [3, 4] # the lid's segment number of normal images
    lid_scratch_segments = [1, 2, 3, 4] # the lid and scratch's segment number of scratch images
    scratch_segments = [2, 3, 4] # the scratch's segment number of scratch images

    # if not os.path.isdir(os.path.join(EXPORT_PATH, "normal_img")):
    #     os.mkdir(os.path.join(EXPORT_PATH, "normal_img"))
    #     os.mkdir(os.path.join(EXPORT_PATH, "segmentation_file"))
    #     os.mkdir(os.path.join(EXPORT_PATH, "segmentation_img"))
    #     os.mkdir(os.path.join(EXPORT_PATH, "basic_unit"))

    generate_scratch_segments(number_of_generated_images, input_normal_segment, input_scratch_segment,
                              lid_normal_segments, lid_scratch_segments, scratch_segments, output_normal_random_image,
                              image_size, factor_shift, factor_rotate,
                              output_scratch_before_after, output_scratch_segment_image, output_scratch_segment_npy)
    cut_basic_unit(output_scratch_segment_npy, output_scratch_basic_unit, image_size)

    print('STEP 3: TRAIN THE GENERATOR')
    os.system('cmd /c "python pix2pixHD/train.py --name straight --dataroot data/4_input_straight --niter 2 --niter_decay 1"')
    os.system('cmd /c "python pix2pixHD/train.py --name curve --dataroot data/4_input_curve --niter 2 --niter_decay 1"')
    os.system('cmd /c "python pix2pixHD/train.py --name end --dataroot data/4_input_end  --niter 2 --niter_decay 1"')

    print('STEP 4: GENERATOR INFERENCE')
    os.system('cmd /c "python pix2pixHD/test.py --name straight --dataroot data/3_output_scratch_basic_unit/straight"')
    os.system('cmd /c "python pix2pixHD/test.py --name curve --dataroot data/3_output_scratch_basic_unit/curve"')
    os.system('cmd /c "python pix2pixHD/test.py --name end --dataroot data/3_output_scratch_basic_unit/end"')

    combine_scratch(output_pix2pix_inference, output_scratch_segment_npy, image_size, output_scratch_combined)
    combine_LID(output_scratch_combined, output_scratch_segment_npy, output_normal_random_image, image_size, output_scratch_lid_combined)