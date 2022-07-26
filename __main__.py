import os
import pickle
# import random
# import gc
# import cv2
# import imutils
import numpy as np
# import matplotlib as mpl
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# from pix2pixHD import train
import os
from itertools import product

from scratch_additional_processing.eval import calculate_activation_statistics
from scratch_additional_processing.scratch_preprocessing import generate_scratch_segments, split_light_dark_dataset
from scratch_additional_processing.cut_basic_units import cut_basic_units
from scratch_additional_processing.scratch_postprocessing import combine_scratch, combine_LID

if __name__ == '__main__':
    image_size = (320, 240)
    number_of_generated_images =840
    factor_shift = 0.3
    factor_rotate = 1

    input_normal_segment = 'data/1_input_scratch/SegmentationClass/'
    input_scratch_segment = 'data/1_input_scratch/SegmentationClass/'
    output_scratch_before_after = 'data/2_output_scratch_before_after'
    output_normal_random_image = 'data/2_output_normal_random_selection'
    output_scratch_segment_image = 'data/2_output_scratch_segment_image'
    output_scratch_segment_npy = 'data/2_output_scratch_segment_npy'
    output_scratch_basic_unit_inference = 'data/3_output_scratch_basic_unit_for_inference'
    input_file_name_datasplit = 'data/4_input_file_name_datasplit'
    input_before_datasplit = 'data/4_input_unsplit_data'
    output_light_dark_datasplit = 'data/6_output_file_datasplit'
    output_scratch_basic_unit_training = 'data/7_output_scratch_basic_unit_for_gen_train'
    output_pix2pix_inference = 'data/9_output_pix2pix_inference'
    output_scratch_combined = 'data/10_output_scratch_combined/'
    output_scratch_lid_combined = 'data/11_output_scratch_lid_combined'

    lid_normal_segments = [1,2,3, 4] # the lid's segment number of normal images
    lid_scratch_segments = [1, 2, 3, 4] # the lid and scratch's segment number of scratch images
    scratch_segments = [2, 3, 4] # the scratch's segment number of scratch images
    scratch_new_segments = [5, 6, 7]  # the new scratch's segment number of scratch images (for scratch on scratch images)
    scratch_basic_units = ['straight', 'curve', 'end', 'all']
    unit_single_color = [5, 4, 120]
    unit_multi_colors = [[128, 0, 0], # straight
                         [0, 128, 128], # curve
                         [0, 128, 0], # end
                         [0, 0, 128]] # all
    dataset_names = [1, 2, 3, 4]
    dataset_training_split = [40, 50, 0, 300]
    dataset_information = ['dark_LID_white_scratch', 'dark_LID_black_scratch', 'light_LID_white_scratch', 'light_LID_black_scratch']

    # steps = [1,2,3,4,5,6,7,8]
    steps = [1,2,6,7,8]

    # NEW SCRATCH GENERATION
    if 1 in steps:
        print("STEP 1: GENERATING " + str(number_of_generated_images) + " NEW SCRATCH SEGMENTS")
        generate_scratch_segments(number_of_generated_images, input_normal_segment, input_scratch_segment,
                                  lid_normal_segments, lid_scratch_segments, scratch_segments, output_normal_random_image,
                                  image_size, factor_shift, factor_rotate, scratch_new_segments,
                                  output_scratch_before_after, output_scratch_segment_image, output_scratch_segment_npy)

    # CUT INTO BASIC UNIT FOR IMAGE INFERENCE
    if 2 in steps:
        print('STEP 2: CUT INTO BASIC UNITS FOR IMAGE INFERENCE')
        cut_basic_units(output_scratch_segment_npy, output_scratch_basic_unit_inference, 'test_A', image_size,
                        scratch_new_segments, scratch_basic_units, unit_single_color, unit_multi_colors)

    # SPLIT LIGHT DARK FOR GENERATOR TRAINING
    if 3 in steps:
        print('STEP 3: SPLIT LIGHT DARK DATASET')
        split_light_dark_dataset(dataset_names, dataset_information, input_before_datasplit,
                                 input_file_name_datasplit, output_light_dark_datasplit)

    # SPLIT INTO BASIC UNIT FOR GENERATOR TRAINING
    if 4 in steps:
        print('STEP 4: SPLIT INTO BASIC UNITS FOR TRAINING GENERATORS')
        import shutil
        try:
            shutil.rmtree(os.path.join(output_scratch_basic_unit_training, 'data_'+str(i)) for i in dataset_names)
        except:
            False
        if not os.path.isdir(os.path.join(output_scratch_basic_unit_training, 'data_1', 'curve')):
            for i in dataset_names:
                os.mkdir(os.path.join(output_scratch_basic_unit_training, 'data_' + str(i)))
                for b in scratch_basic_units:
                    os.mkdir(os.path.join(output_scratch_basic_unit_training, 'data_' + str(i), b))
        for i, j in product(dataset_names, ['train_A','test_A']):
            cut_image = True if j == 'train_A' else False
            cut_basic_units(os.path.join(output_light_dark_datasplit,'data_'+str(i), j),
                            os.path.join(output_scratch_basic_unit_training, 'data_' + str(i)),
                            j, image_size, scratch_segments, scratch_basic_units, unit_single_color, unit_multi_colors,
                            cut_image=cut_image)

    # TRAIN THE GENERATORS
    if 5 in steps:
        print('STEP 5: TRAIN THE GENERATOR')
        os.system('cmd /c "python pix2pixHD/train.py --name straight --dataroot data/7_output_scratch_basic_unit_for_gen_train/data_4/straight --niter 500 --niter_decay 100 --tf_log"')
        os.system('cmd /c "python pix2pixHD/train.py --name curve --dataroot data/7_output_scratch_basic_unit_for_gen_train/data_4/curve --niter 500 --niter_decay 100 --tf_log"')
        os.system('cmd /c "python pix2pixHD/train.py --name end --dataroot data/7_output_scratch_basic_unit_for_gen_train/data_4/end  --niter 500 --niter_decay 100 --tf_log"')

    # SCRATCH INFERENCE
    if 6 in steps:
        print('STEP 6: GENERATOR INFERENCE')
        os.system('cmd /c "python pix2pixHD/test.py --name straight --dataroot data/3_output_scratch_basic_unit_for_inference/straight --how_many "'+str(number_of_generated_images))
        os.system('cmd /c "python pix2pixHD/test.py --name curve --dataroot data/3_output_scratch_basic_unit_for_inference/curve --how_many "'+str(number_of_generated_images))
        os.system('cmd /c "python pix2pixHD/test.py --name end --dataroot data/3_output_scratch_basic_unit_for_inference/end --how_many "'+str(number_of_generated_images))

    # COMBINE BASIC UNITS
    if 7 in steps:
        print('STEP 7: COMBINE BASIC UNITS')
        combine_scratch(output_pix2pix_inference, output_scratch_segment_npy, image_size, output_scratch_combined,scratch_new_segments)

    # COMBINE WITH LID
    if 8 in steps:
        print('STEP 8: COMBINE WITH LIDS')
        combine_LID(output_scratch_combined, output_scratch_segment_npy, output_normal_random_image, image_size, output_scratch_lid_combined)

    # CALCULATE FRECHET DISTANCE
    if 9 in steps:
        d = 'D:\Dropbox\PhD\python\giraffe\data\comprehensive_cars\images_256'
        dd = []
        for f in os.listdir(d): dd.append(os.path.join(d, f))

        m, s = calculate_activation_statistics(dd)
        c = 'D:\Dropbox\PhD\python\giraffe\data\comprehensive_cars\\fid_files_256\cars.npz'
        np.savez(c, m=m, s=s)