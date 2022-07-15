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
from scratch_additional_processing.scratch_preprocessing import generate_scratch_segments
from scratch_additional_processing.cut_basic_units import cut_basic_units
from scratch_additional_processing.scratch_postprocessing import combine_scratch, combine_LID

if __name__ == '__main__':
    image_size = (320, 240)
    number_of_generated_images = 10
    factor_shift = 0.3
    factor_rotate = 1

    input_normal_segment = 'data/1_input_scratch/SegmentationClass/'
    input_scratch_segment = 'data/1_input_scratch/SegmentationClass/'
    output_scratch_before_after = 'data/2_output_scratch_before_after'
    output_normal_random_image = 'data/2_output_normal_random_selection'
    output_scratch_segment_image = 'data/2_output_scratch_segment_image'
    output_scratch_segment_npy = 'data/2_output_scratch_segment_npy'
    output_scratch_basic_unit = 'data/3_output_scratch_basic_unit'
    output_pix2pix_inference = 'data/5_output_pix2pix_inference'
    output_scratch_combined = 'data/6_output_scratch_combined/'
    output_scratch_lid_combined = 'data/7_output_scratch_lid_combined'

    lid_normal_segments = [1,2,3, 4] # the lid's segment number of normal images
    lid_scratch_segments = [1, 2, 3, 4] # the lid and scratch's segment number of scratch images
    scratch_segments = [2, 3, 4] # the scratch's segment number of scratch images
    scratch_new_segments = [5, 6, 7]  # the new scratch's segment number of scratch images (for scratch on scratch images)

    # if not os.path.isdir(os.path.join(EXPORT_PATH, "normal_img")):
    #     os.mkdir(os.path.join(EXPORT_PATH, "normal_img"))
    #     os.mkdir(os.path.join(EXPORT_PATH, "segmentation_file"))
    #     os.mkdir(os.path.join(EXPORT_PATH, "segmentation_img"))
    #     os.mkdir(os.path.join(EXPORT_PATH, "basic_unit"))

    generate_scratch_segments(number_of_generated_images, input_normal_segment, input_scratch_segment,
                              lid_normal_segments, lid_scratch_segments, scratch_segments, output_normal_random_image,
                              image_size, factor_shift, factor_rotate, scratch_new_segments,
                              output_scratch_before_after, output_scratch_segment_image, output_scratch_segment_npy)
    cut_basic_units(output_scratch_segment_npy, output_scratch_basic_unit, image_size, scratch_new_segments)

    print('STEP 3: TRAIN THE GENERATOR')
    os.system('cmd /c "python pix2pixHD/train.py --name straight --dataroot data/4_input_straight --niter 1 --niter_decay 1"')
    os.system('cmd /c "python pix2pixHD/train.py --name curve --dataroot data/4_input_curve --niter 1 --niter_decay 1"')
    os.system('cmd /c "python pix2pixHD/train.py --name end --dataroot data/4_input_end  --niter 1 --niter_decay 1"')

    print('STEP 4: GENERATOR INFERENCE')
    os.system('cmd /c "python pix2pixHD/test.py --name straight --dataroot data/3_output_scratch_basic_unit/straight"')
    os.system('cmd /c "python pix2pixHD/test.py --name curve --dataroot data/3_output_scratch_basic_unit/curve"')
    os.system('cmd /c "python pix2pixHD/test.py --name end --dataroot data/3_output_scratch_basic_unit/end"')

    combine_scratch(output_pix2pix_inference, output_scratch_segment_npy, image_size, output_scratch_combined,scratch_new_segments)
    combine_LID(output_scratch_combined, output_scratch_segment_npy, output_normal_random_image, image_size, output_scratch_lid_combined)