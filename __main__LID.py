import numpy as np
import os
from itertools import product

from s_utils.vgg_model import VGG
from unet.evaluate import evaluate

from s_utils.eval import calculate_m_s, augment_image
from s_utils.frechet_distance import calculate_frechet_distance
from s_utils.scratch_preprocessing import generate_scratch_segments, split_light_dark_dataset
from s_utils.cut_basic_units import cut_basic_units
from s_utils.scratch_postprocessing import combine_scratch, combine_LID
import shutil
import torch
from unet.unet_.unet_model import UNet
from s_utils.data_loading import BasicDataset, ClassifierDataset
from torch.utils.data import DataLoader

from s_utils.utils import slice_dataset

if __name__ == '__main__':
    image_size = (320, 240)
    number_of_generated_images = 1000
    factor_shift = 0.3
    factor_rotate = 1

    input_scratch = 'data/1_input_scratch'
    output_scratch_sliced = 'data/1_output_scratch_sliced'
    input_normal = 'data/1_input_normal'
    output_normal_sliced = 'data/1_output_normal_sliced'
    output_scratch_before_after = 'data/2_output_scratch_before_after'
    output_normal_random_image = 'data/2_output_normal_random_selection'
    output_scratch_segment_image = 'data/2_output_scratch_segment_image'
    output_scratch_segment_npy = 'data/2_output_scratch_segment_npy'
    output_scratch_basic_unit_inference = 'data/3_output_scratch_basic_unit_for_inference'
    input_file_name_datasplit = 'data/4_input_file_name_datasplit'
    input_before_datasplit = 'data/4_input_unsplit_data'
    output_light_dark_datasplit = 'data/6_output_file_datasplit'
    output_light_dark_sliced = 'data/6_output_file_datasplit_sliced'
    output_scratch_basic_unit_training = 'data/7_output_scratch_basic_unit_for_gen_train'
    output_pix2pix_inference = 'data/9_output_pix2pix_inference'
    output_scratch_combined = 'data/10_output_scratch_combined/'
    output_scratch_lid_combined = 'data/11_output_scratch_lid_combined'
    output_fid_scores = 'data/12_output_fid_scores'
    output_std_aug_images = 'data/13_output_std_aug_images'
    output_trained_unet = 'data/14_output_checkpoint_unet'
    output_unet_prediction = 'data/15_output_unet_prediction'
    output_trained_vgg = 'data/16_output_checkpoint_vgg'


    lid_normal_segments = [3, 4] # the lid's segment number of normal images
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
    folds = 5
    data_split = 3

    # steps = [2,3,4,5,6,7,8,14,17]
    steps = [2,3,4, 5, 6, 7, 8, 14, 16, 17]
    # steps = [11, 14, 16, 17]
    # steps = [17]


    # SPLIT LIGHT DARK FOR GENERATOR TRAINING
    if 1.1 in steps:
        print('STEP 1.1: SPLIT LIGHT DARK DATASET')
        split_light_dark_dataset(dataset_names, dataset_information, input_before_datasplit,
                                 input_file_name_datasplit, output_light_dark_datasplit)

    # EXPORT INDEX LIST AND DEFINE SPLIT NUMBER
    if 1.2 in steps:
        print('STEP 1.2: EXPORT INDEX LIST AND DEFINE SPLIT NUMBER')
        slice_dataset(output_light_dark_datasplit, output_light_dark_sliced,
                      input_scratch, output_scratch_sliced,
                      input_normal, output_normal_sliced,
                      dataset_names, folds, data_split)

    for f in range(folds):
    # for f in [0]:
    # NEW SCRATCH GENERATION
        if 2 in steps:
            print("STEP 2: GENERATING " + str(number_of_generated_images) + " NEW SCRATCH SEGMENTS FOLD {}".format(f))
            generate_scratch_segments(number_of_generated_images,
                                      # input_normal_segment, input_scratch_segment,
                                      os.path.join(output_normal_sliced,'split_0_fold_{}'.format(f),'SegmentationClass'),
                                      os.path.join(output_scratch_sliced, 'split_0_fold_{}'.format(f), 'SegmentationClass'),
                                      lid_normal_segments, lid_scratch_segments, scratch_segments,
                                      output_normal_random_image,
                                      image_size, factor_shift, factor_rotate, scratch_new_segments,
                                      output_scratch_before_after, output_scratch_segment_image,
                                      output_scratch_segment_npy, flatten_bg_lid=True, bg=[0,1,2], lid=[3,4])

        # CUT INTO BASIC UNIT FOR IMAGE INFERENCE
        if 3 in steps:
            print('STEP 3: CUT INTO BASIC UNITS FOR IMAGE INFERENCE FOLD {}'.format(f))
            cut_basic_units(output_scratch_segment_npy, output_scratch_basic_unit_inference, 'test_A', image_size,
                            scratch_new_segments, scratch_basic_units, unit_single_color, unit_multi_colors)

        # SPLIT INTO BASIC UNIT FOR GENERATOR TRAINING
        if 4 in steps:
            print('STEP 4: SPLIT INTO BASIC UNITS FOR TRAINING GENERATORS FOLD {}'.format(f))
            try:
                shutil.rmtree(os.path.join(output_scratch_basic_unit_training, 'split_{}_fold_{}_data_{}'.format(str(1), str(f), str(i))) for i in dataset_names)
            except:
                False
            if not os.path.isdir(os.path.join(output_scratch_basic_unit_training, 'split_{}_fold_{}_data_{}'.format(str(1), str(f), str(1)), 'curve')):
                for i in dataset_names:
                    os.mkdir(os.path.join(output_scratch_basic_unit_training, 'split_{}_fold_{}_data_{}'.format(str(1), str(f), str(i))))
                    for b in scratch_basic_units:
                        os.mkdir(os.path.join(output_scratch_basic_unit_training, 'split_{}_fold_{}_data_{}'.format(str(1), str(f), str(i)), b))
            for i, j in product(dataset_names, ['train_A']):#,'test_A']):
                cut_image = True if j == 'train_A' else False
                cut_basic_units(
                    os.path.join(output_light_dark_sliced, 'split_{}_fold_{}_data_{}'.format(str(1), str(f), str(i)), j),
                    os.path.join(output_scratch_basic_unit_training, 'split_{}_fold_{}_data_{}'.format(str(1), str(f), str(i))),
                    j, image_size, scratch_segments, scratch_basic_units, unit_single_color, unit_multi_colors,
                    cut_image=cut_image)

        # TRAIN THE GENERATORS
        if 5 in steps:
            print('STEP 5: TRAIN THE GENERATOR FOLD {}'.format(f))
            os.system('cmd /c "python pix2pixHD/train.py --name straight --dataroot data/7_output_scratch_basic_unit_for_gen_train/split_1_fold_{}_data_4/straight --niter 50 --niter_decay 50 --tf_log"'.format(f))
            os.system('cmd /c "python pix2pixHD/train.py --name curve --dataroot data/7_output_scratch_basic_unit_for_gen_train/split_1_fold_{}_data_4/curve --niter 50 --niter_decay 50 --tf_log"'.format(f))
            os.system('cmd /c "python pix2pixHD/train.py --name end --dataroot data/7_output_scratch_basic_unit_for_gen_train/split_1_fold_{}_data_4/end  --niter 50 --niter_decay 50 --tf_log"'.format(f))

        # SCRATCH INFERENCE
        if 6 in steps:
            print('STEP 6: GENERATOR INFERENCE FOLD {}'.format(f))
            os.system('cmd /c "python pix2pixHD/test.py --name straight --dataroot data/3_output_scratch_basic_unit_for_inference/straight --how_many "'+str(number_of_generated_images))
            os.system('cmd /c "python pix2pixHD/test.py --name curve --dataroot data/3_output_scratch_basic_unit_for_inference/curve --how_many "'+str(number_of_generated_images))
            os.system('cmd /c "python pix2pixHD/test.py --name end --dataroot data/3_output_scratch_basic_unit_for_inference/end --how_many "'+str(number_of_generated_images))

        # COMBINE BASIC UNITS
        if 7 in steps:
            print('STEP 7: COMBINE BASIC UNITS FOLD {}'.format(f))
            combine_scratch(output_pix2pix_inference, output_scratch_segment_npy, image_size, output_scratch_combined,scratch_new_segments)

        # COMBINE WITH LID
        if 8 in steps:
            print('STEP 8: COMBINE WITH LIDS FOLD {}'.format(f))
            combine_LID(output_scratch_combined, output_scratch_segment_npy, output_normal_random_image, image_size, output_scratch_lid_combined, id_lid=1) #TODO id_lid please refer to segment number of lid component

        # CALCULATE M S ORIGINAL IMAGE
        if 9 in steps:
            print('STEP 9: CALCULATE M S ORIGINAL FOLD {}'.format(f))
            calculate_m_s(os.path.join(output_scratch_sliced,'split_0_fold_{}','JPEGimages'.format(f)), os.path.join(output_fid_scores,'original_scratch.npz'))

        # CALCULATE M S SYNTHETIC IMAGE
        if 10 in steps:
            print('STEP 10: CALCULATE M S SYNTHETIC FOLD {}'.format(f))
            calculate_m_s(output_scratch_lid_combined, os.path.join(output_fid_scores, 'sgs_scratch.npz'))

        # STANDARD AUGMENTATION
        if 11 in steps:
            print('STEP 11: BUILDING STANDARD AUGMENTATION IMAGES FOLD {}'.format(f))
            augment_image(os.path.join(output_scratch_sliced,'split_0_fold_{}','JPEGimages').format(f), output_std_aug_images, number_of_generated_images, image_size)

        # CALCULATE M S STD AUG IMAGE
        if 12 in steps:
            print('STEP 120: CALCULATE M S STD AUG FOLD {}'.format(f))
            calculate_m_s(output_std_aug_images, os.path.join(output_fid_scores, 'std_aug_scratch.npz'))

        # CALCULATE FRECHET
        if 13 in steps:
            print('STEP 13: CALCULATE FRECHET FOLD {}'.format(f))
            ori = np.load(os.path.join(output_fid_scores,'original_scratch.npz'))

            syn = np.load(os.path.join(output_fid_scores, 'std_aug_scratch.npz'))
            fid_score = calculate_frechet_distance(ori['m'], ori['s'], syn['m'], syn['s'])
            print('FID SCORES STD AUG = ',fid_score)

            syn = np.load(os.path.join(output_fid_scores, 'sgs_scratch.npz'))
            fid_score = calculate_frechet_distance(ori['m'], ori['s'], syn['m'], syn['s'])
            print('FID SCORES SGS = ', fid_score)

        # RUN UNET
        if 14 in steps:
            print('STEP 14: RUN UNET BATCH 2 FOLD {}'.format(f))
            ori_numbers = int(np.sum(np.genfromtxt(os.path.join(output_light_dark_sliced,'indices','data_length.csv'), delimiter=','))/2)
            os.system('cmd /c "python s_utils/train.py --scale 0.5 --epochs 30 --batch-size 2 --fold {} --purpose segmentation '
                      '--metrics jaccard '
                      # '--image_scratch data/1_output_scratch_sliced/split_1_fold_{}/JPEGImages data/11_output_scratch_lid_combined '
                      # '--label_scratch data/1_output_scratch_sliced/split_1_fold_{}/SegmentationClass data/2_output_scratch_segment_npy '
                      '--image_scratch data/1_output_scratch_sliced/split_1_fold_{}/JPEGImages '
                      '--label_scratch data/1_output_scratch_sliced/split_1_fold_{}/SegmentationClass '
                      '--new_w 320 --new_h 240 --ori_numbers {}"'.format(f,f,f,ori_numbers))

        # RUN EVALUATION
        if 15 in steps:
            print('STEP 15: RUN EVALUATION')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = UNet(n_channels=3, n_classes=2)
            net.to(device=device)
            loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
            dataset = BasicDataset(output_scratch_lid_combined, output_scratch_segment_npy) # SGS data
            # dataset = BasicDataset(input_original_image_scratch, input_scratch_segment)  # original scratch
            eval_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
            for fold in range(1): # range(5)
                net.load_state_dict(torch.load(os.path.join(output_trained_unet,'checkpoint_epoch_30_fold_{}.pth'.format(fold)), map_location=device))
                eval_score = evaluate(net, eval_loader, device, 'jaccard', purpose='segmentation')
                print('EVALUATION JACCARD SCORE: ',eval_score)

        # VISUALIZE UNET
        if 16 in steps:
            print('STEP 16: VISUALIZE UNET')
            os.system('cmd /c "python unet/predict.py --model {} --input {} --output {} --new_w 320 --new_h 240'.format(
                os.path.join(output_trained_unet,'checkpoint_fold.pth'), output_scratch_lid_combined, output_unet_prediction))

        # RUN VGG
        if 17 in steps:
            print('STEP 17: RUN VGG BATCH 16 FOLD {}'.format(f))
            ori_numbers = int(np.sum(
                np.genfromtxt(os.path.join(output_light_dark_sliced, 'indices', 'data_length.csv'), delimiter=',')) / 2)
            os.system(
                'cmd /c "python s_utils/train.py --scale 0.5 --epochs 20 --batch-size 8 --fold {} --purpose classification '
                # '--image_scratch data/1_output_scratch_sliced/split_1_fold_{}/JPEGImages data/11_output_scratch_lid_combined '
                '--image_scratch data/1_output_scratch_sliced/split_1_fold_{}/JPEGImages '
                '--image_normal data/1_output_normal_sliced/split_1_fold_{}/JPEGImages '
                '--metrics dice --new_w 320 --new_h 240 --checkpoint data/16_output_checkpoint_vgg '
                '--ori_numbers {}"'.format(f,f,f,ori_numbers))

        # TEST VGG
        if 18 in steps:
            print('STEP 18: TEST VGG BATCH 16')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = VGG(output_dim=2)
            net.to(device=device)
            loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
            dataset = ClassifierDataset(output_scratch_lid_combined, scale=0.5, new_w=320, new_h=240) # SGS data
            # dataset = BasicDataset(input_original_image_scratch, input_scratch_segment)  # original scratch
            eval_loader = DataLoader(dataset, shuffle=False, batch_size=1)
            for fold in range(1): # range(5)
                net.load_state_dict(torch.load(os.path.join(output_trained_vgg,'checkpoint_epoch_20_fold_{}.pth'.format(fold)), map_location=device))
                eval_score = evaluate(net, eval_loader, device, 'dice', purpose='classification')
                print('EVALUATION DICE SCORE: ',eval_score)
