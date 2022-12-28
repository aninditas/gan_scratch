import os
import shutil

import numpy as np
import torch
from s_models.cls_models import Cls_model
from s_models.unet_model import UNet
from s_utils.cut_basic_units import cut_basic_units
from s_utils.data_loading import SegmentationDataset, ClassifierDataset
from s_utils.eval import calculate_m_s
from s_utils.frechet_distance import calculate_frechet_distance
from s_utils.scratch_generator import split_light_dark_dataset, generate_scratch_segments, combine_scratch, combine_LID
from s_utils.train import train_seg_cls
from s_utils.utils import slice_dataset, slice_dataset_training, augment_image, calculate_diversity
from torch.utils.data import DataLoader
from unet.evaluate import evaluate


class Experiment(object):
    def __init__(self, image_size, number_of_generated_images, factor_shift, factor_rotate, dataset_name, base):
        self.image_size = image_size
        self.number_of_generated_images = number_of_generated_images
        self.factor_shift = factor_shift
        self.factor_rotate = factor_rotate
        self.dataset_name = dataset_name
        self.base = base
        self.input_scratch_raw = 'data/{}_0_input_scratch'.format(dataset_name)
        self.input_normal_raw = 'data/{}_0_input_normal'.format(dataset_name)
        self.input_scratch = 'data/{}_1_input_scratch'.format(dataset_name)
        self.input_normal = 'data/{}_1_input_normal'.format(dataset_name)
        self.output_scratch_sliced = 'data/{}_1_output_scratch_sliced'.format(dataset_name)
        self.output_normal_sliced = 'data/{}_1_output_normal_sliced'.format(dataset_name)
        self.output_scratch_before_after = 'data/{}_2_output_scratch_before_after'.format(dataset_name)
        self.output_normal_random_image = 'data/{}_2_output_normal_random_selection'.format(dataset_name)
        self.output_scratch_segment_image = 'data/{}_2_output_scratch_segment_image'.format(dataset_name)
        self.output_scratch_segment_npy = 'data/{}_2_output_scratch_segment_npy'.format(dataset_name)
        self.output_scratch_basic_unit_inference = 'data/3_output_scratch_basic_unit_for_inference'
        self.input_file_name_datasplit = 'data/{}_4_input_file_name_datasplit'.format(dataset_name)
        self.input_before_datasplit = 'data/{}_4_input_unsplit_data'.format(dataset_name)
        self.output_light_dark_datasplit = 'data/{}_6_output_file_datasplit'.format(dataset_name)
        self.output_light_dark_sliced = 'data/{}_6_output_file_datasplit_sliced'.format(dataset_name)
        self.output_scratch_basic_unit_training = 'data/{}_7_output_scratch_basic_unit_for_gen_train'.format(
            dataset_name)
        self.output_pix2pix_inference = 'data/9_output_pix2pix_inference'
        self.output_scratch_combined = 'data/{}_10_output_scratch_combined/'.format(dataset_name)
        self.output_scratch_lid_combined = 'data/{}_11_output_scratch_lid_combined'.format(dataset_name)
        self.output_fid_scores = 'data/{}_12_output_fid_scores'.format(dataset_name)
        self.output_std_aug_defect_images = 'data/{}_13_output_std_aug_defect_images'.format(dataset_name)
        self.output_std_aug_normal_images = 'data/{}_13_output_std_aug_normal_images'.format(dataset_name)
        self.output_trained_unet = 'data/{}_14_output_checkpoint_unet'.format(dataset_name)
        self.output_unet_prediction = 'data/{}_15_output_unet_prediction'.format(dataset_name)
        self.output_trained_vgg = 'data/{}_16_output_checkpoint_vgg'.format(dataset_name)
        self.output_img_vgg = 'data/{}_17_output_vgg_prediction'.format(dataset_name)

        self.unit_single_color = [5, 4, 120]

        if self.dataset_name == 'lid':  # TODO check dataset
            self.lid_normal_segments = [3, 4]  # the lid's segment number of normal images
            self.lid_scratch_segments = [1, 2, 3, 4]  # the lid and scratch's segment number of scratch images
            self.scratch_segments = [2, 3, 4]  # the scratch's segment number of scratch images
            self.scratch_new_segments = [5, 6,
                                         7]  # the new scratch's segment number of scratch images (for scratch on
            # scratch images)
            self.unit_multi_colors = [[128, 0, 0],  # straight
                                      [0, 128, 128],  # curve
                                      [0, 128, 0],  # end
                                      [0, 0, 128]]  # all
            self.dataset_information = ['dark_LID_white_scratch', 'dark_LID_black_scratch', 'light_LID_white_scratch',
                                        'light_LID_black_scratch']
            self.scratch_basic_units = ['lid_straight', 'lid_curve', 'lid_end', 'lid_all']
            # self.subdataset_numbers = [1,2,3,4]
            self.subdataset_numbers = [1]
            self.pix2pixHD_data_num = 1  # 4

        else:
            self.lid_normal_segments = [0]  # the lid's segment number of normal images
            self.lid_scratch_segments = [0]  # the lid and scratch's segment number of scratch images
            self.scratch_segments = [1]  # the scratch's segment number of scratch images
            self.scratch_new_segments = [
                2]  # the new scratch's segment number of scratch images (for scratch on scratch images)
            self.unit_multi_colors = [[128, 0, 0],
                                      [0, 0, 128]]
            self.subdataset_numbers = [1]
            self.dataset_information = [dataset_name]
            self.pix2pixHD_data_num = 1
            if dataset_name == 'magTile':
                self.scratch_basic_units = ['magTile_crack', 'magTile_all']
            elif dataset_name == 'concrete':
                self.scratch_basic_units = ['conc_crack', 'conc_all']
            elif dataset_name == 'conc2':
                self.scratch_basic_units = ['conc2_crack', 'conc2_all']
            elif dataset_name == 'asphalt':
                self.scratch_basic_units = ['asphalt_gaps', 'asphalt_all']

        self.dataset_training_split = [40, 50, 0, 300]
        self.folds = 5
        self.data_split = 3

    def rename_files(self):
        print('STEP 1.0.1: RENAME FILES')
        for i in ['JPEGImages', 'SegmentationClass']:
            files = os.listdir(os.path.join(self.output_light_dark_datasplit, 'data_1', i))
            for f in files:
                src = os.path.join(self.output_light_dark_datasplit, 'data_1', i, f)
                des = os.path.join(self.output_light_dark_datasplit, 'data_1', i, self.dataset_name + '_' + f)
                os.rename(src, des)

    def split_subdataset(self):
        print('STEP 1.1: SPLIT LIGHT DARK DATASET')
        split_light_dark_dataset(self.subdataset_numbers, self.dataset_information, self.input_before_datasplit,
                                 self.input_file_name_datasplit, self.output_light_dark_datasplit)

    def export_index_split(self):
        print('STEP 1.2: EXPORT INDEX LIST AND DEFINE SPLIT NUMBER')
        if self.dataset_name in ['conc2', 'asphalt']:  # TODO check dataset
            self.input_normal = None
            self.output_normal_sliced = None

        slice_dataset(light_dark_datasplit_path=self.output_light_dark_datasplit,
                      output_dataslice=self.output_light_dark_sliced,
                      input_scratch=self.input_scratch, output_scratch_sliced=self.output_scratch_sliced,
                      input_normal=self.input_normal, output_normal_sliced=self.output_normal_sliced,
                      dataset_names=self.subdataset_numbers, folds=self.folds, data_split=self.data_split,
                      dataset_name=self.dataset_name)

    # 2 NEW SCRATCH GENERATION
    def generate_scratch(self, f, export_before_after=True):
        print("STEP 2: GENERATING " + str(self.number_of_generated_images) + " NEW SCRATCH SEGMENTS FOLD {}".format(f))
        if self.base == 'scratch':
            input_sliced = self.output_scratch_sliced
            lid_segment = self.lid_scratch_segments
            flatten_bg_lid = False
        else:
            input_sliced = self.output_normal_sliced
            lid_segment = self.lid_normal_segments
            flatten_bg_lid = True if self.dataset_name == 'lid' else False
        generate_scratch_segments(number_of_generated_images=self.number_of_generated_images,
                                  input_normal_segment=os.path.join(input_sliced, 'split_1_fold_{}'.format(f),
                                                                    'SegmentationClass'),
                                  input_scratch_segment=os.path.join(self.output_scratch_sliced,
                                                                     'split_1_fold_{}'.format(f), 'SegmentationClass'),
                                  lid_normal_segment=lid_segment, lid_scratch_segment=self.lid_scratch_segments,
                                  scratch_segments=self.scratch_segments,
                                  output_normal_random_image=self.output_normal_random_image,
                                  image_size=self.image_size, factor_shift=self.factor_shift,
                                  factor_rotate=self.factor_rotate,
                                  scratch_new_segments=self.scratch_new_segments,
                                  output_scratch_before_after=self.output_scratch_before_after,
                                  output_scratch_segment_image=self.output_scratch_segment_image,
                                  output_scratch_segment_npy=self.output_scratch_segment_npy,
                                  flatten_bg_lid=flatten_bg_lid,
                                  bg=[0, 1, 2], lid=[3, 4],
                                  dataset_name=self.dataset_name,
                                  export_before_after=export_before_after)

    # 3 CUT INTO BASIC UNIT FOR IMAGE INFERENCE
    def cut_basic_units(self, f, create_defect=True):
        print('STEP 3: CUT INTO BASIC UNITS FOR IMAGE INFERENCE FOLD {}'.format(f))
        segs = self.scratch_new_segments if create_defect else self.scratch_segments
        cut_basic_units(scratch_segment_npy=self.output_scratch_segment_npy,
                        output_scratch_basic_unit=self.output_scratch_basic_unit_inference,
                        folder='test_A', image_size=self.image_size,
                        scratch_segments=segs,
                        scratch_basic_units=self.scratch_basic_units,
                        unit_single_color=self.unit_single_color,
                        unit_multi_colors=self.unit_multi_colors,
                        dataset_name=self.dataset_name)

    # 4 SPLIT INTO BASIC UNIT FOR GENERATOR TRAINING
    def cut_basic_units_train(self, f):
        print('STEP 4: SPLIT INTO BASIC UNITS FOR TRAINING GENERATORS FOLD {}'.format(f))
        slice_dataset_training(self.output_scratch_basic_unit_training, self.subdataset_numbers,
                               self.scratch_basic_units,
                               self.output_light_dark_sliced, self.image_size, self.scratch_segments,
                               self.unit_single_color,
                               self.unit_multi_colors, f, dataset_name=self.dataset_name)

    # 5 TRAIN THE GENERATORS
    def train_generators(self, f):
        print('STEP 5: TRAIN THE GENERATOR FOLD {}'.format(f))
        epochs = 150 if self.dataset_name == 'magTile' else 50

        for sn in self.scratch_basic_units[:-1]:
            os.system('cmd /c "python pix2pixHD/train.py --name {} '
                      '--dataroot data/{}_7_output_scratch_basic_unit_for_gen_train/split_0_fold_{}_data_{}/{} '
                      '--niter {} --niter_decay 50 --tf_log --fold {}"'.format(sn, self.dataset_name, f,
                                                                               self.pix2pixHD_data_num, sn, epochs, f))

    # 6 SCRATCH INFERENCE
    def inference_scratch(self, f, last=False, pix2pix_name=None, final_name=None):
        print('STEP 6: GENERATOR INFERENCE FOLD {}'.format(f))
        sbu = [self.scratch_basic_units[-1]] if last else self.scratch_basic_units[:-1]
        for sn in sbu:
            pname = pix2pix_name if pix2pix_name is not None else sn
            try:
                temp = os.listdir(
                    os.path.join(self.output_pix2pix_inference, pname, 'test_latest_fold_' + str(f), 'images'))
                for t in temp:
                    os.remove(
                        os.path.join(self.output_pix2pix_inference, pname, 'test_latest_fold_' + str(f), 'images', t))
            except FileNotFoundError:
                pass
            os.system('cmd /c "python pix2pixHD/test.py --name {} '
                      '--dataroot data/3_output_scratch_basic_unit_for_inference/{} --how_many {} --fold {} '
                      '--which_epoch latest_fold_{} "'.format(
                pname, sn, len(os.listdir(os.path.join(self.output_scratch_basic_unit_inference, sn, 'test_A'))), f, f))
        if final_name is not None:
            try:
                os.mkdir(os.path.join(self.output_pix2pix_inference, final_name))
            except FileExistsError:
                pass
            try:
                shutil.rmtree(
                    os.path.join(self.output_pix2pix_inference, final_name, 'test_latest_fold_' + str(f), 'images'))
            except FileNotFoundError:
                pass
            try:
                shutil.rmtree(os.path.join(self.output_pix2pix_inference, final_name, 'test_latest_fold_' + str(f)))
            except FileNotFoundError:
                pass
            shutil.copytree(os.path.join(self.output_pix2pix_inference, pix2pix_name, 'test_latest_fold_' + str(f)),
                            os.path.join(self.output_pix2pix_inference, final_name, 'test_latest_fold_' + str(f)))

    # 7 COMBINE BASIC UNITS
    def combine_basic_units(self, f):
        print('STEP 7: COMBINE BASIC UNITS FOLD {}'.format(f))
        combine_scratch(self.output_pix2pix_inference, self.image_size, self.output_scratch_combined,
                        self.scratch_basic_units, f)

    # 8 COMBINE WITH LID
    def combine_with_base(self, f):
        print('STEP 8: COMBINE WITH LIDS FOLD {}'.format(f))
        combine_LID(self.output_scratch_combined, self.output_normal_random_image, self.output_scratch_lid_combined)

    # 9 CALCULATE M S ORIGINAL IMAGE
    def calculate_m_s_ori(self, f):
        print('STEP 9: CALCULATE M S ORIGINAL FOLD {}'.format(f))
        calculate_m_s(os.path.join(self.output_scratch_sliced, 'split_0_fold_{}', 'JPEGimages'.format(f)),
                      os.path.join(self.output_fid_scores, 'original_scratch.npz'))

    # 10 CALCULATE M S SYNTHETIC IMAGE
    def calculate_m_s_syn(self, f):
        print('STEP 10: CALCULATE M S SYNTHETIC FOLD {}'.format(f))
        calculate_m_s(self.output_scratch_lid_combined, os.path.join(self.output_fid_scores, 'sgs_scratch.npz'))

    # 11 STANDARD AUGMENTATION
    def standard_augmentation(self, f, create_defect=True):
        print('STEP 11: BUILDING STANDARD AUGMENTATION IMAGES FOLD {}'.format(f))
        if create_defect:
            augment_image(os.path.join(self.output_scratch_sliced, 'split_1_fold_{}').format(f),
                          self.output_std_aug_defect_images,
                          self.number_of_generated_images, self.image_size, self.dataset_name)
        else:
            augment_image(os.path.join(self.output_normal_sliced, 'split_0_fold_{}').format(f),
                          self.output_std_aug_normal_images,
                          self.number_of_generated_images, self.image_size, self.dataset_name)

    # 12 CALCULATE M S STD AUG IMAGE
    def calculate_m_s_std_aug(self, f):
        print('STEP 120: CALCULATE M S STD AUG FOLD {}'.format(f))
        calculate_m_s(self.output_std_aug_defect_images, os.path.join(self.output_fid_scores, 'std_aug_scratch.npz'))

    # 13 CALCULATE FRECHET
    def calculate_frechet(self, f):
        print('STEP 13: CALCULATE FRECHET FOLD {}'.format(f))
        ori = np.load(os.path.join(self.output_fid_scores, 'original_scratch.npz'))

        syn = np.load(os.path.join(self.output_fid_scores, 'std_aug_scratch.npz'))
        fid_score = calculate_frechet_distance(ori['m'], ori['s'], syn['m'], syn['s'])
        print('FID SCORES STD AUG = ', fid_score)

        syn = np.load(os.path.join(self.output_fid_scores, 'sgs_scratch.npz'))
        fid_score = calculate_frechet_distance(ori['m'], ori['s'], syn['m'], syn['s'])
        print('FID SCORES SGS = ', fid_score)

    # 14 RUN UNET
    def train_unet(self, f, combiner=True, ori=True, std_aug=True, pix2pix=True):
        print('STEP 14: RUN UNET BATCH 2 FOLD {}'.format(f))
        ori_numbers = int(
            np.sum(np.genfromtxt(os.path.join(self.output_light_dark_sliced, 'indices', 'data_length.csv'),
                                 delimiter=',')))
        epochs = 20 if self.dataset_name in ['conc2', 'asphalt', 'lid'] else 30
        learning_rate = 1e-7 if self.dataset_name == 'conc2' else 1e-5
        if combiner:
            train_seg_cls(scale=0.5, epochs=epochs, batch_size=2, fold=f, purpose='segmentation',
                          image_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),
                                                      'JPEGImages'),
                                         self.output_scratch_lid_combined],
                          label_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),
                                                      'SegmentationClass'),
                                         self.output_scratch_segment_npy],
                          metrics='jaccard', new_w=320, new_h=240, dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate,
                          last_scratch_segments=self.scratch_new_segments[-1],
                          output_unet_prediction=self.output_unet_prediction, visualize_unet=True,
                          checkpoint=self.output_trained_unet)
        if ori:
            train_seg_cls(scale=0.5, epochs=epochs, batch_size=2, fold=f, purpose='segmentation',
                          image_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                      'JPEGImages')],
                          label_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                      'SegmentationClass')],
                          metrics='jaccard', new_w=320, new_h=240, dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate,
                          last_scratch_segments=self.scratch_new_segments[-1],
                          output_unet_prediction=self.output_unet_prediction, visualize_unet=True,
                          checkpoint=self.output_trained_unet)
        if std_aug:
            train_seg_cls(scale=0.5, epochs=epochs, batch_size=2, fold=f, purpose='segmentation',
                          image_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                      'JPEGImages'),
                                         os.path.join(self.output_std_aug_defect_images, 'JPEGImages')],
                          label_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                      'SegmentationClass'),
                                         os.path.join(self.output_std_aug_defect_images, 'SegmentationClass')],
                          metrics='jaccard', new_w=320, new_h=240, dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate,
                          last_scratch_segments=self.scratch_new_segments[-1],
                          output_unet_prediction=self.output_unet_prediction, visualize_unet=True,
                          checkpoint=self.output_trained_unet)
        if pix2pix:
            train_seg_cls(scale=0.5, epochs=epochs, batch_size=2, fold=f, purpose='segmentation',
                          image_scratch=[
                              os.path.join(self.output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                           'JPEGImages'),
                              os.path.join(self.output_pix2pix_inference, 'full_defect_' + self.dataset_name,
                                           'test_latest_fold_' + str(f), 'images')],
                          label_scratch=[
                              os.path.join(self.output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                           'SegmentationClass'),
                              self.output_scratch_segment_npy],
                          metrics='jaccard', new_w=320, new_h=240, dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate,
                          last_scratch_segments=self.scratch_new_segments[-1],
                          output_unet_prediction=self.output_unet_prediction, visualize_unet=True,
                          checkpoint=self.output_trained_unet)

    # 15 RUN EVALUATION
    def eval_unet(self):
        print('STEP 15: RUN EVALUATION')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet(n_channels=3, n_classes=2)
        net.to(device=device)
        loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
        dataset = SegmentationDataset(self.output_scratch_lid_combined, self.output_scratch_segment_npy)  # SGS data
        eval_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
        for fold in range(1):  # range(5)
            net.load_state_dict(
                torch.load(os.path.join(self.output_trained_unet, 'checkpoint_epoch_30_fold_{}.pth'.format(fold)),
                           map_location=device))
            eval_score = evaluate(net, eval_loader, device, 'jaccard', purpose='segmentation',
                                  dataset_name=self.dataset_name)
            print('EVALUATION JACCARD SCORE: ', eval_score)

    # 16 VISUALIZE UNET
    def visualize_unet(self):
        print('STEP 16: VISUALIZE UNET')
        os.system(
            'cmd /c "python unet/predict.py --model {} --input {} --output {} --new_w 320 --new_h 240'.format(
                os.path.join(self.output_trained_unet, 'checkpoint_fold.pth'),
                'data/1_output_scratch_sliced/split_0_fold_0/JPEGImages', self.output_unet_prediction))

    # 17 RUN VGG
    def train_cls(self, f, combiner_std=True, combiner_pix=True, ori=True, std_aug=True, pix2pix=True, arch='vgg'):
        print('STEP 17: TRAIN {} BATCH 16 FOLD {}'.format(arch, f))
        ori_numbers = int(np.sum(
            np.genfromtxt(os.path.join(self.output_light_dark_sliced, 'indices', 'data_length.csv'), delimiter=',')))
        learning_rate = 1e-4
        if combiner_std:
            output_img_vgg_ = os.path.join(self.output_img_vgg, 'fold_' + str(f), 'combiner_std')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),
                                                      'JPEGImages'),
                                         self.output_scratch_lid_combined],
                          image_normal=[os.path.join(self.output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                     'JPEGImages'),
                                        os.path.join(self.output_std_aug_normal_images, 'JPEGImages')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=self.output_trained_vgg,
                          dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)

        if combiner_pix:
            output_img_vgg_ = os.path.join(self.output_img_vgg, 'fold_' + str(f), 'combiner_pix')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),
                                                      'JPEGImages'),
                                         self.output_scratch_lid_combined],
                          image_normal=[os.path.join(self.output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                     'JPEGImages'),
                                        os.path.join(self.output_pix2pix_inference, 'full_normal_' + self.dataset_name,
                                                     'test_latest_fold_' + str(f), 'images')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=self.output_trained_vgg,
                          dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)

        if ori:
            output_img_vgg_ = os.path.join(self.output_img_vgg, 'fold_' + str(f), 'ori')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),
                                                      'JPEGImages')],
                          image_normal=[os.path.join(self.output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                     'JPEGImages')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=self.output_trained_vgg,
                          dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)

        if std_aug:
            output_img_vgg_ = os.path.join(self.output_img_vgg, 'fold_' + str(f), 'stdAug')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),
                                                      'JPEGImages'),
                                         os.path.join(self.output_std_aug_defect_images, 'JPEGImages')],
                          image_normal=[os.path.join(self.output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                     'JPEGImages'),
                                        os.path.join(self.output_std_aug_normal_images, 'JPEGImages')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=self.output_trained_vgg,
                          dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)

        if pix2pix:
            output_img_vgg_ = os.path.join(self.output_img_vgg, 'fold_' + str(f), 'pix2pix')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[
                              os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),
                                           'JPEGImages'),
                              os.path.join(self.output_pix2pix_inference, 'full_defect_' + self.dataset_name,
                                           'test_latest_fold_' + str(f), 'images'), ],
                          image_normal=[os.path.join(self.output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f),
                                                     'JPEGImages'),
                                        os.path.join(self.output_pix2pix_inference, 'full_normal_' + self.dataset_name,
                                                     'test_latest_fold_' + str(f), 'images')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=self.output_trained_vgg,
                          dataset_name=self.dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)

    # 18 TEST VGG
    def eval_vgg(self):
        print('STEP 18: TEST VGG BATCH 16')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = Cls_model(output_dim=2)
        net.to(device=device)
        dataset = ClassifierDataset(self.output_scratch_lid_combined, scale=0.5, new_w=320, new_h=240)  # SGS data
        eval_loader = DataLoader(dataset, shuffle=False, batch_size=1)
        for fold in range(1):  # range(5)
            net.load_state_dict(
                torch.load(os.path.join(self.output_trained_vgg, 'checkpoint_epoch_20_fold_{}.pth'.format(fold)),
                           map_location=device))
            eval_score = evaluate(net, eval_loader, device, 'dice', purpose='classification',
                                  dataset_name=self.dataset_name)
            print('EVALUATION DICE SCORE: ', eval_score)

    def train_pix2pixFullImage(self, f):
        os.system(
            'cmd /c "python pix2pixHD/train.py --name full_image_{} '
            '--dataroot data/{}_6_output_file_datasplit_sliced/split_0_fold_{}_data_{} '
            '--niter 1000 --niter_decay 50 --tf_log --fold {}"'.format(
                self.dataset_name, self.dataset_name, f, self.pix2pixHD_data_num, f))

    # 19 CALCULATE DIVERSITY
    def calculate_diversity(self, f, combiner=True, ori=True, std_aug=True, pix2pix=True):
        if combiner:
            scratch_folders = [
                os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                self.output_scratch_lid_combined]
            msssmim_score = calculate_diversity(scratch_folders, new_w=320, new_h=240)
            print("fold {} dataset {} combiner msssim: {}".format(str(f), self.dataset_name, str(msssmim_score)))
        if ori:
            scratch_folders = [
                os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages')]
            msssmim_score = calculate_diversity(scratch_folders, new_w=320, new_h=240)
            print("fold {} dataset {} ori msssim: {}".format(str(f), self.dataset_name, str(msssmim_score)))
        if std_aug:
            scratch_folders = [
                os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                os.path.join(self.output_std_aug_defect_images, 'JPEGImages')]
            msssmim_score = calculate_diversity(scratch_folders, new_w=320, new_h=240)
            print("fold {} dataset {} std_aug msssim: {}".format(str(f), self.dataset_name, str(msssmim_score)))
        if pix2pix:
            scratch_folders = [
                os.path.join(self.output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                os.path.join(self.output_pix2pix_inference, 'full_defect_' + self.dataset_name,
                             'test_latest_fold_' + str(f), 'images')]
            msssmim_score = calculate_diversity(scratch_folders, new_w=320, new_h=240)
            print("fold {} dataset {} pix2pix msssim: {}".format(str(f), self.dataset_name, str(msssmim_score)))
