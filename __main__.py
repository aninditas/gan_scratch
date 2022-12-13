import numpy as np
import os

from s_models.cls_models import Cls_model
from unet.evaluate import evaluate

from s_utils.eval import calculate_m_s
from s_utils.frechet_distance import calculate_frechet_distance
from s_utils.scratch_generator import generate_scratch_segments, split_light_dark_dataset, combine_scratch, combine_LID
from s_utils.cut_basic_units import cut_basic_units
import shutil
import torch
from s_models.unet_model import UNet
from s_utils.data_loading import SegmentationDataset, ClassifierDataset
from torch.utils.data import DataLoader
from s_utils.train import train_seg_cls
from s_utils.utils import slice_dataset, slice_dataset_training, preprocess_dataset, augment_image, calculate_diversity

if __name__ == '__main__':

    image_size = (320, 240)
    number_of_generated_images = 10000
    factor_shift = 0.3
    factor_rotate = 1
    # dataset_name = 'lid'
    dataset_name = 'magTile'
    # dataset_name = 'concrete'
    # dataset_name = 'conc2'
    # dataset_name = 'asphalt'
    base = 'normal'
    # base = 'scratch'

    input_scratch_raw = 'data/{}_0_input_scratch'.format(dataset_name)
    input_normal_raw = 'data/{}_0_input_normal'.format(dataset_name)
    input_scratch = 'data/{}_1_input_scratch'.format(dataset_name)
    input_normal = 'data/{}_1_input_normal'.format(dataset_name)
    output_scratch_sliced = 'data/{}_1_output_scratch_sliced'.format(dataset_name)
    output_normal_sliced = 'data/{}_1_output_normal_sliced'.format(dataset_name)
    output_scratch_before_after = 'data/{}_2_output_scratch_before_after'.format(dataset_name)
    output_normal_random_image = 'data/{}_2_output_normal_random_selection'.format(dataset_name)
    output_scratch_segment_image = 'data/{}_2_output_scratch_segment_image'.format(dataset_name)
    output_scratch_segment_npy = 'data/{}_2_output_scratch_segment_npy'.format(dataset_name)
    output_scratch_basic_unit_inference = 'data/3_output_scratch_basic_unit_for_inference'
    input_file_name_datasplit = 'data/{}_4_input_file_name_datasplit'.format(dataset_name)
    input_before_datasplit = 'data/{}_4_input_unsplit_data'.format(dataset_name)
    output_light_dark_datasplit = 'data/{}_6_output_file_datasplit'.format(dataset_name)
    output_light_dark_sliced = 'data/{}_6_output_file_datasplit_sliced'.format(dataset_name)
    output_scratch_basic_unit_training = 'data/{}_7_output_scratch_basic_unit_for_gen_train'.format(dataset_name)
    output_pix2pix_inference = 'data/9_output_pix2pix_inference'
    output_scratch_combined = 'data/{}_10_output_scratch_combined/'.format(dataset_name)
    output_scratch_lid_combined = 'data/{}_11_output_scratch_lid_combined'.format(dataset_name)
    output_fid_scores = 'data/{}_12_output_fid_scores'.format(dataset_name)
    output_std_aug_defect_images = 'data/{}_13_output_std_aug_defect_images'.format(dataset_name)
    output_std_aug_normal_images = 'data/{}_13_output_std_aug_normal_images'.format(dataset_name)
    output_trained_unet = 'data/{}_14_output_checkpoint_unet'.format(dataset_name)
    output_unet_prediction = 'data/{}_15_output_unet_prediction'.format(dataset_name)
    output_trained_vgg = 'data/{}_16_output_checkpoint_vgg'.format(dataset_name)
    output_img_vgg = 'data/{}_17_output_vgg_prediction'.format(dataset_name)

    unit_single_color = [5, 4, 120]

    if dataset_name=='lid': #TODO check dataset
        lid_normal_segments = [3, 4]  # the lid's segment number of normal images
        lid_scratch_segments = [1, 2, 3, 4]  # the lid and scratch's segment number of scratch images
        scratch_segments = [2, 3, 4]  # the scratch's segment number of scratch images
        scratch_new_segments = [5, 6, 7]  # the new scratch's segment number of scratch images (for scratch on scratch images)
        unit_multi_colors = [[128, 0, 0], # straight
                             [0, 128, 128], # curve
                             [0, 128, 0], # end
                             [0, 0, 128]] # all
        dataset_information = ['dark_LID_white_scratch', 'dark_LID_black_scratch', 'light_LID_white_scratch',
                               'light_LID_black_scratch']
        scratch_basic_units = ['lid_straight', 'lid_curve', 'lid_end', 'lid_all']
        # subdataset_numbers = [1,2,3,4]
        subdataset_numbers = [1]
        pix2pixHD_data_num = 1 #4


    else:
        lid_normal_segments = [0]  # the lid's segment number of normal images
        lid_scratch_segments = [0]  # the lid and scratch's segment number of scratch images
        scratch_segments = [1]  # the scratch's segment number of scratch images
        scratch_new_segments = [2]  # the new scratch's segment number of scratch images (for scratch on scratch images)
        unit_multi_colors = [[128, 0, 0],
                             [0, 0, 128]]
        subdataset_numbers = [1]
        dataset_information = [dataset_name]
        pix2pixHD_data_num = 1
        if dataset_name == 'magTile':
            scratch_basic_units = ['magTile_crack', 'magTile_all']
        elif dataset_name == 'concrete':
            scratch_basic_units = ['conc_crack', 'conc_all']
        elif dataset_name == 'conc2':
            scratch_basic_units = ['conc2_crack', 'conc2_all']
        elif dataset_name =='asphalt':
            scratch_basic_units = ['asphalt_gaps', 'asphalt_all']


    dataset_training_split = [40, 50, 0, 300]
    folds = 5
    data_split = 3


    # 1.0 preprocess dataset
    def _preprocess_dataset():
        print('STEP 1.0: PREPROCESS DATASET')
        preprocess_dataset(input_scratch_raw, input_scratch, image_size, dataset_name)
        preprocess_dataset(input_normal_raw, input_normal, image_size, dataset_name)

    # 1.0.1 rename files
    def _rename_files():
        print('STEP 1.0.1: RENAME FILES')
        for i in ['JPEGImages', 'SegmentationClass']:
            files = os.listdir(os.path.join(output_light_dark_datasplit, 'data_1',i))
            for f in files:
                src = os.path.join(output_light_dark_datasplit, 'data_1', i, f)
                des = os.path.join(output_light_dark_datasplit, 'data_1', i, dataset_name + '_' + f)
                os.rename(src,des)

    # 1.1 SPLIT LIGHT DARK FOR GENERATOR TRAINING
    def _split_subdataset():
        print('STEP 1.1: SPLIT LIGHT DARK DATASET')
        split_light_dark_dataset(subdataset_numbers, dataset_information, input_before_datasplit,
                                 input_file_name_datasplit, output_light_dark_datasplit)

    # 1.2 EXPORT INDEX LIST AND DEFINE SPLIT NUMBER
    def _export_index_split(input_normal=input_normal, output_normal_sliced=output_normal_sliced):
        print('STEP 1.2: EXPORT INDEX LIST AND DEFINE SPLIT NUMBER')
        if dataset_name in ['conc2', 'asphalt']: #TODO check dataset
            input_normal = None
            output_normal_sliced = None

        slice_dataset(light_dark_datasplit_path=output_light_dark_datasplit,
                      output_dataslice=output_light_dark_sliced,
                      input_scratch=input_scratch, output_scratch_sliced=output_scratch_sliced,
                      input_normal=input_normal, output_normal_sliced=output_normal_sliced,
                      dataset_names=subdataset_numbers, folds=folds, data_split=data_split,
                      dataset_name=dataset_name)

    # 2 NEW SCRATCH GENERATION
    def _generate_scratch(export_before_after=True):
        print("STEP 2: GENERATING " + str(number_of_generated_images) + " NEW SCRATCH SEGMENTS FOLD {}".format(f))
        if base == 'scratch':
            input_sliced = output_scratch_sliced
            lid_segment = lid_scratch_segments
            flatten_bg_lid = False
        else:
            input_sliced = output_normal_sliced
            lid_segment = lid_normal_segments
            flatten_bg_lid = True if dataset_name=='lid' else False
        generate_scratch_segments(number_of_generated_images=number_of_generated_images,
                                  input_normal_segment=os.path.join(input_sliced, 'split_1_fold_{}'.format(f),'SegmentationClass'),
                                  input_scratch_segment=os.path.join(output_scratch_sliced, 'split_1_fold_{}'.format(f),'SegmentationClass'),
                                  lid_normal_segment=lid_segment, lid_scratch_segment=lid_scratch_segments,
                                  scratch_segments=scratch_segments,
                                  output_normal_random_image=output_normal_random_image,
                                  image_size=image_size, factor_shift=factor_shift, factor_rotate=factor_rotate,
                                  scratch_new_segments=scratch_new_segments,
                                  output_scratch_before_after=output_scratch_before_after,
                                  output_scratch_segment_image=output_scratch_segment_image,
                                  output_scratch_segment_npy=output_scratch_segment_npy, flatten_bg_lid=flatten_bg_lid,
                                  bg=[0, 1, 2], lid=[3, 4],
                                  dataset_name=dataset_name,
                                  export_before_after=export_before_after)

    # 3 CUT INTO BASIC UNIT FOR IMAGE INFERENCE
    def _cut_basic_units(create_defect=True):
        print('STEP 3: CUT INTO BASIC UNITS FOR IMAGE INFERENCE FOLD {}'.format(f))
        segs = scratch_new_segments if create_defect else scratch_segments
        cut_basic_units(scratch_segment_npy=output_scratch_segment_npy,
                        output_scratch_basic_unit=output_scratch_basic_unit_inference,
                        folder='test_A', image_size=image_size,
                        scratch_segments=segs,
                        scratch_basic_units=scratch_basic_units,
                        unit_single_color=unit_single_color,
                        unit_multi_colors=unit_multi_colors,
                        dataset_name=dataset_name)


    # 4 SPLIT INTO BASIC UNIT FOR GENERATOR TRAINING
    def _cut_basic_units_train():
        print('STEP 4: SPLIT INTO BASIC UNITS FOR TRAINING GENERATORS FOLD {}'.format(f))
        slice_dataset_training(output_scratch_basic_unit_training, subdataset_numbers, scratch_basic_units,
                               output_light_dark_sliced, image_size, scratch_segments, unit_single_color,
                               unit_multi_colors, f, dataset_name=dataset_name)

    # 5 TRAIN THE GENERATORS
    def _train_generators():
        print('STEP 5: TRAIN THE GENERATOR FOLD {}'.format(f))
        # if dataset_name == 'lid': segment_name = ['straight','curve','end'] # TODO check dataset
        # elif dataset_name =='magTile': segment_name = ['crack']
        # elif dataset_name== 'concrete': segment_name =['crack_concrete']
        # elif dataset_name == 'conc2': segment_name = ['crack_conc2']
        # elif dataset_name == 'asphalt': segment_name = ['gaps']
        epochs = 150 if dataset_name=='magTile' else 50
        # data_num = 4 if dataset_name=='lid' else 1

        for sn in scratch_basic_units[:-1]:
            os.system('cmd /c "python pix2pixHD/train.py --name {} '
                      '--dataroot data/{}_7_output_scratch_basic_unit_for_gen_train/split_0_fold_{}_data_{}/{} '
                      '--niter {} --niter_decay 50 --tf_log --fold {}"'.format(sn,dataset_name, f, pix2pixHD_data_num, sn, epochs, f))

    # 6 SCRATCH INFERENCE
    def _inference_scratch(last=False,pix2pix_name=None,final_name=None):
        print('STEP 6: GENERATOR INFERENCE FOLD {}'.format(f))
        # if dataset_name == 'lid': segment_name = ['lid_straight','lid_curve','lid_end'] # TODO check dataset
        # elif dataset_name =='magTile': segment_name = ['crack']
        # elif dataset_name== 'concrete': segment_name =['crack_concrete']
        # elif dataset_name == 'conc2': segment_name = ['crack_conc2']
        # elif dataset_name == 'asphalt': segment_name = ['gaps']
        sbu = [scratch_basic_units[-1]] if last else scratch_basic_units[:-1]
        for sn in sbu:
            pname = pix2pix_name if pix2pix_name != None else sn
            try:
                temp=os.listdir(os.path.join(output_pix2pix_inference,pname,'test_latest_fold_'+str(f),'images'))
                for t in temp: os.remove(os.path.join(output_pix2pix_inference,pname,'test_latest_fold_'+str(f),'images',t))
            except: pass
            os.system('cmd /c "python pix2pixHD/test.py --name {} '
                      '--dataroot data/3_output_scratch_basic_unit_for_inference/{} --how_many {} --fold {} '
                      '--which_epoch latest_fold_{} "'.format(pname,sn,len(os.listdir(os.path.join(output_scratch_basic_unit_inference,sn,'test_A'))), f, f))
                      # '--which_epoch latest_fold_{} "'.format(pname, sn, 10, f, f))
        if final_name!=None:
            try: os.mkdir(os.path.join(output_pix2pix_inference,final_name))
            except: pass
            try: shutil.rmtree(os.path.join(output_pix2pix_inference,final_name,'test_latest_fold_'+str(f),'images'))
            except: pass
            try: shutil.rmtree(os.path.join(output_pix2pix_inference, final_name, 'test_latest_fold_' + str(f)))
            except: pass
            shutil.copytree(os.path.join(output_pix2pix_inference,pix2pix_name,'test_latest_fold_'+str(f)), os.path.join(output_pix2pix_inference,final_name,'test_latest_fold_'+str(f)))

    # 7 COMBINE BASIC UNITS
    def _combine_basic_units():
        print('STEP 7: COMBINE BASIC UNITS FOLD {}'.format(f))
        combine_scratch(output_pix2pix_inference, image_size, output_scratch_combined, scratch_basic_units, f, )

    # 8 COMBINE WITH LID
    def _combine_with_base():
        print('STEP 8: COMBINE WITH LIDS FOLD {}'.format(f))
        id_lid = 1 if dataset_name == 'lid' else 0  # TODO check dataset
        combine_LID(output_scratch_combined, output_scratch_segment_npy, output_normal_random_image, image_size,
                    output_scratch_lid_combined, id_scratch=scratch_new_segments)  # TODO id_lid please refer to segment number of lid component

    # 9 CALCULATE M S ORIGINAL IMAGE
    def _calculate_m_s_ori():
        print('STEP 9: CALCULATE M S ORIGINAL FOLD {}'.format(f))
        calculate_m_s(os.path.join(output_scratch_sliced, 'split_0_fold_{}', 'JPEGimages'.format(f)),
                      os.path.join(output_fid_scores, 'original_scratch.npz'))

    # 10 CALCULATE M S SYNTHETIC IMAGE
    def _calculate_m_s_syn():
        print('STEP 10: CALCULATE M S SYNTHETIC FOLD {}'.format(f))
        calculate_m_s(output_scratch_lid_combined, os.path.join(output_fid_scores, 'sgs_scratch.npz'))

    # 11 STANDARD AUGMENTATION
    def _standard_augmentation(create_defect=True):
        print('STEP 11: BUILDING STANDARD AUGMENTATION IMAGES FOLD {}'.format(f))
        if create_defect:
            augment_image(os.path.join(output_scratch_sliced, 'split_1_fold_{}').format(f), output_std_aug_defect_images,
                          number_of_generated_images, image_size, dataset_name)
        else:
            augment_image(os.path.join(output_normal_sliced, 'split_0_fold_{}').format(f), output_std_aug_normal_images,
                          number_of_generated_images, image_size, dataset_name)

    # 12 CALCULATE M S STD AUG IMAGE
    def _calculate_m_s_std_aug():
        print('STEP 120: CALCULATE M S STD AUG FOLD {}'.format(f))
        calculate_m_s(output_std_aug_defect_images, os.path.join(output_fid_scores, 'std_aug_scratch.npz'))

    # 13 CALCULATE FRECHET
    def _calculate_frechet():
        print('STEP 13: CALCULATE FRECHET FOLD {}'.format(f))
        ori = np.load(os.path.join(output_fid_scores, 'original_scratch.npz'))

        syn = np.load(os.path.join(output_fid_scores, 'std_aug_scratch.npz'))
        fid_score = calculate_frechet_distance(ori['m'], ori['s'], syn['m'], syn['s'])
        print('FID SCORES STD AUG = ', fid_score)

        syn = np.load(os.path.join(output_fid_scores, 'sgs_scratch.npz'))
        fid_score = calculate_frechet_distance(ori['m'], ori['s'], syn['m'], syn['s'])
        print('FID SCORES SGS = ', fid_score)

    # 14 RUN UNET
    def _train_unet(combiner=True, ori=True, std_aug=True, pix2pix=True):
        print('STEP 14: RUN UNET BATCH 2 FOLD {}'.format(f))
        ori_numbers = int(
            np.sum(np.genfromtxt(os.path.join(output_light_dark_sliced, 'indices', 'data_length.csv'),
                                 delimiter=',')))
        epochs = 20 if dataset_name in ['conc2','asphalt','lid'] else 30
        learning_rate = 1e-7 if dataset_name == 'conc2' else 1e-5
        if combiner:
            train_seg_cls(scale=0.5, epochs=epochs, batch_size=2, fold=f, purpose='segmentation',
                          image_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                                 output_scratch_lid_combined],
                          label_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),'SegmentationClass'),
                                output_scratch_segment_npy],
                          metrics='jaccard', new_w=320, new_h=240, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, last_scratch_segments=scratch_new_segments[-1],
                          output_unet_prediction=output_unet_prediction, visualize_unet=True, checkpoint=output_trained_unet)
            # os.system('cmd /c "python s_utils/train.py --scale 0.5 '
            #           '--metrics jaccard --batch-size 2 --fold {} --purpose segmentation '
            #           '--image_scratch {} {} --label_scratch {} {} '  # TODO for SGS
            #           '--dataset_name {} --last_scratch_segments {} --new_w 320 --new_h 240 --ori_numbers {} '
            #           '--output_unet_prediction {} --visualize_unet True --epochs {} --learning-rate {} "'.format
            #           (f,
            #            os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
            #            output_scratch_lid_combined,
            #            os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f),'SegmentationClass'),
            #            output_scratch_segment_npy,
            #            dataset_name, scratch_new_segments[-1], ori_numbers, output_unet_prediction, epochs, learning_rate))
        if ori:
            # learning_rate = 1e-7 if dataset_name == 'conc2' else 1e-2
            train_seg_cls(scale=0.5, epochs=epochs, batch_size=2, fold=f, purpose='segmentation',
                          image_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages')],
                          label_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f),'SegmentationClass')],
                          metrics='jaccard', new_w=320, new_h=240, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, last_scratch_segments=scratch_new_segments[-1],
                          output_unet_prediction=output_unet_prediction, visualize_unet=True, checkpoint=output_trained_unet)
            # os.system('cmd /c "python s_utils/train.py --scale 0.5 '
            #           '--metrics jaccard --batch-size 2 --fold {} --purpose segmentation '
            #           '--image_scratch {} --label_scratch {} '  # TODO for noAug
            #           '--dataset_name {} --last_scratch_segments {} --new_w 320 --new_h 240 --ori_numbers {} '
            #           '--output_unet_prediction {} --visualize_unet True  --epochs {} --learning-rate {} "'.format
            #           (f,
            #            os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
            #            os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f),'SegmentationClass'),
            #            dataset_name, scratch_new_segments[-1], ori_numbers, output_unet_prediction, epochs, learning_rate))
        if std_aug:
            # learning_rate = 1e-7 if dataset_name == 'conc2' else 1e-2
            train_seg_cls(scale=0.5, epochs=epochs, batch_size=2, fold=f, purpose='segmentation',
                          image_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
                                 os.path.join(output_std_aug_defect_images, 'JPEGImages')],
                          label_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'SegmentationClass'),
                                 os.path.join(output_std_aug_defect_images, 'SegmentationClass')],
                          metrics='jaccard', new_w=320, new_h=240, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, last_scratch_segments=scratch_new_segments[-1],
                          output_unet_prediction=output_unet_prediction, visualize_unet=True, checkpoint=output_trained_unet)
            # os.system('cmd /c "python s_utils/train.py --scale 0.5 '
            #           '--metrics jaccard --batch-size 2 --fold {} --purpose segmentation '
            #           '--image_scratch {} {} --label_scratch {} {} '  # TODO for stdAug
            #           '--dataset_name {} --last_scratch_segments {} --new_w 320 --new_h 240 --ori_numbers {} '
            #           '--output_unet_prediction {} --visualize_unet True  --epochs {} --learning-rate {} "'.format
            #           (f,
            #            os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
            #            os.path.join(output_std_aug_defect_images, 'JPEGImages'),
            #            os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'SegmentationClass'),
            #            os.path.join(output_std_aug_defect_images, 'SegmentationClass'),
            #            dataset_name, scratch_new_segments[-1], ori_numbers, output_unet_prediction, epochs, learning_rate))
        if pix2pix:
            # learning_rate = 1e-7 if dataset_name == 'conc2' else 1e-2
            train_seg_cls(scale=0.5, epochs=epochs, batch_size=2, fold=f, purpose='segmentation',
                          image_scratch=[
                      os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
                      os.path.join(output_pix2pix_inference, 'full_defect_'+dataset_name,'test_latest_fold_'+str(f),'images')],
                          label_scratch=[
                      os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'SegmentationClass'),
                      output_scratch_segment_npy],
                          metrics='jaccard', new_w=320, new_h=240, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, last_scratch_segments=scratch_new_segments[-1],
                          output_unet_prediction=output_unet_prediction, visualize_unet=True, checkpoint=output_trained_unet)
            # os.system('cmd /c "python s_utils/train.py --scale 0.5 '
            #           '--metrics jaccard --batch-size 2 --fold {} --purpose segmentation '
            #           '--image_scratch {} {} --label_scratch {} {} '  # TODO for stdAug
            #           '--dataset_name {} --last_scratch_segments {} --new_w 320 --new_h 240 --ori_numbers {} '
            #           '--output_unet_prediction {} --visualize_unet True  --epochs {} --learning-rate {} "'.format
            #           (f,
            #            os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
            #            os.path.join(output_pix2pix_inference, 'full_defect_'+dataset_name,'test_latest_fold_'+str(f),'images'),
            #            os.path.join(output_scratch_sliced, 'split_' + str(1) + '_fold_' + str(f), 'SegmentationClass'),
            #            output_scratch_segment_npy,
            #            dataset_name, scratch_new_segments[-1], ori_numbers, output_unet_prediction, epochs, learning_rate))

    # 15 RUN EVALUATION
    def _eval_unet():
        print('STEP 15: RUN EVALUATION')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet(n_channels=3, n_classes=2)
        net.to(device=device)
        loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
        dataset = SegmentationDataset(output_scratch_lid_combined, output_scratch_segment_npy)  # SGS data
        # dataset = BasicDataset(input_original_image_scratch, input_scratch_segment)  # original scratch
        eval_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
        for fold in range(1):  # range(5)
            net.load_state_dict(
                torch.load(os.path.join(output_trained_unet, 'checkpoint_epoch_30_fold_{}.pth'.format(fold)),
                           map_location=device))
            eval_score = evaluate(net, eval_loader, device, 'jaccard', purpose='segmentation',
                                  dataset_name=dataset_name)
            print('EVALUATION JACCARD SCORE: ', eval_score)

    # 16 VISUALIZE UNET
    def _visualize_unet():
        print('STEP 16: VISUALIZE UNET')
        os.system(
            'cmd /c "python unet/predict.py --model {} --input {} --output {} --new_w 320 --new_h 240'.format(
                # os.path.join(output_trained_unet,'checkpoint_fold.pth'), output_scratch_lid_combined, output_unet_prediction))
                os.path.join(output_trained_unet, 'checkpoint_fold.pth'),
                'data/1_output_scratch_sliced/split_0_fold_0/JPEGImages', output_unet_prediction))

    # 17 RUN VGG
    def _train_cls(combiner_std=True, combiner_pix=True, ori=True, std_aug=True, pix2pix=True,arch='vgg'):
        print('STEP 17: TRAIN {} BATCH 16 FOLD {}'.format(arch,f))
        ori_numbers = int(np.sum(np.genfromtxt(os.path.join(output_light_dark_sliced, 'indices', 'data_length.csv'), delimiter=',')))
        learning_rate = 1e-4 #1e-5
        if combiner_std:
            output_img_vgg_ = os.path.join(output_img_vgg, 'fold_' + str(f), 'combiner_std')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                                 output_scratch_lid_combined],
                          image_normal=[os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
                                os.path.join(output_std_aug_normal_images, 'JPEGImages')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=output_trained_vgg, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)

        #     os.system(
        #         'cmd /c "python s_utils/train.py --scale 0.5 --epochs 10 --batch-size 16 --fold {} --purpose classification '
        #         '--image_scratch {} {} --image_normal {} {} '  # TODO for SGS
        #         '--metrics dice --new_w 320 --new_h 240 --checkpoint {} '
        #         '--dataset_name {} --ori_numbers {} --learning-rate {} --export_path {} "'.format
        #         (f,
        #          os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
        #          output_scratch_lid_combined,
        #          os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
        #          os.path.join(output_std_aug_normal_images, 'JPEGImages'),
        #          output_trained_vgg,
        #          dataset_name, ori_numbers, learning_rate,output_img_vgg_))
        if combiner_pix:
            output_img_vgg_ = os.path.join(output_img_vgg, 'fold_' + str(f), 'combiner_pix')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                                 output_scratch_lid_combined],
                          image_normal=[os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
                                os.path.join(output_pix2pix_inference, 'full_normal_'+dataset_name,'test_latest_fold_'+str(f),'images')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=output_trained_vgg, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)
            # os.system(
            #     'cmd /c "python s_utils/train.py --scale 0.5 --epochs 10 --batch-size 16 --fold {} --purpose classification '
            #     '--image_scratch {} {} --image_normal {} {} '  # TODO for SGS
            #     '--metrics dice --new_w 320 --new_h 240 --checkpoint {} '
            #     '--dataset_name {} --ori_numbers {} --learning-rate {} --export_path {} "'.format
            #     (f,
            #      os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
            #      output_scratch_lid_combined,
            #      os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
            #      os.path.join(output_pix2pix_inference, 'full_normal_'+dataset_name,'test_latest_fold_'+str(f),'images'),
            #      output_trained_vgg,
            #      dataset_name, ori_numbers, learning_rate,output_img_vgg_))
        if ori:
            output_img_vgg_ = os.path.join(output_img_vgg, 'fold_' + str(f), 'ori')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages')],
                          image_normal=[os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=output_trained_vgg, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)
            # os.system(
            #     'cmd /c "python s_utils/train.py --scale 0.5 --epochs 10 --batch-size 16 --fold {} --purpose classification '
            #     '--image_scratch {} --image_normal {} '  # TODO noAug
            #     '--metrics dice --new_w 320 --new_h 240 --checkpoint {} '
            #     '--dataset_name {} --ori_numbers {} --learning-rate {}  --export_path {} "'.format
            #     (f,
            #      os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
            #      os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
            #      output_trained_vgg,
            #      dataset_name, ori_numbers, learning_rate, output_img_vgg_))
        if std_aug:
            output_img_vgg_ = os.path.join(output_img_vgg, 'fold_' + str(f), 'stdAug')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                                 os.path.join(output_std_aug_defect_images, 'JPEGImages')],
                          image_normal=[os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
                                os.path.join(output_std_aug_normal_images, 'JPEGImages')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=output_trained_vgg, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)
            # os.system(
            #     'cmd /c "python s_utils/train.py --scale 0.5 --epochs 10 --batch-size 16 --fold {} --purpose classification '
            #
            #     '--image_scratch {} {} --image_normal {} {} '  # TODO for stdAug
            #     '--metrics dice --new_w 320 --new_h 240 --checkpoint {} '
            #     '--dataset_name {} --ori_numbers {} --learning-rate {} --export_path {}  "'.format
            #     (f,
            #      os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
            #      os.path.join(output_std_aug_defect_images, 'JPEGImages'),
            #      os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
            #      os.path.join(output_std_aug_normal_images, 'JPEGImages'),
            #      output_trained_vgg,
            #      dataset_name, ori_numbers, learning_rate, output_img_vgg_))
        if pix2pix:
            output_img_vgg_ = os.path.join(output_img_vgg, 'fold_' + str(f), 'pix2pix')
            train_seg_cls(scale=0.5, epochs=10, batch_size=16, fold=f, purpose='classification',
                          image_scratch=[
                      os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                      os.path.join(output_pix2pix_inference, 'full_defect_'+dataset_name,'test_latest_fold_'+str(f),'images'),],
                          image_normal=[os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
                                os.path.join(output_pix2pix_inference, 'full_normal_'+dataset_name,'test_latest_fold_'+str(f),'images')],
                          metrics='dice', new_w=320, new_h=240, checkpoint=output_trained_vgg, dataset_name=dataset_name,
                          ori_numbers=ori_numbers, learning_rate=learning_rate, export_path=output_img_vgg_, arch=arch)
            # os.system(
            #     'cmd /c "python s_utils/train.py --scale 0.5 --epochs 10 --batch-size 16 --fold {} --purpose classification '
            #     '--image_scratch {} {} --image_normal {} {} '  # TODO for stdAug
            #     '--metrics dice --new_w 320 --new_h 240 --checkpoint {} '
            #     '--dataset_name {} --ori_numbers {} --learning-rate {} --export_path {}  "'.format
            #     (f,
            #      os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
            #      os.path.join(output_pix2pix_inference, 'full_defect_'+dataset_name,'test_latest_fold_'+str(f),'images'),
            #      os.path.join(output_normal_sliced, 'split_' + str(1) + '_fold_' + str(f), 'JPEGImages'),
            #      os.path.join(output_pix2pix_inference, 'full_normal_'+dataset_name,'test_latest_fold_'+str(f),'images'),
            #      output_trained_vgg,
            #      dataset_name, ori_numbers, learning_rate, output_img_vgg_))


    # 18 TEST VGG
    def _eval_vgg():
        print('STEP 18: TEST VGG BATCH 16')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = Cls_model(output_dim=2)
        net.to(device=device)
        loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
        dataset = ClassifierDataset(output_scratch_lid_combined, scale=0.5, new_w=320, new_h=240)  # SGS data
        # dataset = BasicDataset(input_original_image_scratch, input_scratch_segment)  # original scratch
        eval_loader = DataLoader(dataset, shuffle=False, batch_size=1)
        for fold in range(1):  # range(5)
            net.load_state_dict(
                torch.load(os.path.join(output_trained_vgg, 'checkpoint_epoch_20_fold_{}.pth'.format(fold)),
                           map_location=device))
            eval_score = evaluate(net, eval_loader, device, 'dice', purpose='classification',
                                  dataset_name=dataset_name)
            print('EVALUATION DICE SCORE: ', eval_score)

        # _export_index_split()        # 1.2 export index list


    def _train_pix2pixFullImage():
        os.system('cmd /c "python pix2pixHD/train.py --name full_image_{} --dataroot data/{}_6_output_file_datasplit_sliced/split_0_fold_{}_data_{} --niter 1000 --niter_decay 50 --tf_log --fold {}"'.format(dataset_name,dataset_name,f,pix2pixHD_data_num,f))
    # def _test_pix2pixFullImage(class_type):
    #     os.system('cmd /c "python pix2pixHD/test.py --name full_image_{} --dataroot data/3_output_scratch_basic_unit_for_inference/{}_all/test_latest_fold_{}/images/ --how_many 1000 --fold {} --which_epoch latest_fold_{}"'.format(class_type,dataset_name,dataset_name,f,f,f))

    # 19 CALCULATE DIVERSITY
    def _calculate_diversity(combiner=True, ori=True, std_aug=True, pix2pix=True):
        if combiner:
            scratch_folders = [os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                               output_scratch_lid_combined]
            msssmim_score = calculate_diversity(scratch_folders, new_w=320, new_h=240)
            print("fold {} dataset {} combiner msssim: {}".format(str(f), dataset_name, str(msssmim_score)))
        if ori:
            scratch_folders = [os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages')]
            msssmim_score = calculate_diversity(scratch_folders, new_w=320, new_h=240)
            print("fold {} dataset {} ori msssim: {}".format(str(f), dataset_name,str(msssmim_score)))
        if std_aug:
            scratch_folders = [os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                               os.path.join(output_std_aug_defect_images, 'JPEGImages')]
            msssmim_score = calculate_diversity(scratch_folders, new_w=320, new_h=240)
            print("fold {} dataset {} std_aug msssim: {}".format(str(f), dataset_name,str(msssmim_score)))
        if pix2pix:
            scratch_folders = [os.path.join(output_scratch_sliced, 'split_' + str(2) + '_fold_' + str(f), 'JPEGImages'),
                               os.path.join(output_pix2pix_inference, 'full_defect_'+dataset_name,'test_latest_fold_'+str(f),'images')]
            msssmim_score = calculate_diversity(scratch_folders, new_w=320, new_h=240)
            print("fold {} dataset {} pix2pix msssim: {}".format(str(f), dataset_name, str(msssmim_score)))

    # _rename_files() # only if the file name is number only --> change into dataset name _ number
    # _export_index_split()
    for f in range(folds):
    # for f in [1,2,3,4]:
        print('<<<<<<<<<< FOLD {} >>>>>>>>>>'.format(f))
        _generate_scratch(export_before_after=False)      # 2 scratch generation

        # _cut_basic_units_train() # 4 cut basic unit for training generator
        # _train_generators()      # 5 train generator
        # _train_pix2pixFullImage()

        _cut_basic_units()  # 3 cut basic unit for inference
        _inference_scratch()    # 6 scratch inference scratch only
        _inference_scratch(last=True, pix2pix_name='full_image_' + dataset_name,final_name='full_defect_' + dataset_name)  # 6 scratch inference full scratch img
        _cut_basic_units(create_defect=False)
        _inference_scratch(last=True, pix2pix_name='full_image_' + dataset_name, final_name='full_normal_'+dataset_name) # 6 scratch inference full normal img

        _combine_basic_units()   # 7 combine basic unit
        _combine_with_base()     # 8 combine lid
        # _standard_augmentation() # 11 std aug scratch
        # _standard_augmentation(create_defect=False)

        _train_unet(combiner=True, ori=False, std_aug=False, pix2pix=False)            # 14 train unet
        _train_cls(combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=False, arch='vgg')
        _train_cls(combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=False, arch='resnet')
        _train_cls(combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=False, arch='mobilenet')# 17 train vgg
        _calculate_diversity(combiner=True, ori=True, std_aug=True, pix2pix=True)

        # _visualize_unet()        # 16 visualize unet




    # os.system('cmd /c "python pix2pixHD/train.py --name full_image --dataroot data/lid_6_output_file_datasplit/data_1 --niter 1000 --niter_decay 50 --tf_log --fold 0"')
    # os.system('cmd /c "python pix2pixHD/test.py --name full_image --dataroot data/3_output_scratch_basic_unit_for_inference/full_image --how_many 16 --fold 0 --which_epoch latest_fold_0 "')
