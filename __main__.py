from experiment import Experiment


def main_experiment():
    image_size = (320, 240)
    number_of_generated_images = 1000
    factor_shift = 0.3
    factor_rotate = 1

    # dataset_name = 'lid'
    dataset_name = 'concrete'
    # dataset_name = 'magTile'
    # dataset_name = 'conc2'
    # dataset_name = 'asphalt'

    # base = 'normal'
    base = 'scratch'

    exp = Experiment(image_size, number_of_generated_images, factor_shift, factor_rotate, dataset_name, base=base)

    # rename_files(output_light_dark_datasplit,dataset_name) # only if the file name is number only --> change into dataset name _ number
    # export_index_split(dataset_name,output_light_dark_datasplit,output_light_dark_sliced,input_scratch,output_scratch_sliced,input_normal,output_normal_sliced,subdataset_numbers,folds,data_split)
    for f in range(exp.folds):
        # for f in [0]:
        print('<<<<<<<<<< FOLD {} >>>>>>>>>>'.format(f))
        exp.generate_scratch(f, export_before_after=False)  # 2 scratch generation

        # exp.cut_basic_units_train(f) # 4 cut basic unit for training generator
        # exp.train_generators(f)      # 5 train generator
        # exp.train_pix2pixFullImage(f)

        exp.cut_basic_units(f)  # 3 cut basic unit for inference
        exp.inference_scratch(f)  # 6 scratch inference scratch only
        exp.inference_scratch(f, last=True, pix2pix_name='full_image_' + dataset_name, final_name='full_defect_' + dataset_name)  # 6 scratch inference full scratch img
        exp.cut_basic_units(f, create_defect=False)
        exp.inference_scratch(f, last=True, pix2pix_name='full_image_' + dataset_name, final_name='full_normal_' + dataset_name)  # 6 scratch inference full normal img

        exp.combine_basic_units(f)  # 7 combine basic unit
        exp.combine_with_base(f)  # 8 combine lid
        exp.standard_augmentation(f)  # 11 std aug scratch
        exp.standard_augmentation(f, create_defect=False)

        # exp.train_unet(f, combiner=False, ori=True, std_aug=False, pix2pix=False)  # 14 train unet
        exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=False, arch='vgg')
        exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=False, arch='resnet')
        exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=False, arch='mobilenet')  # 17 train vgg
        # exp.calculate_diversity(f, combiner=True, ori=True, std_aug=True, pix2pix=True)

        # exp.visualize_unet()        # 16 visualize unet

if __name__ == '__main__':
    main_experiment()

