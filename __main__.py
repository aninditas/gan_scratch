from experiment import Experiment, export_heatmap
from itertools import product
import os
from pytorch_cnn_visualizations.src.misc_functions import save_class_activation_images


def main_experiment(dataset_name,base=None, pretrained=False):
    image_size = (320, 240)
    number_of_generated_images = 1000
    factor_shift = 0.3
    factor_rotate = 1

    exp = Experiment(image_size, number_of_generated_images, factor_shift, factor_rotate, dataset_name, base=base, pretrained=pretrained)
    # exp.rename_files(output_light_dark_datasplit,dataset_name) # only if the file name is number only --> change into dataset name _ number
    # exp.export_index_split(dataset_name)
    for f in range(exp.folds):
        if f!=0: # TODO remove
            continue

        print('<<<<<<<<<< FOLD {} >>>>>>>>>>'.format(f))
        exp.generate_scratch(f, export_before_after=False)  # 2 scratch generation

        exp.cut_basic_units_train(f) # 4 cut basic unit for training generator
        exp.train_generators(f)      # 5 train generator
        exp.train_pix2pixFullImage(f)

        exp.cut_basic_units_inf(f)  # 3 cut basic unit for inference
        exp.inference_scratch(f)
        exp.inference_scratch(f, last=True, pix2pix_name='full_image_' + dataset_name, final_name='full_defect_' + dataset_name)  # 6 scratch inference full scratch img
        exp.cut_basic_units_inf(f, create_defect=False)
        exp.inference_scratch(f, last=True, pix2pix_name='full_image_' + dataset_name, final_name='full_normal_' + dataset_name)  # 6 scratch inference full normal img

        exp.combine_basic_units(f)  # 7 combine basic unit
        exp.combine_with_base(f)  # 8 combine lid
        exp.standard_augmentation(f)  # 11 std aug scratch
        exp.standard_augmentation(f, create_defect=False)

        # exp.train_unet(f, combiner=False, ori=True, std_aug=False, pix2pix=False)  # 14 train unet

        # exp.train_cls(f, combiner_std=True, combiner_pix=False, ori=False, std_aug=False, pix2pix=False, arch='vgg')
        if base=='normal': #TODO uncomment
            exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=True, std_aug=True, pix2pix=True, arch='vgg') # 17 train vgg
            exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=True, std_aug=True, pix2pix=True, arch='resnet')
            exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=True, std_aug=True, pix2pix=True, arch='mobilenet')
        elif base=='scratch':
            exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=True, arch='vgg') # 17 train vgg
            exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=True, arch='resnet')
            exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=False, std_aug=False, pix2pix=True, arch='mobilenet')

        # exp.train_cls(f, combiner_std=True, combiner_pix=True, ori=True, std_aug=True, pix2pix=True, arch='vgg')

        # exp.calculate_diversity(f, combiner=True, ori=True, std_aug=True, pix2pix=True)

        # exp.visualize_unet()        # 16 visualize unet



if __name__ == '__main__':
    dataset_name = ['concrete'] #TODO remove
    # dataset_name=['lid','concrete']

    # for i in dataset_name:
    #     main_experiment(i)

    # base = ['scratch '] #TODO remove
    base=['normal','scratch']

    # pretrained = [False] #TODO remove
    pretrained=[True,False]

    # for i in product(dataset_name,base,pretrained):
    #     print('~~~~~~~~~~~~'+i[0]+'~~~~~~~~~~~~'+i[1]+'~~~~~~~~~~~~'+str(i[2])+'~~~~~~~~~~~~')
    #     main_experiment(i[0],i[1],i[2])

    # import try_pytorch
    # try_pytorch.train_pytorch()
    model_path = 'data/lid_16_output_checkpoint_vgg'
    image_path = 'data/lid_17_output_vgg_prediction/fold_0'
    base = ['Normal', 'Scratch']
    # base = ['Scratch']
    scen = ['std_aug', 'ori', 'combiner_pix', 'combiner_std','pix2pix']
    arch = ['vgg','mobilenet','resnet']
    # arch = ['resnet']
    for i in product(base,arch,scen,pretrained):
        print('~~~~~~~~~~~~' + i[0] + '~~~~~~~~~~~~' + i[1] + '~~~~~~~~~~~~' + str(i[2]) + '~~~~~~~~~~~~'+ str(i[3]) + '~~~~~~~~~~~~')
        model_path_ = os.path.join(model_path,'checkpoint_on{}_{}_{}_pre{}_fold_0.pth'.format(i[0],i[1],i[2],str(i[3])))
        target_layer = 30 if i[1]=='vgg' else 18 if i[1]=='mobilenet' else 7 if i[1]=='resnet' else 0
        export_heatmap(model_path_,image_path,arch=i[1],target_layer=target_layer)

