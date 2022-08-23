import copy
import os
import shutil
import random
from itertools import product

import numpy as np


def slice_dataset(light_dark_datasplit_path, output_dataslice, input_scratch, output_scratch_sliced, input_normal, output_normal_sliced, dataset_names, folds, data_split):
    data_length_scratch=[]
    for t in [output_dataslice, output_normal_sliced]:
        try:
            os.mkdir(os.path.join(t, 'indices'))
        except:
            pass

    for dn in dataset_names:
        file_list_scratch = os.listdir(os.path.join(light_dark_datasplit_path,'data_'+str(dn),'train_A'))
        data_length_scratch.append(len(file_list_scratch))
        for f in range(folds):
            random.seed(f)
            temp_scratch = copy.deepcopy(file_list_scratch)
            random.shuffle(temp_scratch)

            try:
                [os.mkdir(os.path.join(output_dataslice, 'split_{}_fold_{}_data_{}'.format(str(ds), str(f), str(dn)))) for ds in range(data_split)]
            except:
                pass

            try:
                [os.mkdir(os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)))) for ds in range(data_split)]
            except:
                pass

            for t in ['train_A', 'train_B']:
                try:
                    [os.mkdir(os.path.join(output_dataslice,'split_{}_fold_{}_data_{}'.format(str(ds),str(f),str(dn)),t)) for ds in range(data_split)]
                except:
                    pass

            for t in ['JPEGImages', 'SegmentationClass']:
                try:
                    [os.mkdir(os.path.join(output_scratch_sliced,'split_{}_fold_{}'.format(str(ds),str(f),str(dn)),t)) for ds in range(data_split)]
                except:
                    pass

            for ds in range(data_split):
                file_slice_scratch = temp_scratch[int(ds*(len(temp_scratch)/data_split)):int(((ds+1)*(len(temp_scratch)/data_split)))]
                for fs in file_slice_scratch:
                    shutil.copy(os.path.join(light_dark_datasplit_path, 'data_' + str(dn), 'train_A', fs),
                                os.path.join(output_dataslice,'split_{}_fold_{}_data_{}'.format(str(ds),str(f),str(dn)), 'train_A'))

                    shutil.copy(os.path.join(light_dark_datasplit_path, 'data_' + str(dn), 'train_B', fs[:-4]+'.jpg'),
                                os.path.join(output_dataslice,'split_{}_fold_{}_data_{}'.format(str(ds), str(f), str(dn)), 'train_B'))

                    shutil.copy(os.path.join(input_scratch, 'JPEGImages', fs[:-8] + '.jpg'),
                                os.path.join(output_scratch_sliced,'split_{}_fold_{}'.format(str(ds), str(f)), 'JPEGImages'))

                    shutil.copy(os.path.join(input_scratch, 'SegmentationClass', fs[:-8] + '.npy'),
                                os.path.join(output_scratch_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)), 'SegmentationClass'))

            np.savetxt(os.path.join(output_dataslice,'indices','data_'+str(dn)+'_fold_'+str(f)+'.csv'), temp_scratch, fmt='% s')


    file_list_normal = os.listdir(os.path.join(input_normal,'JPEGImages'))
    for f in range(folds):
        random.seed(f)
        temp_normal = copy.deepcopy(file_list_normal)
        random.shuffle(temp_normal)

        try:
            [os.mkdir(os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(ds), str(f)))) for ds in range(data_split-1)]
        except:
            pass

        for t in ['JPEGImages', 'SegmentationClass']:
            try:
                [os.mkdir(os.path.join(output_normal_sliced,'split_{}_fold_{}'.format(str(ds),str(f)),t)) for ds in range(data_split-1)]
            except:
                pass

        temp = int((np.sum(np.array(data_length_scratch)/data_split))/2)
        data_length_normal = [temp, len(file_list_normal)-temp]
        for idx, dl in enumerate(data_length_normal):
            file_slice_normal = temp_normal[int(idx*data_length_normal[idx-1]):int((idx*data_length_normal[idx-1])+dl)]
            for fs in file_slice_normal:
                shutil.copy(os.path.join(input_normal, 'JPEGImages', fs),
                            os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(idx), str(f)), 'JPEGImages'))

                shutil.copy(os.path.join(input_normal, 'SegmentationClass', fs[:-4] + '.npy'),
                            os.path.join(output_normal_sliced, 'split_{}_fold_{}'.format(str(idx), str(f)), 'SegmentationClass'))

            np.savetxt(os.path.join(output_normal_sliced,'indices','data_'+str(dn)+'_fold_'+str(f)+'.csv'), temp_scratch, fmt='% s')

    np.savetxt(os.path.join(output_dataslice,'indices','data_length.csv'), (np.array(data_length_scratch)/data_split).astype(np.uint), fmt='% i')
