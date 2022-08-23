import os
import numpy as np
import cv2
import tqdm, random

from s_utils.frechet_distance import calculate_activation_statistics


def calculate_m_s(image_folder, output_fid_scores):
    dd = []
    for f in os.listdir(image_folder): dd.append(os.path.join(image_folder, f))
    m_syn, s_syn = calculate_activation_statistics(dd)
    np.savez(output_fid_scores, m=m_syn, s=s_syn)


def augment_image(ori_img_path, aug_img_path, num_of_aug, image_shape):
    ori_img_list = os.listdir(ori_img_path)
    for idx in tqdm.tqdm(range(num_of_aug)):
        image = random.choice(ori_img_list)
        image = cv2.imread(os.path.join(ori_img_path,image))
        image = cv2.resize(image,image_shape)
        # M = np.float32([[1, 0, image_shape[0]*0.02*random.randint(-1,1)], [0, 1, image_shape[1]*0.02*random.randint(-1,1)]])
        # image = cv2.warpAffine(image, M, (image_shape[0],image_shape[1]), borderMode=cv2.BORDER_REFLECT)
        image = cv2.flip(image, -1)
        cv2.imwrite(os.path.join(aug_img_path,str(idx)+'.jpg'),image)