import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_img(path, save_path, category):
    print("Loading")
    name_list = os.listdir(path)
    if not os.path.isdir(save_path + '\\' + category):
        os.mkdir(save_path + '\\' + category)
    for name in name_list:
        img = cv2.imread(path + "\\" + name, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (320, 240))

        cv2.imwrite(save_path + "\\" + category + "\\" + name, img)


def resize_npy(path, save_path):
    print("Loading")
    name_list = os.listdir(path)
    if not os.path.isdir(save_path + "\\npy_file"):
        os.mkdir(save_path + "\\npy_file")
    for name in name_list:
        npy = np.load(path + "\\" + name)
        npy = cv2.resize(npy, (320, 240), interpolation=cv2.INTER_NEAREST)

        np.save(save_path + "\\npy_file\\" + name, npy)


if __name__ == '__main__':
    #   .jpg
    real_img_path = r"C:\Users\Chen\PycharmProjects\data_annotation\new_scratch_out\JPEGImages"
    #   .png
    segmentation_img_path = r"C:\Users\Chen\PycharmProjects\data_annotation\new_scratch_out\SegmentationClassPNG"
    #   .npy
    npy_path = r"C:\Users\Chen\PycharmProjects\data_annotation\new_scratch_out\SegmentationClass"
    #   save path
    save_path = r"E:\github_dataset"

    # resize_img(real_img_path, save_path, "real image")
    # resize_img(segmentation_img_path, save_path, "segmentation image")
    resize_npy(npy_path, save_path)

