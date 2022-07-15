import os
import cv2
import numpy as np


def cut_basic_unit(path, npy_path, save_path):
    name_list = os.listdir(path)
    if not os.path.isdir(save_path + '\\' + "straight"):
        os.mkdir(save_path + '\\straight')
        os.mkdir(save_path + '\\curve')
        os.mkdir(save_path + '\\end')
    for name in name_list:
        segmentation_image = cv2.imread(path + "\\" + name, cv2.IMREAD_COLOR)
        npy_file = np.load(npy_path + "\\" + name[:-4] + ".npy")
        str_part = segmentation_image.copy()
        cur_part = segmentation_image.copy()
        end_part = segmentation_image.copy()
        for i in range(240):
            for j in range(320):
                if npy_file[i][j] != 2:
                    str_part[i][j] = str_part[i][j] * 0
        # for i in range(240):
        #     for j in range(320):
                if npy_file[i][j] != 3:
                    cur_part[i][j] = cur_part[i][j] * 0
        # for i in range(240):
        #     for j in range(320):
                if npy_file[i][j] != 4:
                    end_part[i][j] = end_part[i][j] * 0


        cv2.imwrite(save_path + "\\straight\\" + name, str_part)
        cv2.imwrite(save_path + "\\curve\\" + name, cur_part)
        cv2.imwrite(save_path + "\\end\\" + name, end_part)


if __name__ == '__main__':
    segmentation_path = r"E:\github_dataset\segmentation image"
    npy_path = r"E:\github_dataset\npy_file"
    save_path = r"E:\github_dataset\basic unit segmentation"

    cut_basic_unit(segmentation_path, npy_path, save_path)
