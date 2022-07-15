import numpy as np
import os
import matplotlib.pyplot as plt


ori_path = r"E:\PycharmProject\unets\new_real_data\test_npy"
save_path = r"E:\PycharmProject\unets\new_real_data\test_npy"

npy_list = os.listdir(ori_path)
count = 0
for file in npy_list:
    print(count)
    img = np.load(ori_path + "\\" + file)
    img[img > 1] = 2
    np.save(save_path + "\\" + file, img)
    count = count + 1

