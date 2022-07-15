import os, cv2
import numpy as np
def cut_basic_units(output_scratch_segment_npy, output_scratch_basic_unit, image_size, scratch_new_segments):
    print('STEP 2: CUT INTO BASIC UNITS')
    name_list = os.listdir(output_scratch_segment_npy)
    if not os.path.isdir(output_scratch_basic_unit + '\\' + "straight\\test_A"):
        os.mkdir(output_scratch_basic_unit + '\\straight\\test_A')
        os.mkdir(output_scratch_basic_unit + '\\curve\\test_A')
        os.mkdir(output_scratch_basic_unit + '\\end\\test_A')
        os.mkdir(output_scratch_basic_unit + '\\all\\test_A')
    for name in name_list:
        npy_file = np.load(output_scratch_segment_npy + "\\" + name)
        straight = np.zeros(shape=tuple(reversed(image_size)) + (3,))
        curve = np.zeros(shape=tuple(reversed(image_size)) + (3,))
        end = np.zeros(shape=tuple(reversed(image_size)) + (3,))
        LID = np.zeros(shape=tuple(reversed(image_size)) + (3,))
        for i in range(image_size[1]):
            for j in range(image_size[0]):
                if npy_file[i, j] == scratch_new_segments[0]:
                    straight[i, j] = [5, 4, 120]
                    LID[i, j] = [128, 0, 0]
                elif npy_file[i, j] == scratch_new_segments[1]:
                    curve[i, j] = [5, 4, 120]
                    LID[i, j] = [0, 128, 128]
                elif npy_file[i, j] == scratch_new_segments[2]:
                    end[i, j] = [5, 4, 120]
                    LID[i, j] = [0, 128, 0]
                elif npy_file[i, j] == 1:
                    LID[i, j] = [0, 0, 128]
        cv2.imwrite(output_scratch_basic_unit + "\\straight\\test_A\\" + name[:-4] + ".png", straight)
        cv2.imwrite(output_scratch_basic_unit + "\\curve\\test_A\\" + name[:-4] + ".png", curve)
        cv2.imwrite(output_scratch_basic_unit + "\\end\\test_A\\" + name[:-4] + ".png", end)
        cv2.imwrite(output_scratch_basic_unit + "\\all\\test_A\\" + name[:-4] + ".png", LID)

