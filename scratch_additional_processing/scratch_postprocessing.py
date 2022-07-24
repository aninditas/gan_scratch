import os, cv2
import numpy as np

def combine_scratch(output_pix2pix_inference, output_scratch_segment_npy, image_size, output_scratch_combined, scratch_new_segments):
    gen_list = os.listdir(os.path.join(output_pix2pix_inference, "curve/test_latest/images/"))

    for num in range(len(gen_list)):
        npy = np.load(os.path.join(output_scratch_segment_npy, gen_list[num][:-22] + ".npy"))
        curve_img = cv2.imread(os.path.join(output_pix2pix_inference, "curve/test_latest/images/", gen_list[num]), cv2.IMREAD_COLOR)
        straight_img = cv2.imread(os.path.join(output_pix2pix_inference, "straight/test_latest/images/", gen_list[num]), cv2.IMREAD_COLOR)
        end_img = cv2.imread(os.path.join(output_pix2pix_inference, "end/test_latest/images/", gen_list[num]), cv2.IMREAD_COLOR)

        result = np.zeros(shape=tuple(reversed(image_size)) + (3,))

        for i in range(image_size[1]):
            for j in range(image_size[0]):
                if npy[i][j] == scratch_new_segments[0]:
                    result[i][j] = straight_img[i][j]
                elif npy[i][j] == scratch_new_segments[1]:
                    result[i][j] = curve_img[i][j]
                elif npy[i][j] == scratch_new_segments[2]:
                    result[i][j] = end_img[i][j]
        # result = result / 255
        cv2.imwrite(os.path.join(output_scratch_combined, str(gen_list[num][:-22]) + ".jpg"), result)


def combine_LID(output_scratch_combined, output_scratch_segment_npy, output_normal_random_image, image_size, output_scratch_lid_combined):
    name_list = os.listdir(output_scratch_combined)

    for name in name_list:
        npy = np.load(os.path.join(output_scratch_segment_npy, name[:-4] + ".npy"))
        lid = cv2.imread(os.path.join(output_normal_random_image, name), cv2.IMREAD_COLOR)
        gen = cv2.imread(os.path.join(output_scratch_combined, name), cv2.IMREAD_COLOR)
        # bnd = np.load(os.path.join(BND_PATH, name[:-4] + ".npy"))

        FIN = lid.copy()

        for i in range(image_size[1]):
            for j in range(image_size[0]):
                if npy[i][j] > 1 and gen[i, j, 0] != 0:
                    FIN[i][j] = gen[i][j]
                # if bnd[i, j] == 1 and gen[i, j, 0] < 65:
                #     FIN[i, j] = lid[i, j]
        cv2.imwrite(os.path.join(output_scratch_lid_combined, name), FIN)
