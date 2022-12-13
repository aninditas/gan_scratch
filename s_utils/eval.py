import os
import numpy as np

from s_utils.frechet_distance import calculate_activation_statistics


def calculate_m_s(image_folder, output_fid_scores):
    dd = []
    for f in os.listdir(image_folder): dd.append(os.path.join(image_folder, f))
    m_syn, s_syn = calculate_activation_statistics(dd)
    np.savez(output_fid_scores, m=m_syn, s=s_syn)


