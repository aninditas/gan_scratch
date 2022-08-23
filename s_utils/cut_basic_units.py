import os, cv2
import numpy as np
def cut_basic_units(output_scratch_segment_npy, output_scratch_basic_unit, folder, image_size, scratch_segments,
                    scratch_basic_units,unit_single_color, unit_multi_colors, cut_image=False):

    def _cut(parameters, loc):
        units[scratch_basic_units[-1]][loc] = unit_multi_colors[-1]
        for bu, ss, co in zip(scratch_basic_units[:len(scratch_segments)], parameters,
                              unit_multi_colors[:len(scratch_segments)], ):
            loc = np.where((segment_file==ss).all(axis=-1)) if cut_image else np.where(segment_file==ss)
            units[bu][loc] = unit_single_color
            units[scratch_basic_units[-1]][loc] = co
            if cut_image: units['image_cut'][bu][loc] = units['image'][bu][loc]
        return units

    name_list = os.listdir(output_scratch_segment_npy)
    if not os.path.isdir(os.path.join(output_scratch_basic_unit, 'straight',folder)):
        for s in scratch_basic_units:
            os.mkdir(os.path.join(output_scratch_basic_unit, s, folder))
            if cut_image: os.mkdir(os.path.join(output_scratch_basic_unit, s, folder[:-1]+'B'))

    for name in name_list:
        units = {}
        units['image'] = {}
        units['image_cut'] = {}
        for s in scratch_basic_units:
            units[s] = np.zeros(shape=tuple(reversed(image_size)) + (3,))
            if cut_image:
                units['image'][s] = cv2.imread(os.path.join(output_scratch_segment_npy[:-1]+'B', name[:-3]+'jpg'))
                units['image_cut'][s] = np.zeros_like(units['image'][s])
        try:
            segment_file = np.load(os.path.join(output_scratch_segment_npy, name))
            loc = np.where(segment_file > 0)
            units = _cut(scratch_segments, loc)
        except:
            segment_file = cv2.imread(os.path.join(output_scratch_segment_npy, name))
            loc = np.where(np.argmax(segment_file, axis=-1) > 0)
            units = _cut(unit_multi_colors, loc)
        for s in scratch_basic_units:
            cv2.imwrite(os.path.join(output_scratch_basic_unit, str(s), folder, name[:-4] + ".png"), units[s])
            if cut_image: cv2.imwrite(os.path.join(output_scratch_basic_unit, str(s), folder[:-1]+'B', name[:-4] + ".png"), units['image_cut'][s])
