import os, cv2
import numpy as np
import shutil
def cut_basic_units(scratch_segment_npy, output_scratch_basic_unit, folder, image_size, scratch_segments,
                    scratch_basic_units, unit_single_color, unit_multi_colors, cut_image=False, dataset_name=None):

    def _cut(parameters, loc):
        units[scratch_basic_units[-1]][loc] = unit_multi_colors[-1]
        for bu, ss, co in zip(scratch_basic_units[:len(scratch_segments)], parameters,
                              unit_multi_colors[:len(scratch_segments)], ):
            # loc = np.where((segment_file==ss).all(axis=-1)) if len(parameters)>1 else
            loc = np.where(segment_file==ss)
            units[bu][loc] = unit_single_color #if units[bu].max()>1 else unit_single_color/255
            units[scratch_basic_units[-1]][loc] = co
            if cut_image: units['image_cut'][bu][loc] = units['image'][bu][loc]
        return units

    name_list = os.listdir(scratch_segment_npy)
    # if not os.path.isdir(os.path.join(output_scratch_basic_unit, scratch_basic_units[-1],folder)):
    for s in scratch_basic_units:
        try:
            os.mkdir(os.path.join(output_scratch_basic_unit, s))
        except: pass
        try:
            os.mkdir(os.path.join(output_scratch_basic_unit, s, folder))
        except:
            temp = os.listdir(os.path.join(output_scratch_basic_unit, s, folder))
            for t in temp: os.remove(os.path.join(output_scratch_basic_unit, s, folder,t))
        if cut_image:
            try:
                os.mkdir(os.path.join(output_scratch_basic_unit, s, folder[:-1]+'B'))
            except:
                temp = os.listdir(os.path.join(output_scratch_basic_unit, s, folder[:-1]+'B'))
                for t in temp: os.remove(os.path.join(output_scratch_basic_unit, s, folder[:-1]+'B',t))


    for name in name_list:
        units = {}
        units['image'] = {}
        units['image_cut'] = {}
        for s in scratch_basic_units:
            units[s] = np.zeros(shape=tuple(reversed(image_size)) + (3,)).astype(np.uint8)
            if cut_image:
                units['image'][s] = cv2.resize(cv2.imread(os.path.join(scratch_segment_npy[:-1] + 'B', name[:-3] + 'jpg')), image_size)
                units['image_cut'][s] = np.zeros_like(units['image'][s])
        try:
            segment_file = cv2.resize(np.array(np.load(os.path.join(scratch_segment_npy, name)), dtype='uint8'), dsize=image_size,interpolation=cv2.INTER_NEAREST)
            loc = np.where(segment_file > 0)
            units = _cut(scratch_segments, loc)
        except:
            if dataset_name == 'lid': #TODO check dataset
                segment_file = cv2.resize(cv2.imread(os.path.join(scratch_segment_npy, name)), image_size)
                loc = np.where(np.argmax(segment_file, axis=-1) > 0)
                units = _cut(unit_multi_colors, loc)
            else:
                segment_file = np.rint(cv2.resize(cv2.imread(os.path.join(scratch_segment_npy, name))[:, :, 0] / 255, image_size))
                loc = np.where(segment_file > 0)
                units = _cut(scratch_segments, loc)
        for s in scratch_basic_units:
            cv2.imwrite(os.path.join(output_scratch_basic_unit, str(s), folder, name[:-4] + ".png"), units[s])
            if cut_image: cv2.imwrite(os.path.join(output_scratch_basic_unit, str(s), folder[:-1]+'B', name[:-4] + ".png"), units['image_cut'][s])
