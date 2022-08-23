import logging
from os import listdir
from os.path import splitext, join
from pathlib import Path
import itertools
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '',new_w=None, new_h=None):
        self.images_dir = [Path(id) for id in images_dir]
        self.masks_dir = [Path(id) for id in masks_dir]
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.new_w = new_w
        self.new_h = new_h
        self.mask_suffix = mask_suffix
        temp = [listdir(id) for id in images_dir]
        temp = list(itertools.chain(*temp))
        self.ids = [splitext(file)[0] for file in temp if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask, new_w, new_h):
        w, h = pil_img.size
        if new_w is not None and new_h is not None:
            newW = new_w
            newH = new_h
        else:
            newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        # mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        mask_file=[]
        img_file=[]
        for mk, id in zip(self.masks_dir, self.images_dir):
            try:
                mask_file.append(list(mk.glob(name + '.*'))[0])
                img_file.append(list(id.glob(name + '.*'))[0])
            except:
                pass

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False, new_w=self.new_w, new_h=self.new_h)
        mask = self.preprocess(mask, self.scale, is_mask=True, new_w=self.new_w, new_h=self.new_h)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class ClassifierDataset(Dataset):
    def __init__(self, scratch_path, normal_path=None, scale: float = 1.0, new_w=None, new_h=None):
        self.scale = scale
        self.new_w=new_w
        self.new_h=new_h
        self.scratch_path = scratch_path
        self.normal_path = normal_path
        # self.files_scratch = listdir(scratch_path)
        temp = [listdir(id) for id in scratch_path]
        self.files_scratch = list(itertools.chain(*temp))
        self.files_normal = listdir(normal_path) if normal_path!=None else []
        self.X_data = self.files_scratch+self.files_normal
        self.y_data = np.concatenate((np.repeat(1,len(self.files_scratch)),np.repeat(0,len(self.files_normal))))
        assert len(self.X_data)==len(self.y_data), f'image and label length is different'

    def __getitem__(self, index):
        lbl = self.y_data[index]
        parent = self.scratch_path if index<len(self.files_scratch) else self.normal_path
        try:
            img = BasicDataset.load(join(parent, self.X_data[index]))
        except:
            for pa in parent:
                try:
                    img = BasicDataset.load(join(pa,self.X_data[index]))
                    break
                except:
                    pass

        img = BasicDataset.preprocess(img, self.scale, is_mask=False, new_w=self.new_w, new_h=self.new_h)
        # lbl = BasicDataset.preprocess(lbl, self.scale, is_mask=True)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(lbl.copy()).long().contiguous()
        }
        # return img, lbl

    def __len__(self):
        return len(self.y_data)