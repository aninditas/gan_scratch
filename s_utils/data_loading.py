import logging
from os import listdir
from os.path import splitext, join
from pathlib import Path
import itertools
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '',new_w=None, new_h=None, last_scratch_segments=None):
        self.images_dir = [Path(id) for id in images_dir]
        self.masks_dir = [Path(id) for id in masks_dir]
        self.last_scratch_segments = last_scratch_segments
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
    def preprocess(pil_img, scale, is_mask, new_w, new_h, last_scratch_segments=1):
        w, h = pil_img.size
        if new_w is not None and new_h is not None:
            newW = new_w
            newH = new_h
        else:
            newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        if not is_mask:
            pil_img = pil_img.convert('RGB')
        img_ndarray = np.asarray(pil_img).squeeze()

        if is_mask and np.max(img_ndarray)>last_scratch_segments:
            img_ndarray =np.rint(img_ndarray/255).astype(np.int32)

        if is_mask and img_ndarray.shape[-1]==3:
            img_ndarray = img_ndarray[:,:,0]

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
            # cv2.imread(str(filename))

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

        img = self.preprocess(img, self.scale, is_mask=False, new_w=self.new_w, new_h=self.new_h, last_scratch_segments=self.last_scratch_segments)
        mask = self.preprocess(mask, self.scale, is_mask=True, new_w=self.new_w, new_h=self.new_h, last_scratch_segments=self.last_scratch_segments)

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
        try:
            temp = [listdir(id) for id in normal_path]
            self.files_normal = list(itertools.chain(*temp))
        except: self.files_normal = []
        # self.files_normal = listdir(normal_path) if normal_path!=None else []
        self.X_data = self.files_scratch+self.files_normal
        self.y_data = np.concatenate((np.repeat(1,len(self.files_scratch)),np.repeat(0,len(self.files_normal))))
        assert len(self.X_data)==len(self.y_data), f'image and label length is different'

    def __getitem__(self, index):
        lbl = self.y_data[index]
        parent = self.scratch_path if index<len(self.files_scratch) else self.normal_path
        try:
            img = SegmentationDataset.load(join(parent, self.X_data[index]))
        except:
            for pa in parent:
                try:
                    img = SegmentationDataset.load(join(pa, self.X_data[index]))
                    break
                except: pass

        img = SegmentationDataset.preprocess(img, self.scale, is_mask=False, new_w=self.new_w, new_h=self.new_h)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch
                .as_tensor(lbl.copy()).long().contiguous()
        }

    def __len__(self):
        return len(self.y_data)