import numpy as np
import torch
import cv2
import torch.nn.functional as F

from joblib import Parallel, delayed
from tqdm import tqdm

from torch.utils.data import Dataset, sampler
from glob import glob

import pandas as pd
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

from sklearn.preprocessing import LabelEncoder

class BalanceSampler(sampler.Sampler):
    def __init__(self, dataset, ratio=3):
        self.r = ratio-1
        self.dataset = dataset
        self.pos_index = np.where(dataset.df.cancer>0)[0]
        self.neg_index = np.where(dataset.df.cancer==0)[0]

        self.length = self.r*int(np.floor(len(self.neg_index)/self.r))

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[:self.length].reshape(-1,self.r)
        pos_index = np.random.choice(pos_index, self.length//self.r).reshape(-1,1)

        index = np.concatenate([pos_index,neg_index],-1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.length

class MMGDataset(Dataset):
    def __init__(self, 
                data_dir:str,
                fold:int,
                aux_classes,
                aux_label,
                mode:str='train',
                augmentation:bool=False,
                resize_shape:list=[256,256],
                ):

        assert mode in ['train', 'valid']
        df = pd.read_csv(data_dir)
        df = df.query('fold!=9')
        if mode == 'train':
            self.df = df.query(f'fold != {fold}').reset_index(drop=True)
        elif mode == 'valid':
            self.df = df.query(f'fold == {fold}').reset_index(drop=True)
        else: 
            assert "Setting correctly fold number"

        self.aug_targets = aux_classes
        self.augmentation = augmentation
        self.mode = mode
        self.resize_shape = resize_shape
        self.aug = {'train': Compose([
                Resize(*self.resize_shape, p=1.0),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                ),
                Blur(p=0.3), 
                OpticalDistortion(p=0.3), 
                GridDistortion(p=0.3),
                RandomBrightnessContrast(p=0.3),
                # IAASharpen(p=0.3),
                # IAAAdditiveGaussianNoise(p=0.3),
                # CLAHE(p=0.3),
                # HueSaturationValue(p=0.3),
                ToTensorV2(p=1.0),
                
            ],p=1.)
            ,
            'valid' : Compose([
                Resize(*self.resize_shape, p=1.0),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0
                ),
                ToTensorV2(p=1.0),
            ], p=1.)
            }


        # self.df = df[:200]

    def __getitem__(self, index):
        # load image
        data = {}
        img   = cv2.imread(self.df.iloc[index].l_crop_img_path)
        
        label = self.df.iloc[index].cancer
        aux   = self.df.iloc[index][self.aug_targets].values.astype(np.float32)
        # img = cv2.resize(img,)
        augimg = self.aug[self.mode](image=img)['image']
        # print(augimg.shape,'12309182301928')
        # img = torch.tensor(img, dtype=torch.float)
        data['img']   = augimg
        data['label'] = label
        data['aux']   = torch.from_numpy(aux)
        data['origin'] = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),self.resize_shape)[None]
        data['patient_id'] = self.df.iloc[index].patient_id
        data['site_id'] = self.df.iloc[index].site_id

        return data

    def __len__(self):
        return len(self.df)