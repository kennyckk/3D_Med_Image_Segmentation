import logging
import os
import nibabel as nb
from nibabel.processing import resample_to_output
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchio as tio
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset,ConcatDataset
from tqdm import tqdm
import torchvision.transforms as T
import elasticdeform.torch as etorch
from utils import show_mid_slice


def transformation(scale_size,patch_size,random_flip_prob=1,rotate_val=0,shift_val=0,scale_val=0):

    train_transform=tio.transforms.Compose([
        tio.transforms.RandomFlip(axes=(0,1),p=random_flip_prob), # 1,H,W,D,
        tio.transforms.RandomAffine(scales=scale_val,
                                    degrees=rotate_val,
                                    translation=(scale_size[0]*shift_val[0],scale_size[1]*shift_val[1],0),
                                    ),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),#make the value to be mean 0 norm
        tio.transforms.CropOrPad(scale_size, mask_name='mask'), #will crop around the mask value not equal to 0
    ])

    eval_transform=tio.transforms.Compose([
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),  # make the value to be mean 0 norm
        tio.transforms.CropOrPad(scale_size, mask_name=None), #to do center crop
    ])

    return {'train':train_transform,'eval':eval_transform}

def load_nifti_img(filepath, voxel_size=2,mask=True):

    nim = nb.load(filepath)
    order,dtype= (0,np.uint8) if mask else (3,np.int16)

    nim=resample_to_output(nim,voxel_size,order=order)
    out_nii_array = nim.get_fdata().astype(dtype,copy=False) # shape: (H,W,D) if its grey
    out_nii_array = np.squeeze(out_nii_array)  # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.affine,
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array,meta

class BasicDataset3D(Dataset): #already split to train and eval, so dir should indicate the train or eval
    def __init__(self, images_dir: str, mask_dir: str,transform:dict,train:bool,mask_suffix: str = '',img_prefix:str='',binary=False):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.mask_suffix = mask_suffix
        self.img_prefix=img_prefix
        self.binary=binary
        self.transform=transform['train'] if train else transform['eval']


        # first need to check if the train or mask data is less
        images_files=os.listdir(self.images_dir)
        masks_files=os.listdir(self.mask_dir)
        self.img_limit=(len(images_files)< len(masks_files))
        self.total=images_files if self.img_limit else masks_files
        logging.info(f'Creating dataset with {len(self.total)} examples')

        # find number of unique mask class
        if self.binary: # if know the dataset is binary class
            self.mask_values=[0,1]
        else:  # if having multiple class, will have to run through all images to check
            pass
        logging.info(f'Unique mask values: {self.mask_values}')


    def __len__(self):
        return len(self.total)

    def preprocess(self, mask_values, img, is_mask,binary=False):
        # input will be mask or img with H,W,D np array
        if is_mask:
            #img already 3D array with class in uint8 in np array
            mask = img
            if not binary:
                for i, v in enumerate(mask_values):
                    mask[mask==v]=i
            mask=mask[np.newaxis,...] # 1,H,W,D
            return torch.as_tensor(mask.copy()).long().contiguous() #1,H,W,D
        else:
            # img already 3D array in int16 in np array
            if img.ndim==3: #its H,W,D , gray scale
                img= img[np.newaxis,...] # become 1,H,W,D
            else: # H,W,D,C, RGB
                img=img.transpose((3,0,1,2)) #become C,H,W,D

            return torch.as_tensor(img.copy()).float().contiguous() # 1,H,W,D or C,H,W,D

    def __getitem__(self, idx):
        name= self.total[idx] #get the filename depending img or mask is limiting
        if self.img_limit: #image is less than mask e.g. name=PANCREAS_0001.nii.gz
            img_file= list(self.images_dir.glob(name))
            #need to get corresponding mask file i.e. PANCREAS_0001.nii.gz-->label0001.nii.gz
            mask_file=list(self.mask_dir.glob(name.replace(self.img_prefix,self.mask_suffix)))
        else:#mask is less or equal to img e.g. name=label0001.nii.gz
            img_file= list(self.images_dir.glob(name.replace(self.mask_suffix,self.img_prefix)))
            mask_file=list(self.mask_dir.glob(name))
        assert len(img_file)==1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file)==1,f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        #proceed to load the nii.gz file to 3D images np array
        mask,_=load_nifti_img(mask_file[0],mask=True) #H,W,D
        img,_=load_nifti_img(img_file[0],mask=False) # H,W,D
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img= self.preprocess(self.mask_values,img, is_mask=False,binary=True)
        mask = self.preprocess(self.mask_values, mask, is_mask=True, binary=True)
        # img (1,H,W,D) or (C,H,W,D) float ; mask (1,H,W,D) long Tenosr


        img_n_mask=tio.Subject( #create tio subject for transformation
            img=tio.ScalarImage(tensor=img),
            mask=tio.LabelMap(tensor=mask))
        assert self.transform!=None, "transform is required for 3D data"
        transformed=self.transform(img_n_mask) # to undergo necessary transformation
        #img (1, H,W,D) float32 mask (1,H,W,D) float -->(1, H,W,D) float32, mask (H,W,D) long
        img= transformed.img.tensor; mask=torch.squeeze(transformed.mask.tensor.long(),0)

        return {
            'image': img, #(1, H,W,D) float32
            'mask': mask #(H,W,D) long
        }

class CT_82(BasicDataset3D):
    def __init__(self,images_dir, mask_dir,train):
        # transformation/augmentation from original setting
        transform=transformation([160, 160, 96], [160, 160, 96], 0.5, 15.0, [0.1, 0.1], [0.7, 1.3])
        mask_suffix= 'label'
        img_prefix='PANCREAS_'
        binary=True
        super().__init__(images_dir,mask_dir,transform,train,mask_suffix,img_prefix,binary)

if __name__ =="__main__":

    # mask,_=load_nifti_img(list(Path('../data/imgs/train/').glob("PANCREAS_0001.nii.gz"))[0],mask=False)
    # print(mask.dtype,mask)
    #scale_size,patch_size,random_flip_prob=1,rotate_val=0,shift_val=[],scale_val=0

    dataset=CT_82('../data/imgs/train/','../data/masks/train/',train=True)
    print(len(dataset))
    data=next(iter(dataset))
    img=data['image'];mask=data['mask']

    print(torch.max(img), torch.mean(img),torch.max(mask), img.size(),mask.size())
    show_mid_slice(torch.squeeze(img).numpy())
    show_mid_slice(torch.squeeze(mask).numpy())

