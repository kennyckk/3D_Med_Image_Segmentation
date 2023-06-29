import os
import nibabel as nb
from nibabel.processing import resample_from_to,resample_to_output,conform
import numpy as np
import matplotlib.pyplot as plt


def show_slices(slices):
    """
    Function to display a row of image slices
    Input is a list of numpy 2D image slices
    """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

    plt.show()

def show_mid_slice(img_numpy, title='img'):
   """
   Accepts an 3D numpy array and shows median slices in all three planes
   """
   assert img_numpy.ndim == 3
   n_i, n_j, n_k = img_numpy.shape

   # sagittal (left image)
   center_i1 = int((n_i - 1) / 2)
   # coronal (center image)
   center_j1 = int((n_j - 1) / 2)
   # axial slice (right image)
   center_k1 = int((n_k - 1) / 2)

   show_slices([img_numpy[center_i1, :, :],
                img_numpy[:, center_j1, :],
                img_numpy[:, :, center_k1]])
   plt.suptitle(title)

datapath1="./masks/train/label0001.nii.gz"
datapath2="./imgs/train/PANCREAS_0001.nii.gz"

data1=nb.load(datapath1)
print(data1)

target_voxel_size = (2.0, 2.0, 2.0)
target_shape = np.ceil(np.array(data1.dataobj.shape) * np.array(data1.header.get_zooms()) / target_voxel_size)

# target_shape=target_shape.astype(np.uint, copy=False)
# print('target shape:', target_shape)


data1=resample_to_output(data1,2)
print(data1)
#data1=conform(data1,target_shape,target_voxel_size)
# world_data2=resample_to_output(data2,voxel_sizes=2)
# print(world_data1.get_fdata().shape,world_data2.get_fdata().shape)

data1=data1.get_fdata()
print(data1.shape)
#show_mid_slice(data1[::-1,:,:])
show_mid_slice(data1)