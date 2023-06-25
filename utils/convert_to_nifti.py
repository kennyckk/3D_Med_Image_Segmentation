"""this python file is to unzip and convert the raw DICOM file from Official Archive to nii.gz"""

import os
import glob
import dicom2nifti
import tqdm
from multiprocessing import Pool

out_dir='./imgs/' # to be changed for custom dir
os.makedirs(out_dir,exist_ok=True)
curr=os.getcwd()+'/Pancreas-CT'
folders=os.listdir(curr)

def convert(folder):
    curr_folder=os.path.join(curr,folder)
    #print(curr_folder)
    curr_folder+='/*/*/*/*.dcm'
    dcms=glob.glob(curr_folder,recursive=True)
    dcm_folder=os.path.dirname(dcms[0]) # this is the folder containing all dcms for one 3d image
    #print(dcm_folder)
    out_filename=folder+'.nii.gz'
    dicom2nifti.dicom_series_to_nifti(dcm_folder,os.path.join(out_dir,out_filename))

if __name__ =="__main__":
    with Pool(processes=os.cpu_count()) as p:
        tqdm.tqdm(
            p.map(convert, folders),
            total=len(folders)
        )
