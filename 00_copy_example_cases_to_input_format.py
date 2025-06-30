import SimpleITK as sitk
import json
import os
from os.path import join
import shutil
import numpy as np


"""
script to convert training case(s) to the format as set up by the sockets on the Grand Challenge platform
When running the algorithm, the file structure is different to the paths as provided in the training dataset,
however the data itself is identical in terms of PET/CT, threshold, organ segmentation, and rigid registration parameters
once accounting for some path/file conversions.

Training Dataset Structure Input Files:
/case (train_XXXX)/
    FDG/
        CT.nii.gz
        PET.nii.gz
        rigid.tfm (FDG fixed, PSMA moving)
        threshold.json {"suv_threshold": 2.1165041586631523}
        totseg_24.nii.gz
    PSMA/
        CT.nii.gz
        PET.nii.gz
        rigid.tfm (PSMA fixed, FDG moving, inverse of other rigid transform)
        threshold.json {"suv_threshold": 3.0}
        totseg_24.nii.gz        
        


algorithm files as formated on grand-challenge platform: (Note: mapped to /input/ at runtime with docker run --volume ... parameters)
/test/input/interf0/   
            fdg-pet-suv-threshold.json  (2.1165041586631523) #just float value, not dictionary {"suv_threshold":2.11)
            psma-pet-suv-threshold.json (3.0)
            psma-to-fdg-registration.json (parameters to match with /FDG/rigid.tfm, FDG as fixed image PSMA is moving, reverse is SITK inverse once set as Euler 3D transform, conversion between parameters and sitk transform method provided separately)
            images/
                fdg-ct/*.mha  #single .mha file in /images/fdg-ct/ subdirectory. Equivalent to Nifti in train_XXXX/FDG/CT.nii.gz
                fdg-ct-organ-segmentation/*mha #equivalent to /fdg/totseg_24.nii.gz
                fdg-pet/*.mha #equivalent to /fdg/PET.nii.gz
                psma-ct/*.mha #/psma/CT.nii.gz
                psma-ct-organ-segmentation/*.mha #/psma/totseg_24.nii.gz
                psma-pet-ga-68/*.mha #/psma/PET.nii.gz


Training Dataset Output Files:
/case (train_XXXX)/
    FDG/
        TTB.nii.gz
    PSMA/
        TTB.nii.gz

Expected Algorithm Output Files: (Note: mapped to /output/ at runtime according to docker run --volume ... parameters)
/test/output/interf0/
    images/
        fdg-pet-ttb/*.mha  #maps to train_XXXX/FDG/TTB.nii.gz. any filename should suit as long as included in the images/fdg-pet-ttb/ folder (eg, images/fdg-pet-ttb/fdg_segmentation.mha)
        psma-pet-ttb/*.mha #as above, the output socket just checks for a .mha image in the pmsa-pet-ttb subdirectory


Change the data_top variable below to match the location of your training dataset and select a case to test on. Debug dataset is cases 0025, 0060, and 0083.
This includes a selection of cases with largest file size to test wall time limit of 30 minutes.

"""



top='../CHALLENGE_DATA' #update to location of training data folder top directory as distributed. Subdirectories per case of the form 'train_XXXX'
case='train_0025' #large filesize case
##case='train_0083' #large filesize case
##case='train_0060' #"Average" case

out_top='test/input/interf0'


def convert_tfm_to_json(rigid_tfm,json_fname):
    if isinstance(rigid_tfm,str):
        rigid_tfm=sitk.ReadTransform(rigid_tfm)
    ar=np.zeros((4,4))
    mx=np.array(rigid_tfm.GetMatrix()).reshape(3,3)
    ar[:3,:3]=mx #write transform matrix to first 3x3 values of 4x4 array
    cent=np.array(rigid_tfm.GetCenter())
    tx=np.array(rigid_tfm.GetTranslation())
    ar[:3,3]=tx #write translation to last column of 4x4 array
    ar[3,:3]=cent #write centre to bottom row of 4x4 array
    out_dat={'3d_affine_transform':ar.tolist()}
    print(out_dat)
    with open(json_fname,'w') as f:
        json.dump(out_dat, f, indent=4)
    return

def convert_registration_json_to_transforms(json_fname):
    #function to convert grand challenge registration archive json to simple itk Euler 3D registration (*.tfm) as provided in training dataset
    #note 4x4 matrix stored on archive json files is unique to the challenge (may not correspond with other documentation) so use
    #this function to convert if required for image processing pipelines
    archive_json=read_json_file(json_fname) #read in dictionary from archive storage json format, stored as PSMA (moving) to FDG (fixed)
    matrix=np.array(archive_json['3d_affine_transform']).reshape(4,4) #get stored parameter matrix
    mx=matrix[:3,:3] #first 3x3 corresponds to Euler matrix
    center=matrix[3,:3] #xyz centre encoded in first 3 values of bottom row
    tx=matrix[:3,3] #xyz translation encoded in first 3 values of last column
    fdg_fixed_transform=sitk.Euler3DTransform() #create new SITK Euler 3D Transform
    fdg_fixed_transform.SetMatrix(mx.reshape(9)) #Set matrix values
    fdg_fixed_transform.SetCenter(center) #Set Center of Rotation
    fdg_fixed_transform.SetTranslation(tx) #Set Translation 
    psma_fixed_transform=fdg_fixed_transform.GetInverse() #compute inverse for FDG to PSMA rigid registration
    return fdg_fixed_transform, psma_fixed_transform

def load_json_file(location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())

def format_threshold_json(in_path,out_path):
    threshold=load_json_file(in_path)['suv_threshold']
    with open(out_path,'w') as f:
        f.write(json.dumps(threshold))
    return   

def copy_nii_to_mha(src,dst):
    im=sitk.ReadImage(src)
    sitk.WriteImage(im,dst,useCompression=True)
    return


def main():
    print(case)
    out_case=case #case name train_XXXX
    cdir=out_top #case dir

    #gtdir=join(gt_top,out_case)

    dirs=[cdir,join(cdir,'images'),join(cdir,'images','fdg-ct'),
          join(cdir,'images','fdg-pet'),join(cdir,'images','fdg-ct-organ-segmentation'),
          join(cdir,'images','psma-ct'),join(cdir,'images','psma-pet-ga-68'),
          join(cdir,'images','psma-ct-organ-segmentation')]
    for d in dirs:
        os.makedirs(d,exist_ok=True)
    
    rigid_path=join(top,case,'FDG','rigid.tfm')
    print(rigid_path)
    r1=sitk.ReadTransform(rigid_path)
    json_fname=join(cdir,'psma-to-fdg-registration.json')
    convert_tfm_to_json(rigid_path,json_fname) #converts euler tfm to 4x4 json matrix
    #r2=read_tfm_from_json(json_fname)
    format_threshold_json(join(top,case,'PSMA','threshold.json'),join(cdir,'psma-pet-suv-threshold.json'))
    format_threshold_json(join(top,case,'FDG','threshold.json'),join(cdir,'fdg-pet-suv-threshold.json'))
    for d in os.listdir(join(cdir,'images')): #loop to clean up previous existing images...
        for f in os.listdir(join(cdir,'images',d)):
            if f.endswith('.mha'):
                os.unlink(join(cdir,'images',d,f))
    copy_nii_to_mha(join(top,case,'FDG','CT.nii.gz'),join(cdir,'images','fdg-ct',case+'-fdg-ct.mha'))
    copy_nii_to_mha(join(top,case,'FDG','PET.nii.gz'),join(cdir,'images','fdg-pet',case+'-fdg-pet.mha'))
    copy_nii_to_mha(join(top,case,'FDG','totseg_24.nii.gz'),join(cdir,'images','fdg-ct-organ-segmentation',case+'-fdg-organs.mha'))
    copy_nii_to_mha(join(top,case,'PSMA','CT.nii.gz'),join(cdir,'images','psma-ct',case+'-psma-ct.mha'))
    copy_nii_to_mha(join(top,case,'PSMA','PET.nii.gz'),join(cdir,'images','psma-pet-ga-68',case+'-psma-pet.mha'))
    copy_nii_to_mha(join(top,case,'PSMA','totseg_24.nii.gz'),join(cdir,'images','psma-ct-organ-segmentation',case+'-psma-organs.mha'))

    #lastly copy generic inputs.json, usually generated by GC platform backend. Helpful for parsing input sockets and used in example inference script
    shutil.copyfile(join('test','inputs.json'),join(cdir,'inputs.json'))
    
if __name__ == "__main__":
    main()
