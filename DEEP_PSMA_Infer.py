import os
from os.path import join,isdir
import SimpleITK as sitk
import shutil
import pandas as pd
import numpy as np
import time

import nnunet_config_paths #sets flag for nnunet results (resources/nnUNet_data/results)



"""
Script to run inference based on trained models from DEEP-PSMA Baseline Model. Only Fold 0 included for simple testing...
This script will return a predicted TTB label as sitk image format based on PET/CT input data and defined threshold.
The "run_inference" function is imported into the inference.py script which manages most of the filehandling at runtime.

"""


##nn_predict_exe='nnUNetv2_predict' #if not available in %PATH update to appropriate location
nn_predict_exe='/home/user/.local/bin/nnUNetv2_predict' #if not available in %PATH update to appropriate location, nn-unet executable locations if pip installed with --user flag
##nn_predict_exe=r"F:\TotalSegmentator_win\totseg_v24\Scripts\nnUNetv2_predict.exe"


def expand_contract_label(label,distance=5.0):
    """expand or contract sitk label image by distance indicated.
    negative values will contract, positive values expand.
    returns sitk image of adjusted label"""
    lar=sitk.GetArrayFromImage(label)
    label_single=sitk.GetImageFromArray((lar>0).astype('int16'))
    label_single.CopyInformation(label)
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetUseImageSpacing(True)
    distance_filter.SquaredDistanceOff()
    dmap=distance_filter.Execute(label_single)
    dmap_ar=sitk.GetArrayFromImage(dmap)
    new_label_ar=(dmap_ar<=distance).astype('int16')
    new_label=sitk.GetImageFromArray(new_label_ar)
    new_label.CopyInformation(label)
    return new_label

def run_inference(pt,ct,tracer='PSMA',suv_threshold=3.0,output_fname='deep-psma_inferred.nii.gz',
                   return_ttb_sitk=False,temp_dir='temp',fold='0',
                   expansion_radius=7.): #fold='all'
    start_time=time.time()
    print('RUNNING DEEP PSMA BASELINE')
    if isinstance(pt,str):
        pt_suv=sitk.ReadImage(pt)
    else:
        pt_suv=pt
    if isinstance(ct,str):
        ct=sitk.ReadImage(ct)
    module_dir=os.path.dirname(__file__)

    rs=sitk.ResampleImageFilter()
    rs.SetReferenceImage(pt_suv)
    rs.SetDefaultPixelValue(-1000)
    ct_rs=rs.Execute(ct)
    pt_rs=pt_suv/suv_threshold
    os.makedirs(temp_dir,exist_ok=True) #create/empty nnU-Net temporary dir
    for f in os.listdir(temp_dir):
        if isdir(join(temp_dir,f)):
            shutil.rmtree(join(temp_dir,f))
        else:
            os.unlink(join(temp_dir,f))
    #filenames to end in _0000.nii.gz and _0001.nii.gz, 0=rescaled PET, 1=CT
    os.makedirs(join(temp_dir,'nn_input'),exist_ok=True)
    sitk.WriteImage(pt_rs,join(temp_dir,'nn_input','deep-psma_0000.nii.gz'))
    sitk.WriteImage(ct_rs,join(temp_dir,'nn_input','deep-psma_0001.nii.gz'))

    call=nn_predict_exe+' -i '+join(temp_dir,'nn_input')+' -o '+join(temp_dir,'nn_output')
    if tracer=='PSMA': #select which nnunet model to use based on tracer flag
        call+=' -d '+'801'
    elif tracer=='FDG':
        call+=' -d '+'802'
    call+=' -c 3d_fullres'
    if not fold=='all':
        call+=' -f '+str(fold)

    print('images loaded',round(time.time()-start_time,1))
    print('Calling nnU-Net')
    print(call)
    
    os.system(call)
    

    ct_ar=sitk.GetArrayFromImage(ct_rs)
    pt_ar=sitk.GetArrayFromImage(pt_rs)
    tar=(pt_ar>=1.0).astype('int8') #above threshold array, mask from PET image
    pred_label=sitk.ReadImage(join(temp_dir,'nn_output','deep-psma.nii.gz')) #label predicted by nnUNet
    pred_ttb_ar=(sitk.GetArrayFromImage(pred_label)==1).astype('int8') #Predicted mask region for disease
    pred_norm_ar=(sitk.GetArrayFromImage(pred_label)==2).astype('int8') #predicted mask region for physiological
    
    pred_ttb_label=sitk.GetImageFromArray(pred_ttb_ar) #convert predicted TTB label to sitk format with spacing information to run grow/expansion function
    pred_ttb_label.CopyInformation(pred_label)
    pred_ttb_label_expanded=expand_contract_label(pred_ttb_label,distance=expansion_radius) #expand nnU-Net predicted disease region
    pred_ttb_ar_expanded=sitk.GetArrayFromImage(pred_ttb_label_expanded) #get array from TTB expanded sitk image
    pred_ttb_ar_expanded=np.logical_and(pred_ttb_ar_expanded>0,tar>0) #re-threshold expanded disease region
    
    output_ar=np.logical_and(pred_ttb_ar_expanded>0,pred_norm_ar==0).astype('int8') #final label, expanded areas inferred as physiological set to background (0)

    output_label=sitk.GetImageFromArray(output_ar) #construct final output label
    output_label.CopyInformation(pred_label) #pred_ttb_label
    output_label=sitk.Resample(output_label,pt,sitk.TranslationTransform(3), sitk.sitkNearestNeighbor, 0)
    
    sitk.WriteImage(output_label,join(temp_dir,'output_label.nii.gz'))
    if output_fname:
        sitk.WriteImage(output_label,output_fname)
    print('Processing complete',round(time.time()-start_time,1))
    if return_ttb_sitk:
        return output_label
    else:
        return





    
##Example Usage:
##ct_fname=r"W:\DEEP-PSMA\CHALLENGE_DATA\train_0001\PSMA\CT.nii.gz"
##pt_fname=r"W:\DEEP-PSMA\CHALLENGE_DATA\train_0001\PSMA\PET.nii.gz"
##pt=sitk.ReadImage(pt_fname)
##ct=sitk.ReadImage(ct_fname)
##suv_threshold=3.0
##
##ttb=run_inference(pt,ct,tracer='PSMA',output_fname=r"inference_test.nii.gz",
##                   return_ttb_sitk=True,
##                   fold=0,suv_threshold=suv_threshold)
##
