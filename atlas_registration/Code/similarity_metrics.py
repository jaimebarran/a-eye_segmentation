from asyncore import write
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import Line2D
from sqlalchemy import true
from scipy import stats

# nDSC (normalized DSC)
def dice_norm_metric(ground_truth, predictions):
    '''
    For a single example returns DSC_norm, fpr, fnr
    '''

    # Reference for normalized DSC
    r = 0.001 # It should be 1/N*(np.sum(voxels_label[i])/np.sum(voxels_image[i])) i belonging to training set
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0, 1.0, 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2 * tp / (fp_scaled + 2 * tp + fn)

        fpr = fp / (len(gt.flatten()) - np.sum(gt))
        if np.sum(gt) == 0:
            fnr = 1.0
        else:
            fnr = fn / np.sum(gt)
        return dsc_norm # fpr, fnr

''' Data frame file generation
base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/' # {1, 5, 7, 9}
gt_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/a123/' # GT

# List of best subjects
best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33'] # 5
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-30','sub-33','sub-34'] # 7
# best_subjects_cc = ['sub-02','sub-03','sub-08','sub-09','sub-20','sub-29','sub-30','sub-33','sub-34'] # 9

# List of remaining subjects
all_subjects = list()
for i in range(35):
    all_subjects.append('sub-'+str(i+1).zfill(2))
rest_subjects = [elem for elem in all_subjects if elem not in best_subjects_cc]

# Save values in an array
# All labels
val_dsc = np.zeros(len(rest_subjects))
# val_hau = np.zeros(len(rest_subjects))
val_hau_avg = np.zeros(len(rest_subjects))
val_vol = np.zeros(len(rest_subjects))
val_ndsc = np.zeros(len(rest_subjects))
val_size = np.zeros(len(rest_subjects))
# Lens
val_dsc_lens = np.zeros(len(rest_subjects))
val_hau_avg_lens = np.zeros(len(rest_subjects))
val_vol_lens = np.zeros(len(rest_subjects))
val_ndsc_lens = np.zeros(len(rest_subjects))
val_size_lens = np.zeros(len(rest_subjects))
# Globe
val_dsc_globe = np.zeros(len(rest_subjects))
val_hau_avg_globe = np.zeros(len(rest_subjects))
val_vol_globe = np.zeros(len(rest_subjects))
val_ndsc_globe = np.zeros(len(rest_subjects))
val_size_globe = np.zeros(len(rest_subjects))
# Optic nerve
val_dsc_nerve = np.zeros(len(rest_subjects))
val_hau_avg_nerve = np.zeros(len(rest_subjects))
val_vol_nerve = np.zeros(len(rest_subjects))
val_ndsc_nerve = np.zeros(len(rest_subjects))
val_size_nerve = np.zeros(len(rest_subjects))
# Intraconal fat
val_dsc_int_fat = np.zeros(len(rest_subjects))
val_hau_avg_int_fat = np.zeros(len(rest_subjects))
val_vol_int_fat = np.zeros(len(rest_subjects))
val_ndsc_int_fat = np.zeros(len(rest_subjects))
val_size_int_fat = np.zeros(len(rest_subjects))
# Extraconal fat
val_dsc_ext_fat = np.zeros(len(rest_subjects))
val_hau_avg_ext_fat = np.zeros(len(rest_subjects))
val_vol_ext_fat = np.zeros(len(rest_subjects))
val_ndsc_ext_fat = np.zeros(len(rest_subjects))
val_size_ext_fat = np.zeros(len(rest_subjects))
# Lateral rectus muscle
val_dsc_lat_mus = np.zeros(len(rest_subjects))
val_hau_avg_lat_mus = np.zeros(len(rest_subjects))
val_vol_lat_mus = np.zeros(len(rest_subjects))
val_ndsc_lat_mus = np.zeros(len(rest_subjects))
val_size_lat_mus = np.zeros(len(rest_subjects))
# Medial rectus muscle
val_dsc_med_mus = np.zeros(len(rest_subjects))
val_hau_avg_med_mus = np.zeros(len(rest_subjects))
val_vol_med_mus = np.zeros(len(rest_subjects))
val_ndsc_med_mus = np.zeros(len(rest_subjects))
val_size_med_mus = np.zeros(len(rest_subjects))
# Inferior rectus muscle
val_dsc_inf_mus = np.zeros(len(rest_subjects))
val_hau_avg_inf_mus = np.zeros(len(rest_subjects))
val_vol_inf_mus = np.zeros(len(rest_subjects))
val_ndsc_inf_mus = np.zeros(len(rest_subjects))
val_size_inf_mus = np.zeros(len(rest_subjects))
# Superior rectus muscle
val_dsc_sup_mus = np.zeros(len(rest_subjects))
val_hau_avg_sup_mus = np.zeros(len(rest_subjects))
val_vol_sup_mus = np.zeros(len(rest_subjects))
val_ndsc_sup_mus = np.zeros(len(rest_subjects))
val_size_sup_mus = np.zeros(len(rest_subjects))
### Grouped labels ###
# # Fats
# val_dsc_fats = np.zeros(len(rest_subjects))
# val_hau_avg_fats = np.zeros(len(rest_subjects))
# val_vol_fats = np.zeros(len(rest_subjects))
# val_ndsc_fats = np.zeros(len(rest_subjects))
# # Muscles
# val_dsc_muscles = np.zeros(len(rest_subjects))
# val_hau_avg_muscles = np.zeros(len(rest_subjects))
# val_vol_muscles = np.zeros(len(rest_subjects))
# val_ndsc_muscles = np.zeros(len(rest_subjects))
    
reader = sitk.ImageFileReader()

for i in range(len(rest_subjects)):

    # Prediction image to compare to GT
    pr_path = base_dir + 'reg_cropped_other_subjects/' + rest_subjects[i] + '_reg_cropped/labels2subject3.nii.gz' # Labels' image to compare to GT
    reader.SetFileName(pr_path)
    pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
    pr_arr = sitk.GetArrayFromImage(pr_sitk) # in numpy format
    pr_size = pr_arr.shape[0]*pr_arr.shape[1]*pr_arr.shape[2]

    # Ground truth
    gt_path = gt_dir + rest_subjects[i] + '/input/' + rest_subjects[i] + '_labels_cropped.nii.gz' # GT
    reader.SetFileName(gt_path)
    gt_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
    gt_arr = sitk.GetArrayFromImage(gt_sitk) # en numpy format

    # LENS
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==1, pr_sitk==1)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_lens[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_lens[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk, pr_sitk)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_lens[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==1, pr_arr==1)
    val_ndsc_lens[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==1) / pr_size
    val_size_lens[i] = size
    
    # GLOBE EX LENS
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==2, pr_sitk==2)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_globe[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_globe[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk, pr_sitk)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_globe[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==2, pr_arr==2)
    val_ndsc_globe[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==2) / pr_size
    val_size_globe[i] = size

    # OPTIC NERVE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==3, pr_sitk==3)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_nerve[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_nerve[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk, pr_sitk)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_nerve[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==3, pr_arr==3)
    val_ndsc_nerve[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==3) / pr_size
    val_size_nerve[i] = size

    # INTRACONAL FAT
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==4, pr_sitk==4)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_int_fat[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_int_fat[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==4, pr_sitk==4)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_int_fat[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==4, pr_arr==4)
    val_ndsc_int_fat[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==4) / pr_size
    val_size_int_fat[i] = size

    # EXTRACONAL FAT
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==5, pr_sitk==5)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_ext_fat[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_ext_fat[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==5, pr_sitk==5)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_ext_fat[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==5, pr_arr==5)
    val_ndsc_ext_fat[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==5) / pr_size
    val_size_ext_fat[i] = size

    # LATERAL RECTUS MUSCLE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==6, pr_sitk==6)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_lat_mus[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_lat_mus[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==6, pr_sitk==6)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_lat_mus[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==6, pr_arr==6)
    val_ndsc_lat_mus[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==6) / pr_size
    val_size_lat_mus[i] = size

    # MEDIAL RECTUS MUSCLE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==7, pr_sitk==7)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_med_mus[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_med_mus[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==7, pr_sitk==7)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_med_mus[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==7, pr_arr==7)
    val_ndsc_med_mus[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==7) / pr_size
    val_size_med_mus[i] = size

    # INFERIOR RECTUS MUSCLE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==8, pr_sitk==8)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_inf_mus[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_inf_mus[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==8, pr_sitk==8)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_inf_mus[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==8, pr_arr==8)
    val_ndsc_inf_mus[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==8) / pr_size
    val_size_inf_mus[i] = size

    # SUPERIOR RECTUS MUSCLE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==9, pr_sitk==9)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_sup_mus[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_sup_mus[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==9, pr_sitk==9)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_sup_mus[i] = hausdorf_distance_avg
    # nDSCvol
    nDSC = dice_norm_metric(gt_arr==9, pr_arr==9)
    val_ndsc_sup_mus[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr==9) / pr_size
    val_size_sup_mus[i] = size

    # ALL LABELS
    # DSC
    dsc = (val_dsc_lens[i]+val_dsc_globe[i]+val_dsc_nerve[i]+val_dsc_int_fat[i]+val_dsc_ext_fat[i]+val_dsc_lat_mus[i]+val_dsc_med_mus[i]+val_dsc_inf_mus[i]+val_dsc_sup_mus[i])/9
    val_dsc[i] = dsc
    # Volume
    vol = (val_vol_lens[i]+val_vol_globe[i]+val_vol_nerve[i]+val_vol_int_fat[i]+val_vol_ext_fat[i]+val_vol_lat_mus[i]+val_vol_med_mus[i]+val_vol_inf_mus[i]+val_vol_sup_mus[i])/9
    val_vol[i] = vol
    # Hausdorff distance
    hau_avg = (val_hau_avg_lens[i]+val_hau_avg_globe[i]+val_hau_avg_nerve[i]+val_hau_avg_int_fat[i]+val_hau_avg_ext_fat[i]+val_hau_avg_lat_mus[i]+val_hau_avg_med_mus[i]+val_hau_avg_inf_mus[i]+val_hau_avg_sup_mus[i])/9
    val_hau_avg[i] = hau_avg
    # nDSC
    nDSC = (val_ndsc_lens[i]+val_ndsc_globe[i]+val_ndsc_nerve[i]+val_ndsc_int_fat[i]+val_ndsc_ext_fat[i]+val_ndsc_lat_mus[i]+val_ndsc_med_mus[i]+val_ndsc_inf_mus[i]+val_ndsc_sup_mus[i])/9
    val_ndsc[i] = nDSC
    # Volume structure
    size = np.count_nonzero(pr_arr) / pr_size
    val_size[i] = size

    ### GROUPED LABELS ###
    # # FATS
    # # Measures Image Filter 
    # overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    # overlap_measures_filter.Execute(gt_sitk==4 or gt_sitk==5, pr_sitk==4 or pr_sitk==5)
    # # DSC
    # dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    # val_dsc_fats[i] = dsc
    # # Volume
    # vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    # val_vol_fats[i] = vol
    # # Hausdorff distance
    # hausdorf = sitk.HausdorffDistanceImageFilter()
    # hausdorf.Execute(gt_sitk==4 or gt_sitk==5, pr_sitk==4 or pr_sitk==5)
    # hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    # val_hau_avg_fats[i] = hausdorf_distance_avg
    # # nDSC
    # gt_mask = np.logical_or(gt_arr==4, gt_arr==5)
    # pr_mask = np.logical_or(pr_arr==4, pr_arr==5)
    # nDSC = dice_norm_metric(gt_mask, pr_mask)
    # val_ndsc_fats[i] = nDSC

    # # MUSCLES
    # # Measures Image Filter 
    # overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    # overlap_measures_filter.Execute(gt_sitk>=6 or gt_sitk<=9, pr_sitk>=6 or pr_sitk<=9)
    # # DSC
    # dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    # val_dsc_muscles[i] = dsc
    # # Volume
    # vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    # val_vol_muscles[i] = vol
    # # Hausdorff distance
    # hausdorf = sitk.HausdorffDistanceImageFilter()
    # hausdorf.Execute(gt_sitk>=6 or gt_sitk<=9, pr_sitk>=6 or pr_sitk<=9)
    # hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    # val_hau_avg_muscles[i] = hausdorf_distance_avg
    # # nDSC
    # gt_mask = np.logical_or.reduce((gt_arr==6, gt_arr==7, gt_arr==8, gt_arr==9))
    # pr_mask = np.logical_or.reduce((pr_arr==6, pr_arr==7, pr_arr==8, pr_arr==9))
    # nDSC = dice_norm_metric(gt_mask, pr_mask)
    # val_ndsc_muscles[i] = nDSC

# Save values to a csv
metrics = ['Subject', 'DSC_all', 'Haus_avg_all', 'Volume_all', 'nDSC_all', 'Size_all', 
            'DSC_lens', 'Haus_avg_lens', 'Volume_lens', 'nDSC_lens', 'Size_lens',
            'DSC_globe', 'Haus_avg_globe', 'Volume_globe', 'nDSC_globe', 'Size_globe',
            'DSC_nerve', 'Haus_avg_nerve', 'Volume_nerve', 'nDSC_nerve', 'Size_nerve',
            'DSC_int_fat', 'Haus_avg_int_fat', 'Volume_int_fat', 'nDSC_int_fat', 'Size_int_fat',
            'DSC_ext_fat', 'Haus_avg_ext_fat', 'Volume_ext_fat', 'nDSC_ext_fat', 'Size_ext_fat',
            'DSC_lat_mus', 'Haus_avg_lat_mus', 'Volume_lat_mus', 'nDSC_lat_mus', 'Size_lat_mus',
            'DSC_med_mus', 'Haus_avg_med_mus', 'Volume_med_mus', 'nDSC_med_mus', 'Size_med_mus',
            'DSC_inf_mus', 'Haus_avg_inf_mus', 'Volume_inf_mus', 'nDSC_inf_mus', 'Size_inf_mus',
            'DSC_sup_mus', 'Haus_avg_sup_mus', 'Volume_sup_mus', 'nDSC_sup_mus', 'Size_sup_mus']
vals = np.array([rest_subjects, val_dsc, val_hau_avg, val_vol, val_ndsc, val_size,
                val_dsc_lens, val_hau_avg_lens, val_vol_lens, val_ndsc_lens, val_size_lens,
                val_dsc_globe, val_hau_avg_globe, val_vol_globe, val_ndsc_globe, val_size_globe,
                val_dsc_nerve, val_hau_avg_nerve, val_vol_nerve, val_ndsc_nerve, val_size_nerve,
                val_dsc_int_fat, val_hau_avg_int_fat, val_vol_int_fat, val_ndsc_int_fat, val_size_int_fat,
                val_dsc_ext_fat, val_hau_avg_ext_fat, val_vol_ext_fat, val_ndsc_ext_fat, val_size_ext_fat,
                val_dsc_lat_mus, val_hau_avg_lat_mus, val_vol_lat_mus, val_ndsc_lat_mus, val_size_lat_mus,
                val_dsc_med_mus, val_hau_avg_med_mus, val_vol_med_mus, val_ndsc_med_mus, val_size_med_mus,
                val_dsc_inf_mus, val_hau_avg_inf_mus, val_vol_inf_mus, val_ndsc_inf_mus, val_size_inf_mus,
                val_dsc_sup_mus, val_hau_avg_sup_mus, val_vol_sup_mus, val_ndsc_sup_mus, val_size_sup_mus])
vals = vals.T
# print(vals)
# print(f"type: {vals.dtype}, shape: {vals.shape}")

with open('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/sim_metrics_labels2subject3_N5.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(metrics)
    writer.writerows(vals)

# '''

# ''' Plot per metric
path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/'
filename = 'DSC_nDSC_VolSim_Haus_labels2subject3_N5.png'
df5 = pd.read_csv(path + 'sim_metrics_labels2subject3_N5.csv')

# # Dataframes {DSC, nDSC, Volume (voxels)} separate labels for N=5 only
data_dsc = [df5['DSC_all'], df5['DSC_lens'], df5['DSC_globe'], df5['DSC_nerve'], df5['DSC_int_fat'], df5['DSC_ext_fat'], df5['DSC_lat_mus'], df5['DSC_med_mus'], df5['DSC_inf_mus'], df5['DSC_sup_mus']]
data_ndsc = [df5['nDSC_all'],  df5['nDSC_lens'], df5['nDSC_globe'], df5['nDSC_nerve'], df5['nDSC_int_fat'], df5['nDSC_ext_fat'], df5['nDSC_lat_mus'], df5['nDSC_med_mus'], df5['nDSC_inf_mus'], df5['nDSC_sup_mus']]
data_vol = [df5['Volume_all'], df5['Volume_lens'], df5['Volume_globe'], df5['Volume_nerve'], df5['Volume_int_fat'], df5['Volume_ext_fat'], df5['Volume_lat_mus'], df5['Volume_med_mus'], df5['Volume_inf_mus'], df5['Volume_sup_mus']]
data_haus = [df5['Haus_avg_all'], df5['Haus_avg_lens'], df5['Haus_avg_globe'], df5['Haus_avg_nerve'], df5['Haus_avg_int_fat'], df5['Haus_avg_ext_fat'], df5['Haus_avg_lat_mus'], df5['Haus_avg_med_mus'], df5['Haus_avg_inf_mus'], df5['Haus_avg_sup_mus']]

labels = ['lens', 'globe', 'nerve', 'intraconal fat', 'extraconal fat', 'lateral rectus muscle', 'medial rectus muscle', 'inferior rectus muscle', 'superior rectus muscle']
median = [np.around(np.median(x), 2) for x in data_dsc]
for i in range(len(labels)):
    print(labels[i], median[i])

# Figure 1
fig, axs = plt.subplots(4, figsize=(20,10), sharex=True)
fig.canvas.set_window_title('Similarity metrics N=5')
# fig.suptitle('Similarity metrics N=5')

# Boxplot & Swarmplot (points)
ax1 = sns.boxplot(data=data_dsc, ax=axs[0]).set(ylabel="Value")
ax1 = sns.swarmplot(data=data_dsc, ax=axs[0])
ax2 = sns.boxplot(data=data_ndsc, ax=axs[1]).set(ylabel="Value")
ax2 = sns.swarmplot(data=data_ndsc, ax=axs[1])
ax3 = sns.boxplot(data=data_vol, ax=axs[2]).set(ylabel="Value")
ax3 = sns.swarmplot(data=data_vol, ax=axs[2])
ax4 = sns.boxplot(data=data_haus, ax=axs[3]).set(ylabel="Value")
ax4 = sns.swarmplot(data=data_haus, ax=axs[3])

# Set labels and titles
ax1.set_xticklabels(['all','lens','globe','nerve','int_fat','ext_fat','lat_mus','med_mus','inf_mus','sup_mus'])
ax1.set_title('DSC')
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax2.set_title('nDSC')
ax2.set_yticks(np.arange(0, 1.1, 0.1))
ax3.set_title('Volume similarity')
ax3.set_yticks(np.arange(-2, 2.5, 0.5))
ax4.set_title('Hausdorff distance')
ax4.set_yticks(np.arange(0, 1.7, 0.2))

plt.show()

# Save figure
# plt.savefig(path + filename, bbox_inches='tight')

# '''

''' Spearman correlation plots and values

# Data per label
df5 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5_avgAll_nDSC_sizeprvsim_separate_labels.csv')

# Figure 2
fig, axs = plt.subplots(2, 5)
fig.suptitle('Spearman correlation N=5')

# Plots
ax0 = sns.scatterplot(y=df5['nDSC_all'], x=df5['Size_all'], ax=axs[0,0], legend='brief')
ax1 = sns.scatterplot(y=df5['nDSC_lens'], x=df5['Size_lens'], ax=axs[0,1], legend='brief')
ax2 = sns.scatterplot(y=df5['nDSC_globe'], x=df5['Size_globe'], ax=axs[0,2], legend='brief')
ax3 = sns.scatterplot(y=df5['nDSC_nerve'], x=df5['Size_nerve'], ax=axs[0,3], legend='brief')
ax4 = sns.scatterplot(y=df5['nDSC_int_fat'], x=df5['Size_int_fat'], ax=axs[0,4], legend='brief')
ax5 = sns.scatterplot(y=df5['nDSC_ext_fat'], x=df5['Size_ext_fat'], ax=axs[1,0], legend='brief')
ax6 = sns.scatterplot(y=df5['nDSC_lat_mus'], x=df5['Size_lat_mus'], ax=axs[1,1], legend='brief')
ax7 = sns.scatterplot(y=df5['nDSC_med_mus'], x=df5['Size_med_mus'], ax=axs[1,2], legend='brief')
ax8 = sns.scatterplot(y=df5['nDSC_inf_mus'], x=df5['Size_inf_mus'], ax=axs[1,3], legend='brief')
ax9 = sns.scatterplot(y=df5['nDSC_sup_mus'], x=df5['Size_sup_mus'], ax=axs[1,4], legend='brief')

# Spearman correlation coefficients per label
scc0 = round(stats.spearmanr(df5['nDSC_all'],df5['Size_all'])[0],4)
scc1 = round(stats.spearmanr(df5['nDSC_lens'],df5['Size_lens'])[0],4)
scc2 = round(stats.spearmanr(df5['nDSC_globe'],df5['Size_globe'])[0],4)
scc3 = round(stats.spearmanr(df5['nDSC_nerve'],df5['Size_nerve'])[0],4)
scc4 = round(stats.spearmanr(df5['nDSC_int_fat'],df5['Size_int_fat'])[0],4)
scc5 = round(stats.spearmanr(df5['nDSC_ext_fat'],df5['Size_ext_fat'])[0],4)
scc6 = round(stats.spearmanr(df5['nDSC_lat_mus'],df5['Size_lat_mus'])[0],4)
scc7 = round(stats.spearmanr(df5['nDSC_med_mus'],df5['Size_med_mus'])[0],4)
scc8 = round(stats.spearmanr(df5['nDSC_inf_mus'],df5['Size_inf_mus'])[0],4)
scc9 = round(stats.spearmanr(df5['nDSC_sup_mus'],df5['Size_sup_mus'])[0],4)

# Set labels and titles
ax0.set_title('Spearman='+str(scc0))
ax1.set_title('Spearman='+str(scc1))
ax2.set_title('Spearman='+str(scc2))
ax3.set_title('Spearman='+str(scc3))
ax4.set_title('Spearman='+str(scc4))
ax5.set_title('Spearman='+str(scc5))
ax6.set_title('Spearman='+str(scc6))
ax7.set_title('Spearman='+str(scc7))
ax8.set_title('Spearman='+str(scc8))
ax9.set_title('Spearman='+str(scc9))

plt.show()

# '''