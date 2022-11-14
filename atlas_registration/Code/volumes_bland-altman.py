from asyncore import write
from turtle import title
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import Line2D
from sqlalchemy import true
from scipy import stats
import statsmodels.api as sm
import pyCompare as pc
import nibabel as nb

# ''' Data frame file generation

pr_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/' # {1, 5, 7, 9}
gt_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/a123/'
filename = 'volumes_bland-altman_Zscore_labels2subject4_N5.csv'

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
val_vol_pr_all = np.zeros(len(rest_subjects))
val_vol_gt_all = np.zeros(len(rest_subjects))
# Lens
val_vol_pr_lens = np.zeros(len(rest_subjects))
val_vol_gt_lens = np.zeros(len(rest_subjects))
# Globe
val_vol_pr_globe = np.zeros(len(rest_subjects))
val_vol_gt_globe = np.zeros(len(rest_subjects))
# Optic nerve
val_vol_pr_nerve = np.zeros(len(rest_subjects))
val_vol_gt_nerve = np.zeros(len(rest_subjects))
# Intraconal fat
val_vol_pr_int_fat = np.zeros(len(rest_subjects))
val_vol_gt_int_fat = np.zeros(len(rest_subjects))
# Extraconal fat
val_vol_pr_ext_fat = np.zeros(len(rest_subjects))
val_vol_gt_ext_fat = np.zeros(len(rest_subjects))
# Lateral rectus muscle
val_vol_pr_lat_mus = np.zeros(len(rest_subjects))
val_vol_gt_lat_mus = np.zeros(len(rest_subjects))
# Medial rectus muscle
val_vol_pr_med_mus = np.zeros(len(rest_subjects))
val_vol_gt_med_mus = np.zeros(len(rest_subjects))
# Inferior rectus muscle
val_vol_pr_inf_mus = np.zeros(len(rest_subjects))
val_vol_gt_inf_mus = np.zeros(len(rest_subjects))
# Superior rectus muscle
val_vol_pr_sup_mus = np.zeros(len(rest_subjects))
val_vol_gt_sup_mus = np.zeros(len(rest_subjects))

reader = sitk.ImageFileReader()

for i in range(len(rest_subjects)):
    
    # Prediction image
    pr_path = pr_dir + 'reg_cropped_other_subjects/' + rest_subjects[i] + '_reg_cropped/labels2subject4.nii.gz'
    reader.SetFileName(pr_path)
    pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
    pr_arr = sitk.GetArrayFromImage(pr_sitk)
    # pr_arr = nb.load(pr_path).get_fdata()
    # pr_size = pr_arr.shape[0] * pr_arr.shape[1] * pr_arr.shape[2] # pr_size == gt_size == image_size

    # Ground truth
    gt_path = gt_dir + rest_subjects[i] + '/input/' + rest_subjects[i] + '_labels_cropped.nii.gz' # GT
    reader.SetFileName(gt_path)
    gt_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
    gt_arr = sitk.GetArrayFromImage(gt_sitk) # en numpy format
    # gt_arr = nb.load(gt_path).get_fdata()
    gt_size = gt_arr.shape[0] * gt_arr.shape[1] * gt_arr.shape[2] # pr_size == gt_size == image_size

    # LENS
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==1)
    val_vol_pr_lens[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==1)
    val_vol_gt_lens[i] = vol_gt

    # GLOBE EX LENS
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==2)
    val_vol_pr_globe[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==2)
    val_vol_gt_globe[i] = vol_gt

    # OPTIC NERVE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==3)
    val_vol_pr_nerve[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==3)
    val_vol_gt_nerve[i] = vol_gt

    # INTRACONAL FAT
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==4)
    val_vol_pr_int_fat[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==4)
    val_vol_gt_int_fat[i] = vol_gt

    # EXTRACONAL FAT
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==5)
    val_vol_pr_ext_fat[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==5)
    val_vol_gt_ext_fat[i] = vol_gt

    # LATERAL RECTUS MUSCLE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==6)
    val_vol_pr_lat_mus[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==6)
    val_vol_gt_lat_mus[i] = vol_gt

    # MEDIAL RECTUS MUSCLE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==7)
    val_vol_pr_med_mus[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==7)
    val_vol_gt_med_mus[i] = vol_gt

    # INFERIOR RECTUS MUSCLE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==8)
    val_vol_pr_inf_mus[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==8)
    val_vol_gt_inf_mus[i] = vol_gt

    # SUPERIOR RECTUS MUSCLE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==9)
    val_vol_pr_sup_mus[i] = vol_pr
    vol_gt = np.count_nonzero(gt_arr==9)
    val_vol_gt_sup_mus[i] = vol_gt

    # ALL LABELS
    # Volume structure
    val_vol_pr_all[i] = (val_vol_pr_lens[i] + val_vol_pr_globe[i] + val_vol_pr_nerve[i] + val_vol_pr_int_fat[i] + val_vol_pr_ext_fat[i] +
                            val_vol_pr_lat_mus[i] + val_vol_pr_med_mus[i] + val_vol_pr_inf_mus[i] + val_vol_pr_sup_mus[i]) / 9
    val_vol_gt_all[i] = (val_vol_gt_lens[i] + val_vol_gt_globe[i] + val_vol_gt_nerve[i] + val_vol_gt_int_fat[i] + val_vol_gt_ext_fat[i] +
                            val_vol_gt_lat_mus[i] + val_vol_gt_med_mus[i] + val_vol_gt_inf_mus[i] + val_vol_gt_sup_mus[i]) / 9

# Z-score
# All labels
gt_all_mean = np.mean(val_vol_gt_all)
gt_all_std = np.std(val_vol_gt_all)
val_vol_gt_all_z = (val_vol_gt_all - gt_all_mean) / gt_all_std
pr_all_mean = np.mean(val_vol_pr_all)
pr_all_std = np.std(val_vol_pr_all)
val_vol_pr_all_z = (val_vol_pr_all - pr_all_mean) / pr_all_std
# Lens
gt_lens_mean = np.mean(val_vol_gt_lens)
gt_lens_std = np.std(val_vol_gt_lens)
val_vol_gt_lens_z = (val_vol_gt_lens - gt_lens_mean) / gt_lens_std
pr_lens_mean = np.mean(val_vol_pr_lens)
pr_lens_std = np.std(val_vol_pr_lens)
val_vol_pr_lens_z = (val_vol_pr_lens - pr_lens_mean) / pr_lens_std
# Globe
gt_globe_mean = np.mean(val_vol_gt_globe)
gt_globe_std = np.std(val_vol_gt_globe)
val_vol_gt_globe_z = (val_vol_gt_globe - gt_globe_mean) / gt_globe_std
pr_globe_mean = np.mean(val_vol_pr_globe)
pr_globe_std = np.std(val_vol_pr_globe)
val_vol_pr_globe_z = (val_vol_pr_globe - pr_globe_mean) / pr_globe_std
# Optic nerve
gt_nerve_mean = np.mean(val_vol_gt_nerve)
gt_nerve_std = np.std(val_vol_gt_nerve)
val_vol_gt_nerve_z = (val_vol_gt_nerve - gt_nerve_mean) / gt_nerve_std
pr_nerve_mean = np.mean(val_vol_pr_nerve)
pr_nerve_std = np.std(val_vol_pr_nerve)
val_vol_pr_nerve_z = (val_vol_pr_nerve - pr_nerve_mean) / pr_nerve_std
# Intraconal fat
gt_int_fat_mean = np.mean(val_vol_gt_int_fat)
gt_int_fat_std = np.std(val_vol_gt_int_fat)
val_vol_gt_int_fat_z = (val_vol_gt_int_fat - gt_int_fat_mean) / gt_int_fat_std
pr_int_fat_mean = np.mean(val_vol_pr_int_fat)
pr_int_fat_std = np.std(val_vol_pr_int_fat)
val_vol_pr_int_fat_z = (val_vol_pr_int_fat - pr_int_fat_mean) / pr_int_fat_std
# Extraconal fat
gt_ext_fat_mean = np.mean(val_vol_gt_ext_fat)
gt_ext_fat_std = np.std(val_vol_gt_ext_fat)
val_vol_gt_ext_fat_z = (val_vol_gt_ext_fat - gt_ext_fat_mean) / gt_ext_fat_std
pr_ext_fat_mean = np.mean(val_vol_pr_ext_fat)
pr_ext_fat_std = np.std(val_vol_pr_ext_fat)
val_vol_pr_ext_fat_z = (val_vol_pr_ext_fat - pr_ext_fat_mean) / pr_ext_fat_std
# Lateral rectus muscle
gt_lat_mus_mean = np.mean(val_vol_gt_lat_mus)
gt_lat_mus_std = np.std(val_vol_gt_lat_mus)
val_vol_gt_lat_mus_z = (val_vol_gt_lat_mus - gt_lat_mus_mean) / gt_lat_mus_std
pr_lat_mus_mean = np.mean(val_vol_pr_lat_mus)
pr_lat_mus_std = np.std(val_vol_pr_lat_mus)
val_vol_pr_lat_mus_z = (val_vol_pr_lat_mus - pr_lat_mus_mean) / pr_lat_mus_std
# Medial rectus muscle
gt_med_mus_mean = np.mean(val_vol_gt_med_mus)
gt_med_mus_std = np.std(val_vol_gt_med_mus)
val_vol_gt_med_mus_z = (val_vol_gt_med_mus - gt_med_mus_mean) / gt_med_mus_std
pr_med_mus_mean = np.mean(val_vol_pr_med_mus)
pr_med_mus_std = np.std(val_vol_pr_med_mus)
val_vol_pr_med_mus_z = (val_vol_pr_med_mus - pr_med_mus_mean) / pr_med_mus_std
# Inferior rectus muscle
gt_inf_mus_mean = np.mean(val_vol_gt_inf_mus)
gt_inf_mus_std = np.std(val_vol_gt_inf_mus)
val_vol_gt_inf_mus_z = (val_vol_gt_inf_mus - gt_inf_mus_mean) / gt_inf_mus_std
pr_inf_mus_mean = np.mean(val_vol_pr_inf_mus)
pr_inf_mus_std = np.std(val_vol_pr_inf_mus)
val_vol_pr_inf_mus_z = (val_vol_pr_inf_mus - pr_inf_mus_mean) / pr_inf_mus_std
# Superior rectus muscle
gt_sup_mus_mean = np.mean(val_vol_gt_sup_mus)
gt_sup_mus_std = np.std(val_vol_gt_sup_mus)
val_vol_gt_sup_mus_z = (val_vol_gt_sup_mus - gt_sup_mus_mean) / gt_sup_mus_std
pr_sup_mus_mean = np.mean(val_vol_pr_sup_mus)
pr_sup_mus_std = np.std(val_vol_pr_sup_mus)
val_vol_pr_sup_mus_z = (val_vol_pr_sup_mus - pr_sup_mus_mean) / pr_sup_mus_std

# Save values to a csv
metrics = ['Subject','vol_pr_all','vol_gt_all','vol_pr_lens','vol_gt_lens','vol_pr_globe','vol_gt_globe','vol_pr_nerve','vol_gt_nerve',
            'vol_pr_int_fat','vol_gt_int_fat','vol_pr_ext_fat','vol_gt_ext_fat','vol_pr_lat_mus','vol_gt_lat_mus','vol_pr_med_mus','vol_gt_med_mus',
            'vol_pr_inf_mus','vol_gt_inf_mus','vol_pr_sup_mus','vol_gt_sup_mus']
vals = np.array([rest_subjects, val_vol_pr_all_z, val_vol_gt_all_z, val_vol_pr_lens_z, val_vol_gt_lens_z, val_vol_pr_globe_z, val_vol_gt_globe_z, val_vol_pr_nerve_z,
                val_vol_gt_nerve_z,val_vol_pr_int_fat_z, val_vol_gt_int_fat_z, val_vol_pr_ext_fat_z, val_vol_gt_ext_fat_z, val_vol_pr_lat_mus_z, val_vol_gt_lat_mus_z,
                val_vol_pr_med_mus_z, val_vol_gt_med_mus_z, val_vol_pr_inf_mus_z, val_vol_gt_inf_mus_z, val_vol_pr_sup_mus_z, val_vol_gt_sup_mus_z])
vals = vals.T

with open(pr_dir + filename, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(metrics)
    writer.writerows(vals)

# '''

# ''' Bland-Altman plot
path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/'
filename = 'Bland-Altman_labels2subject4_Zscore_shared_axis_N5.png'
df_vol = pd.read_csv(path + 'volumes_bland-altman_Zscore_labels2subject4_N5.csv')                                              

# Subplots
k = 1.09 # Figure size to preserve ratio 16:9
fig, ax = plt.subplots(2, 5, figsize=(16*k, 9*k))
fig.canvas.set_window_title('Volume difference - Bland-Altman plots')
# fig.suptitle('Volume difference')
fix_axis = True
x_axis = [-2, 2]
y_axis = [-2.5, 3.5]

# all labels
sm.graphics.mean_diff_plot(df_vol['vol_pr_all'], df_vol['vol_gt_all'], ax=ax[0][0])
ax[0][0].set_title('all')
if fix_axis:
    ax[0][0].set_xlim(x_axis)
    ax[0][0].set_ylim(y_axis)
# lens
sm.graphics.mean_diff_plot(df_vol['vol_pr_lens'], df_vol['vol_gt_lens'], ax=ax[0][1])
ax[0][1].set_title('lens')
if fix_axis:
    ax[0][1].set_xlim(x_axis)
    ax[0][1].set_ylim(y_axis)
# globe
sm.graphics.mean_diff_plot(df_vol['vol_pr_globe'], df_vol['vol_gt_globe'], ax=ax[0][2])
ax[0][2].set_title('globe')
if fix_axis:
    ax[0][2].set_xlim(x_axis)
    ax[0][2].set_ylim(y_axis)
# nerve
sm.graphics.mean_diff_plot(df_vol['vol_pr_nerve'], df_vol['vol_gt_nerve'], ax=ax[0][3])
ax[0][3].set_title('nerve')
if fix_axis:
    ax[0][3].set_xlim(x_axis)
    ax[0][3].set_ylim(y_axis)
# int fat
sm.graphics.mean_diff_plot(df_vol['vol_pr_int_fat'], df_vol['vol_gt_int_fat'], ax=ax[0][4])
ax[0][4].set_title('int fat')
if fix_axis:
    ax[0][4].set_xlim(x_axis)
    ax[0][4].set_ylim(y_axis)
# ext fat
sm.graphics.mean_diff_plot(df_vol['vol_pr_ext_fat'], df_vol['vol_gt_ext_fat'], ax=ax[1][0])
ax[1][0].set_title('ext fat')
if fix_axis:
    ax[1][0].set_xlim(x_axis)
    ax[1][0].set_ylim(y_axis)
# lat mus
sm.graphics.mean_diff_plot(df_vol['vol_pr_lat_mus'], df_vol['vol_gt_lat_mus'], ax=ax[1][1])
ax[1][1].set_title('lat mus')
if fix_axis:
    ax[1][1].set_xlim(x_axis)
    ax[1][1].set_ylim(y_axis)
# med mus
sm.graphics.mean_diff_plot(df_vol['vol_pr_med_mus'], df_vol['vol_gt_med_mus'], ax=ax[1][2])
ax[1][2].set_title('med mus')
if fix_axis:
    ax[1][2].set_xlim(x_axis)
    ax[1][2].set_ylim(y_axis)
# inf mus
sm.graphics.mean_diff_plot(df_vol['vol_pr_inf_mus'], df_vol['vol_gt_inf_mus'], ax=ax[1][3])
ax[1][3].set_title('inf mus')
if fix_axis:
    ax[1][3].set_xlim(x_axis)
    ax[1][3].set_ylim(y_axis)
# sup mus
sm.graphics.mean_diff_plot(df_vol['vol_pr_sup_mus'], df_vol['vol_gt_sup_mus'], ax=ax[1][4])
ax[1][4].set_title('sup mus')
if fix_axis:
    ax[1][4].set_xlim(x_axis)
    ax[1][4].set_ylim(y_axis)

# plt.tight_layout()

# Single plot
# all labels
# pc.blandAltman(df_vol['vol_gt_nerve'], df_vol['vol_pr_nerve'])
# sm.graphics.mean_diff_plot(df_vol['vol_gt_nerve'], df_vol['vol_pr_nerve'])

# plt.show()

# Save figure
plt.savefig(path + filename)

# '''