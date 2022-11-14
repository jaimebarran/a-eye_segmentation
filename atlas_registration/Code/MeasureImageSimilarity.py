import os
import subprocess
import numpy as np
from numpy import genfromtxt
import SimpleITK as sitk
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/'
output_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/Output/Metrics/'
colin_path = '/mnt/sda1/Repos/a-eye/Data/templateflow/colin27/tpl-MNIColin27_T1w.nii.gz'

best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33']

metric_weight = 1
radius = 10
sampling_strategy = 'None' # None (full sampling), random, regular
sampling_percentage = 1 # Must be [0,1]

output_file_cc = output_dir + 'CC_N5.txt'
output_file_mi = output_dir + 'MI_N5.txt'
output_file_demons = output_dir + 'Demons_cropped.txt'

''' Loop registration results vs Colin template
# i = 0
for folder1 in os.listdir(base_dir):
    # print(folder1)    
    # reg_result = base_dir + folder1 + '/output_colin27/BrainExtractionNormalized.nii.gz'
    colin_path = base_dir + folder1 + '/input/tpl-MNIColin27_T1w_cropped.nii.gz'
    reg_result = base_dir + folder1 + '/output_colin27_cropped/Warped.nii.gz'
    reader.SetFileName(reg_result)
    pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)

    # ANTS - Measure Image Similarity

    # Cross correlation
    command = 'MeasureImageSimilarity -d 3' + \
    ' -m ' + 'CC["' + colin_path + '", "' + reg_result  + \
    '", ' + str(metric_weight) + ', ' + str(radius) + \
    ', ' + str(sampling_strategy) + ', ' + str(sampling_percentage) + ']'  + \
    ' >> ' + output_file_cc # + \
    # ' -v 1'
    # print(command)
    output_system = os.system(command)

    # Mutual Information
    command = 'MeasureImageSimilarity -d 3' + \
    ' -m ' + 'MI["' + colin_path + '", "' + reg_result  + \
    '", ' + str(metric_weight) + ', ' + str(radius) + \
    ', ' + str(sampling_strategy) + ', ' + str(sampling_percentage) + ']'  + \
    ' >> ' + output_file_mi # + \
    # ' -v 1'
    # print(command)
    output_system = os.system(command)

    # Demons
    # command = 'MeasureImageSimilarity -d 3' + \
    # ' -m ' + 'Demons["' + colin_path + '", "' + reg_result  + \
    # '", ' + str(metric_weight) + ', ' + str(radius) + \
    # ', ' + str(sampling_strategy) + ', ' + str(sampling_percentage) + ']'  + \
    # ' >> ' + output_file_demons # + \
    # # ' -v 1'
    # print(command)
    # output_system = os.system(command)

    # Command as list (not working)
    # metric_string = 'CC["' + colin_path + '", "' + reg_result + '", ' + str(metric_weight) + ', ' + str(radius) + ', ' + str(sampling_strategy) + ', ' + str(sampling_percentage) + ']'
    # print(metric_string)
    # output_subprocess = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
    # print(output[i])
    # i += 1
'''

''' Loop registration results vs custom template
for i in range(len(best_subjects_cc)):
    # print(folder1)
    # reg_result = base_dir + folder1 + '/output_colin27/BrainExtractionNormalized.nii.gz'
    # colin_path = base_dir + folder1 + '/input/tpl-MNIColin27_T1w_cropped.nii.gz'
    reg_result = base_dir + 'best_subjects_eye_cc/CustomTemplate_5_n0/' + best_subjects_cc[i] + '_reg_croppedWarped.nii.gz'
    template_cc_5 = base_dir + 'best_subjects_eye_cc/CustomTemplate_5_n0/' + best_subjects_cc[i] + '_template_cropped.nii.gz'

    # ANTS - Measure Image Similarity

    # Cross correlation
    command = 'MeasureImageSimilarity -d 3' + \
    ' -m ' + 'CC["' + template_cc_5 + '", "' + reg_result  + \
    '", ' + str(metric_weight) + ', ' + str(radius) + \
    ', ' + str(sampling_strategy) + ', ' + str(sampling_percentage) + ']'  + \
    ' >> ' + output_file_cc # + \
    # ' -v 1'
    print(command)
    output_system = os.system(command)
'''

# ''' Get the n higher scores
num_best_subjects = 5
sub_txt = output_dir + 'subjects.txt'
arr_sub = genfromtxt(sub_txt, dtype=str)
# print(arr_sub)

# CC
cc_txt = output_dir + 'CC_cropped.txt' # CC, MI, Demons
arr_cc = genfromtxt(cc_txt)
# print(arr_cc)
arr_cc_abs = np.abs(arr_cc)
# Get the n best subjects
arr5_cc = np.sort(arr_cc_abs)[::1][:num_best_subjects] # nth higher values
print(f'CC: {arr5_cc}')
arr_ind = np.argsort(arr_cc_abs)[::1][:num_best_subjects] # Indexes of first n higher values
# print(arr_ind)
arr5_sub = arr_sub[arr_ind]
print(f'CC: {arr5_sub}')

# MI
mi_txt = output_dir + 'MI_cropped.txt'
arr_mi = genfromtxt(mi_txt)
# print(arr_mi)
arr_mi_abs = np.abs(arr_mi)
# Get the n best subjects
arr5_mi = np.sort(arr_mi_abs)[::1][:num_best_subjects] # nth higher values
print(f'MI: {arr5_mi}')
arr_ind = np.argsort(arr_mi_abs)[::1][:num_best_subjects] # Indexes of first n higher values
# print(arr_ind)
arr5_sub = arr_sub[arr_ind]
print(f'MI: {arr5_sub}')

# Demons
# demons_txt = output_dir + 'Demons.txt'
# arr_demons = genfromtxt(demons_txt)
# # print(arr_demons)
# arr5_demons = np.sort(arr_demons)[::-1][:num_best_subjects] # num_besth higher values
# print(f'Demons: {arr5_demons}')
# arr_ind = np.argsort(arr_demons)[::-1][:num_best_subjects] # Indexes of first n higher values
# # print(arr_ind)
# arr5_sub = arr_sub[arr_ind]
# print(f'Demons: {arr5_sub}')
# '''

''' PLOT
# metrics = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/Output/Metrics/metrics.csv')
metrics = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/Output/Metrics/metrics_custom_template.csv')
ax = sns.boxplot(data=metrics).set(xlabel="Metric", ylabel="Value")
ax = sns.swarmplot(data=metrics)
plt.show()

'''