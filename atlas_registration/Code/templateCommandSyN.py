import ants
import os
import glob
import numpy as np

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/a123/'
# input_im_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/'
input_im_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/worst_subjects_eye_cc/'
output_dir = input_im_dir + 'CustomTemplate_5_n0/'

# '''
# best_subjects_cc = ['sub-29','sub-20','sub-33','sub-03','sub-02','sub-34','sub-30','sub-08','sub-09']
best_subjects_cc = ['sub-21','sub-26','sub-15','sub-23','sub-19']
best_subjects_mi = ['sub-20','sub-33','sub-09','sub-32','sub-29']

# Create output directories
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(input_im_dir):
    os.makedirs(input_im_dir)

'''
# List of paths to best subjects
subs = [] # list
for i in range(len(best_subjects_cc)):
    # print(i)
    subs.append(base_dir + best_subjects_cc[i] + '/input/' + best_subjects_cc[i] + '_T1_aff.nii.gz')
    # print(subs[i])
# print(subs)

# Copy best subjects images into separate folder for the algorithm in bash to run
for i in range(len(subs)):
    command = 'cp ' + subs[i] + ' ' + input_im_dir
    # print(command)
    # os.system(command)

# List of images read by ants
# population = list()
# for i in range(len(subs)):
#     population.append(ants.image_read(subs[i], dimension=3))
#     # print(population[i])

# By ANTs function (working but not saving anything and not so customizable as bash)
# btp = ants.build_template(
#     initialTemplate = None,
#     image_list = population,
#     iterations = 4,
#     gradient_step = 0.2,
#     verbose = True,
#     syn_metric = 'CC',
#     reg_iterations = (100, 70, 50, 0) 
# )

# ants.plot(btp, filename=output_dir+'custom_template.nii.gz')
'''

# '''
# v2
command = 'antsMultivariateTemplateConstruction2.sh -d 3' + \
    ' -o ' + output_dir + \
    ' -i ' + '4' + \
    ' -g ' + '0.2' + \
    ' -j ' + '12' + \
    ' -c ' + '2' + \
    ' -k ' + '1' + \
    ' -w ' + '1' + \
    ' -n ' + '0' + \
    ' -r ' + '1' + \
    ' -m ' + 'CC[2]' + \
    ' -q ' + '100x70x50x20' + \
    ' -f ' + '8x4x2x1' + \
    ' -s ' + '3x2x1x0' + \
    ' -t ' + 'SyN' + \
    ' ' + input_im_dir + '*.nii.gz'
print(command)
os.system(command)

# v1
# command = 'antsMultivariateTemplateConstruction.sh -d 3' + \
#     ' -o ' + output_dir + 'template_' + \
#     ' -i ' + '3' + \
#     ' -g ' + '0.2' + \
#     ' -j ' + '12' + \
#     ' -c ' + '2' + \
#     ' -k ' + '1' + \
#     ' -w ' + '1' + \
#     ' -m ' + '100x50x10' + \
#     ' -n ' + '0' + \
#     ' -r ' + '1' + \
#     ' -s ' + 'CC' + \
#     ' -t ' + 'GR' + \
#     + ' ' + input_im_dir + '*.nii.gz'
# print(command)
# os.system(command)
# '''