import nibabel as nb
import numpy as np
import SimpleITK as sitk
import os, glob
from pathlib import Path

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/'

# List of best subjects to do the registration
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33'] # 5
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-30','sub-33','sub-34'] # 7
# best_subjects_cc = ['sub-02','sub-03','sub-08','sub-09','sub-20','sub-29','sub-30','sub-33','sub-34'] # 9
rest_subjects = ['sub-08','sub-09','sub-30','sub-34']

# List of remaining subjects
# all_subjects = list()
# for i in range(35):
#     all_subjects.append('sub-'+str(i+1).zfill(2))
# rest_subjects = [elem for elem in all_subjects if elem not in best_subjects_cc]

labels = ['lens','globe','nerve','intfat','extfat','latmus','medmus','infmus','supmus']

# ''' Loop of subjects
for i in range(0, len(rest_subjects)):
# for i in range(0,1):

    # Header from labels
    segments = []
    for l in range(0, len(labels)):
        subject_labels_path = base_dir + 'reg_cropped_other_subjects/' + rest_subjects[i] + '_reg_cropped/Per_Class_2/' + labels[l] + '2subject.nii.gz'
        # print(subject_labels_path)
        sub_lab = nb.load(Path(subject_labels_path))
        segments.append(sub_lab)
    header = segments[0].header.copy()
    header.set_data_dtype('uint8')

    # Array of transformed probabilities (to subject space)
    prob_arr = np.zeros(len(segments)) # 9 classes

    # Resulting image initialization
    result_im = np.zeros_like(segments[0].dataobj, dtype='uint8')

    # Coordinates loop
    for x in range(0, result_im.shape[0]):
        for y in range(0, result_im.shape[1]):
            for z in range(0, result_im.shape[2]):

                # Filling the array of transformed probabilities 
                for j in range(len(prob_arr)):
                    prob_arr[j] = segments[j].get_fdata()[x,y,z]
                # if prob_arr[0] > 0 : print(f'there is lens!')

                # Maximum a posteriori only in cases > 0
                if np.any(prob_arr):
                    result_im[x,y,z] = np.argmax(prob_arr)+1
    
    # Transform matrix to image and save file
    nii = nb.Nifti1Image(result_im, segments[0].affine, header)
    nii.to_filename(base_dir + 'reg_cropped_other_subjects/' + rest_subjects[i] + '_reg_cropped/labels2subject4.nii.gz')
