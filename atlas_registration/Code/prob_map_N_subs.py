from shutil import which
import nibabel as nb
import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
from scipy.stats import mode

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/'
output_dir = base_dir+'Probability_Maps/'
# Create output directories
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

template_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/template0.nii.gz'
mask_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/all_segments_mask.nii.gz'
# template_cropped_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/template0_cropped_15vox.nii.gz'
# mask_cropped_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/all_segments_mask_cropped.nii.gz'

best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33'] # 5
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-30','sub-33','sub-34'] # 7
# best_subjects_cc = ['sub-02','sub-03','sub-08','sub-09','sub-20','sub-29','sub-30','sub-33','sub-34'] # 9
num_subjects = len(best_subjects_cc) # number of subjects
threshold = 0/num_subjects # to compute the probabilities

im_template = sitk.ReadImage(template_path)
im_mask = sitk.ReadImage(mask_path)

# Subjects' labels
segments = [nb.load(f) for f in Path(base_dir).rglob("*labels2template5.nii.gz")]
# print(segments[0].get_fdata()[0,0,0])
header = segments[0].header.copy()
header.set_data_dtype("uint8")

# Matrix of zeros of the size of the image
# matrix = np.zeros_like(segments[0].dataobj, dtype="uint8")
prob_matrix = np.zeros_like(segments[0].dataobj, dtype="uint8")

# Bounding box
lsif = sitk.LabelStatisticsImageFilter() # It requires intensity and label images
lsif.Execute(im_template, im_mask) # Mask! Where all the labels are 1!
bounding_box = np.array(lsif.GetBoundingBox(1)) # GetBoundingBox(label)
print(f"Bounding box:  {bounding_box}") # [xmin, xmax, ymin, ymax, zmin, zmax]

# Loop
for x in range(bounding_box[0], bounding_box[1]+1):
    for y in range(bounding_box[2], bounding_box[3]+1):
        for z in range(bounding_box[4], bounding_box[5]+1):
            arr = np.zeros(num_subjects)
            for i in range(num_subjects):
                arr[i] = segments[i].get_fdata()[x,y,z] # Array of votes from each subject for specific point [x,y,z]
            prob = np.zeros(9) # 9 classes
            if np.any(arr): # check if array has any nonzero value
                for j in range(len(prob)):
                    prob[j] = np.count_nonzero(arr ==  j+1) / len(arr) # Array of probabilities for each class
                # sum_prob = np.sum(prob) # Sum of non-zero probabilities (less than 1)
                prob_matrix[x,y,z] = np.argmax(prob)+1 # if np.amax(prob) >= threshold else 0 # Most likely class (highest probability) between 1 and 9 meeting a threshold
                # prob_matrix[x,y,z] = np.amax(prob) # np.interp(np.amax(prob), [0,1], [1,9])
                # prob_matrix[x,y,z] = np.interp(np.amax(prob), [0,1], [1,9])
            else:
                prob_matrix[x,y,z] = 0 # Background
            # prob = len(arr_non_zero == most_frequent)/len(arr)
            # matrix[x,y,z] = np.argmax(prob)

# Probability map representation
nii = nb.Nifti1Image(prob_matrix, segments[0].affine, header)
# nii.to_filename(output_dir+"prob_map_cropped_th"+str(int(threshold*100))+".nii.gz")
nii.to_filename(output_dir+"prob_map_th0_2.nii.gz")