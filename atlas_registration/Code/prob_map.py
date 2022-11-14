from shutil import which
import nibabel as nb
import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
from scipy.stats import mode

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/a123/'
output_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/Output/'
colin_path = '/mnt/sda1/Repos/a-eye/Data/templateflow/colin27/tpl-MNIColin27_T1w.nii.gz'
mask_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/Output/all_segments_mask.nii.gz'

num_subjects = 35 # number of subjects
threshold = 5/num_subjects # to compute the probabilities

im_colin = sitk.ReadImage(colin_path)
im_mask = sitk.ReadImage(mask_path)

# Subjects' labels
segments = [nb.load(f) for f in Path(base_dir).glob("*/output_colin27/all_segments_template.nii.gz")]
# print(segments[0].get_fdata()[0,0,0])
header = segments[0].header.copy()
header.set_data_dtype("uint8")

# Matrix of zeros of the size of the image
# matrix = np.zeros_like(segments[0].dataobj, dtype="uint8")
prob_matrix = np.zeros_like(segments[0].dataobj, dtype="uint8")

# Bounding box
lsif = sitk.LabelStatisticsImageFilter() # It requires intensity and label images
lsif.Execute(im_colin, im_mask) # Mask! Where all the labels are 1!
bounding_box = np.array(lsif.GetBoundingBox(1)) # GetBoundingBox(label)
print(f"Bounding box:  {bounding_box}") # [xmin, xmax, ymin, ymax, zmin, zmax]


# Loop
for x in range(bounding_box[0], bounding_box[1]+1):
    for y in range(bounding_box[2], bounding_box[3]+1):
        for z in range(bounding_box[4], bounding_box[5]+1):
            arr = np.zeros(num_subjects)
            for i in range(num_subjects):
                arr[i] = segments[i].get_fdata()[x,y,z] # Array of votes from each subject for specific point [x,y,z]
            # arr_non_zero = arr[np.nonzero(arr)]
            # median = np.median(arr_non_zero) # [0,9]
            # most_frequent = mode(arr_non_zero)[0][0]
            prob = np.zeros(9) # 9 classes
            if np.any(arr): # check if array has any nonzero value
                for j in range(len(prob)):
                    prob[j] = np.count_nonzero(arr ==  j+1) / len(arr) # Array of probabilities for each class
                # sum_prob = np.sum(prob) # Sum of non-zero probabilities (less than 1)
                prob_matrix[x,y,z] = np.argmax(prob)+1 if np.amax(prob) >= threshold else 0 # Most likely class between 1 and 9 meeting a threshold
            else:
                prob_matrix[x,y,z] = 0 # Background
            # prob = len(arr_non_zero == most_frequent)/len(arr)
            # matrix[x,y,z] = np.argmax(prob)

# Probability map representation
# nii = nb.Nifti1Image(matrix, segments[0].affine, header)
# nii = nb.Nifti1Image(prob_matrix, segments[0].affine, header)
# nii.to_filename(output_dir+"prob_map_th"+str(int(threshold*100))+".nii.gz")