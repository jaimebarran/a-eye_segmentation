import os
import dicom2nifti

input_directory = '/mnt/sda1/Repos/a-eye/Data/SHIP_dataset/non_labeled_dataset/'
output_directory = '/mnt/sda1/Repos/a-eye/Data/SHIP_dataset/non_labeled_dataset_nifti/'
filename = 'T1.nii'

# Converting a directory with dicom files to nifti files
dicom2nifti.convert_directory(input_directory, output_directory)

# Converting a directory with only 1 series to 1 nifti file
# dicom2nifti.dicom_series_to_nifti(input_directory, output_directory+filename, reorient_nifti=True)