import os
import dicom2nifti
from sqlalchemy import false
import subprocess

input_directory = '/mnt/sda1/BIDS/sourcedata/'
output_directory = '/mnt/sda1/ANTs/a123/'
# output_directory = '/home/jaimebarranco/Desktop/Test/reorient_false/'

# Loop
for folder1 in os.listdir(input_directory):
    # print(folder)
    filename = folder1 + '_T1.nii.gz'
    for folder2 in os.listdir(input_directory+folder1):
        if 't1' in folder2 and not folder2.startswith('.'):
            input_dicom_folder = input_directory+folder1+'/'+folder2
            # print(input_dicom_folder)
            output_nifti_folder = output_directory+folder1+'/input/'
            # print(output_nifti_folder)
            # os.makedirs(output_nifti_folder)
            # Converting a directory with dicom files to nifti files
            # dicom2nifti.convert_directory(input_dicom_folder, output_nifti_folder+filename)
            # Converting a directory with only 1 series to 1 nifti file
            dicom2nifti.dicom_series_to_nifti(input_dicom_folder, output_nifti_folder+filename, reorient_nifti=True)
            # cmd = ["dcm2niix", "-f", filename, "-z", "y", "-o", output_nifti_folder, input_dicom_folder]
            # process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
            # _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = proc.communicate()