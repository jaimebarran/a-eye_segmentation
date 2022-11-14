import os, glob
from pickletools import uint1
import subprocess
import numpy as np
import shutil
import dicom2nifti

# 1) copy to sourcedata
# input_directory = './a123/.'
# output_directory = './sourcedata/'

# Copy all files to /sourcedata
# cmd = ["cp", '-a', input_directory, output_directory]
# process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
# _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = proc.communicate()

# Ascending order list
# arr = np.array([])
# for folder in os.listdir(output_directory):
#     print(folder)
#     arr = np.append(arr, folder)
# # print(arr)
# arr = np.sort(arr)
# print(arr)

# Rename folders in order
# i=1
# for j in arr:
#     os.rename(output_directory+j, output_directory+'sub-'+str(i).zfill(2))
#     i+=1


# 2) dicom
# input_directory = './sourcedata/'
# output_directory = './derivatives/'

# Copy loop
# for folder1 in os.listdir(input_directory):
#     for folder2 in os.listdir(input_directory+folder1):
#         if 't1' in folder2 and not folder2.startswith('.'):
#             # print(folder2)
#             path = input_directory+folder1+'/'+folder2+'/'
#             # out_path = output_directory+'sub-'+str(i).zfill(2)
#             out_path = output_directory+folder1
#             # os.makedirs(out_path)
#             command_copy = 'find ' + path + ' -name "*.gz" -not -name ".*" -exec cp {} ' + out_path + ' \;'
#             # print(command)
#             os.system(command_copy)
# Rename loop
# for folder in os.listdir(output_directory):
#     for filename in os.listdir(output_directory+folder):
#         # print(f'{filename}, {folder}')
#         os.rename(output_directory+folder+'/'+filename, output_directory+folder+'/'+folder+'_labels.nii.gz')


# 3) dcm2bids
# input_directory = './sourcedata/'
# output_directory = './'

# Copy loop
# for folder1 in os.listdir(input_directory):
#     path1 = input_directory+folder1+'/'+'dicomdir'
#     # os.makedirs(path1)
#     for folder2 in os.listdir(input_directory+folder1):
#         path2 = input_directory+folder1+'/'+folder2+'/'
#         command_copy = 'find ' + path2 + ' -name "*.dcm" -exec cp {} ' + path1 + ' \;'
#         os.system(command_copy)

# dcm2bids loop
# for folder in os.listdir(input_directory):
#     print(folder)
#     path1 = input_directory+folder+'/'+'dicomdir'+'/'
#     # print(path1)
#     sub = folder[-2:]
#     # print(sub)
#     path2 = output_directory+folder+'/'
#     # print(path2)
#     # os.makedirs(path2)
#     command = 'dcm2bids -d ' + path1 + ' -p ' + sub + ' -c ./BIDS_config.json -o ' + path2 + ' --forceDcm2niix'
#     # print(command)
#     os.system(command)
#     # cmd = ['dcm2bids -d ' + path1 + ' -p ' + sub + ' -c ./BIDS_config.json -o ' + path2 + ' --forceDcm2niix']
#     # process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
#     # _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = proc.communicate()


# 4) clean directory
# dir = './'
# # Remove tmp folders and move sub-xx content to upper sub-xx folder
# for folder1 in os.listdir(dir):
#     if os.path.isdir(dir+folder1) and folder1.startswith('sub'):
#         for folder2 in os.listdir(dir+folder1):
#             # print(folder2)
#             # if folder2.startswith('tmp'):
#             #     shutil.rmtree(dir+folder1+'/'+folder2)
#             path_in = dir+folder1+'/'+folder2+'/'
#             path_out = dir+folder1
#             # command = 'mv ' + path_in + '* ' + path_out
#             # os.system(command)
#             if folder2.startswith('sub'):
#                 # print(folder2)
#                 os.removedirs(path_in)

# Extra 1: copy labels from each subject to ANTs input directory
# input_dir = '/mnt/sda1/BIDS/derivatives/'
# output_directory = '/mnt/sda1/ANTs/a123/'

# for folder in os.listdir(input_dir):
#     input_folder = input_dir+folder
#     output_folder = output_directory+folder+'/input/'
#     # os.makedirs(output_folder)
#     command = 'cp ' + input_folder + '/*.gz ' + output_folder
#     os.system(command)

# Extra 2: a123_BIDs but only with T1 images
input_dir = '/mnt/sda1/Repos/a-eye/Data/SHIP_dataset/non_labeled_dataset/'
output_dir = '/mnt/sda1/Repos/a-eye/Data/SHIP_dataset/non_labeled_dataset_nifti/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop
i=0
for folder1 in sorted(os.listdir(input_dir)):
    i+=1
    filename = folder1
    for folder2 in os.listdir(input_dir+folder1):
        if i==10: break
        if 't1' in folder2 and not folder2.startswith('.'):
            input_dicom_folder = input_dir+folder1+'/'+folder2
            
            # output
            output_nifti_folder = output_dir+folder1
            if not os.path.exists(output_nifti_folder):
                os.makedirs(output_nifti_folder)

            # dcm2niix
            # dicom2nifti.dicom_series_to_nifti(input_dicom_folder, output_nifti_folder+filename, reorient_nifti=True)
            cmd = ["dcm2niix", "-f", filename, "-z", "y", "-o", output_nifti_folder, input_dicom_folder]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
            _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = proc.communicate()
            # Dealing with files in that folder
            # for f in glob.glob(input_dir+folder1+'/anat/'+folder1+'_T1.nii.gz'):
            #     os.remove(f)