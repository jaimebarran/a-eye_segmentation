import numpy as np
import os,sys
import nibabel as nib
import SimpleITK as sitk
import pandas as pd

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/'
# image_path = '/mnt/sda1/ANTs/input/mni152/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz' # ref mni
# image_path = '/mnt/sda1/ANTs/input/colin27/tpl-MNIColin27_T1w.nii.gz' # ref colin
image_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/template0.nii.gz'
# image_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_7_n1/template0.nii.gz'
# image_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_9_n1/template0.nii.gz'

best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33']
labels_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/all_segments_mask.nii.gz'

for i in range(len(best_subjects_cc)):

    # Image
    # image_path = base_dir+folder+'/input/'+folder+'_T1_oriented.nii.gz'
    # Labels
    # labels_path = base_dir+folder+'/input/'+folder+'_labels.nii.gz'
    # labels_path = base_dir+folder+'/output_mni152/all_segments_template.nii.gz' # sub-xx's labels in mni space
    # labels_path = base_dir+folder+'/output_colin27/all_segments_template.nii.gz' # sub-xx's labels in colin space
    # labels_path = base_dir + 'best_subjects_eye_cc/' + best_subjects_cc[i] + '_labels2template5.nii.gz'

    # Boundary for the bounding box
    bound = 15

    image = sitk.ReadImage(image_path)
    all_segments = sitk.ReadImage(labels_path)
    image_x_size, image_y_size, image_z_size = image.GetSize()
    print(f"image_x_size {image_x_size} image_y_size {image_y_size} image_z_size{image_z_size}")

    # Mask
    all_segments_mask = all_segments > 0
    # sitk.WriteImage(all_segments_mask, base_dir+folder+'/input/'+folder+'_labels_mask.nii.gz')

    # Bounding box
    lsif = sitk.LabelStatisticsImageFilter() # It requires intensity and label images
    lsif.Execute(image, all_segments_mask) # Mask! Where all the labels are 1!
    bounding_box = np.array(lsif.GetBoundingBox(1)) # GetBoundingBox(label)
    print(f"Bounding box:  {bounding_box}") # [xmin, xmax, ymin, ymax, zmin, zmax]
    bounding_box_expanded = bounding_box.copy()
    bounding_box_expanded[0::2] -= bound # even indexes
    bounding_box_expanded[1::2] += bound # odd indexes
    print(f"Expanded bounding box: {bounding_box_expanded}")

    # Limits
    if bounding_box_expanded[0] < 0: bounding_box_expanded[0] = 0
    if bounding_box_expanded[1] > image_x_size: bounding_box_expanded[1] = image_x_size
    if bounding_box_expanded[2] < 0: bounding_box_expanded[2] = 0
    if bounding_box_expanded[3] > image_y_size: bounding_box_expanded[3] = image_y_size
    if bounding_box_expanded[4] < 0: bounding_box_expanded[4] = 0
    if bounding_box_expanded[5] > image_z_size: bounding_box_expanded[5] = image_z_size
    print(f"Expanded bounding box after limits: {bounding_box_expanded}")

    # Crop
    image_crop = image[int(bounding_box_expanded[0]):int(bounding_box_expanded[1]), # x
                    int(bounding_box_expanded[2]):int(bounding_box_expanded[3]), # y
                    int(bounding_box_expanded[4]):int(bounding_box_expanded[5])] # z
    # sitk.WriteImage(image_crop, base_dir+folder+'/input/'+folder+'_T1_cropped.nii.gz')
    # sitk.WriteImage(image_crop, base_dir+folder+'/input/tpl-MNI152NLin2009cAsym_res-01_T1w_cropped.nii.gz')
    # sitk.WriteImage(image_crop, base_dir+folder+'/input/tpl-MNIColin27_T1w_cropped.nii.gz')
    # sitk.WriteImage(image_crop, base_dir + '/best_subjects_eye_cc/CustomTemplate_5_n1/' + best_subjects_cc[i] + '_template_cropped.nii.gz')
    sitk.WriteImage(image_crop, base_dir + '/best_subjects_eye_cc/CustomTemplate_5_n1/template0_cropped.nii.gz')

    # The following is only needed with the non reference T1 image
    # all_segments_crop = all_segments[int(bounding_box_expanded[0]):int(bounding_box_expanded[1]), # x
    #                                 int(bounding_box_expanded[2]):int(bounding_box_expanded[3]), # y
    #                                 int(bounding_box_expanded[4]):int(bounding_box_expanded[5])] # z
    # sitk.WriteImage(all_segments_crop, base_dir+folder+'/input/'+folder+"_labels_cropped.nii.gz" )                