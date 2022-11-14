import os, glob

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/'
# ref_mni152 = '/mnt/sda1/ANTs/input/mni152/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'
# ref_colin27 = '/mnt/sda1/ANTs/input/colin27/tpl-MNIColin27_T1w.nii.gz'
# eye_mask_mni = '/mnt/sda1/ANTs/input/mni152/tpl-MNI152NLin2009cAsym_res-01_desc-eye_mask.nii.gz'
template_cc = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/template0.nii.gz'
# template_cc_cropped = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/template0_cropped_15vox.nii.gz'
# template_labels_cropped = base_dir + 'best_subjects_eye_cc/CustomTemplate_5_n1/Probability_Maps/Per_Class_2/prob_map_cropped_preMaxAPost_sup_mus.nii.gz'

# List of best subjects to do the registration
best_subjects_cc = ['sub-33'] # 5
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33'] # 5
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-30','sub-33','sub-34'] # 7
# best_subjects_cc = ['sub-02','sub-03','sub-08','sub-09','sub-20','sub-29','sub-30','sub-33','sub-34'] # 9
# rest_subjects = ['sub-08','sub-09','sub-30','sub-34']

# List of remaining subjects
# all_subjects = list()
# for i in range(35):
#     all_subjects.append('sub-'+str(i+1).zfill(2))
# rest_subjects = [elem for elem in all_subjects if elem not in best_subjects_cc]
# print(rest_subjects)

# List of worst subjects
# best_subjects_cc = ['sub-15','sub-19','sub-21','sub-23','sub-26']
# worst_subjects_cc = ['sub-15','sub-19','sub-21','sub-23','sub-26','sub-17','sub-16','sub-28','sub-10'] # 9

# ''' Loop for best subjects
j = 0
# antsEyeExtraction and antsApplyTransforms
for i in range(len(best_subjects_cc)):
    input_t1 = base_dir + 'best_subjects_eye_cc/' + best_subjects_cc[i] + '_T1_aff.nii.gz'
    input_labels = base_dir + 'a123/' + best_subjects_cc[i] + '/input/' + best_subjects_cc[i] + '_labels.nii.gz'
    # input_t1_cropped = base_dir + 'a123/' + best_subjects_cc[i] + '/input/' + best_subjects_cc[i] + '_T1_cropped.nii.gz'
    # input_labels_cropped = base_dir + 'a123/' + best_subjects_cc[i] + '/input/' + best_subjects_cc[i] + '_labels_cropped.nii.gz'
    # ref_mni152_cropped = base_dir + best_subjects_cc[i] + '/input/' + 'tpl-MNI152NLin2009cAsym_res-01_T1w_cropped.nii.gz'
    # ref_colin27_cropped = base_dir + best_subjects_cc[i] + '/input/' + 'tpl-MNIColin27_T1w_cropped.nii.gz'
    output = base_dir +  'best_subjects_eye_cc/CustomTemplate_5_n1/new_registration/' # Change this when doing new extractions
    # output_reg_cropped_path = output + 'reg_cropped_best_subjects/' + best_subjects_cc[i] + '_reg_cropped/'
    if not os.path.exists(output):
        os.makedirs(output)

    # Brain extraction (only to test)
    # command1 = 'antsBrainExtraction.sh -d 3' + \
    # ' -a ' + input_t1 + \
    # ' -e ' + ref_mni152 + \
    # ' -m ' + eye_mask_mni + \
    # ' -o ' + output + \
    # ' -k ' + '1' # 1 = keep temporary files, 0 = remove them
    # print(command1)
    # # os.system(command1)

    # Eye extraction
    # command1 = 'antsEyeExtraction.sh -d 3' + \
    # ' -a ' + input_t1                      + \
    # ' -e ' + template_cc                   + \
    # ' -f ' + eye_mask_mni                  + \
    # ' -g ' + input_labels                  + \
    # ' -o ' + output                        + \
    # ' -k ' + '1' # 1 = keep temporary files, 0 = remove them
    # print(command1)
    # os.system(command1)

    # ApplyTransforms
    command2 = 'antsApplyTransforms -d 3' + \
    ' -i ' +  input_labels + \
    ' -r ' +  template_cc + \
    ' -n ' + 'MultiLabel' + \
    ' -t ' + output + '1Warp.nii.gz' + \
    ' -t ' + output + '0GenericAffine.mat' + \
    ' -o ' +  output + best_subjects_cc[i] +'_labels2template5_2.nii.gz' + \
    ' --float 0 --verbose 1'
    print(command2)
    os.system(command2)
    # j += 1

    # antsRegistrationSyNQuick
    # command1 = 'antsRegistrationSyNQuick.sh -d 3' + \
    # ' -m ' + input_t1                             + \
    # ' -f ' + template_cc                          + \
    # ' -t ' + 's'                                  + \
    # ' -o ' + output                               + \
    # ' -n ' + '8'
    # print(command1)
    # os.system(command1)

    # antsRegistrationSyN (for cropped images)
    # command1 = 'antsRegistrationSyN.sh -d 3' + \
    # ' -m ' + input_t1_cropped   + \
    # ' -f ' + template_cc_cropped + \
    # ' -t ' + 's'                + \
    # ' -o ' + output_reg_cropped_path
    # # print(command1)
    # os.system(command1)

    # # ApplyTransforms (for cropped images)
    # command2 = 'antsApplyTransforms -d 3 ' + \
    # ' -i ' +  input_labels + \
    # ' -o ' +  output_reg_cropped_path + 'labels2template2.nii.gz' + \
    # ' -r ' +  template_cc_cropped + \
    # ' -n ' + 'MultiLabel' + \
    # ' -t ' + output_reg_cropped_path + '1Warp.nii.gz' + \
    # ' -t ' + '[' + output_reg_cropped_path + '0GenericAffine.mat, 0 ]' + \
    # ' --float 0 --verbose 1'
    # # print(command2)
    # os.system(command2)
# '''

''' Loop for the rest of the subjects
# antsRegistrationSyN and antsApplyTransforms
for i in range(len(rest_subjects)):
    # input_t1 = base_dir + 'best_subjects_eye_cc/' + best_subjects_cc[i] + '_T1_aff.nii.gz'
    # input_labels = base_dir + 'a123/' + rest_subjects[i] + '/input/' + rest_subjects[i] + '_labels.nii.gz'
    input_t1_cropped = base_dir + 'a123/' + rest_subjects[i] + '/input/' + rest_subjects[i] + '_T1_cropped.nii.gz'
    # input_labels_cropped = base_dir + 'a123/' + rest_subjects[i] + '/input/' + rest_subjects[i] + '_labels_cropped.nii.gz'
    output = base_dir +  'best_subjects_eye_cc/CustomTemplate_5_n1/reg_cropped_other_subjects/' # Change this when doing new extractions
    output_reg_cropped_path = output + rest_subjects[i] + '_reg_cropped/Per_Class_2/'
    warp_paths = output + rest_subjects[i] + '_reg_cropped/'
    if not os.path.exists(output_reg_cropped_path):
        os.makedirs(output_reg_cropped_path)
    
    ## antsRegistrationSyN (for cropped images)
    # command1 = 'antsRegistrationSyN.sh -d 3' + \
    # ' -m ' + input_t1_cropped   + \
    # ' -f ' + template_cc_cropped + \
    # ' -t ' + 's'                + \
    # ' -o ' + output_reg_cropped_path
    # print(command1)
    # os.system(command1)

    # # ApplyTransforms (for cropped images)
    # command2 = 'antsApplyTransforms -d 3 ' + \
    # ' -i ' +  input_labels_cropped + \
    # ' -o ' +  output_reg_cropped_path + 'labels2template2.nii.gz' + \
    # ' -r ' +  template_cc_cropped + \
    # ' -n ' + 'MultiLabel' + \
    # ' -t ' + output_reg_cropped_path + '1Warp.nii.gz' + \
    # ' -t ' + '[' + output_reg_cropped_path + '0GenericAffine.mat, 0 ]' + \
    # ' --float 0 --verbose 1'
    # print(command2)
    # os.system(command2)

    # ApplyTransforms (for cropped images) with inverse transform to get the template labels into subject space
    command2 = 'antsApplyTransforms -d 3 ' + \
    ' -i ' +  template_labels_cropped + \
    ' -o ' +  output_reg_cropped_path + 'supmus2subject.nii.gz' + \
    ' -r ' +  input_t1_cropped + \
    ' -t ' + '[' + warp_paths + '0GenericAffine.mat, 1 ]' + \
    ' -t ' + warp_paths + '1InverseWarp.nii.gz' + \
    ' --float 0 --verbose 1'
    # print(command2)
    os.system(command2)

    # Dealing with files in that folder
    # for f in glob.glob(output_reg_cropped_path + '2subject.nii.gz'):
    #     os.remove(f)

# '''