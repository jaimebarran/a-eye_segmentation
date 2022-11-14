"""
Created on Jun 09, 2022

This script contains useful functions to convert the dataset from the PACS format (i.e. the format the data has once extracted from the PACS) to the BIDS format

"""

import os
import subprocess
import SimpleITK as sitk
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import TextIO
import argparse
from ants import registration, image_read, apply_transforms, image_similarity


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def print_both(f, *args):
    """This variant of the print function both prints to console and to an output file that is saved locally.
    Args:
        f (TextIO): file where the function prints
        * args: list of elements that will be converted to text
    Returns:
        None
    """
    toprint = ' '.join([str(arg) for arg in args])  # concatenate everything into one string
    print(toprint)  # print to console
    print(toprint, file=f)  # print in the specified file


def dcm2niix_wrapper(out_dir: str,
                     volume_path: str,
                     out_name: str,
                     out_file: TextIO) -> None:
    """This function is a wrapper for the commandline command dcm2niix.
    It is used to convert a dicom file into the corresponding nifti+json
    Args:
        out_dir (str): folder where the nifti (and json) will be saved
        volume_path (str): path to dicom volume
        out_name (str): name of output files
        out_file (TextIO): output log file where we save the status of the script
    Returns:
        None
    """
    # N.B. dcm2niix must be installed
    create_dir_if_not_exist(out_dir)

    if not os.path.exists(os.path.join(out_dir, "{}.nii.gz".format(out_name))):
        cmd = ["dcm2niix", "-f", out_name, "-z", "y", "-o", out_dir, volume_path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
        _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = proc.communicate()
    else:
        print_both(out_file, "Nifti volume already exists")


def fsl_brain_extraction_tool_wrapper(input_volume_path: str,
                                      out_dir: str,
                                      out_volume_path: str,
                                      fractional_intensity_threshold: str,
                                      out_file: TextIO) -> None:
    """This function is a wrapper for the brain extraction tool (BET) from fsl. The skull-stripped volume
    is created in the same directory of the original volume.
    Args:
        input_volume_path (str): path of input volume (that we want to skull-strip)
        out_dir (str): path to output folder (will be created if it doesn't exist)
        out_volume_path (str): path of output skull-stripped file
        fractional_intensity_threshold (str): has to be in the range [0,1]; default=0.5; smaller values give larger brain outline estimates
        out_file (TextIO): output log file where we save the status of the script
    Returns:
        None
    """
    # N.B. fsl must be installed
    create_dir_if_not_exist(out_dir)

    if not os.path.exists(out_volume_path):  # if output volume does not exist
        cmd = ['bet', input_volume_path, out_volume_path, '-f', fractional_intensity_threshold]  # type: list # create list as if it was a command line expression
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # pass the list as input to Popen
        _ = process.communicate()[0]  # the [0] is to return just the output, because otherwise it would be outs, errs = process.communicate()
    else:
        print_both(out_file, "Skull-stripped volume already exists")


def bias_field_correct(input_img_path: str,
                       output_path: str) -> None:
    input_img = sitk.ReadImage(input_img_path)  # read image
    mask_img = sitk.OtsuThreshold(input_img, 0, 1, 200)  # create binary mask with Otsu method
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)  # cast to float32
    corrector = sitk.N4BiasFieldCorrectionImageFilter()  # create corrector object
    output = corrector.Execute(input_img, mask_img)  # apply corrector to obtain output image
    sitk.WriteImage(output, output_path)


def write_to_json_file(path: str,
                       filename: str,
                       data: dict) -> None:
    file_path_name = path + '/' + filename + '.json'
    with open(file_path_name, 'w') as fp:
        json.dump(data, fp)


def n4_bias_field_correction(derivatives_n4bfc_folder: str,
                             sub: str,
                             ses: str,
                             orig_anat_dir: str,
                             out_file: TextIO,
                             desired_axes_orientations: tuple) -> None:
    out_dir = os.path.join(derivatives_n4bfc_folder, sub, ses, "anat")
    create_dir_if_not_exist(out_dir)

    # apply bias field correction to TOF volume
    n4_tof_name = "{}_{}_desc-angio_N4bfc_mask".format(sub, ses)
    out_n4_tof_path = os.path.join(out_dir, "{}.nii.gz".format(n4_tof_name))
    if not os.path.exists(out_n4_tof_path):  # if path does not exist
        bias_field_correct(os.path.join(orig_anat_dir, '{}_{}_angio.nii.gz'.format(sub, ses)), out_n4_tof_path)

        # json file creation
        data = {'Space': 'orig_TOF', 'SkullStripped': 'False', 'N4_bias_field_corrected': 'True'}  # write json file information
        write_to_json_file(out_dir, n4_tof_name, data)  # create json file by invoking function
    else:
        print_both(out_file, "{}.nii.gz already exists".format(n4_tof_name))

    # ensure orientation is correct
    n4_tof_nifti = nib.load(out_n4_tof_path)  # load as nibabel object
    assert get_axes_orientations(n4_tof_nifti) == desired_axes_orientations, "Axes orientations wrong for {}".format(n4_tof_name)

    # ensure bias-field-corrected volume is of type float32
    cast_nifti_volume_to_float_32(n4_tof_nifti, out_n4_tof_path)

    # COMPUTE brain_mask of N4 bias-field-corrected TOF volume
    out_n4_bet_tof_path = out_n4_tof_path.replace("mask", "brain_mask")
    if not os.path.exists(out_n4_bet_tof_path):
        fsl_brain_extraction_tool_wrapper(input_volume_path=out_n4_tof_path,
                                          out_dir=out_dir,
                                          out_volume_path=out_n4_bet_tof_path,
                                          fractional_intensity_threshold="0.1",
                                          out_file=out_file)
        # json file creation
        data = {'Space': 'orig_TOF', 'SkullStripped': 'True', 'N4_bias_field_corrected': 'True'}  # write json file information
        write_to_json_file(out_dir, n4_tof_name.replace("mask", "brain_mask"), data)  # create json file by invoking function
    else:
        print_both(out_file, "{}.nii.gz already exists".format(n4_tof_name.replace("mask", "brain_mask")))

    # ensure orientation is correct
    n4_tof_bet_nifti = nib.load(out_n4_bet_tof_path)  # load as nibabel object
    assert get_axes_orientations(n4_tof_bet_nifti) == desired_axes_orientations, "Axes orientations wrong for {}".format(out_n4_bet_tof_path)

    # ensure dtype is also correct
    cast_nifti_volume_to_float_32(n4_tof_bet_nifti, out_n4_bet_tof_path)


def save_reg_quality_metrics(metrics_list: list,
                             out_file_path: str):
    parent_dir = str(Path(out_file_path).parent)
    df = pd.DataFrame([metrics_list], columns=["neigh_corr", "mut_inf"])
    create_dir_if_not_exist(parent_dir)
    df.to_csv(out_file_path, index=False)


def compute_registration_parameters(t1_mni_atlas_path: str,
                                    orig_anat_dir: str,
                                    sub: str,
                                    ses: str,
                                    reg_params_dir: str,
                                    registr_metrics_folder: str,
                                    out_file: TextIO) -> None:
    if not os.path.exists(os.path.join(reg_params_dir, "out_MNI_2_T1_Warped.nii.gz")):  # if registration file is empty
        # first, compute registration mni --> struct
        out_dict_mni_2_struct = registration(moving=image_read(t1_mni_atlas_path),
                                             fixed=image_read(os.path.join(orig_anat_dir, '{}_{}_T1w.nii.gz'.format(sub, ses))),
                                             type_of_transform="antsRegistrationSyNQuick[s]",
                                             outprefix=os.path.join(reg_params_dir, "out_MNI_2_T1_"))

        out_dict_mni_2_struct["warpedmovout"].to_file(os.path.join(reg_params_dir, "out_MNI_2_T1_Warped.nii.gz"))
        out_dict_mni_2_struct["warpedfixout"].to_file(os.path.join(reg_params_dir, "out_MNI_2_T1_InverseWarped.nii.gz"))
    else:
        print_both(out_file, "{} already exists".format("out_MNI_2_T1_Warped.nii.gz"))

    # SAVE registration quality metrics
    out_file_metrics_mni2struct = os.path.join(registr_metrics_folder, "{}_{}_reg_quality_metrics_{}.csv".format(sub, ses, "mni2struct"))
    if not os.path.exists(out_file_metrics_mni2struct):
        mni_2_struct_cc_metric = image_similarity(image_read(os.path.join(reg_params_dir, "out_MNI_2_T1_Warped.nii.gz")),
                                                  image_read(os.path.join(orig_anat_dir, '{}_{}_T1w.nii.gz'.format(sub, ses))),
                                                  metric_type='ANTsNeighborhoodCorrelation')
        mni_2_struct_mi_metric = image_similarity(image_read(os.path.join(reg_params_dir, "out_MNI_2_T1_Warped.nii.gz")),
                                                  image_read(os.path.join(orig_anat_dir, '{}_{}_T1w.nii.gz'.format(sub, ses))),
                                                  metric_type="MattesMutualInformation")

        # SAVE REGISTRATION QUALITY METRICS
        save_reg_quality_metrics([mni_2_struct_cc_metric, mni_2_struct_mi_metric], out_file_metrics_mni2struct)
    else:
        print_both(out_file, "{} already exists".format("{}_{}_reg_quality_metrics_{}.csv".format(sub, ses, "mni2struct")))

    if not os.path.exists(os.path.join(reg_params_dir, "out_T1_2_TOF_Warped.nii.gz")):
        # then, compute registration struct --> TOF
        out_dict_struct_2_tof = registration(moving=image_read(os.path.join(orig_anat_dir, '{}_{}_T1w.nii.gz'.format(sub, ses))),
                                             fixed=image_read(os.path.join(orig_anat_dir, '{}_{}_angio.nii.gz'.format(sub, ses))),
                                             type_of_transform="Affine",
                                             outprefix=os.path.join(reg_params_dir, "out_T1_2_TOF_"))

        out_dict_struct_2_tof["warpedmovout"].to_file(os.path.join(reg_params_dir, "out_T1_2_TOF_Warped.nii.gz"))
        out_dict_struct_2_tof["warpedfixout"].to_file(os.path.join(reg_params_dir, "out_T1_2_TOF_InverseWarped.nii.gz"))
    else:
        print_both(out_file, "{} already exists".format("out_T1_2_TOF_Warped.nii.gz"))

    # SAVE registration quality metrics
    out_file_metrics_struct2tof = os.path.join(registr_metrics_folder, "{}_{}_reg_quality_metrics_{}.csv".format(sub, ses, "struct2tof"))
    if not os.path.exists(out_file_metrics_struct2tof):
        struct_2_tof_cc_metric = image_similarity(image_read(os.path.join(reg_params_dir, "out_T1_2_TOF_Warped.nii.gz")),
                                                  image_read(os.path.join(orig_anat_dir, '{}_{}_angio.nii.gz'.format(sub, ses))),
                                                  metric_type='ANTsNeighborhoodCorrelation')
        struct_2_tof_mi_metric = image_similarity(image_read(os.path.join(reg_params_dir, "out_T1_2_TOF_Warped.nii.gz")),
                                                  image_read(os.path.join(orig_anat_dir, '{}_{}_angio.nii.gz'.format(sub, ses))),
                                                  metric_type="MattesMutualInformation")

        # SAVE REGISTRATION QUALITY METRICS
        save_reg_quality_metrics([struct_2_tof_cc_metric, struct_2_tof_mi_metric], out_file_metrics_struct2tof)
    else:
        print_both(out_file, "{} already exists".format("{}_{}_reg_quality_metrics_{}.csv".format(sub, ses, "struct2tof")))


def register_vessel_atlas_mni_2_tof(vesselmni_2_tof_dir: str,
                                    sub: str,
                                    ses: str,
                                    vessel_mni_atlas_path: str,
                                    orig_anat_dir: str,
                                    reg_params_dir: str,
                                    desired_axes_orientations: tuple) -> None:
    # define filenames and paths
    out_file_mni_2_struct = os.path.join(vesselmni_2_tof_dir, '{}_{}_desc-vesselMNI2struct_deformed.nii.gz').format(sub, ses)
    vesselmni2angio_deformed_name = "{}_{}_desc-vesselMNI2angio_deformed".format(sub, ses)
    out_file_struct_2_tof = os.path.join(vesselmni_2_tof_dir, '{}.nii.gz').format(vesselmni2angio_deformed_name)

    if not os.path.exists(out_file_mni_2_struct) and not os.path.exists(out_file_struct_2_tof):  # if paths do not exist
        # first, register the atlas from mni space to struct space
        mni_2_struct = apply_transforms(moving=image_read(vessel_mni_atlas_path),
                                        fixed=image_read(os.path.join(orig_anat_dir, '{}_{}_T1w.nii.gz'.format(sub, ses))),
                                        transformlist=[os.path.join(reg_params_dir, 'out_MNI_2_T1_1Warp.nii.gz'),
                                                       os.path.join(reg_params_dir, 'out_MNI_2_T1_0GenericAffine.mat')])

        # create file from ANTsImage
        mni_2_struct.to_file(out_file_mni_2_struct)

    if not os.path.exists(out_file_struct_2_tof):  # if output path does not exist
        # then, register the obtained warped volume from struct space to TOF space
        struct_2_tof = apply_transforms(moving=image_read(out_file_mni_2_struct),
                                        fixed=image_read(os.path.join(orig_anat_dir, '{}_{}_angio.nii.gz'.format(sub, ses))),
                                        transformlist=[os.path.join(reg_params_dir, 'out_T1_2_TOF_0GenericAffine.mat')])

        # create file from ANTsImage
        struct_2_tof.to_file(out_file_struct_2_tof)
        # remove intermediate volume (mni_2_struct) since we don't need it
        os.remove(out_file_mni_2_struct)

        # write json file information
        data = {"Space": "orig", "from": "MNI", "to": "orig", "SkullStripped": "False"}
        write_to_json_file(vesselmni_2_tof_dir, vesselmni2angio_deformed_name, data)  # create json file

    # ensure orientation is correct
    vesselmni_2_tof_nifti = nib.load(out_file_struct_2_tof)  # load as nibabel object
    assert get_axes_orientations(vesselmni_2_tof_nifti) == desired_axes_orientations, "Axes orientations wrong for {}".format(vesselmni2angio_deformed_name)
    cast_nifti_volume_to_float_32(vesselmni_2_tof_nifti, out_file_struct_2_tof)


def mni_2_tof_registration(derivatives_path_registrations: str,
                           sub: str,
                           ses: str,
                           t1_mni_atlas_path: str,
                           orig_anat_dir: str,
                           vessel_mni_atlas_path: str,
                           out_file: TextIO,
                           desired_axes_orientations: tuple):

    # create vesselMNI_2_TOF derivative folder
    vesselmni_2_tof_dir = os.path.join(derivatives_path_registrations, "vesselMNI_2_angioTOF", sub, ses, "anat")
    create_dir_if_not_exist(vesselmni_2_tof_dir)

    reg_params_sub_ses_dir = os.path.join(derivatives_path_registrations, "reg_params", sub, ses)
    create_dir_if_not_exist(reg_params_sub_ses_dir)

    reg_metrics_sub_ses_dir = os.path.join(derivatives_path_registrations, "reg_metrics", sub, ses)
    create_dir_if_not_exist(reg_metrics_sub_ses_dir)

    # COMPUTE REGISTRATION PARAMETERS
    compute_registration_parameters(t1_mni_atlas_path, orig_anat_dir, sub, ses, reg_params_sub_ses_dir, reg_metrics_sub_ses_dir, out_file)

    # APPLY REGISTRATION for vesselMNI atlas
    register_vessel_atlas_mni_2_tof(vesselmni_2_tof_dir, sub, ses, vessel_mni_atlas_path, orig_anat_dir, reg_params_sub_ses_dir, desired_axes_orientations)


def get_axes_orientations(input_nifti_volume: nib.Nifti1Image) -> tuple:
    """This function returns the axes orientations as a tuple
    Args:
        input_nifti_volume (nib.Nifti1Image): the input volume for which we want the axes orientations
    Returns:
        orientations (tuple): the axes orientations
    """
    aff_mat = input_nifti_volume.affine  # type: np.ndarray # extract affine matrix
    orientations = nib.aff2axcodes(aff_mat)

    return orientations


def create_dir_if_not_exist(dir_to_create: str) -> None:
    """This function creates the input dir if it doesn't exist.
    Args:
        dir_to_create (str): directory that we want to create
    Returns:
        None
    """
    if not os.path.exists(dir_to_create):  # if dir doesn't exist
        os.makedirs(dir_to_create)  # create it


def cast_nifti_volume_to_uint_8(nibabel_object: nib.Nifti1Image,
                                nibabel_object_path: str):
    """This function casts the input volume to uint8
    Args:
        nibabel_object (nib.Nifti1Image): input nibabel volume
        nibabel_object_path (str): path to nifti volume that we want to cast to uint8
    Returns:
        None
    """
    volume_np = np.asanyarray(nibabel_object.dataobj)  # extract voxel values
    if volume_np.dtype != np.uint8:  # if dtype is not uint8
        volume_np_int8 = np.array(volume_np, dtype=np.uint8)  # cast to uint8
        volume_obj_int8 = nib.Nifti1Image(volume_np_int8, affine=nibabel_object.affine)  # create nibabel object
        nib.save(volume_obj_int8, nibabel_object_path)  # overwrite previous volume


def cast_nifti_volume_to_int_32(nibabel_object: nib.Nifti1Image,
                                  nibabel_object_path: str):
    """This function casts the input volume to int32
    Args:
        nibabel_object (nib.Nifti1Image): input nibabel volume
        nibabel_object_path (str): path to nifti volume that we want to cast to int32
    Returns:
        None
    """
    volume_np = np.asanyarray(nibabel_object.dataobj)  # extract voxel values
    if volume_np.dtype != np.int32:  # if dtype is not int32
        volume_np_int32 = np.array(volume_np, dtype=np.float32)  # cast to int32
        volume_obj_int32 = nib.Nifti1Image(volume_np_int32, affine=nibabel_object.affine)  # create nibabel object
        nib.save(volume_obj_int32, nibabel_object_path)  # overwrite previous volume


def cast_nifti_volume_to_float_32(nibabel_object: nib.Nifti1Image,
                                  nibabel_object_path: str):
    """This function casts the input volume to float32
    Args:
        nibabel_object (nib.Nifti1Image): input nibabel volume
        nibabel_object_path (str): path to nifti volume that we want to cast to float32
    Returns:
        None
    """
    volume_np = np.asanyarray(nibabel_object.dataobj)  # extract voxel values
    if volume_np.dtype != np.float32:  # if dtype is not float32
        volume_np_f32 = np.array(volume_np, dtype=np.float32)  # cast to float32
        volume_obj_f32 = nib.Nifti1Image(volume_np_f32, affine=nibabel_object.affine)  # create nibabel object
        nib.save(volume_obj_f32, nibabel_object_path)  # overwrite previous volume


def keep_only_digits(input_string: str) -> str:
    """This function takes as input a string and returns the same string but only containing digits
    Args:
        input_string (str): the input string from which we want to remove non-digit characters
    Returns:
        output_string (str): the output string that only contains digit characters
    """
    numeric_filter = filter(str.isdigit, input_string)
    output_string = "".join(numeric_filter)

    return output_string


def get_parser():
    """This function creates a parser for handling input arguments"""
    p = argparse.ArgumentParser(description='Aneurysm_Net')
    p.add_argument('--config', type=str, required=True, help='Path to json configuration file.')
    return p


def load_config_file():
    """This function loads the input config file
    Returns:
        config_dictionary (dict): it contains the input arguments
    """
    parser = get_parser()  # create parser
    args = parser.parse_args()  # convert argument strings to objects
    with open(args.config, 'r') as f:
        config_dictionary = json.load(f)

    return config_dictionary


def main():
    pass


if __name__ == '__main__':
    # N.B: THIS CODE IS NOT CLEAN, AND STRONGLY DEPENDS ON HOW YOUR DICOM DATASET WAS EXTRACTED FROM YOUR LOCAL PACS;
    # THE IDEA IS MORE TO PROVIDE SMALL FUNCTIONS TO REPLICATE THE PREPROCESSING STEPS OF THE PIPELINE USED FOR THIS WORK
    main()
