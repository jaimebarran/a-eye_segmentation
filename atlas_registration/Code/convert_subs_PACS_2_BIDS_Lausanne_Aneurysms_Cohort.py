"""
Created on Jun 09, 2022

This script converts the dataset from the PACS format (i.e. the format the data has once extracted from the PACS) to the BIDS format

"""

from typing import TextIO
import os
import sys
sys.path.append('/home/to5743/aneurysm_project/Aneurysm_Clinical_Paper/')  # this line is needed on the HPC cluster to recognize the dir as a python package
import re
import time
from joblib import Parallel, delayed
from datetime import datetime
import nibabel as nib
from typing import Tuple
from pathlib import Path
from shutil import copy
from BIDS_Dataset_Creation.utils_preprocessing_BIDS_dataset import dcm2niix_wrapper, fsl_brain_extraction_tool_wrapper, write_to_json_file,\
    get_axes_orientations, mni_2_tof_registration, print_both, n4_bias_field_correction, cast_nifti_volume_to_uint_8,\
    create_dir_if_not_exist, keep_only_digits, cast_nifti_volume_to_int_32, load_config_file


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def print_running_time(start_time,
                       end_time,
                       process_name,
                       out_file):
    """This function takes as input the start and the end time of a process and prints to console the time elapsed for this process
    Args:
        start_time (float): instant when the timer is started
        end_time (float): instant when the timer was stopped
        process_name (string): name of the process
        out_file (TextIO): path where we save the logs of this script
    Returns:
        None
    """
    sentence = str(process_name)  # convert to string whatever the user inputs as third argument
    temp = end_time - start_time  # compute time difference
    hours = temp // 3600  # compute hours
    temp = temp - 3600 * hours  # if hours is not zero, remove equivalent amount of seconds
    minutes = temp // 60  # compute minutes
    seconds = temp - 60 * minutes  # compute minutes
    print_both(out_file, '\n%s time: %d hh %d mm %d ss' % (sentence, hours, minutes, seconds))
    return


def create_orig_anat_folder(new_bids_path: str,
                            sub: str,
                            ses: str,
                            tof_series_path: str,
                            t1_series_path: str,
                            desired_axes_orientations: tuple,
                            out_file: TextIO) -> Tuple[str, str]:
    """This function creates the original anat folder (i.e. the one that contains the original, unmodified sequences"""
    bids_sub_ses_anat_path = os.path.join(new_bids_path, sub, ses, "anat")
    create_dir_if_not_exist(bids_sub_ses_anat_path)

    # convert TOF volume
    tof_bids_name = '{}_{}_angio'.format(sub, ses)
    dcm2niix_wrapper(out_dir=bids_sub_ses_anat_path,
                     volume_path=tof_series_path,
                     out_name=tof_bids_name,
                     out_file=out_file)

    # ensure orientation is correct
    tof_nifti_volume_path = os.path.join(bids_sub_ses_anat_path, tof_bids_name + ".nii.gz")
    tof_nifti_volume_obj = nib.load(tof_nifti_volume_path)
    assert get_axes_orientations(tof_nifti_volume_obj) == desired_axes_orientations, "Axes orientations wrong for {}".format(tof_bids_name)

    # convert T1w volume
    t1w_bids_name = '{}_{}_T1w'.format(sub, ses)
    dcm2niix_wrapper(out_dir=bids_sub_ses_anat_path,
                     volume_path=t1_series_path,
                     out_name=t1w_bids_name,
                     out_file=out_file)

    # ensure orientation is correct
    t1w_nifti_volume = nib.load(os.path.join(bids_sub_ses_anat_path, t1w_bids_name + ".nii.gz"))
    assert get_axes_orientations(t1w_nifti_volume) == desired_axes_orientations, "Axes orientations wrong for {}".format(tof_bids_name)

    return bids_sub_ses_anat_path, tof_bids_name


def create_manual_masks_dir(orig_anat_dir: str,
                            tof_bids_name: str,
                            sub: str,
                            ses: str,
                            lesions_list: list,
                            derivatives_path_manual_masks: str,
                            out_file: TextIO,
                            desired_axes_orientations: tuple):
    print_both(out_file, "\n----- Computing FSL-BET for TOF volume...".format(sub, ses))

    # create folder specific for this sub-ses
    manual_masks_sub_ses_dir = os.path.join(derivatives_path_manual_masks, sub, ses, "anat")

    tof_volume_path = os.path.join(orig_anat_dir, tof_bids_name + ".nii.gz")
    out_volume_path = os.path.join(manual_masks_sub_ses_dir, tof_bids_name.replace("angio", "desc-brain_mask") + ".nii.gz")

    # generate skull-stripped volume
    fsl_brain_extraction_tool_wrapper(input_volume_path=tof_volume_path,
                                      out_dir=manual_masks_sub_ses_dir,
                                      out_volume_path=out_volume_path,
                                      fractional_intensity_threshold="0.1",
                                      out_file=out_file)

    # write json file information
    data = {'RawSources': '{}.nii.gz'.format(tof_bids_name), 'Space': 'orig', 'Type': 'brain'}
    new_name_json = tof_bids_name.replace("angio", "desc-brain_mask")
    write_to_json_file(manual_masks_sub_ses_dir, new_name_json, data)  # create json file

    # ensure orientation is correct
    angio_bet_nifti = nib.load(out_volume_path)  # load as nibabel object
    assert get_axes_orientations(angio_bet_nifti) == desired_axes_orientations, "Axes orientations wrong for {}".format(out_volume_path)

    # ensure dtype is correct
    cast_nifti_volume_to_int_32(angio_bet_nifti, out_volume_path)

    if lesions_list:  # if lesions_list is not empty --> the subject has one (or more) aneurysms
        print_both(out_file, "\n----- Renaming aneurysm mask(s)...".format(sub, ses))
        for lesion_path in lesions_list:
            match_lesion = re.findall(r"[Ll]esion\d+", lesion_path)
            lesion_number = keep_only_digits(match_lesion[0])
            lesion_name = "{}_{}_desc-Lesion_{}_mask.nii.gz".format(sub, ses, lesion_number)
            dst_path = os.path.join(manual_masks_sub_ses_dir, lesion_name)
            if not os.path.exists(dst_path):  # if file doesn't exist
                copy(src=lesion_path, dst=dst_path)  # copy and rename file

            # write json file information
            data = {'RawSources': '{}.nii.gz'.format(tof_bids_name), 'Space': 'orig', 'Type': 'Lesion'}
            new_lesion_name_json = lesion_name.replace(".nii.gz", "")
            write_to_json_file(manual_masks_sub_ses_dir, new_lesion_name_json, data)  # create json file

            # ensure orientation is correct
            lesion_nifti = nib.load(dst_path)  # load as nibabel object
            assert get_axes_orientations(lesion_nifti) == desired_axes_orientations, "{}_{}): Axes orientations {}, while we expect {}".format(sub,
                                                                                                                                               ses,
                                                                                                                                               get_axes_orientations(lesion_nifti),
                                                                                                                                               desired_axes_orientations)

            # ensure dtype is correct
            cast_nifti_volume_to_uint_8(lesion_nifti, dst_path)

    return manual_masks_sub_ses_dir


def create_derivatives(tof_bids_name: str,
                       derivatives_path_manual_masks: str,
                       derivatives_path_registrations: str,
                       lesions_list: list,
                       orig_anat_dir: str,
                       sub: str,
                       ses: str,
                       t1_mni_atlas_path: str,
                       vessel_mni_atlas_path: str,
                       derivatives_n4bfc_folder: str,
                       out_file: TextIO,
                       desired_axes_orientations: tuple) -> None:

    # create manual masks folder (i.e. brain extraction and manual annotations)
    create_manual_masks_dir(orig_anat_dir,
                            tof_bids_name,
                            sub,
                            ses,
                            lesions_list,
                            derivatives_path_manual_masks,
                            out_file,
                            desired_axes_orientations)

    # create registration folder
    print_both(out_file, "\n----- Register vessel atlas from MNI to TOF...")
    mni_2_tof_registration(derivatives_path_registrations,
                           sub,
                           ses,
                           t1_mni_atlas_path,
                           orig_anat_dir,
                           vessel_mni_atlas_path,
                           out_file,
                           desired_axes_orientations)

    # create N4 bias-field-corrected volumes
    print_both(out_file, "\n----- Compute N4 bias-field-corrected TOF...")
    n4_bias_field_correction(derivatives_n4bfc_folder,
                             sub,
                             ses,
                             orig_anat_dir,
                             out_file,
                             desired_axes_orientations)


def bidsify_one_subject(new_bids_path: str,
                        sub_ses_path: str,
                        vessel_mni_atlas_path: str,
                        t1_mni_atlas_path: str,
                        out_log_file_path: str,
                        desired_axes_orientations: tuple):
    start = time.time()  # start timer; used to compute the time needed to run this script
    sub = re.findall(r"sub-\d+", sub_ses_path)[0]  # extract sub
    ses = re.findall(r"ses-\w{6}\d+", sub_ses_path)[0][0:12]  # extract ses
    assert len(ses) == 12  # make sure that ses string has correct length
    output_file = open(out_log_file_path, "w")  # open a file where we'll store all the prints of the script; "w" guarantees that every time the script is run, the file is overwritten
    print_both(output_file, "\n---------------------- Converting {}_{}...".format(sub, ses))

    all_series = os.listdir(sub_ses_path)  # group all sequences in a list
    if len(all_series) > 6:
        raise ValueError("In this script, we expect only two sequences per session (T1w scan and TOF scan) + a maximum of 4 aneurysm masks")

    r_tof = re.compile("(.+)?[Tt][Oo][Ff]")  # define regex to match
    tof_series = list(filter(r_tof.match, all_series))  # only extract sequences that match the regex

    r_t1 = re.compile("(.+)?[Tt]1")  # define regex to match
    t1w_series = list(filter(r_t1.match, all_series))

    assert len(tof_series) == len(t1w_series) == 1, "Only one TOF and one T1w should match"

    tof_series_path = os.path.join(sub_ses_path, tof_series[0])
    t1_series_path = os.path.join(sub_ses_path, tof_series[0])

    sub_ses_parent_dir = Path(sub_ses_path).parent.absolute()  # extract parent directory
    r_lesion = re.compile("(.+)?[Ll]esion")  # define regex to match
    lesion_masks = list(filter(r_lesion.match, os.listdir(sub_ses_parent_dir)))  # try to match

    lesions = []  # type: list # will contain the full path(s) to the manual aneurysm mask(s); will be empty for controls
    for aneurysm in lesion_masks:
        lesions.append(os.path.join(sub_ses_parent_dir, aneurysm))

    # ---------- CREATE ORIGINAL ANAT FOLDER
    orig_anat_dir, tof_bids_name = create_orig_anat_folder(new_bids_path,
                                                           sub,
                                                           ses,
                                                           tof_series_path,
                                                           t1_series_path,
                                                           desired_axes_orientations,
                                                           output_file)

    # ---------- CREATE DERIVATIVES FOLDER
    # define "derivatives" folders
    derivatives_path_manual_masks = os.path.join(new_bids_path, "derivatives", "manual_masks")
    derivatives_path_registrations = os.path.join(new_bids_path, "derivatives", "registrations")
    derivatives_n4bfc = os.path.join(new_bids_path, "derivatives", "N4_bias_field_corrected")

    create_derivatives(tof_bids_name,
                       derivatives_path_manual_masks,
                       derivatives_path_registrations,
                       lesions,
                       orig_anat_dir,
                       sub,
                       ses,
                       t1_mni_atlas_path,
                       vessel_mni_atlas_path,
                       derivatives_n4bfc,
                       output_file,
                       desired_axes_orientations)

    # ------------------------------------------------ALL THE CODE BELOW IS JUST FOR PRINTING THE RUNNING TIME--------------------------------------------------
    end = time.time()  # stop timer
    print_running_time(start, end, "Preprocessing time for {}_{}".format(sub, ses), output_file)


def main():
    # the code inside here is run only when THIS script is run, and not just imported
    config_dict = load_config_file()  # load input config file

    # extract input args
    new_bids_path = config_dict['new_bids_path']  # type: str
    new_patients_path = config_dict['new_patients_path']  # type: str
    vessel_mni_atlas_path = config_dict['vessel_mni_atlas_path']
    t1_mni_atlas_path = config_dict['t1_mni_atlas_path']
    out_log_file_path = config_dict['out_log_file_path']
    desired_axes_orientations = tuple(config_dict['desired_axes_orientations'])
    nb_parallel_jobs = config_dict['nb_parallel_jobs']

    date = (datetime.today().strftime('%b_%d_%Y'))  # save today's date
    new_bids_path = new_bids_path.replace("BIDS_Aneurysm", "BIDS_Aneurysm_{}".format(date))
    out_log_file_path = out_log_file_path.replace("_creation", "_creation_{}".format(date))

    all_sub_ses = []
    for sub in sorted(os.listdir(new_patients_path)):
        if os.path.isdir(os.path.join(new_patients_path, sub)):  # only take directories
            for ses in sorted(os.listdir(os.path.join(new_patients_path, sub))):
                if os.path.isdir(os.path.join(new_patients_path, sub, ses)):  # only take directories
                    all_sub_ses.append(os.path.join(new_patients_path, sub, ses))

    assert all_sub_ses, "List of subs should not be empty"

    Parallel(n_jobs=nb_parallel_jobs, backend='loky')(delayed(bidsify_one_subject)(new_bids_path,
                                                                                   sub_ses_path,
                                                                                   vessel_mni_atlas_path,
                                                                                   t1_mni_atlas_path,
                                                                                   out_log_file_path,
                                                                                   desired_axes_orientations) for sub_ses_path in all_sub_ses)


if __name__ == '__main__':
    main()
