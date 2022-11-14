import nibabel as nb
import numpy as np
from pathlib import Path

# base_dir = '/mnt/sda1/ANTs/a123/'
base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/'

# Create mask from all subjects' labels
segments = [nb.load(f) for f in Path(base_dir).glob("*_labels2template5.nii.gz")]
# segments = [nb.load(f) for f in Path(base_dir).rglob("reg_cropped_best_subjects/*/*labels2template.nii.gz")]
# print(len(segments))
segmentation = np.zeros_like(segments[0].dataobj, dtype="uint8")
for label, volume in enumerate(segments):
    segmentation[np.asanyarray(volume.dataobj) > 0] = 1
header = segments[0].header.copy()
header.set_data_dtype("uint8")
nii = nb.Nifti1Image(segmentation, segments[0].affine, header)
# print(nii.header)
nii.to_filename(base_dir+"all_segments_mask_PRUEBAS.nii.gz")