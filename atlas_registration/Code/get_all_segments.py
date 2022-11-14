'''
Converts a set of labels in one unique labels file
'''

import nibabel as nb
from pathlib import Path
import numpy as np

t1 = nb.load("./Data/T1.nii.gz")
print(t1.header)

segments = [nb.load(f) for f in Path("./Data/LABELS_nii").glob("*.nii")]
# segments
print(segments[0])

segmentation = np.zeros_like(segments[0].dataobj, dtype="uint8")
for label, volume in enumerate(segments):
    segmentation[np.asanyarray(volume.dataobj) > 0] = label + 1

header = segments[0].header.copy()
header.set_data_dtype("uint8")

nii = nb.Nifti1Image(segmentation, segments[0].affine, header)
print(nii.header)

nii.to_filename("./Output/all_segments.nii.gz")