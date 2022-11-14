import spm as spm
import numpy as np
import os

# rmfield in python
# 1
def rmfield( a, *fieldnames_to_remove ):
    return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]
# 2
def remove_field(a, name):
    names = list(a.dtype.names)
    if name in names:
        names.remove(name)
    b = a[names]
    return b

imageFixed = '/home/yaleman/testBene/YA_05mm_slice_1.nii'
V = spm.vol(imageFixed)
I = spm.read_vols(V)
V = rmfield(V,'private')
spm.write_vol(V,I)
Vtotal = V
Vtotal.fname = '/home/yaleman/testBene/motion_correction_YA_05mm.nii'
Itotal = I # ??
Itotal[:,:,1,1] = I
spm.write_vol(Vtotal(1),np.squeeze(Itotal[:,:,1,1]))
   
Nimages=6
for i in range(2,Nimages-1):
        imageMove = '/home/yaleman/testBene/YA_05mm_slice_' + str(i) + '.nii'
        V1 = spm.vol(imageMove)
        I1 = spm.read_vols(V1)
        V1 = rmfield(V1,'private')
        spm.write_vol(V1,I1)

        defFile = '/home/yaleman/testBene/defo_' + str(i) + '.nii.gz'
        outMove = '/home/yaleman/testBene/YA_05mm_slice_' + str(i) + '_Warp.nii'

        cad = 'antsRegistrationSyN.sh -d 2 -f ' + imageFixed + ' -m ' + imageMove + ' -t s -n 8 -o ' + defFile[1:-8]
        os.system(cad)

        cad = 'antsApplyTransforms -d 2 -i ' + imageMove + ' -o ' + outMove + ' -r ' + imageFixed + ' -t ' + defFile[1:-8] + '1Warp.nii.gz -n BSpline[2] -v '
        os.system(cad)
        Vz = spm.vol(outMove)
        Vz = rmfield(Vz,'private')
        Iz = spm.read_vols_gzip(Vz)
        Vz.mat = V.mat
        spm.write_vol(Vz,Iz)
        Itotal[:,:,1,i] = Iz
        Vtotal[i] = Vz
        Vtotal[i].fname = '/home/yaleman/testBene/motion_correction_YA_05mm.nii'
        Vtotal[i].n = np.array([i,1])
        spm.write_vol(Vtotal(i),np.squeeze(Itotal[:,:,1,i]))

a=1