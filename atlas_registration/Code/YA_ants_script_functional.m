
imageFixed = '/home/yaleman/testBene/YA_05mm_slice_1.nii';
V = spm_vol(imageFixed);
I  = spm_read_vols(V);
V = rmfield(V,'private');
spm_write_vol(V,I);
Vtotal = V;
Vtotal.fname = '/home/yaleman/testBene/motion_correction_YA_05mm.nii';
Itotal(:,:,1,1) = I;
spm_write_vol(Vtotal(1),squeeze(Itotal(:,:,1,1))); 

Nimages = 6;
for i = 2:Nimages
   
    imageMove = ['/home/yaleman/testBene/YA_05mm_slice_' num2str(i) '.nii'];
    V1 = spm_vol(imageMove);
    I1 = spm_read_vols(V1);
    V1 = rmfield(V1,'private');
    spm_write_vol(V1,I1);
    
    defFile = ['/home/yaleman/testBene/defo_' num2str(i) '.nii.gz'];
    outMove = ['/home/yaleman/testBene/YA_05mm_slice_' num2str(i) '_Warp.nii'];
    
    cad = ['antsRegistrationSyN.sh -d 2 -f ' imageFixed ' -m ' imageMove ' -t s -n 8 -o ' defFile(1:end-7)];
    system(cad);
    
    cad = ['antsApplyTransforms -d 2 -i ' imageMove ' -o ' outMove ' -r ' imageFixed ' -t ' defFile(1:end-7) '1Warp.nii.gz -n BSpline[2] -v ' ];
    system(cad);
    Vz = spm_vol(outMove);
    Vz = rmfield(Vz,'private');
    Iz = spm_read_vols_gzip(Vz);
    Vz.mat = V.mat;
    spm_write_vol(Vz,Iz);
    Itotal(:,:,1,i) = Iz;
    Vtotal(i) = Vz;
    Vtotal(i).fname = '/home/yaleman/testBene/motion_correction_YA_05mm.nii';
    Vtotal(i).n = [i 1];
    spm_write_vol(Vtotal(i),squeeze(Itotal(:,:,1,i))); 
end
a = 1;

