import numpy
import os,sys
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
#import getEyeCenter

# bash command
# $ bash /opt/ANTs/bin/antsBrainExtraction.sh -d 3 -a ../input/tpl-MNIColin27_T1w_cropEye.nii.gz -e ../input/SoenkeSample_cropEye.nii.gz -m ../input/tpl-MNIColin27_desc-brain_mask.nii.gz -o output -k 1

def cropVolumeImage(image,x,y,z,border1,border2,border3):
    
    xSize, ySize, zSize = image.GetSize()
    #xMin,yMin,zMin = image.TransformPhysicalPointToIndex([x-border,y-border,z-border])
    #xMax,yMax,zMax = image.TransformPhysicalPointToIndex([x+border,y+border,z+border])

    #x,y,z = image.TransformPhysicalPointToIndex([x,y,z])
    xMin = x - int(border1/2) 
    xMax = x + int(border1/2) 

    yMin = y - int(border2/2) 
    yMax = y + int(border2/2)  

    zMin = z - int(border3/2) 
    zMax = z + int(border3/2)
    
    # Define limits
    if xMin< 0 :
        xMin = 0  
    if xMax> xSize:
        xMax = xSize

    if yMin< 0 :
        yMin = 0
    if yMax > ySize:
        yMax = ySize

    if zMin< 0 :
        zMin = 0
    if zMax > zSize:
        zMax = zSize
   
    imageReturn = image[int(xMin):int(xMax),int(yMin):int(yMax),int(zMin):int(zMax)]
    return imageReturn

    
if __name__ == "__main__":
    #argv list: MriFile(.nii), scleraFile(.nii), size, output

    main_path = '/mnt/sda1/Repos/a-eye/Data/SHIP_dataset/'
    # mask filenames as second column.
    csv_path = '/mnt/sda1/Ocular_tumor_detection/MainProcess/PythonCode/1.Pre-process/Data/idx1.csv' #main_path + 'idx1.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    path = csv_path[:csv_path.rfind('/')] + '/' #+ '/'
    print(path)
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
        # if (item[0][0:1]=="E") or(item[0][0:1]=="P"):
            if(numpy.isnan(item[1]) == False):
                mriFile  = "/T1.nii.gz" #main_path + 'non_labeled_dataset_nifti/'
                print(sys.argv[1]+mriFile)
    #             x = item[1]
    #             y = item[2]
    #             z = item[3]
                
    #             # image = sitk.ReadImage(sys.argv[1]+item[0]+mriFile)
    #             image = sitk.ReadImage(sys.argv[1]+mriFile)
    #             # directory = "./Output/OnlyEyeRegion/"+item[0]
    #             directory = "./cropped/"+item[0]
    #             if not os.path.exists(directory):
    #                 os.makedirs(directory)
    #             binaryImage = cropVolumeImage(image,x,y,z,96,96,96)
    #         sitk.WriteImage( binaryImage, directory+mriFile )
    #         if (i==10): 
    #             break
            # if(numpy.isnan(item[4]) == False):
            #     mriFile  = "/T2.nii"
            #     x = item[4]
            #     y = item[5]
            #     z = item[6]
                
            #     image = sitk.ReadImage(sys.argv[1]+item[0]+mriFile)
            #     directory = "./Output/OnlyEyeRegion/"+item[0]
            #     if not os.path.exists(directory):
            #         os.makedirs(directory)
            #     binaryImage = cropVolumeImage(image,x,y,z,64,64,64)
            #     sitk.WriteImage( binaryImage, directory+mriFile )

