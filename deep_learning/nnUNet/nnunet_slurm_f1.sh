#!/bin/bash
#SBATCH --job-name=nn_bs2_f1
#SBATCH --chdir=/home/ja4922/AEye
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaime.barranco-hernandez@chuv.ch
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:00
#SBATCH --output=eye_3d_f1_out_%N.%j.%a.out
#SBATCH --error=eye_3d_f1_err_%N.%j.%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
## Apparently these lines are needed for GPU execution
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --gres=gpu:1
singularity run --bind /data/bach/AEye/nnUNet:/opt/nnunet_resources --nv docker://petermcgor/nnunet:0.0.1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task313_Eye 1 --npz
