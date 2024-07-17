#!/bin/bash
#SBATCH --job-name=nn_best_conf
#SBATCH --chdir=/home/ja4922/AEye
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaime.barranco-hernandez@chuv.ch
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:00
#SBATCH --output=eye_3d_inf_out_%N.%j.%a.out
#SBATCH --error=eye_3d_inf_err_%N.%j.%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --gres=gpu:1
singularity run --bind /data/bach/Jaime/nnUNet_guillaume:/opt/nnunet_resources --nv docker://petermcgor/nnunetv2:0.4.0 nnUNet_find_best_configuration 313