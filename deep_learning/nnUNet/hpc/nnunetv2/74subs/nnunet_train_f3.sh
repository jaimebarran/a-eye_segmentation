#!/bin/bash
#SBATCH --job-name=nn_f3
#SBATCH --chdir=/cluster/home/ja4922/AEye/nnunetv2/35subs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaime.barranco-hernandez@chuv.ch
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:00
#SBATCH --output=f3_%N_%j_%a.out
#SBATCH --error=f3_%N_%j_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --gres=gpu:rtx3090:1

export SINGULARITY_TMPDIR=/data/bach/Jaime/singularity/tmp
export SINGULARITY_CACHEDIR=/data/bach/Jaime/singularity/cache
export nnUNet_compile=f

singularity run --bind /data/bach/Jaime/nnUNet_guillaume:/opt/nnunet_resources --nv docker://jaimebarran/nnunet:0.1.0 nnUNetv2_train 313 3d_fullres 3 --npz