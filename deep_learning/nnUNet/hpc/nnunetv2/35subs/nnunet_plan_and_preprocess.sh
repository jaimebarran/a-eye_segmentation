#!/bin/bash
#SBATCH --job-name=nn_pp
#SBATCH --chdir=/cluster/home/ja4922/AEye/nnunetv2/35subs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaime.barranco-hernandez@chuv.ch
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:00
#SBATCH --output=pp_%N_%j_%a.out
#SBATCH --error=pp_%N_%j_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --gres=gpu:rtx3090:1

export SINGULARITY_TMPDIR=/data/bach/Jaime/singularity/tmp
export SINGULARITY_CACHEDIR=/data/bach/Jaime/singularity/cache

singularity run --bind /data/bach/Jaime/nnUNet_guillaume:/opt/nnunet_resources --nv docker://jaimebarran/nnunet:0.1.0 nnUNetv2_plan_and_preprocess -d 313 -c 3d_fullres --verify_dataset_integrity