#!/bin/bash
#SBATCH --job-name=nn_pe
#SBATCH --chdir=/cluster/home/ja4922/AEye/nnunetv2/35subs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaime.barranco-hernandez@chuv.ch
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:00
#SBATCH --output=pe_%N_%j_%a.out
#SBATCH --error=pe_%N_%j_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --gres=gpu:1
export SINGULARITY_TMPDIR=/data/bach/Jaime/singularity/tmp
export SINGULARITY_CACHEDIR=/data/bach/Jaime/singularity/cache
singularity run --bind /data/bach/Jaime/nnUNet_guillaume:/opt/nnunet_resources --nv docker://petermcgor/nnunetv2:0.4.0 nnUNetv2_plan_experiment -d 313 -pl nnUNetPlannerResEncM