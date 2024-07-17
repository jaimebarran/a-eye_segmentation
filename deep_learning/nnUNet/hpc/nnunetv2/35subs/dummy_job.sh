#!/bin/bash
#SBATCH --job-name=dj
#SBATCH --chdir=/cluster/home/ja4922/AEye/nnunetv2/35subs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaime.barranco-hernandez@chuv.ch
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:00
#SBATCH --output=dj_%N_%j_%a.out
#SBATCH --error=dj_%N_%j_%a.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --account rad
#SBATCH --partition rad
# #SBATCH --gres=gpu:rtx3090:1

# Infinite loop that does nothing
while true
do
    :
done
