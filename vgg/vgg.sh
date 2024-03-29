#!/bin/bash
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=48:00:00
#PBS -l wd
#PBS -l storage=scratch/po21
#PBS -m abe
#PBS -N temp
#PBS -l jobfs=100GB

module load cuda/10.1
module load cudnn/7.6.5-cuda10.1 
source /home/549/jz1585/.miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate tf

cd /scratch/po21/jz1585/VGG/

#generate .pickle 
python generate_data_list.py
#train vgg
python train.py
#gradcam
python gradcampp.py

