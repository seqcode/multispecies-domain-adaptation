#!/bin/bash
#PBS -W umask=0007  
#PBS -W group_list=sam77_collab
#PBS -j oe
#PBS -l walltime=20:00:00
#PBS -l nodes=1:ppn=1:gpus=1:rhel7
#PBS -l pmem=1gb
#PBS -A sam77_e_g_gc_default

module load cuda/11.1.0

#Check GPUs

nvidia-smi

nvcc --version
