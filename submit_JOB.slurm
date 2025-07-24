#!/bin/bash
#
# Simple SLURM script for submitting multiple serial commands (e.g. parametric studies) 
# using a script wrapper to launch the commands.
#
# To use, change this job script to accommodate 
# running your serial application(s) in your WORKDIR directory (usually the directory of submission).
# Edit the commands file to specify the executions for the launcher (paramrun) to perform.
#
# To run this script, run the follow in the working directory
# sbatch submit_JOB.slurm
#
#-----------------------------------------------------------------------
#                      Get the job Info
#-----------------------------------------------------------------------
# showq -u                  # to show all job under my ID
# squeue -u userID --start  # to get a estimated time
# sstat --jobs=jobID        # stats about active jobs
# scancel jobID             # to cancel a job using its ID
#
#
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#
#         <------------- Setup Parameters ------------>
#                        DO NOT UNCOMMENT
#
#SBATCH -J ESM2ne_MLM                      # Job name
#SBATCH -e jobs/%x_%j_stderr.txt        # Name of stderr error file
#SBATCH -o jobs/%x_%j_stdout.txt        # Name of stdout output file
#SBATCH -p gpu-h100                     # Queue (partition) name
#SBATCH --nodes=1                       # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=2
#SBATCH -t 48:00:00                     # Run time (hh:mm:ss)
#SBATCH -A MCB24023                     # Project/Allocation name (req'd if you have more than 1)
#-----------------------------------------------------------------------
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --mail-user=luiz.vieira@utexas.edu





#-----------------------------------------------------------------------
#                             USING SLURM
# Below, is the script that actually do the job.
# Activate the env, with all the necessery dependencies.
# Pass all the path to input and output files, and scripts.
#-----------------------------------------------------------------------


# Activate the virtual enviroment need to run the job
source venv2/bin/activate


########################### This works to use two gpus in one Node ###########################
# -u for unbuffered, so I can see the output with tail -f

# HIS7_YEAST_Kondrashov2017_muts
# CUDA_VISIBLE_DEVICES=0 python -u scripts/extract.py esm2_t6_8M_UR50D data_muts_part1.fasta embeddings/esm2_8M_mean --repr_layers 6 --include mean &
# CUDA_VISIBLE_DEVICES=1 python -u scripts/extract.py esm2_t6_8M_UR50D data_muts_part2.fasta embeddings/esm2_8M_mean --repr_layers 6 --include mean


########################### Loop in bash file ###########################
# This will call a bash script that will run the commands above but in a loop from a bash fil
#python src/training/train.py -o vicam_600m/CRVDBv29_maxLen2046_Full_lr1e6_RLRP
#python src/training/train.py -o ViCAM_300M/CRVDBv29_maxLen2046_Full_lr5e4_RLRP_test
srun python src/training/train_esm2.py -o esm2_vicam_650m/CRVDBv29_maxLen1022_Full_lr4e4_RLRP
