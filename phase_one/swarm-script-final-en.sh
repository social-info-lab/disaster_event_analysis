#!/bin/bash
#
#SBATCH --job-name=mediacloud
#SBATCH --output=/mnt/nfs/work1/brenocon/ecai/mediacloud-output-11.21/%x-%j.out  # output file
#SBATCH --error=/mnt/nfs/work1/brenocon/ecai/mediacloud-output-11.21/%x-%j.err       # File to which STDERR will be written
#SBATCH --partition=defq    # Partition to submit to
#SBATCH --ntasks-per-node=8
#SBATCH --time=12:00:00
#SBATCH --mem=800
module add python/3.11.5
python grep-phase1-exp-final-en.py $1 $2 $3
