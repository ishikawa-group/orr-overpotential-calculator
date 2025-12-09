#!/bin/bash
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=24:00:00
#$ -N ORR_sample
#$ -o ./example/log/RPBE_output.log
#$ -e ./example/log/RPBE_error.log

# Load required modules
module load intel
module load intel-mpi
module load cuda

# Activate Python virtual environment
source /path/to/venv/bin/activate

# Set VASP pseudopotential path for ASE
export VASP_PP_PATH=/path/to/vasp/pseudopotentials
export VASP_SCRIPT=/path/to/repo/src/orr_overpotential_calculator/run_vasp/run_vasp.py

# Run python script
python3 ./example/run_test.py
