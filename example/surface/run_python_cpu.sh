#!/bin/bash
#$ -cwd
#$ -l cpu_160=1
#$ -l h_rt=24:00:00
#$ -N ORR_Rh111
#$ -o /gs/fs/tga-ishikawalab/wakamiya/orr_overpotential_calculator/example/surface/Rh111/log/RPBE_output.log
#$ -e /gs/fs/tga-ishikawalab/wakamiya/orr_overpotential_calculator/example/surface/Rh111/log/RPBE_error.log

# Load required modules
module load intel
module load intel-mpi
module load cuda

# Activate Python virtual environment
source /gs/fs/tga-ishikawalab/wakamiya/python_virtual_env/ORR_catalyst_generator_env/bin/activate

# Set VASP pseudopotential path for ASE
export VASP_PP_PATH=/gs/fs/tga-ishikawalab/vasp/potential
export VASP_SCRIPT=/gs/fs/tga-ishikawalab/wakamiya/orr_overpotential_calculator/src/orr_overpotential_calculator/run_vasp/run_vasp.py

# Run python script
python3 /gs/fs/tga-ishikawalab/wakamiya/orr_overpotential_calculator/example/surface/run_Rh111.py