#!/bin/bash

#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --tasks-per-node=12
#SBATCH --exclusive
#SBATCH --time=01:00:00

module purge

set -e
module load mpich/3.3.2

mpiexec -np 12 ./capar 3600 500
