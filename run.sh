#!/bin/bash

module load PrgEnv-cray
module load icl-magma
#module load magma
module load openblas
module load cuda10.1
module load gcc

srun -n1 --exclusive -p v100 ./many-small-dgemms.x
