#!/bin/bash

module load PrgEnv-cray
#module load magma
module load icl-magma
module load openblas
module load cuda10.1
module load gcc

make clean
make
