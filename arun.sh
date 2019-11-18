#!/bin/bash

module load gcc openblas rocm
ulimit -c unlimited
rm -f core*
srun -n1 --exclusive -p amdMI60 ./many-small-dgemms.x
