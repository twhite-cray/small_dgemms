#!/bin/bash

module load gcc
module load openblas
module load rocm

export KMDUMPISA=1

make -f Makefile.amd clean
make -f Makefile.amd
