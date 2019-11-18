#!/bin/bash

module load gcc
module load openblas
module load rocm

#make -f Makefile.amd clean
make -f Makefile.amd
