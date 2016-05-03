#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=8:00:00
#PBS -l mem=64GB
#PBS -N torch_runner
#PBS -M ajr619@nyu.edu
#PBS -m e
#PBS -j oe

SRCDIR=$HOME/workspace/Deep-Learning/hw4/
RUNDIR=$SCRATCH/Deep-Learning/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
cp -R $SRCDIR/* $RUNDIR

cd $RUNDIR

module purge
module load torch/intel/20151009

th main.lua
