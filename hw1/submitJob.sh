#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l walltime=100:00:00
#PBS -l mem=64GB
#PBS -N doall
#PBS -M mc3784@nyu.edu,ajr619@nyu.edu
#PBS -j oe

module purge

SRCDIR=$HOME/workspace/Deep-Learning/hw1/
RUNDIR=$SCRATCH/hw1/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
cp -R $SRCDIR/* $RUNDIR

cd $RUNDIR

module load torch

th doall.lua
