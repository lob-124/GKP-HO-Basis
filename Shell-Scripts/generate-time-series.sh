#!/bin/sh
#PBS -N time-series 
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=1:00:00:00
#PBS -l mem=5gb

cd $PBS_O_WORKDIR

Omega=6.283185
omega0=.0001
Lambda=10000

t_start=0
t_stop=.0001
num_t_points=101
num_realizations=5

save_path=../../../../Data/Time-Series/oh-


conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


python time_series_driver.py $Omega $omega0 $Lambda $t_start $t_stop $num_t_points $num_realizations $save_path 

conda deactivate 
