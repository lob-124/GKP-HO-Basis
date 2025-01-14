#!/bin/sh
#PBS -N time-series
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00:00
#PBS -l mem=1gb
#PBS -t 3-99%5


cd $PBS_O_WORKDIR


Omega=6.283185
omega0=.0001
Lambda=1e6

t_start=0
t_stop=.0001
num_t_points=101
num_realizations=1

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


for i in $(seq 0 1 9);
do

file_num=$(echo "$i + 10*$PBS_ARRAYID" | bc)

save_path=../../../../Data/Time-Series/series-test-${file_num}-

python time_series_driver.py $Omega $omega0 $Lambda $t_start $t_stop $num_t_points $num_realizations $save_path

done

conda deactivate 
