#!/bin/sh
#PBS -N sqrtT-gate-mistargeted-fidelities-manyGHz-2.5uH-7.5-2cleanup-f=.008
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=14:00:00:00
#PBS -l mem=3gb
#PBS -t 0-24%10


cd $PBS_O_WORKDIR

f=0.008


omega=5.28
nu=2
E_J=100
Gamma=1.5 #GHz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=25000
N_periods=2
N_wells=111
N_rungs=50
drive_res_seg_1=25
drive_res_LCJ=20

param_num=7.5

LCJ_save_path=../../../../Data/Lattice-Data/

seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv

#for E_J in 100 150 200 250
#do
for i in $(seq 1 1 10)
do

file_num=$(echo "$i + 10*$PBS_ARRAYID" | bc)
coeffs_file=../../../../Data/Gate-Coefficients/Mistargeted/${param_num}-microsec/coeffs-${param_num}microsec-L=2.5uH-f=${f}-${file_num}.txt

data_save_path=../../../../Data/Simulation-Results/sqrtT-gate-mistargeted-fidelities-MANY-TRAJECTORIES-${param_num}microsec-${N_periods}periods-f=${f}-${file_num}-


python gkp_sqrtT_gate_dynamics_driver.py $omega $nu $E_J $Gamma $Temp $Lambda $N_samples $N_periods $N_wells $N_rungs $drive_res_seg_1 $drive_res_LCJ $coeffs_file $LCJ_save_path $data_save_path 

done

#done

conda deactivate
