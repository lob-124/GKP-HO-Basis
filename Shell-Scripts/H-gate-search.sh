#!/bin/sh
#PBS -N H-gate-search
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00:00
#PBS -l mem=3gb


cd $PBS_O_WORKDIR
denom=1


#omega=1.32
nu=2
delta_t=16.667
Gamma=2 #GHz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=250
N_periods=4
N_wells=111
N_rungs=50
drive_res_LC=25
drive_res_LCJ=20
LC_output_res=0
LCJ_output_res=0

LCJ_save_path=../../../../Data/Lattice-Data/

record_weights=0
seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


for i in $(seq 1 1 99);
do
omega=$(echo "scale=3; 1.32 + 0.132*$i" | bc -l)
E_J=$(echo "scale=3; 10*$omega/1.32" | bc -l)

data_save_path=../../../../Data/Simulation-Results/H-gate-search

python gkp_H_gate_mistimed_dynamics_driver.py $omega $nu $E_J $delta_t $Gamma $Temp $Lambda $N_samples $N_periods $N_wells $N_rungs $drive_res_LC $drive_res_LCJ $LC_output_res $LCJ_save_path $data_save_path $record_weights

done

conda deactivate 
