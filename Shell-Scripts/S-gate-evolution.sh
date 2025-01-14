#!/bin/sh
#PBS -N S-gate-fidelities
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00:00
#PBS -l mem=3gb
#PBS -t 0-9


cd $PBS_O_WORKDIR


omega=5.28
nu=2
E_J=200
Gamma=1.5 #GHz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=1000
N_periods=2
N_wells=111
N_rungs=50
drive_res_LC=25
drive_res_LCJ=20
drive_res_mistiming=10

LCJ_save_path=../../../../Data/Lattice-Data/

seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


for i in $(seq 1 1 5);
do

delta_t=$(echo "scale=1; $PBS_ARRAYID + 0.2*$i"| bc -l)

data_save_path=../../../../Data/Simulation-Results/S-gate-fidelities-mistimed-

python gkp_S_gate_mistimed_dynamics_driver.py $omega $nu $E_J $delta_t $Gamma $Temp $Lambda $N_samples $N_periods $N_wells $N_rungs $drive_res_LC $drive_res_LCJ $drive_res_mistiming $LCJ_save_path $data_save_path

done

conda deactivate 
