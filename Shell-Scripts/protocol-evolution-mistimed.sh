#!/bin/sh
#PBS -N protocol-mistimed-fidelities-10-periods-2.5uH-high
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=5:00:00:00
#PBS -l mem=3gb
#PBS -t 1-10


cd $PBS_O_WORKDIR


omega=5.28
nu=2
E_J=$(echo "40*$PBS_ARRAYID" | bc)
Gamma=1.5 #GHz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=10000
N_periods=10
N_wells=111
N_rungs=50
drive_res_LCJ=20
drive_res_mistimed=10

LCJ_save_path=../../../../Data/Lattice-Data/
data_save_path=../../../../Data/Simulation-Results/protocol-mistimed-fidelities-10-periods-

record_weights=0
seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


for i in $(seq 16 1 30)
do

delta_t=$(echo "scale=1; 4 + 0.2*$i" | bc -l)

python gkp_mistimed_dynamics_driver.py $omega $nu $E_J $delta_t $Gamma $Temp $Lambda $N_samples $N_periods $N_wells $N_rungs $drive_res_LCJ $drive_res_mistimed $LCJ_save_path $data_save_path $record_weights

done

conda deactivate
