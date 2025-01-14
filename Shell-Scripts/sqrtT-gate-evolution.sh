#!/bin/sh
#PBS -N sqrtT-gate-fidelities-L=2.5uH-13
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00:00
#PBS -l mem=4gb


cd $PBS_O_WORKDIR
denom=1


omega=5.28
nu=2
E_J=80
Gamma=1.5 #GHz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=1000
N_periods=4
N_wells=111
N_rungs=50
drive_res_seg_1=25
drive_res_LCJ=20
seg_1_output_res=0
LCJ_output_res=0
stab_output_res=0
free_output_res=0

coeffs_file=../../../../Data/Gate-Coefficients/coeffs-13microsec-L=2.5uH.txt
LCJ_save_path=../../../../Data/Lattice-Data/
data_save_path=../../../../Data/Simulation-Results/sqrtT-gate-fidelities-13microsec-

record_weights=0
seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


python gkp_sqrtT_gate_dynamics_driver.py $omega $nu $E_J $Gamma $Temp $Lambda $N_samples $N_periods $N_wells $N_rungs $drive_res_seg_1 $drive_res_LCJ $seg_1_output_res $LCJ_output_res $coeffs_file $LCJ_save_path $data_save_path $record_weights


conda deactivate 
