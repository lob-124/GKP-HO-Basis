#!/bin/sh
#PBS -N T-gate-native-long
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=28:00:00:00
#PBS -l mem=20gb

cd $PBS_O_WORKDIR
denom=1


omega=1.32
E_J=200
nu=2
z_s=10000
#Gamma_0=1.0 #GHz
#Gamma=$(echo "scale=2; $Gamma_0 * $PBS_ARRAYID / $num_jobs" | bc -l | awk '{printf "%f",$0}')
Gamma=1.0 #GHz
#Gamma=Gamma_0
gamma_q=1e-6 #e^2/THz
#gamma_pL_0=100.0 #kHz
#gamma_pL=$(echo "scale=3; $gamma_pL_0 * $PBS_ARRAYID / $denom" | bc -l | awk '{printf "%f",$0}')
gamma_pL=0
Temp=0.04 #K
Lambda=500 #GHz

num_samples=1
periods=1
max_wells=111
max_rungs=75
drive_res=20
stab_output_res=20
free_output_res=0

LCJ_save_path=../../../../Data/Lattice-Data/
data_save_path=../../../../Data/Simulation-Results/T-gate-native-long-

record_weights=0
#seedSSE=$(echo "2048 * $PBS_ARRAYID" | bc -l)
seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


python gkp_dynamics_driver.py $omega $E_J $nu $z_s $Gamma $gamma_q $gamma_pL $Temp $Lambda $num_samples $periods $max_wells $max_rungs $drive_res $stab_output_res $free_output_res $LCJ_save_path $data_save_path $record_weights $seedSSE $seedinit

conda deactivate 
