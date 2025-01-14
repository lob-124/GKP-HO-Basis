#!/bin/sh
#PBS -N mistiming-dt=10-data
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=14:00:00:00
#PBS -l mem=5gb
#PBS -t 1-3

cd $PBS_O_WORKDIR
denom=1


omega=1.32
E_J=200
nu=2
z_s=9
delta_t=10
#Gamma_0=1.0 #GHz
#Gamma=$(echo "scale=2; $Gamma_0 * $PBS_ARRAYID / $num_jobs" | bc -l | awk '{printf "%f",$0}')
Gamma=0.5 #GHz
#Gamma=Gamma_0
gamma_q=1e-5 #e^2/THz
#gamma_pL_0=100.0 #kHz
#gamma_pL=$(echo "scale=3; $gamma_pL_0 * $PBS_ARRAYID / $denom" | bc -l | awk '{printf "%f",$0}')
gamma_pL=0
Temp=0.04 #K
Lambda=500 #GHz

num_samples=1
periods=1000
max_wells=111
max_rungs=100
drive_res=20
stab_output_res=0
free_output_res=0

LCJ_save_path=../../../../Data/Lattice-Data/
data_save_path=../../../../Data/Simulation-Results/mistiming-$PBS_ARRAYID-with-zerod-weights-

record_weights=0
seedSSE=$(echo "2048 * $PBS_ARRAYID" | bc -l)
#seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


python gkp_dynamics_driver_2.py $omega $E_J $nu $z_s $delta_t $Gamma $gamma_q $gamma_pL $Temp $Lambda $num_samples $periods $max_wells $max_rungs $drive_res $stab_output_res $free_output_res $LCJ_save_path $data_save_path $record_weights $seedSSE $seedinit

conda deactivate 
