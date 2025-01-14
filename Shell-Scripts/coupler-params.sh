#!/bin/sh
#PBS -N coupler
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=14:00:00:00
#PBS -l mem=4gb
#PBS -t 1-20

cd $PBS_O_WORKDIR


E_J_0=0.9
dE_J=.01

omega_start=0.9	#In  GHz
omega_stop=1.1
omega_step=.005
Z_start=0.45	#In h/4e^2
Z_stop=0.55
Z_step=.005
flux_start=-0.1	#In Phi_0/2
flux_stop=0.8
flux_step=.005

data_save_path=../../../../Data/Simulation-Results/coupler-params-


conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv

E_J=$(echo "scale=2; $E_J_0 + $dE_J*$PBS_ARRAYID" | bc -l | awk '{printf "%f",$0}')

python coupler_params.py $E_J $omega_start $omega_stop $omega_step $Z_start $Z_stop $Z_step $flux_start $flux_stop $flux_step $data_save_path

conda deactivate 
