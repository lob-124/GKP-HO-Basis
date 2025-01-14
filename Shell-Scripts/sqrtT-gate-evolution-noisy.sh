#!/bin/sh
#PBS -N sqrtT-noisy-fidelities-200GHz-N=8-2.5uH-2cleanup
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=5:00:00:00
#PBS -l mem=1gb
#PBS -t 0-99%10

cd $PBS_O_WORKDIR


omega=5.28
nu=2
E_J=200
gamma_q=0 #e^2/THz
#gamma_phi=0.1 #Phi_0^2/THz
Gamma=1.5 #GHz
Temp=0.04 #K
Lambda=500 #GHz
Omega=6.283185
omega_0=.0001 
_Lambda=1e6

N_decoupling=8
N_samples=250
N_periods=2
N_wells=111
N_rungs=50
drive_res_seg1=25
drive_res_LCJ=20
#output_res_seg1=0
#output_res_LCJ=0

param_num=7.5

coeffs_file=../../../../Data/Gate-Coefficients/coeffs-${param_num}microsec-L=2.5uH.txt
#noise_file=../../../../Data/Time-Series/series-test-${num}-Omega=${Omega}HZ-omega0=${omega_0}Hz-Lambda=${_Lambda}Hz.dat
LCJ_save_path=../../../../Data/Lattice-Data/
#data_save_path=../../../../Data/Simulation-Results/sqrtT-gate-noise-${param_num}microsec-${N_periods}periods-${num}

seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv

for i in $(seq 1 1 20)
do 

gamma_phi=$(echo "scale=1; 0.1*$i"| bc -l)

for j in $(seq 0 1 9)
do

num=$(echo "10*$PBS_ARRAYID + $j" | bc)

noise_file=../../../../Data/Time-Series/series-test-${num}-Omega=${Omega}Hz-omega0=${omega_0}Hz-Lambda=${_Lambda}Hz.dat
data_save_path=../../../../Data/Simulation-Results/sqrtT-gate-noise-UDD-${param_num}microsec-${N_periods}periods-${num}-

python gkp_sqrtT_gate_noisy_dynamics_driver_v5.py $omega $nu $E_J $gamma_phi $Gamma $Temp $Lambda $N_decoupling $N_samples $N_periods $N_wells $N_rungs $drive_res_seg1 $drive_res_LCJ $coeffs_file $noise_file $LCJ_save_path $data_save_path 

done

done

conda deactivate
