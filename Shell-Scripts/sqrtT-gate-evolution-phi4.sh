#!/bin/sh
#PBS -N sqrtT-gate-phi4-fidelity-new
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00:00
#PBS -l mem=3gb
#PBS -t 96


cd $PBS_O_WORKDIR
denom=1


omega=5.28
nu=2
#E_J=200
E_J_prime=0.1
#coeff_4=$(echo "scale=11; .00000001*$PBS_ARRAYID" | bc -l | awk '{printf "%.8f",$0}') 
#coeff_4=$(echo "scale=12; e(-20.94387527+.06907755*($PBS_ARRAYID-10))" | bc -l | awk '{printf "%.12f",$0}')
coeff_4=$(echo "scale=12; e(-20.94387527+.06907755*$PBS_ARRAYID)" | bc -l | awk '{printf "%.12f",$0}')
Gamma=1.5 #GHz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=10000
N_periods=2
N_wells=111
N_rungs=50
drive_res_phi4=25
drive_res_LCJ=20
phi4_output_res=0
LCJ_output_res=0
stab_output_res=0
free_output_res=0

LCJ_save_path=../../../../Data/Lattice-Data/
data_save_path=../../../../Data/Simulation-Results/phi4-sqrtT-gate-fidelities-

record_weights=0
seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv

#for i in $(seq 235 -2 35);
#for i in $(seq 150 -1 15);
#for i in $(seq 1500 -10 150);
for i in $(seq 100 50 300);
do
E_J=$(echo "1*$i" | bc)
python gkp_sqrtT_gate_dynamics_phi4_driver.py $omega $nu $E_J $coeff_4 $Gamma $Temp $Lambda $N_samples $N_periods $N_wells $N_rungs $drive_res_phi4 $drive_res_LCJ $phi4_output_res $LCJ_output_res $stab_output_res $free_output_res $LCJ_save_path $data_save_path $record_weights

done

conda deactivate 
