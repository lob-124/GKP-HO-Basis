#!/bin/sh
#PBS -N sqrtT-gate-mistimed-fidelities-125GHz-2.5uH-2.7-2cleanup
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=14:00:00:00
#PBS -l mem=3gb
#PBS -t 1-10


cd $PBS_O_WORKDIR


omega=5.28
nu=2
E_J=125
Gamma=1.5 #GHz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=100000
N_periods=2
N_wells=111
N_rungs=50
drive_res_seg_1=25
drive_res_LCJ=20
drive_res_mistimed=10

coeffs_file=../../../../Data/Gate-Coefficients/coeffs-7.5microsec-L=2.5uH.txt
LCJ_save_path=../../../../Data/Lattice-Data/
data_save_path=../../../../Data/Simulation-Results/sqrtT-gate-mistimed-fidelities-MANY-TRAJECTORIES-${PBS_ARRAYID}-7.5microsec-2periods-

record_weights=0
seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv


#for i in $(seq 2 2 20)
for i in 3 5 7 ;
do

#delta_t=$(echo "scale=1; 0.1*($i + 20*$PBS_ARRAYID)" | bc -l)
delta_t=$i

python gkp_sqrtT_gate_mistimed_dynamics_driver_v2.py $omega $nu $E_J $delta_t $Gamma $Temp $Lambda $N_samples $N_periods $N_wells $N_rungs $drive_res_seg_1 $drive_res_LCJ $drive_res_mistimed $coeffs_file $LCJ_save_path $data_save_path

done

conda deactivate
