#!/bin/sh
#PBS -N revival-decay-L=2.5uH
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00:00
#PBS -l mem=2gb
#PBS -t 1-10


cd $PBS_O_WORKDIR


omega=5.28
f=$(echo "scale=3; $omega/(6.283)" | bc -l | awk '{printf "%.3f",$0}')
nu=2
#E_J=100
Gamma=1 #GHz
gamma_q=.01 #e^2/THz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=10
N_periods=4
N_revivals=20000
N_wells=111
N_rungs=50
drive_res=20

LCJ_save_path=../../../../Data/Lattice-Data/
LCJ_save_data=0
data_save_path=../../../../Data/Simulation-Results/revival-decay-${PBS_ARRAYID}-

seed_SSE=$(echo "2048*$PBS_ARRAYID" | bc)
seed_init=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv

#for lambda in $(seq 400 200 2000)
for i in $(seq 1 1 10)
do

lambda=$(echo "scale=3; 0.05 + $i*.005" | bc -l | awk '{printf "%.3f",$0}')
#E_J=$(echo "scale=3; $f/(124.025*($lambda*$lambda*$lambda*$lambda))" | bc -l | awk '{printf "%.3f",$0}') 
E_J=$(echo "scale=12; $f/(124.025*$lambda*$lambda*$lambda*$lambda)" | bc -l | awk '{printf "%.3f",$0}')

python gkp_revival_decay_driver.py $omega $E_J $nu $Gamma $gamma_q $Temp $Lambda $N_samples $N_periods $N_revivals $N_wells $N_rungs $drive_res $LCJ_save_path $LCJ_save_data $data_save_path $seed_SSE $seed_init

done

conda deactivate 
