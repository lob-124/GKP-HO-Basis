#!/bin/sh
#PBS -N T-gate-fidelity-probes
#PBS -e ../../../../Error/$PBS_JOBNAME.err
#PBS -o ../../../../Output/$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00:00
#PBS -l mem=2gb
#PBS -t 106-125


cd $PBS_O_WORKDIR
denom=1


omega=1.32
nu=2
#L_frac=$(echo "scale=4; 0.98 + .0002*$PBS_ARRAYID" | bc -l | awk '{printf "%f",$0}')
L_frac=$(echo "scale=4; 0.9875 + .0001*$PBS_ARRAYID" | bc -l | awk '{printf "%f",$0}')
#L_frac=0.99
E_J=60
#E_J_prime=0.1
#E_J_prime=$(echo "scale=3; 0.1 + .01*$PBS_ARRAYID" | bc -l | awk '{printf "%f",$0}') 
Gamma=1.5 #GHz
Temp=0.04 #K
Lambda=500 #GHz

N_samples=50
N_periods=4
N_wells=91
N_rungs=50
drive_res_LCJJ=25
drive_res_LCJ=20
LCJJ_output_res=0
LCJ_output_res=0
stab_output_res=0
free_output_res=0

LCJJ_save_path=../../../../Data/Lattice-Data/
data_save_path=../../../../Data/Simulation-Results/more-samples-T-gate-fidelities-timing-

record_weights=0
#seedSSE=$(echo "2048 * $PBS_ARRAYID" | bc -l)
seedSSE=2048
seedinit=1024

conda_path=/home/lobrien/miniconda3
source ${conda_path}/etc/profile.d/conda.sh
conda activate myenv

for i in $(seq 1 50);
do
E_J_prime=$(echo "scale=3; .01*$i" | bc -l | awk '{printf "%f",$0}') 

python gkp_T_gate_dynamics_driver.py $omega $nu $L_frac $E_J $E_J_prime $Gamma $Temp $Lambda $N_samples $N_periods $N_wells $N_rungs $drive_res_LCJJ $drive_res_LCJ $LCJJ_output_res $LCJ_output_res $stab_output_res $free_output_res $LCJJ_save_path $data_save_path $record_weights

done
conda deactivate 
