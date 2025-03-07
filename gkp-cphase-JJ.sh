#!/bin/sh


omega_A=5.28
J_A=200
nu_A=2
Gamma_A=1
omega_B=5.28
J_B=200
nu_B=2
Gamma_B=1
J_C=0.1
f_A=0.5
f_B=0.5
Temp=.04
Lambda=500
N_samples=1
N_wells_A=31
N_rungs_A=10
N_wells_B=31
N_rungs_B=10
drive_resolution_order=20
output_resolution_order=4
LCJ_save_path=./Data/LCJ-Data/
data_save_path=./Data/Simulation-Results/

seed_SSE=1024
seed_init=2048

#conda_path=
source activate base 
conda activate GKP-env


python gkp_cphase_gate_JJ_dynamics_driver.py $omega_A $J_A $nu_A $Gamma_A $omega_B $J_B $nu_B $Gamma_B $J_C $f_A $f_B $Temp $Lambda $N_samples $N_wells_A $N_rungs_A $N_wells_B $N_rungs_B $drive_resolution_order $output_resolution_order $LCJ_save_path $data_save_path $seed_SSE $seed_init

# for n_rungs in 10 15 20 25 30;
# do
# python gkp_cphase_gate_JJ_dynamics_driver.py $omega_A $J_A $nu_A $Gamma_A $omega_B $J_B $nu_B $Gamma_B $J_C $f_A $f_B $Temp $Lambda $N_samples $N_wells_A $n_rungs $N_wells_B $n_rungs $drive_resolution_order $output_resolution_order $LCJ_save_path $data_save_path $seed_SSE $seed_init
# done

conda deactivate