#!/bin/bash
#SBATCH --job-name=swave128
#SBATCH --partition=shared
## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition
#SBATCH --time=1-05:00:00 ## time format is DD-HH:MM:SS
## task-per-node x cpus-per-task should not typically exceed core count on an individual node
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G ## max amount of memory per node you require

##SBATCH --core-spec=0 ## Uncomment to allow jobs to request all cores on a node    

#SBATCH --error=swavepsh-%A.err ## %A - filled with jobid
#SBATCH --output=swavepsh-%A.out ## %A - filled with jobid
## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=runburg@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
source ~/.bash_profile

module load lang/Python/3.7.2-intel-2018.5.274

pip3 install scipy

python3 -u source/n0_pc.py source/n0.params output/n0_pshfunc_beta19_Mmin1e2.npz beta 1.9 M_min 1e2
