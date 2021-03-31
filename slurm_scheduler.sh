#!/bin/bash
for filename in n-1*.slurm; do
	echo $filename;
	sbatch $filename;
done
