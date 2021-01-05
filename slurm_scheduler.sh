#!/bin/bash
for filename in n*.slurm; do
	echo $filename;
	sbatch $filename;
done
