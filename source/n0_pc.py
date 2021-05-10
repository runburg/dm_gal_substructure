#!/usr/bin/env python3
"""
File: n0_pc.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: 
"""
import numpy as np
from scipy import integrate, stats
import sys
print(sys.path)
from scipy.interpolate import interp1d
import prob_dists as pd
import utils

params = p = utils.read_param_file(sys.argv[1])

for v in range(3, len(sys.argv), 2):
    print("read", sys.argv[v], "as", sys.argv[v+1])
    p[sys.argv[v]] = float(sys.argv[v+1])

print('beta', p['beta'])
print('M_min', p['M_min'])

pshop, fluxes, psis = pd.psh_of_psi(p, num_psi=50, infer_values=False)

psh_2dfunc = pd.interp_and_save_psh(p, pshop, fluxes, psis, outfile=sys.argv[2])
