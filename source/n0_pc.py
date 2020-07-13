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

pshop, fluxes, psis = pd.psh_of_psi(p, num_psi=15, infer_values=True)

psh_2dfunc = pd.interp_and_save_psh(p, pshop, fluxes, psis, outfile=sys.argv[2])
