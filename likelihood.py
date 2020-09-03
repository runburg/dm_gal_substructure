#!/usr/bin/env python3
"""
File: likelihood.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Compute the likelihood over fwimp
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats
import matplotlib.cm as cm
from numpy import fft
from scipy import interpolate as intp
import importlib

sys.path.append('./source/')

from source import plots
from source import prob_dists as pd
from utils import read_param_file, update_params

param_file = './source/n0.params'
params = read_param_file(param_file)
p = params

with np.load('./output/n0_pshfunc.npz') as f:
    psi = f['psi']
    fluxes = f['flux']
    psh2d = f['psh']

# restrict to valid range of flux calculation
valid_lim = -55
fluxes = fluxes[:valid_lim]
psh2d = psh2d[:valid_lim]

# generate data with different fwimp
f = 10

# get psh data as function of psi and flux
counts = np.arange(0, 20)
psh2d[psh2d < 0] = 0
pshfunc2d = intp.interp2d(psi, fluxes, np.nan_to_num(psh2d), bounds_error=False, fill_value=0)

# make the simulated skymap
psis = np.linspace(40, 180, num=50)
pc_psi = np.array([np.trapz(1 / f * pshfunc2d(abs(psi), fluxes).flatten() * stats.poisson.pmf(counts[:, np.newaxis], p['exposure'] * f * fluxes), f * fluxes, axis=-1) for psi in psis])
pc_of_psi = intp.interp1d(psis, pc_psi, axis=0)

subcounts, ang_dists = pd.generate_skymap_sample_pc(p, pc_of_psi, return_subcounts=True, save_output=True)

# get psh for the angles we are considering
psh = pshfunc2d(np.abs(ang_dists), fluxes)
psh /= np.trapz(psh, fluxes, axis=0)

num_search = 50
fwimp_search = (np.log10(f)-2, np.log10(f)+1, num_search)

# get pixels outside of galactic plane
nside = p['nside']
npix = healpy.nside2npix(nside)
lon, lat = healpy.pix2ang(nside, range(npix), lonlat=True)
good_indices = (abs(lat) >= 40)

# backgrounds
gal_bg = np.load(p['gal_flux_bg_file'])[good_indices] * p['exposure']
iso_bg = p['iso_flux_bg'] * p['exposure']
bg_count = gal_bg + iso_bg

counts = np.arange(0, 150)

S, fwimps = pd.likelihood(p, psh, subcounts.astype(np.int16), fluxes, counts, fwimp_limits=fwimp_search, bg_count=bg_count)

np.savez('./output/likelihood_vals.npz', S=S, fwimps=fwimps)

fig, axs = plt.subplots()
axs.plot(fwimps, -(S - S.min()), color='xkcd:tangerine', label='P(C)')
axs.set_xlabel(r'f$_{SUSY}$', fontsize=20)
axs.set_xscale('log')
axs.set_ylabel(r'$\mathcal{L}$', fontsize=20)

fig.savefig('./output/likelihood_plot.png')
