#!/usr/bin/env python3
"""
File: prob_dists.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: Probability distributions and helper functions for DM annihilating in galactic substructure
"""
import numpy as np
from scipy import integrate, stats
from scipy.interpolate import interp2d
import plots
from scipy import interpolate as intp
import random


def gaussian_mean(M, r, a=77.4, b=0.87, c=-0.23, fsusy=1):
    """Return mean of conditional luminosity distribution.

    Inputs:
        - M: mass of subhalo [solar masses]
        - r: distance from galactic center [kpc]
        - a: coefficient for first term [default=77.4]
        - b: coefficient for second term [default=0.97]
        - c: coefficient for third term [default=-0.23]
        - fsusy: value of fsusy [default=1e-28 [cm**3 s**-1 GeV**-2]]

    Returns:
        - mean of the gaussian distribution
    """
    return a + b * np.log(M / 1e5) + c * np.log(r / 50) + np.log(fsusy)


def gaussian_sd(M, r, a=0.74, b=-0.003, c=0.011):
    """Return the sd of the conditional luminosity function.

    Inputs:
        - M: mass of subhalo [solar masses]
        - r: distance from galactic center [kpc]
        - a: coefficient for first term [default=0.74]
        - b: coefficient for second term [default=-0.003]
        - c: coefficient for third term [default=-0.011]

    Returns:
        - sd of gaussian
    """
    return a + b * np.log(M / 1e5) + c * np.log(r / 50)


def conditional_luminosity_function(lnL, M, r, mean_params={}, sd_params={}, fsusy=1):
    """Gives the value of the conditional luminosity function.

    Inputs:
        - lnL: natural log of lumionsity of the subhalo
        - M: mass of the subhalo [solar masses]
        - r: galactocentric radius of the subhalo [kpc]
        - mean_params: coefficients for parameter dependence for the distribution mean
        - sd_params: coefficients for parameter dependence for the sd of the distribution

    Returns:
        - value of the clf
    """
    mean = gaussian_mean(M, r, **mean_params, fsusy=fsusy)
    sigma = gaussian_sd(M, r, **sd_params)

    return stats.norm.pdf(lnL, loc=mean, scale=sigma)


def mass_distribution(M, r, A=1.2e4, beta=1.9, rs=21):
    """Return the mass distribution value.

    Inputs:
        - M: mass of the subhalo [solar masses]
        - r: galactocentric radius of the subhalo [kpc]
        - A: amplitude [default=1.2e4 [solar mass**-1 kpc**-3]]
        - beta: strength of dependence on mass [default value=1.9]
        - rs: scale radius of Milky Way [default value=21 [kpc]]

    Returns:
        - value of mass distribution
    """
    r_tilde = r / rs

    return A * M**(-beta) / (r_tilde * (1 + r_tilde)**2)


def p1(F, psi, num=100, R_G=220, M_min=0.01, M_max=1e10, d_solar=8.5, mean_params={}, sd_params={}, fsusy=1, **kwargs):
    """Return the unnormalized value of the probability distribution of one subhalo.

    Inputs:
        - F: flux of subhalo [photons/cm^2/year]
        - psi: angle from galactic center [degrees]
        - R_G: extent of dm halo [default value=220 [kpc]]
        - M_min: lower limit on mass integral [default=0.01 [solar masses]]
        - M_max: upper limit on mass integral [default=1e10 [solar masses]]
        - d_solar: galactocentric distance of sun [default=8.5 [kpc]]
        - mean_params: coefficients for parameter dependence for the distribution mean
        - sd_params: coefficients for parameter dependence for the sd of the distribution

    Returns:
        - value of probability of one subhalo for a given flux and angle
    """
    # Get upper bound for l integral
    psi_rad = np.deg2rad(psi)
    l_max = d_solar * (np.cos(psi_rad) + np.sqrt(-(np.sin(psi_rad)**2) + (R_G / d_solar)**2))

    # Convert years to seconds
    seconds_in_a_year = 3.154e7

    # Convert cm to kpc
    cm_in_a_kpc = 3.086e21

    def integrand_2d(M, l):
        # Luminosity
        Lsh = 4 * np.pi * l**2 * F / seconds_in_a_year * cm_in_a_kpc**2

        # radius from GC
        r = np.sqrt(l**2 + d_solar**2 - 2 * l * d_solar * np.cos(psi_rad))

        return l**4 * mass_distribution(M, r) / Lsh * conditional_luminosity_function(np.log(Lsh), M, r, mean_params=mean_params, sd_params=sd_params, fsusy=fsusy)

    lvals = np.logspace(np.log10(1e-8), np.log10(l_max), num=num)
    mvals = np.logspace(np.log10(M_min), np.log10(M_max), num=num)

    func = integrand_2d(mvals[:, np.newaxis, np.newaxis], lvals[:, np.newaxis])

    int_exp2d = np.trapz(np.trapz(func, mvals, axis=0), lvals, axis=0)

    return int_exp2d / np.trapz(int_exp2d, F)


def mu(psi, R_G=220, M_min=0.01, M_max=1e10, d_solar=8.5, omega_pixel=(np.pi / 180)**2, num=100, **kwargs):
    """Compute mean value of subhalos in a pixel."""
    # Get upper bound for l integral
    psi_rad = np.deg2rad(psi)
    l_max = d_solar * (np.cos(psi_rad) + np.sqrt(-(np.sin(psi_rad)**2) + (R_G / d_solar)**2))

    def integrand_2d(M, l):
        # radius from GC
        r = np.sqrt(l**2 + d_solar**2 - 2 * l * d_solar * np.cos(psi_rad))

        return l**2 * mass_distribution(M, r)

    lvals = np.logspace(np.log10(1e-8), np.log10(l_max), num=num)
    mvals = np.logspace(np.log10(M_min), np.log10(M_max), num=num)

    # perform the 2d integration by first integrating of mass than luminosity
    int_exp2d = np.trapz(np.trapz(integrand_2d(mvals[:, np.newaxis, np.newaxis], lvals[:, np.newaxis]), mvals, axis=0), lvals, axis=0)

    return omega_pixel * int_exp2d


def ft_p1(p1_vals, fluxes, log_k_min, log_k_max, N_k, **kwargs):
    """Return Fourier Transform of P1(F)."""
    k = np.logspace(log_k_min, log_k_max, num=N_k)
    func = p1_vals * np.exp(2j * np.pi * fluxes * k[:, np.newaxis])

    ft_p1 = np.trapz(func, fluxes, axis=1)

    return ft_p1, k


def pf(muu, ft_p1_vals, k, psh_log_f_min, psh_log_f_max, N_psh, **kwargs):
    """Return IFT of exp(mu(ft_p1 - 1))."""
    ks = np.concatenate((-np.flip(k), k))
    ft_p1s = np.concatenate((np.conj(np.flip(ft_p1_vals)), ft_p1_vals))

    fluxes = np.logspace(psh_log_f_min, psh_log_f_max, num=N_psh)
    func = np.exp(muu * (ft_p1s - 1)) * np.exp(-2j * np.pi * fluxes[:, np.newaxis] * ks)

    ift_func = np.trapz(func, ks, axis=1)

    return ift_func, fluxes


def psh(params_dict, infer_values=False, plot=False):
    """Return the function P_sh(F) for a given psi.

    Inputs:
        - psi: angle from galactic center [degrees]
        - F_min: minimum flux for FFT
        - F_max: maximum flux for FFT
        - num_bins: number of points to perform FFT (power of 2 for speed)

    Returns:
        - function of probability distribution psh at psi
    """
    p = params_dict
    fluxes = np.logspace(p['log_flux_min'], p['log_flux_max'], num=p['N'])

    # Get P1 values
    p1_vals = p1(fluxes, **p)
    print("Got P1(F) values")

    if plot is True:
        fig, _ = plots.check_p1_plot(fluxes, p1_vals)
        fig.savefig('./output/check_p1.png')

        fig, _ = plots.check_fp1_plot(fluxes, p1_vals)
        fig.savefig('./output/check_fp1.png')

    if infer_values is True:
        N_k = 1
        log_k_min = fluxes[p1_vals.argmax()]
        log_k_max = log_k_min
        log_diff = 0.5
        threshold = 0.9
        val = 1
        while val > threshold:
            log_k_min += log_diff
            log_k_max += log_diff
            val, _ = ft_p1(p1_vals, fluxes, log_k_min, log_k_max, N_k)

        # p['log_k_min'] = 3
        p['log_k_max'] = log_k_max
        print('inferred log_k_max', log_k_max)

    # Perform FT
    muu = mu(**p)[0]
    ft_p1_vals, k = ft_p1(p1_vals, fluxes, **p)
    print("Finished FT of P1")

    if plot is True:
        fig, _ = plots.check_ft_p1_plot(k, ft_p1_vals)
        fig.savefig('./output/check_ft_p1.png')

        fig, _ = plots.check_psh_integrand_plot(k, ft_p1_vals, muu)
        fig.savefig('./output/check_psh_integrand.png')

    # if infer_values is True:
        # p['psh_log_f_min'] = p['log_flux_min'] + 7
        # p['psh_log_f_max'] = p['log_flux_max']

    # Compute Psh
    pf_vals, flux = pf(muu, ft_p1_vals, k, **p)

    if plot is True:
        fig, _ = plots.check_psh_plot(flux, pf_vals)
        fig.savefig('./output/check_psh.png')

    if infer_values is True:
        func = pf_vals.real * flux
        max_index = np.argmax(func)
        print(max_index, len(func))
        if max_index < len(func)/2:
            finish = max_index + np.argmax(func[max_index+1:] < (func[max_index] / 1e6))
            start = max_index - np.argmax(np.flip(func[:max_index]) < (func[max_index] / 1e6))
            # if finish <= max_index:
            #     finish = -1
            # else:
            #     finish += max_index
            pf_vals = pf_vals[start:finish]
            flux = flux[start:finish]

    print("Finished Psh computation")
    expec = muu * np.trapz(fluxes * p1_vals, fluxes)
    reali = np.trapz(pf_vals.real * flux, flux)
    print(f'\mu \int F P1: {expec}')
    print(f' \int F Psh: {reali}')
    print(f'percent error = {round((reali - expec)/(expec)*100, 2)}%\n')

    return pf_vals, flux


def psh_of_psi(params, psi_min=40, psi_max=180, num_psi=14, plot=False, infer_values=True):
    """Get psh values as a function of psi."""
    psis = np.linspace(psi_min, psi_max, num=num_psi)

    psh_vals_over_psi = []
    fluxes = []

    for psi in psis:
        params['psi'] = psi
        print('Angle:', psi)

        pshvals, fxs = psh(params, plot=plot, infer_values=infer_values)
        psh_vals_over_psi.append(pshvals)
        fluxes.append(fxs)

    return np.array(psh_vals_over_psi), np.array(fluxes), psis


def psh_s(ang_dists, input_file='./output/n0_pshfunc.npz', return_all=False):
    """
    Compute P_sh for sommerfeld annihilation.
    """
    with np.load(input_file) as f:
        psi = f['psi']
        fluxes = f['flux']
        psh2d = f['psh']

    # restrict to valid range of flux calculation
    low_lim = 0
    valid_lim = -55
    fluxes = fluxes[low_lim:valid_lim]
    psh2d = psh2d[low_lim:valid_lim]

    # get psh data as function of psi and flux
    psh2d[psh2d < 0] = 0
    pshfunc2d = intp.interp2d(psi, fluxes, np.nan_to_num(psh2d), bounds_error=False, fill_value=0)

    # get psh for the angles we are considering
    psh = pshfunc2d(np.abs(ang_dists), fluxes)
    psh /= np.trapz(psh, fluxes, axis=0)

    if return_all is True:
        return psh, pshfunc2d, fluxes, psi

    return psh


def psh_som(ang_dists, input_file='./output/n-1_pshfunc.npz', return_all=False):
    """
    Compute P_sh for sommerfeld annihilation.
    """
    with np.load(input_file) as f:
        psi = f['psi']
        fluxes = f['flux']
        psh2d = f['psh']

    # restrict to valid range of flux calculation
    low_lim = 150
    valid_lim = -50
    fluxes = fluxes[low_lim:valid_lim]
    psh2d = psh2d[low_lim:valid_lim]

    # get psh data as function of psi and flux
    psh2d[psh2d < 0] = 0
    pshfunc2d = intp.interp2d(psi, fluxes, np.nan_to_num(psh2d), bounds_error=False, fill_value=0)

    # get psh for the angles we are considering
    psh = pshfunc2d(np.abs(ang_dists), fluxes)
    psh /= np.trapz(psh, fluxes, axis=0)

    if return_all is True:
        return psh, pshfunc2d, fluxes, psi

    return psh


def pc_of_psi(p, pshfunc2d, fluxes, counts, bg_count=np.array([0])):
    psis = np.linspace(40, 180, num=60)

    f = p['fwimp']
    exposure = p['exposure']

    mean_f = f * np.trapz(fluxes * pshfunc2d(psis[0], fluxes).flatten(), fluxes)
    mean_count = int(mean_f * exposure + np.max(bg_count))
    counts = np.arange(0, mean_count + 4 * np.sqrt(mean_count) + 40)
    if counts[-1] > 50:
        print('largest count is', counts[-1], '...this may take a while')

    pc_psi = np.trapz(1/f * pshfunc2d(psis, fluxes)[:, :, np.newaxis] * stats.poisson.pmf(counts[np.newaxis, np.newaxis, :], exposure * f * fluxes[:, np.newaxis, np.newaxis] + bg_count[np.newaxis, :, np.newaxis]), f * fluxes, axis=0)

    pc_of_psi = intp.interp1d(psis, np.array(pc_psi, dtype=np.float64), axis=0)

    if np.any(np.isnan(pc_psi)):
        print('found nan in pc')

    return pc_of_psi


def interp_and_save_psh(params, psh, flux, psi, return_interp2d=True, outfile='./output/pshfunc.npz'):
    """Get a convenient form for psh and save arrays."""
    fluxes = np.logspace(params['psh_log_f_min'], params['psh_log_f_max'], num=250)
    interp_array = np.zeros((len(fluxes), len(psi)))

    for i, pshvals in enumerate(psh):
        func = intp.interp1d(flux[i], pshvals.real, fill_value=0, bounds_error=0)
        interp_array[:, i] = func(fluxes)

    np.savez(outfile, flux=fluxes, psi=psi, psh=interp_array)

    func2d = intp.interp2d(psi, fluxes, interp_array, bounds_error=False, fill_value=0)

    np.save(outfile + "2d_interp.npz", func2d)

    if return_interp2d is True:
        print("Returned P_sh(\psi, flux)")
        return func2d

    return interp_array, fluxes, psi


def pc(psh_2dfunc, fluxes, psi, exposure, background=False, countmax=20):
    """Compute P(C) for ensemble of halos."""
    from scipy.stats import poisson

    counts = np.arange(0, countmax + 1)
    pcvals = np.trapz(psh_2dfunc(psi, fluxes).flatten() * poisson.pmf(counts[:, np.newaxis], exposure * fluxes), fluxes, axis=-1)

    return pcvals, counts


def generate_skymap_sample_pc_2(p, pshfunc2d, fluxes, cut_out_band=40, output_path='./output/', print_updates=False, save_output=False, return_subcounts=False, with_bg=True):
    import healpy
    f = p['fwimp']
    exposure = p['exposure']
    nside = p['nside']
    npix = healpy.nside2npix(nside)
    pixel_counts = np.ones(npix) * healpy.pixelfunc.UNSEEN

    lon, lat = healpy.pix2ang(nside, range(npix), lonlat=True)

    good_indices = (abs(lat) >= cut_out_band)

    ang_dists = np.rad2deg(np.arccos(np.cos(np.deg2rad(lon[good_indices])) * np.cos(np.deg2rad(lat[good_indices]))))

    sub_counts = np.zeros(len(ang_dists))

    if with_bg is True:
        gal_bg = np.load(p['gal_flux_bg_file'])[good_indices] * p['exposure']
        iso_bg = p['iso_flux_bg'] * p['exposure']
        # print(np.mean(gal_bg), iso_bg)
        bg_counts = gal_bg + iso_bg
        print('got background', np.mean(bg_counts))
    else:
        bg_counts = 0
        gal_bg = 0
        iso_bg = 0

    mean_f = f * np.trapz(fluxes[:, np.newaxis] * pshfunc2d(np.abs(ang_dists), fluxes), fluxes[:, np.newaxis], axis=0)
    mean_f *= exposure
    mean_f += bg_counts
    # nominal_counts = np.arange(int(-2*np.std(mean_f)), int(2*np.std(mean_f)))
    nominal_counts = np.arange(-40, 40)
    print('count len', len(nominal_counts))
    counts = nominal_counts[np.newaxis, :].astype(np.int16) + mean_f[:, np.newaxis].astype(np.int16)
    counts[counts<0] = 0

    # mean_f = f * np.trapz(fluxes * pshfunc2d(np.abs(ang_dists).min(), fluxes).flatten(), fluxes)
    # mean_count = (mean_f * exposure + np.mean(bg_counts)).astype(np.int64)
    # lower_count = mean_count
    # lower_count -= 4 * np.sqrt(lower_count)
    # if lower_count < 0:
    #     lower_count = 0
    # upper_count = mean_count
    # upper_count += 4 * np.sqrt(upper_count)
    # counts = np.arange(lower_count, upper_count)
    # if counts[-1] > 50:
    #     print('largest count is', counts[-1], '...this may take a while')

    pcvals = np.trapz(1/f * pshfunc2d(ang_dists, fluxes)[:, :, np.newaxis] * stats.poisson.pmf(counts[np.newaxis, :, :], exposure * f * fluxes[:, np.newaxis, np.newaxis] + bg_counts[np.newaxis, :, np.newaxis]), f * fluxes, axis=0)
    
    if np.any(np.sum(pcvals, axis=-1) < 0.9):
        print('calculated pc does not integrate to 1. potentially due to insufficient count range. consider increasing the min/max allowed counts', np.sum(np.sum(pcvals, axis=-1) < 0.9))
    pcvals /= np.sum(pcvals, axis=-1)[:, np.newaxis]

    cum_pc = pcvals.cumsum(axis=1)
    rand_vals = np.random.rand(len(cum_pc), 1)
    sub_counts = (rand_vals < cum_pc).argmax(axis=1)
    sub_counts += counts[:, 0]
    # sub_counts += lower_count

    # for i, psi in enumerate(ang_dists):
    #     if print_updates is True:
    #         if i % 10000 == 0:
    #             print(i, '/', len(ang_dists))
    #     #     print(psi)
    #     pcvals = pc_of_psi(abs(psi))
    #     sub_counts[i] += np.random.choice(np.arange(len(pcvals)), size=1, p=pcvals/np.sum(pcvals))

    pixel_counts[good_indices] = sub_counts

    if save_output is True:
        outfile = f"{output_path}n{p['n']}_skymap_{str(random.randint(0, 99999)).rjust(5, '0')}.npy"
        np.save(outfile, pixel_counts)
        print("saved in", outfile)

    if return_subcounts is True:
        return sub_counts, ang_dists, gal_bg + iso_bg

    return pixel_counts


def generate_skymap_sample_pc(p, pc_of_psi, cut_out_band=40, output_path='./output/', print_updates=False, save_output=False, return_subcounts=False, with_bg=True):
    import healpy
    nside = p['nside']
    npix = healpy.nside2npix(nside)
    pixel_counts = np.ones(npix) * healpy.pixelfunc.UNSEEN

    lon, lat = healpy.pix2ang(nside, range(npix), lonlat=True)

    good_indices = (abs(lat) >= cut_out_band)

    ang_dists = np.rad2deg(np.arccos(np.cos(np.deg2rad(lon[good_indices])) * np.cos(np.deg2rad(lat[good_indices]))))

    if with_bg is True:
        gal_bg = np.load(p['gal_flux_bg_file'])[good_indices] * p['exposure']
        iso_bg = p['iso_flux_bg'] * p['exposure']
        # print(np.mean(gal_bg), iso_bg)

        bg_counts = stats.poisson.rvs(gal_bg + iso_bg)
    else:
        bg_counts = 0
        gal_bg = 0
        iso_bg = 0

    sub_counts = np.zeros(len(ang_dists))
    sub_counts += bg_counts

    # pcvals = pc_of_psi(np.abs(ang_dists))
    # pcvals /= np.sum(pcvals, axis=-1)[:, np.newaxis]

    # cum_pc = pcvals.cumsum(axis=1)
    # rand_vals = np.random.rand(len(cum_pc), 1)
    # sub_counts += (rand_vals < pcvals).argmax(axis=1)

    for i, psi in enumerate(ang_dists):
        if print_updates is True:
            if i % 10000 == 0:
                print(i, '/', len(ang_dists))
        #     print(psi)
        pcvals = pc_of_psi(abs(psi))
        sub_counts[i] += np.random.choice(np.arange(len(pcvals)), size=1, p=pcvals/np.sum(pcvals))

    pixel_counts[good_indices] = sub_counts

    if save_output is True:
        outfile = f"{output_path}n{p['n']}_skymap_{str(random.randint(0, 99999)).rjust(5, '0')}.npy"
        np.save(outfile, pixel_counts)
        print("saved in", outfile)

    if return_subcounts is True:
        return sub_counts, ang_dists, gal_bg + iso_bg

    return pixel_counts


def likelihood(p, psh, subcounts, fluxes, counts, fwimp_limits=(-5, 5, 20), bg_count=np.array([0]), verbose=False):
    from scipy.stats import poisson

    exposure = p['exposure']

    fwimps = np.logspace(*fwimp_limits)
    # print(fwimps)
    S = np.zeros(fwimps.shape)

    for i, f in enumerate(fwimps):
        if verbose is True:
            print(i, '/', len(fwimps))
        # p['fwimp'] = f

        pc = integrate.simps(1/f * psh * poisson.pmf(subcounts[np.newaxis, :], exposure * f * fluxes[:, np.newaxis] + bg_count[np.newaxis, :]), f * fluxes, axis=0)
        # pc = np.trapz(1/f * psh * poisson.pmf(subcounts[np.newaxis, :], exposure * f * fluxes[:, np.newaxis] + bg_count[np.newaxis, :]), f * fluxes, axis=0)
        # pc = np.trapz(1 / f * psh[:, :, np.newaxis] * poisson.pmf(counts[np.newaxis, np.newaxis, :], exposure * f * fluxes[:, np.newaxis, np.newaxis] + bg_count[np.newaxis, :, np.newaxis]), f * fluxes, axis=0)

        # print(pc)
        # pixel_probs = pc[np.arange(len(pc)), subcounts]
        pixel_probs = pc

        print('zero prob pixels', np.sum(pixel_probs <= 0))
        S[i] = -2 * np.sum(np.log(pixel_probs, where=(pixel_probs > 0)))

    return S, fwimps


def poisson_likelihood(p, psh, subcounts, fluxes, counts, fwimp_limits=(-5, 5, 20), bg_count=np.array([0]), verbose=False):
    from scipy.stats import poisson

    exposure = p['exposure']

    fwimps = np.logspace(*fwimp_limits)
    S = np.zeros(fwimps.shape)

    mean_f = np.trapz(fluxes[:, np.newaxis] * psh, fluxes[:, np.newaxis], axis=0)

    for i, f in enumerate(fwimps):
        if verbose is True:
            print(i, '/', len(fwimps))

        pc = poisson.pmf(subcounts, exposure * f * mean_f + bg_count)

        # pixel_probs = pc[np.arange(len(pc)), subcounts]
        pixel_probs = pc

        print('zero prob pixels', np.sum(pixel_probs <= 0))

        S[i] = -2 * np.sum(np.log(pixel_probs, where=(pixel_probs>0)))

    return S, fwimps

