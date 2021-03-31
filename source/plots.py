#!/usr/bin/env python3
"""
File: plots.py
Author: Jack Runburg
Email: jack.runburg@gmail.com
Github: https://github.com/runburg
Description: 
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import utils
import prob_dists as pd


def p1_plot(params, psi=40, n_list=[0, 2, 4, -1], outfile='./output/p1_plot.png', color=None, shift=False, betas=None):
    """Plot P1(F)."""
    fig, ax = utils.plot_setup(1, 1, figsize=(8, 6), set_global_params=True)

    n_labels = {-1: r"Som. enh. ($n=-1$)", 0: r"$s$-wave ($n=0$)", 2: r"$p$-wave", 4: r"$d$-wave"}
    if betas is None:
        betas = [params['beta']]

    for beta in betas:
        shiftval = 1
        if color is None:
            colors = iter(cm.plasma(np.linspace(0.4, 1, num=len(n_list))))
        else:
            colors = iter(color)

        for n in n_list:
            mean_params = {'a': 77.4, 'b': 0.87 + 0.31 * n, 'c': -0.23 - 0.04 * n}
            logmin = -24
            logmax = -3
            fluxes = np.logspace(logmin, logmax, num=(logmax - logmin) * 20)
            probs = pd.p1(fluxes, psi, mean_params=mean_params, num=200, beta=beta)
            # probs = [p1(flux, 40) for flux in fluxes]
            func = fluxes * probs

            if shift is True:
                if n == 0:
                    shiftval = fluxes[func.argmax()]
                    print('shift is', shiftval)
                else:
                    print('Fluxes shifted by', shiftval/fluxes[func.argmax()], fluxes[func.argmax()])
                    fluxes *= shiftval / fluxes[func.argmax()]

        #     print(normalization)

            ax.plot(fluxes, func, label=n_labels[n] + r'$\beta$=' + str(beta), color=next(colors))
        # print(f'slope near end for n={n}: {(func[-60]-func[-40])/(fluxes[-60]-fluxes[-40])}')

    ax.set_xscale('log')
    ax.set_xlabel(r'Flux [photons cm$^{-2}$ yr$^{-1}$]')
    ax.set_ylabel(rf'$ F \times P^n_1(F)$ at $\psi={psi}^\circ$')
    ax.set_yscale('log')
    # ax.set_title(rf"P_1(F) for $M_{min}={{params['M_min']:.2f}} M_\odot$, $\Psi={{psi:.2f}}^\circ$")

    ax.grid()
    ax.set_xticks([1e-25 * 10**i for i in range(21)])
    ax.set_xlim(left=1e-16, right=1e-5)
    ax.set_ylim(bottom=1e-3, top=1)

    lgd = ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))

    fig.savefig(outfile, bbox_extra_artists=(lgd,), bbox_inches='tight')

    return fig, ax


def p1_slope_plot(params, psi=40, n_list=[-1, 0, 2, 4], outfile='./output/p1_slope_plot.png'):
    """Plot slope of P1(F)."""
    fig, ax = utils.plot_setup(1, 1, figsize=(12, 8), set_global_params=True)

    n_labels = {-1: "Som. enh.", 0: r"$s$-wave", 2: r"$p$-wave", 4: r"$d$-wave"}
    colors = iter(cm.plasma(np.linspace(0.1, 1, num=len(n_list))))

    for n in n_list:
        mean_params = {'a': 77.4, 'b': 0.87 + 0.31 * n, 'c': -0.23 - 0.04 * n}
        logmin = -24
        logmax = -3
        fluxes = np.logspace(logmin, logmax, num=(logmax - logmin) * 20)
        probs = pd.p1(fluxes, psi, mean_params=mean_params, num=200)

        func = np.log10(probs)
        col = next(colors)

        ax.plot(fluxes[:-1], (func[:-1]-func[1:])/(np.log10(fluxes[:-1])-np.log10(fluxes[1:])), label=n_labels[n], color=col)
        ax.axhline(-1.03/(1+.36*n)-1, color=col)

    ax.set_xscale('log')
    ax.set_xlabel(r'Flux [photons cm$^{-2}$ yr$^{-1}$]')
    ax.set_ylabel('Slope of robability')
    ax.set_ylim(bottom=-3, top=1)
    # ax.set_xlim(left=1e-22, right=1e-2)

    ax.set_title(r"Slope of probability distribution for $M_{min}=0.01 M_\odot$, $\Psi=40^\circ$")

    ax.grid(which='both')
    ax.set_yticks(np.linspace(-3, 1, num=21))
    ax.legend()

    fig.savefig(outfile)

    return fig, ax


def check_fp1_plot(fluxes, p1_vals, i1=0, i2=-1, j=[]):
    """Plot F x P1 to check results."""
    fig, axs = utils.plot_setup(1, 1)
    axs.plot(fluxes[i1:i2], fluxes[i1:i2] * p1_vals[i1:i2])

    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.set_ylabel('F x P1')
    axs.set_xlabel('F')

    for jj in j:
        axs.axvline(fluxes[jj], color='xkcd:orchid')

    return fig, axs


def check_p1_plot(fluxes, p1_vals, i1=0, i2=-1, j=[]):
    """Plot P1 to check results."""
    fig, ax = utils.plot_setup(1, 1)
    ax.plot(fluxes[i1:i2], p1_vals[i1:i2])

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('F')
    ax.set_ylabel('P1')

    for jj in j:
        ax.axvline(fluxes[jj], color='xkcd:orchid')

    return fig, ax


def check_ft_p1_plot(k, ft_p1, i1=0, i2=-1, j=[]):
    """Plot FT of P1(F) to check results."""
    fig, ax = utils.plot_setup(1, 1)
    ax.plot(k[i1:i2], ft_p1[i1:i2].real)
    ax.plot(k[i1:i2], ft_p1[i1:i2].imag)
    ax.set_xscale('log')
    ax.set_xlabel('k')
    ax.set_ylabel('P1(k)')

    for jj in j:
        ax.axvline(fluxes[jj], color='xkcd:orchid')

    return fig, ax


def check_psh_integrand_plot(k, ft_p1, muu, i1=0, i2=-1, j=[]):
    """Plot integrand of Psh to check results."""
    fig, ax = utils.plot_setup(1, 1)
    ax.plot(k[i1:i2], np.exp(muu * (ft_p1 - 1))[i1:i2].real)
    ax.plot(k[i1:i2], np.exp(muu * (ft_p1 - 1))[i1:i2].imag)
    ax.set_xscale('log')
    ax.set_xlabel('k')
    ax.set_ylabel(r'exp($\mu(\psi)$(P1(k)-1))')

    for jj in j:
        ax.axvline(fluxes[jj], color='xkcd:orchid')

    return fig, ax


def check_psh_plot(fluxes, psh_vals, i1=0, i2=-1, j=[]):
    """Plot Psh to check results."""
    fig, ax = utils.plot_setup(1, 1)
    ax.plot(fluxes[i1:i2], (fluxes * psh_vals)[i1:i2].real)
    ax.plot(fluxes[i1:i2], (fluxes * psh_vals)[i1:i2].imag)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('F')
    ax.set_ylabel(r'P$_{sh}\times F$')

    for jj in j:
        ax.axvline(fluxes[jj], color='xkcd:orchid')

    return fig, ax


def psh_func_psi_plot(psh_vals_over_psi, fluxes, psis, outfile='./output/psh_of_psi.png'):
    """Plot Psh as a function of psi."""
    from matplotlib import cm
    colors = cm.viridis(np.linspace(0, 1, num=len(psis)))

    fig, ax = plt.subplots()

    for pshvals, fxs, psi, col in zip(psh_vals_over_psi, fluxes, psis, colors):
        ax.plot(fxs, fxs * pshvals.real, label=rf"{psi}$^\circ$", color=col)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'F [photons cm$^{-2}$ yr$^{-1}$]')
    ax.set_ylabel('F x Psh')

    cbarlabs = np.arange(min(psis), max(psis) + 1, step=10)
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.viridis), ticks=np.linspace(0, 1, num=len(cbarlabs)))
    cbar.set_label(r'$\psi$[$^\circ$]', rotation=270, labelpad=2)
    cbar.ax.set_yticklabels(cbarlabs)

    fig.savefig(outfile)

    return fig, ax
