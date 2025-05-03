import numpy as np
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib import pyplot as plt
import pandas as pd
from nec2array import (ArrayModel, VoltageSource, FreqSteps, Wire,
                  ExecutionBlock, RadPatternSpec, EEPdata)
from time import time
# from pygdsm import GlobalSkyModel16, GSMObserver16, GlobalSkyModel, GSMObserver, LowFrequencySkyModel, LFSMObserver
from pygdsm import LowFrequencySkyModel, LFSMObserver
from datetime import datetime
import healpy as hp
from tqdm import tqdm
from astropy.time import Time
from astropy.coordinates import EarthLocation, Longitude
import astropy.units as u
from scipy.interpolate import SmoothSphereBivariateSpline, interp1d


np.set_printoptions(precision=4, linewidth=80)
raw_data_path = '../data/'
data_path = './figures_thesis_materials/'


def directivity_phis():
    save_figure = False

    #### Build array element antenna
    # Everything scaled to lambda so actual lambda is arbitrary
    # but for simplicity set it to 1m.
    lambda_ = 1
    p1 = (0., 0., -lambda_ / 2 / 2)
    p2 = (0., 0., +lambda_ / 2 / 2)
    l12 = (p1, p2)
    wire_radius = 1e-6 * lambda_

    model_name = __file__.replace('.py', '')
    warnick6 = ArrayModel(model_name)
    warnick6.set_commentline(warnick6.name)
    warnick6.set_commentline('Author: T. Carozzi')
    warnick6.set_commentline('Date: 2024-03-20')
    warnick6['ant_Z']['w'] = Wire(*l12, wire_radius)
    warnick6['ant_Z']['w'].add_port(0.5, 'the_port')

    # Set up execution settings
    nr_freqs = 1
    frq_cntr = 3.0e8 / lambda_ / 1e6
    frq_cntr_step = FreqSteps('lin', nr_freqs, frq_cntr)
    warnick6.segmentalize(201, frq_cntr)
    _port_ex = ('the_port', VoltageSource(1.0))
    nph = 100
    rps = RadPatternSpec(thets=90., nph=nph, dph=360 / nph, phis=0.)
    eb_arr = ExecutionBlock(frq_cntr_step, _port_ex, rps)

    #### Set up array of 6 elements with two different spacings
    #### and reference antenna to nr. 2
    nr_ants = 6
    ref_ant = 2
    frq_idx = 0
    pats_OC = []
    pats_SC = []
    pows_act_OC = []
    pows_act_SC = []
    # The two spacings in units of lambda:
    spacefacs = [0.5, 0.2]
    for spfac in spacefacs:
        d = spfac * lambda_
        arr_pos = [[(antnr * d - (nr_ants - 1) / 2 * d) * 1, 0., 0.] for antnr in range(nr_ants)]
        warnick6.arrayify(element=['ant_Z'], array_positions=arr_pos)
        eepSCdat = warnick6.excite_1by1(eb_arr, save_necfile=False)
        pow_act_SC = eepSCdat.get_pow_arr()[frq_idx, ref_ant, ref_ant]
        pows_act_SC.append(pow_act_SC)
        # Get OC from SC through transformation:
        eepOCdat = eepSCdat.transform_to('OC')
        pow_act_OC = (np.abs(eepOCdat.current_excite) ** 2
                      * np.real(eepOCdat.impedances[frq_idx, ref_ant, ref_ant]) / 2.)
        pows_act_OC.append(pow_act_OC)
        # [n_antennas, n_frequencies, n_thetas, n_phis, n_polarizations]
        _antspats = eepSCdat.get_antspats_arr()
        pats_SC.append(_antspats[:, 0, 0, :, :].squeeze())
        _antspats = eepOCdat.get_antspats_arr()
        pats_OC.append(_antspats[:, 0, 0, :, :].squeeze())

    def dbi(efield, pow):
        eta = 377
        U = (np.abs(efield[:, 0]) ** 2 + np.abs(efield[:, 1]) ** 2) / (2 * eta)
        dbi_ = 10 * np.log10(U / pow * 4 * np.pi)
        return dbi_

    #### Plot the results
    phis = rps.as_thetaphis()[1]

    base_fontsize = 16
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    fig, axs = plt.subplots(len(spacefacs), 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    print(pows_act_SC)
    for spidx in range(len(spacefacs)):
        pat_SC_dbi = dbi(pats_SC[spidx][ref_ant, :], pows_act_SC[spidx])
        axs[spidx].plot(phis, pat_SC_dbi, 'y-.')
        pat_OC_dbi = dbi(pats_OC[spidx][ref_ant, :], pows_act_OC[spidx])
        axs[spidx].plot(phis, pat_OC_dbi, 'r-.')
        axs[spidx].set_title(f'$d = {spacefacs[spidx]}$ m')
        axs[spidx].set_ylabel('Directivity [dBi]')
        axs[spidx].set_xlim([0, 360])
        axs[spidx].set_ylim([-7., 7.])
        axs[spidx].legend(['SC', 'OC (from SC)'], fontsize=base_fontsize, loc='upper right', bbox_to_anchor=(0.88, 0.4))
    axs[-1].set_xlabel(r'$\phi$ [deg]')
    plt.tick_params(labelsize=base_fontsize)
    if save_figure:
        plt.savefig(f'results/directivity_phis.eps', dpi=300, facecolor='w')
    plt.show()


def eels_phis():
    save_figure = False

    #### Build array element antenna
    # Everything scaled to lambda so actual lambda is arbitrary
    # but for simplicity set it to 1m.
    lambda_ = 1
    p1 = (0., 0., -lambda_ / 2 / 2)
    p2 = (0., 0., +lambda_ / 2 / 2)
    l12 = (p1, p2)
    wire_radius = 1e-6 * lambda_

    model_name = __file__.replace('.py', '')
    warnick6 = ArrayModel(model_name)
    warnick6.set_commentline(warnick6.name)
    warnick6.set_commentline('Author: T. Carozzi')
    warnick6.set_commentline('Date: 2024-03-20')
    warnick6['ant_Z']['w'] = Wire(*l12, wire_radius)
    warnick6['ant_Z']['w'].add_port(0.5, 'the_port')

    # Set up execution settings
    nr_freqs = 1
    frq_cntr = 3.0e8 / lambda_ / 1e6
    frq_cntr_step = FreqSteps('exp', nr_freqs, frq_cntr, incr=1.022)
    warnick6.segmentalize(201, frq_cntr)
    _port_ex = ('the_port', VoltageSource(1.0))
    nph = 100
    rps = RadPatternSpec(thets=90., nph=nph, dph=360 / nph, phis=0.)
    eb_arr = ExecutionBlock(frq_cntr_step, _port_ex, rps)

    #### Set up array of 6 elements with two different spacings
    #### and reference antenna to nr. 2
    nr_ants = 6
    ref_ant = 2
    Hscs_abs = []
    hocs_abs = []
    # The two spacings in units of lambda:
    spacefacs = [0.5, 0.2]

    for spfac in spacefacs:
        d = spfac * lambda_
        arr_pos = [[antnr * d - (nr_ants - 1) / 2 * d, 0., 0.] for antnr in range(nr_ants)]
        warnick6.arrayify(element=['ant_Z'], array_positions=arr_pos)
        eepSCdat = warnick6.excite_1by1(eb_arr)
        eepOCdat = eepSCdat.transform_to('OC')

        eelscdat = eepSCdat.get_EELs()
        Hsc_abs = np.sqrt(np.abs(eelscdat.eels[ref_ant].f_tht) ** 2
                          + np.abs(eelscdat.eels[ref_ant].f_phi) ** 2)
        Hscs_abs.append(Hsc_abs[0, 0, :])
        eelocdat = eepOCdat.get_EELs()
        hoc_abs = np.sqrt(np.abs(eelocdat.eels[ref_ant].f_tht) ** 2
                          + np.abs(eelocdat.eels[ref_ant].f_phi) ** 2)
        hocs_abs.append(hoc_abs[0, 0, :])

    #### Plot the results
    phis = rps.as_thetaphis()[1]

    base_fontsize = 16
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    for spidx in range(len(spacefacs)):
        ax[spidx].plot(phis, Hscs_abs[spidx] * 377, 'y-.', label='SC EEL')
        ax[spidx].plot(phis, hocs_abs[spidx], 'r-.', label='OC (from SC) EEL')
        ax[spidx].set_title(f'$d = {spacefacs[spidx]}$ m')
        ax[spidx].set_ylabel('EEL [m]')
        ax[spidx].set_xlim([0, 360])
        ax[spidx].set_ylim([0., 3.])
        ax[spidx].legend(fontsize=base_fontsize, loc='upper right', bbox_to_anchor=(0.65, 0.95))
    ax[-1].set_xlabel(r'$\phi$ [deg]')
    plt.tick_params(labelsize=base_fontsize)
    if save_figure:
        plt.savefig(f'results/EELs_phis.eps', dpi=300, facecolor='w')
    plt.show()


def directivity_frqs():
    save_figure = False

    #### Build array element antenna
    # Everything scaled to lambda so actual lambda is arbitrary
    # but for simplicity set it to 1m.
    lambda_ = 1
    p1 = (0., 0., -lambda_ / 2 / 2)
    p2 = (0., 0., +lambda_ / 2 / 2)
    l12 = (p1, p2)
    wire_radius = 1e-6 * lambda_

    model_name = __file__.replace('.py', '')
    warnick6 = ArrayModel(model_name)
    warnick6.set_commentline(warnick6.name)
    warnick6.set_commentline('Author: T. Carozzi')
    warnick6.set_commentline('Date: 2024-03-20')
    warnick6['ant_Z']['w'] = Wire(*l12, wire_radius)
    warnick6['ant_Z']['w'].add_port(0.5, 'the_port')

    # Set up execution settings
    nr_freqs = 100
    frq_cntr = 3.0e8 / lambda_ / 1e6 / 2
    frq_cntr_step = FreqSteps('exp', nr_freqs, frq_cntr, incr=1.022)
    warnick6.segmentalize(201, frq_cntr)
    _port_ex = ('the_port', VoltageSource(1.0))
    nph = 4
    rps = RadPatternSpec(thets=90., nph=nph, dph=360 / nph, phis=0.)
    eb_arr = ExecutionBlock(frq_cntr_step, _port_ex, rps)

    ref_phi = 90
    diff = np.abs(np.array(rps.as_thetaphis()[1]) - ref_phi)
    ref_phi_index = np.where(diff == np.min(diff))[0][0]
    acc_ref_phi = rps.as_thetaphis()[1][ref_phi_index]
    print(f'reference phi: {acc_ref_phi}')

    #### Set up array of 6 elements with two different spacings
    #### and reference antenna to nr. 2
    nr_ants = 6
    ref_ant = 2
    pats_OC = []
    pats_SC = []
    pats_NO = []
    pows_act_OC = []
    pows_act_SC = []
    pows_act_NO = []
    # The two spacings in units of lambda:
    spacefacs = [10, 0.5, 0.2]

    for spfac in spacefacs:
        d = spfac * lambda_
        arr_pos = [[antnr * d - (nr_ants - 1) / 2 * d, 0., 0.] for antnr in range(nr_ants)]
        warnick6.arrayify(element=['ant_Z'], array_positions=arr_pos)

        eepSCdat = warnick6.excite_1by1(eb_arr)
        pow_act_SC = eepSCdat.get_pow_arr()[:, ref_ant, ref_ant]
        pows_act_SC.append(pow_act_SC)
        # Get OC from SC through transformation:
        eepOCdat = eepSCdat.transform_to('OC')
        pow_act_OC = (np.abs(eepOCdat.current_excite) ** 2
                      * np.real(eepOCdat.impedances[:, ref_ant, ref_ant]) / 2.)
        pows_act_OC.append(pow_act_OC)
        eepNOdat = eepSCdat.transform_to('NO', imp_load=np.diag(np.ones(nr_ants)*50))
        pow_act_NO = (np.abs(eepNOdat.current_excite) ** 2
                      * np.real(eepNOdat.get_impedances()[:, ref_ant, ref_ant]) / 2.)
        pows_act_NO.append(pow_act_NO)

        # [n_antennas, n_frequencies, n_thetas, n_phis, n_polarizations]
        _antspats = eepSCdat.get_antspats_arr()
        pats_SC.append(_antspats[:, :, 0, ref_phi_index, :].squeeze())
        _antspats = eepOCdat.get_antspats_arr()
        pats_OC.append(_antspats[:, :, 0, ref_phi_index, :].squeeze())
        _antspats = eepNOdat.get_antspats_arr()
        pats_NO.append(_antspats[:, :, 0, ref_phi_index, :].squeeze())

    def dbi(efield, pow):
        eta = 377
        U = (np.abs(efield[:, 0]) ** 2 + np.abs(efield[:, 1]) ** 2) / (2 * eta)
        dbi_ = 10 * np.log10(U / pow * 4 * np.pi)
        return dbi_

    #### Plot the results
    frqs = frq_cntr_step.aslist()

    from matplotlib.ticker import ScalarFormatter
    base_fontsize = 16
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    fig, axs = plt.subplots(len(spacefacs), 1, figsize=(12, 4*len(spacefacs)), sharex=True)
    for spidx in range(len(spacefacs)):
        pat_SC_dbi = dbi(pats_SC[spidx][ref_ant, :], pows_act_SC[spidx])
        axs[spidx].plot(frqs, pat_SC_dbi, 'y-.')
        pat_OC_dbi = dbi(pats_OC[spidx][ref_ant, :], pows_act_OC[spidx])
        axs[spidx].plot(frqs, pat_OC_dbi, 'r-.')
        pat_NO_dbi = dbi(pats_NO[spidx][ref_ant, :], pows_act_NO[spidx])
        axs[spidx].plot(frqs, pat_NO_dbi, 'b-.')
        axs[spidx].set_xscale('log')
        axs[spidx].set_title(f'$d = {spacefacs[spidx]*lambda_}$ m')
        axs[spidx].set_ylabel('Directivity [dBi]')
        axs[spidx].set_ylim([-90, 15])
        axs[spidx].legend(['SC', 'OC (from SC)', r'imp load = 50 $\Omega$'], fontsize=base_fontsize)
    axs[-1].set_xticks([200, 400, 600, 800, 1000, 1200])
    axs[-1].get_xaxis().set_major_formatter(ScalarFormatter())
    axs[-1].ticklabel_format(style='plain', axis='x')
    axs[-1].set_xlabel(r'$\nu$ [MHz]')
    plt.tick_params(labelsize=base_fontsize)
    plt.subplots_adjust(hspace=0.3)
    if save_figure:
        plt.savefig(f'results/directivity_frqs.eps', dpi=300, facecolor='w')
    plt.show()


def power_antenna():
    save_figure = False

    # N(antennas*polarizations) * N(timings) * N(frequencies)
    # polarization - even: x, odd:y
    d = np.load(raw_data_path + 'SE607_20240430_093342_spw3_int1_dur60_sst.npy')
    f_index = 230
    mean = np.mean(d, axis=1)
    std = np.std(d, axis=1)
    antennas = np.arange(96)
    print(np.shape(d), np.shape(mean))

    x_pol = mean[::2, :]
    mean_x = np.mean(x_pol, axis=0)
    std_x = np.std(x_pol, axis=0)
    # mutual_impx = np.load('results/dual_xpol_f60_s121_numa96.npy')
    # self_impx = np.real(np.diag(mutual_impx[0, :, :]))
    relat_stdx = std_x[f_index]/mean_x[f_index]
    # print(std_x[f_index]/mean_x[f_index], np.std(self_impx)/np.mean(self_impx))

    base_fontsize = 20
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(antennas, mean[::2, f_index], yerr=std[::2, f_index], fmt='-o', color='b', ecolor='r', elinewidth=2,
                capsize=4, capthick=2, markersize=2, linewidth=1)
    ax.text(70, 1.4e7, f'relative std = {format(relat_stdx, ".2%")}', fontsize=base_fontsize)
    ax.set_title('x polarization')
    ax.set_xlabel('No. antenna')
    ax.set_ylabel('Self power')
    if save_figure:
        plt.savefig(f'results/power_xpol.eps', dpi=300, facecolor='w')
    plt.show()

    y_pol = np.concatenate((mean[1::2, :][:31], mean[1::2, :][32:]))
    mean_y = np.mean(y_pol, axis=0)
    std_y = np.std(y_pol, axis=0)
    mutual_impy = np.load('results/dual_ypol_f60_s121_numa96.npy')
    # self_impy = np.real(np.diag(mutual_impy[0, :, :]))
    # self_impy = np.concatenate((self_impy[:31], self_impy[32:]))
    relat_stdy = std_y[f_index] / mean_y[f_index]
    # print(std_y[f_index]/mean_y[f_index], np.std(self_impy)/np.mean(self_impy))

    fig, ax = plt.subplots(figsize=(12, 8))
    mean[1::2, f_index][31] = np.nan
    ax.errorbar(antennas, mean[1::2, f_index], yerr=std[1::2, f_index], fmt='-o', color='b', ecolor='r', elinewidth=2,
                capsize=4, capthick=2, markersize=2, linewidth=1)
    ax.text(70, 1.42e7, f'relative std = {format(relat_stdy, ".2%")}', fontsize=base_fontsize)
    ax.set_title('y polarization')
    ax.set_xlabel('No. antenna')
    ax.set_ylabel('Self power')
    if save_figure:
        plt.savefig(f'results/power_ypol.eps', dpi=300, facecolor='w')
    plt.show()


def imp_ants():
    save_figure = False

    xpol_100 = np.load(f'{data_path}/dual_xpol_100_f44.9_s20_numa96_imp.npy')
    xpol = np.load(f'{data_path}/dual_xpol_f44.9_s20_numa96_imp.npy')
    ypol = np.load(f'{data_path}/dual_ypol_f44.9_s20_numa96_imp.npy')
    # xpol_100 = np.load(f'{data_path}/dual_xpol_100_f44.9_s100_numa96_imp.npy')
    # xpol = np.load(f'{data_path}/dual_xpol_f44.9_s100_numa96_imp.npy')
    # ypol = np.load(f'{data_path}/dual_ypol_f44.9_s100_numa96_imp.npy')
    print(np.shape(xpol))
    ants = np.arange(96)

    base_fontsize = 18
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    fig, ax = plt.subplots(figsize=(12, 8))

    self_xpol = np.real(np.diag(xpol[0, :, :]))
    self_ypol = np.real(np.diag(ypol[0, :, :]))
    self_ypol = np.concatenate((self_ypol[:31], self_ypol[32:]))
    self_xpol_100 = np.real(np.diag(xpol_100[0, :, :]))
    ax.plot(ants, np.diag(np.real(xpol[0, :, :])), 'y-.', label='x pol')
    ax.plot(ants, np.diag(np.real(ypol[0, :, :])), 'r-.', label='y pol')
    ax.plot(ants, np.diag(np.real(xpol_100[0, :, :])), 'b-.', label='x pol (spacing * 100)')
    text = f'x pol relative std = {format(np.std(self_xpol) / np.mean(self_xpol), ".2%")} \n' \
           f'y pol relative std = {format(np.std(self_ypol) / np.mean(self_ypol), ".2%")}'
    ax.text(38, 12.43, text, fontsize=base_fontsize)
    ax.set_xlabel('No. antennas')
    ax.set_ylabel(r'Impedance ($\Omega$)')
    legend = ax.legend(loc='lower left')
    legend.set_bbox_to_anchor((0, 0.07))
    if save_figure:
        plt.savefig(f'results/lofar_imp.eps', dpi=300, facecolor='w')
    plt.show()


if __name__ == '__main__':
    # directivity_phis()
    # eels_phis()
    # directivity_frqs()
    # power_antenna()
    imp_ants()
