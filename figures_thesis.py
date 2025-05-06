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


def _load_data(data_set, f_index, polar):
    """
    Args:
        data_set (int): 1 or 2. 1 means loading the data in 2020, and 2 means 2024.
        f_index (int): The frequency channel.
        polar (str): 'X' or 'Y'. 'X' is x polarization and 'Y' is y polarization.
    Returns:
        data (2-D array of real values): The data based on the given input params.
        times (1-D array of real values): Time stamps (second).
        flag (1-D array of bool values): Flagging the bad antennas.
    """
    data = None
    times = None
    # Flagging the antennas that are removed
    origin_flags = np.full(96, False, dtype=bool)
    if data_set == 1:
        # N(antennas*polarizations) * N(timings) * N(frequencies)
        # polarization - even: x, odd:y
        ds = np.load(raw_data_path + 'SE607_20201202_115839_spw3_int600_dur86147_sst.npy')
        data = ds[:, :, f_index]
        if polar == 'X':
            data = data[0::2, :]
        elif polar == 'Y':
            data = data[1::2, :]
        else:
            raise ValueError('Invalid param: "polar" must be "X" or "Y".')
        times = np.linspace(0, 24*3600, np.shape(data)[1], endpoint=False)
    elif data_set == 2:
        ds = np.load(raw_data_path + 'SE607_20240916_180834_spw3_int519_dur86400_sst.npz')
        # files = ds.files()  # heads
        times = ds['delta_secs'][:, 0]
        data = np.zeros((192, len(times)))
        for i in range(len(times)):
            data[:, i] = ds[f'arr_{i}'][0, f_index, :]
        if polar == 'X':
            data = data[0::2, :]
        elif polar == 'Y':
            data = data[1::2, :]
        else:
            raise ValueError('Invalid params: "polar" must be "X" or "Y".')
        # removed antennas because they are broken
        origin_flags = np.min(data, axis=1) < 1.e7
    if data_set is None:
        raise ValueError('Invalid param: "data_set" must be 1 or 2.')

    return data, times, origin_flags


def _ant_coord_trans(nside, thetas, phis, beam):
    index = np.arange(0, 12 * nside ** 2, 1)

    """
    Filling the gains below the sky in the source data with zeros and verifying the one-to-one match between gains 
    and coordinates (co-latitude and longitude). 
    """
    thetas = np.linspace(0, 180, 91, endpoint=True) * np.pi / 180
    phis = np.linspace(0, 360, 180, endpoint=False) * np.pi / 180
    thetas_mesh, phis_mesh = np.meshgrid(thetas, phis)
    beam = np.hstack((beam, np.zeros((180, 45))))
    beam = beam.ravel()
    gain_colat = np.arccos(np.sin(thetas_mesh.ravel()) * np.sin(phis_mesh.ravel()))
    gain_lon = np.arctan2(
        - np.sin(thetas_mesh.ravel()) * np.cos(phis_mesh.ravel()), np.cos(thetas_mesh.ravel()))
    gain_lon[gain_lon < 0] = gain_lon[gain_lon < 0] + 2 * np.pi

    """
    Verifying the one-to-one match between gains and pixels and interpolating the gains to expand them to that with 
    'self.nside'. n is the nside to guarantee that every pixel has at least one gain in the source data. 
    """
    n = 16
    pix = hp.ang2pix(n, gain_colat, gain_lon, nest=False)
    gain_array = np.vstack((beam, pix))  # one-to-one match between gains and pixels
    # Ordering to interpolate (Constructing the corresponding relation with index)
    gain_array = gain_array[:, gain_array[1].argsort()]
    gain_fix = np.zeros(12 * n ** 2)
    counts = np.zeros(12 * n ** 2)
    for i in range(gain_array.shape[1]):
        gain_fix[int(gain_array[1, i])] += gain_array[0, i]
        counts[int(gain_array[1, i])] += 1
    gain_fix /= counts
    colatitude_rad, lon_rad = hp.pix2ang(nside, index, nest=False)  # set prime vertical as "equator"
    beam = hp.pixelfunc.get_interp_val(gain_fix, colatitude_rad, lon_rad, nest=False)

    return index, beam


def _random_antenna(nr_samples, frq_cntr, rel_std=0.01, xpol=True, ypol=True, excite='X', ground=True):
    seed = 42
    nr_freqs = 1
    nr_thetas = 46
    nr_phis = 180
    segmentalize = 101

    wire_radius = 0.0003
    sep = 2.5 * wire_radius
    pw = 0.090
    ph = 1.6
    aal = 1.38
    pal = aal / np.sqrt(2)
    # The first raw is puck width. The second raw is puck height. The bottom four raws are arms (projection).
    params = np.tile(np.array([[pw], [ph], [pal], [pal], [pal], [pal]]), (1, nr_samples))
    np.random.seed(seed)
    errors = np.random.normal(loc=0., scale=rel_std, size=np.shape(params))
    params = params + params * errors

    imps = np.zeros(nr_samples, dtype=complex)
    EEPs = []
    EELs = []
    for i in tqdm(range(nr_samples), desc='Generating the random antennas'):
        model_name = __file__.replace('.py', '')
        lba_model = ArrayModel(model_name)
        lba_model.set_commentline(lba_model.name)
        lba_model.set_commentline('Author: T. Liu')
        lba_model.set_commentline('Date: 2024-05-09')

        element = []
        if xpol:
            px1 = (-np.cos(np.deg2rad(45)) * (params[2, i] + params[0, i] / 2),
                   -np.sin(np.deg2rad(45)) * (params[2, i] + params[0, i] / 2), params[1, i] - params[2, i])
            px2 = (-np.cos(np.deg2rad(45)) * params[0, i] / 2, -np.sin(np.deg2rad(45)) * params[0, i] / 2, params[1, i])
            px3 = (np.cos(np.deg2rad(45)) * params[0, i] / 2, np.sin(np.deg2rad(45)) * params[0, i] / 2, params[1, i])
            px4 = (np.cos(np.deg2rad(45)) * (params[3, i] + params[0, i] / 2),
                   np.sin(np.deg2rad(45)) * (params[3, i] + params[0, i] / 2), params[1, i] - params[3, i])
            # px3 = (-px2[0], -px2[1], px2[2])
            # px4 = (-px1[0], -px1[1], px1[2])
            lx12 = (px1, px2)
            lx23 = (px2, px3)
            lx34 = (px3, px4)
            element.append('ant_X')

            lba_model['ant_X']['-X'] = Wire(*lx12, wire_radius)
            lba_model['ant_X']['puck_X'] = Wire(*lx23, wire_radius)
            lba_model['ant_X']['+X'] = Wire(*lx34, wire_radius)
            if excite == 'X':
                lba_model['ant_X']['puck_X'].add_port(0.5, 'LNA_X', VoltageSource(1.0))
                _port_ex = ('LNA_X', VoltageSource(1.0))

        if ypol:
            py1 = (-np.sin(np.deg2rad(45)) * (params[4, i] + params[0, i] / 2),
                   np.cos(np.deg2rad(45)) * (params[4, i] + params[0, i] / 2), params[1, i] - params[4, i] + sep)
            py2 = (
                -np.sin(np.deg2rad(45)) * params[0, i] / 2, np.cos(np.deg2rad(45)) * params[0, i] / 2,
                params[1, i] + sep)
            py3 = (
                np.sin(np.deg2rad(45)) * params[0, i] / 2, -np.cos(np.deg2rad(45)) * params[0, i] / 2,
                params[1, i] + sep)
            py4 = (np.sin(np.deg2rad(45)) * (params[5, i] + params[0, i] / 2),
                   -np.cos(np.deg2rad(45)) * (params[5, i] + params[0, i] / 2), params[1, i] - params[5, i] + sep)
            # py3 = (-py2[0], -py2[1], py2[2])
            # py4 = (-py1[0], -py1[1], py1[2])
            ly12 = (py1, py2)
            ly23 = (py2, py3)
            ly34 = (py3, py4)
            element.append('ant_Y')

            lba_model['ant_Y']['-Y'] = Wire(*ly12, wire_radius)
            lba_model['ant_Y']['puck_Y'] = Wire(*ly23, wire_radius)
            lba_model['ant_Y']['+Y'] = Wire(*ly34, wire_radius)
            if excite == 'Y':
                lba_model['ant_Y']['puck_Y'].add_port(0.5, 'LNA_Y', VoltageSource(1.0))
                _port_ex = ('LNA_Y', VoltageSource(1.0))
        lba_model.arrayify(element=element, array_positions=np.array([[0, 0, 0]]))

        _frq_cntr_step = FreqSteps('lin', nr_freqs, frq_cntr, 2.0)
        lba_model.segmentalize(segmentalize, frq_cntr)
        if ground:
            lba_model.set_ground()
        if xpol:
            if lba_model['ant_X']['puck_X'].nr_seg < 11:
                lba_model['ant_X']['puck_X'].nr_seg = 11
        if ypol:
            if lba_model['ant_Y']['puck_Y'].nr_seg < 11:
                lba_model['ant_Y']['puck_Y'].nr_seg = 11
        _epl = RadPatternSpec(nth=nr_thetas, dth=2., nph=nr_phis, dph=2.)
        eb_arr = ExecutionBlock(_frq_cntr_step, _port_ex, _epl)
        # diag_segs = lba_model.seglamlens(_frq_cntr_step)
        lba_model.add_executionblock('exec', eb_arr)
        # diag_thin = lba_model.segthinness()
        eepSCdat = lba_model.excite_1by1(eb_arr, save_necfile=False)
        eepNOdat = eepSCdat.transform_to('NO', imp_load=np.diag(np.ones(1, dtype=complex) * (5.6 - 236.7j)))
        # eepOCdat = eepSCdat.transform_to('OC')
        # eepNOdat = eepSCdat.transform_to('NO', imp_load=np.diag(np.ones(nr_ants, dtype=complex)*(5.6-236.7j)))
        # # print(_epl.as_thetaphis())
        # eelOCdat = eepOCdat.get_EELs()
        # eepdat = eepOCdat
        Z = eepSCdat.get_impedances()
        EEP = eepNOdat.get_antspats_arr()
        EEPs.append(EEP[0, :, :, :, :])
        EEL = eepNOdat.get_EELs().eels[0]
        EELs.append(EEL)
        imps[i] = Z[0, 0, 0]
    EEPs = np.array(EEPs)

    return EEPs, EELs, imps


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
    relat_stdx = std_x[f_index]/mean_x[f_index]

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
    relat_stdy = std_y[f_index] / mean_y[f_index]

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


def auto_corr_data():
    save_figure = False

    num_grids = 2000

    f_index = 230  # f = 44.92 MHz
    polar = 'X'
    base_time_2020 = '2020-12-02 11:58:39.000'
    base_time_2024 = '2024-09-16 18:08:34.000'

    data_2020, times_2020, origin_flags_2020 = _load_data(1, f_index, polar)
    data_2024, times_2024, origin_flags_2024 = _load_data(2, f_index, polar)

    origin_flags_2024[31] = True  # The data from the antenna 31 in data_2024 are invalid
    num_ants_2020 = np.sum(~origin_flags_2020)
    num_ants_2024 = np.sum(~origin_flags_2024)

    data_2020 = data_2020[~origin_flags_2020, :]
    data_2024 = data_2024[~origin_flags_2024, :]
    times_2020 = Time(base_time_2020, format='iso', scale='utc') + times_2020 * u.second
    times_2024 = Time(base_time_2024, format='iso', scale='utc') + times_2024 * u.second

    location = EarthLocation(lon=11.917778 * u.deg, lat=57.393056 * u.deg)
    times_2020.location = location
    times_2024.location = location

    lst_2020 = times_2020.sidereal_time('mean').hour  # Transform to sidereal time
    lst_2024 = times_2024.sidereal_time('mean').hour  # Transform to sidereal time

    lst_grid = np.linspace(0, 24, num_grids)
    interp_2020 = interp1d(lst_2020, data_2020, kind='linear', fill_value="extrapolate")
    interp_2024 = interp1d(lst_2024, data_2024, kind='linear', fill_value="extrapolate")

    data_interp_2020 = interp_2020(lst_grid)
    data_interp_2024 = interp_2024(lst_grid)

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_2020.T)
    ax.set_title(f'{num_ants_2020} LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Self power')
    if save_figure:
        plt.savefig(f'results/24hautocorr_raw.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_2024.T)
    ax.set_title(f'{num_ants_2024} LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Self power')
    if save_figure:
        plt.savefig(f'results/24hautocorr_raw2.eps', dpi=300, facecolor='w')
    plt.show()


def lofar_layout():
    save_figure = False

    f_index = 230  # f = 44.92 MHz
    polar = 'X'

    data_2020, times_2020, origin_flags_2020 = _load_data(1, f_index, polar)
    data_2024, times_2024, origin_flags_2024 = _load_data(2, f_index, polar)

    either_ants_flags = origin_flags_2020 + origin_flags_2024
    either_ants_flags[31] = True  # The data from the antenna 31 in data_2024 are invalid

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    ants_valid = ~either_ants_flags.copy()
    ants_broken = either_ants_flags.copy()
    ants_broken[31] = False  # The data from the antenna 31 in data_2024 are invalid but not broken
    index_valid = np.where(ants_valid)[0]
    index_broken = np.where(ants_broken)[0]
    index_invalid = np.array([31])

    arr_origin = np.loadtxt(raw_data_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_name = arr_origin[:, 0]
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0]
    arr_y = arr_pos[:, 1]

    arr_xvalid = arr_x.copy()[index_valid]
    arr_yvalid = arr_y.copy()[index_valid]
    arr_xbroken = arr_x.copy()[index_broken]
    arr_ybroken = arr_y.copy()[index_broken]
    arr_xinvalid = arr_x.copy()[index_invalid]
    arr_yinvalid = arr_y.copy()[index_invalid]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(arr_xvalid, arr_yvalid, s=50, c='red')
    ax.scatter(arr_xbroken, arr_ybroken, s=50, c='black', marker='x')
    ax.scatter(arr_xinvalid, arr_yinvalid, s=50, c='blue')
    ax.plot([-28, -22], [-28, -22], color='blue')
    ax.text(-30, -30, 'x pol', fontsize=12)
    ax.plot([-28, -22], [-22, -28], color='blue')
    ax.text(-30, -20, 'y pol', fontsize=12)
    for i, name in enumerate(arr_name):
        ax.annotate(name, (arr_x[i], arr_y[i]), fontsize=12, textcoords="offset points", xytext=(0, 5), ha='center')
    ax.set_title('SE607 layout')
    ax.set_xlabel('p Axis (m)')
    ax.set_ylabel('q Axis (m)')
    if save_figure:
        plt.savefig(f'results/SE607_layout.eps', dpi=300, facecolor='w')
    plt.show()


def imp_ants():
    save_figure = False

    xpol_100 = np.load(f'{data_path}dual_xpol_96_100_f44.92_s101_numa96_imp.npy')
    xpol = np.load(f'{data_path}dual_xpol_96_f44.92_s101_numa96_imp.npy')
    ypol = np.load(f'{data_path}dual_ypol_96_f44.92_s101_numa96_imp.npy')

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
    ax.plot(ants, np.diag(np.real(xpol[0, :, :])), 'y-.', label='x pol')
    ax.plot(ants, np.diag(np.real(ypol[0, :, :])), 'r-.', label='y pol')
    ax.plot(ants, np.diag(np.real(xpol_100[0, :, :])), 'b-.', label='x pol (spacing * 100)')
    text = f'x pol relative std = {format(np.std(self_xpol) / np.mean(self_xpol), ".2%")} \n' \
           f'y pol relative std = {format(np.std(self_ypol) / np.mean(self_ypol), ".2%")}'
    ax.text(36, 27.17, text, fontsize=base_fontsize)
    ax.set_xlabel('No. antennas')
    ax.set_ylabel(r'Impedance ($\Omega$)')
    ax.legend(loc='lower left')
    if save_figure:
        plt.savefig(f'results/lofar_imp.eps', dpi=300, facecolor='w')
    plt.show()


def comp_power():
    save_figure = False

    num_grids = 2000

    f_index = 230  # f = 44.92 MHz
    polar = 'X'
    base_time_2020 = '2020-12-02 11:58:39.000'

    data_2020, times_2020, origin_flags_2020 = _load_data(1, f_index, polar)

    times_2020 = Time(base_time_2020, format='iso', scale='utc') + times_2020 * u.second

    location = EarthLocation(lon=11.917778 * u.deg, lat=57.393056 * u.deg)
    times_2020.location = location

    lst_2020 = times_2020.sidereal_time('mean').hour  # Transform to sidereal time

    lst_grid = np.linspace(0, 24, num_grids)
    interp_2020 = interp1d(lst_2020, data_2020, kind='linear', fill_value="extrapolate")
    data_interp_2020 = interp_2020(lst_grid)

    ks_2020 = np.mean(data_interp_2020) / np.mean(data_interp_2020, axis=1)
    data_interp_norm_2020 = data_interp_2020 * ks_2020[:, None]

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_2020.T)
    ax.set_title(f'96 LBA antennas (x polarization)')
    std = np.mean(np.std(data_interp_2020, axis=0) / np.mean(data_interp_2020, axis=0))
    ax.text(6, 2.3e7, f"relative std = {std * 100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Self power')
    if save_figure:
        plt.savefig(f'results/24hautocorr.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_norm_2020.T)
    ax.set_title(f'96 LBA antennas (x polarization)')
    std = np.mean(np.std(data_interp_norm_2020, axis=0) / np.mean(data_interp_norm_2020, axis=0))
    ax.text(6, 1.9e7, f"relative std = {std * 100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Self power')
    if save_figure:
        plt.savefig(f'results/24hautocorr_norm.eps', dpi=300, facecolor='w')
    plt.show()


def power_simulation():
    # LOFAR: x to east (several degrees difference), y to north (several degrees difference), z to up
    # healpix: x to up, y to east, z to north
    save_figure = False

    frq = 44.92
    nside = 256  # at least 256 to avoid repetition of pixels
    lon = 11.917778
    lat = 57.393056
    f_index = 230

    nr_thetas = 46
    nr_phis = 180
    thetas = np.linspace(0, 90, nr_thetas, endpoint=True) * np.pi / 180
    phis = np.linspace(0, 360, nr_phis, endpoint=False) * np.pi / 180

    eep96 = np.load(f'{data_path}dual_xpol_96_f44.92_s101_numa96_EEP.npy')[:, 0, :, :, :]
    # eep96 = np.load('results/dual_xpol_96_thick_wire_f44.92_s101_numa96_EEP.npy')[:, 0, :, :, :]
    _1, _2, origin_flags = _load_data(2, f_index, 'X')
    index_invalid = np.sum(~origin_flags[:31])
    eep61 = np.load(f'{data_path}dual_xpol_62_broken_f44.92_s101_numa62_EEP.npy')[:, 0, :, :, :]
    # eep61 = np.load('results/dual_xpol_62_parts_EEP_300.npy')[:, 0, :, :, :]
    eep61 = np.delete(eep61, index_invalid, axis=0)

    eep96 = np.abs(eep96[:, :, :, 0]) ** 2 + np.abs(eep96[:, :, :, 1]) ** 2
    eep96_uni_healpix = np.zeros((96, 12 * nside ** 2))
    eep96_norm_healpix = np.zeros((96, 12 * nside ** 2))
    _, beam960 = _ant_coord_trans(nside, thetas, phis, eep96[0, :, :].T)
    eep61 = np.abs(eep61[:, :, :, 0]) ** 2 + np.abs(eep61[:, :, :, 1]) ** 2
    eep61_uni_healpix = np.zeros((61, 12 * nside ** 2))

    nr_samples = 96
    EEPs, _, imps = _random_antenna(nr_samples, frq)
    EEPs = np.abs(EEPs[:, 0, :, :, 0]) ** 2 + np.abs(EEPs[:, 0, :, :, 1]) ** 2
    EEPs_uni_single_healpix = np.zeros((nr_samples, 12 * nside ** 2))
    EEPs_norm_single_healpix = np.zeros((nr_samples, 12 * nside ** 2))
    _, beam0_single = _ant_coord_trans(nside, thetas, phis, EEPs[0, :, :].T)

    for i in range(96):
        _, beam = _ant_coord_trans(nside, thetas, phis, eep96[i, :, :].T)
        _, beam_single = _ant_coord_trans(nside, thetas, phis, EEPs[i, :, :].T)
        beam_uni = beam / np.sum(beam)
        beam_norm = beam / np.sum(beam960)
        beam_uni_single = beam_single / np.sum(beam_single)
        beam_norm_single = beam_single / np.sum(beam0_single)
        eep96_uni_healpix[i, :] = beam_uni
        eep96_norm_healpix[i, :] = beam_norm
        EEPs_uni_single_healpix[i, :] = beam_uni_single
        EEPs_norm_single_healpix[i, :] = beam_norm_single
        print(1, i)

    for i in range(61):
        _, beam61 = _ant_coord_trans(nside, thetas, phis, eep61[i, :, :].T)
        beam_uni61 = beam61 / np.sum(beam61)
        eep61_uni_healpix[i, :] = beam_uni61
        print(2, i)

    times = np.linspace(0, 24 * 3600, 1000, endpoint=False)
    base_time = '2020-12-02 11:58:39.000'
    times = Time(base_time, format='iso', scale='utc') + times * u.second
    location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    times.location = location
    lst = times.sidereal_time('mean').hour  # Transform to sidereal time
    base_time = times.datetime[np.where(lst == np.min(lst))[0]][0]

    timings = 140
    ants96_temps_uni = np.zeros((timings, 96))
    ants96_temps_norm = np.zeros((timings, 96))
    ants_temps_uni_single = np.zeros((timings, nr_samples))
    ants_temps_norm_single = np.zeros((timings, nr_samples))
    ants61_temps_uni = np.zeros((timings, 61))
    for t in range(timings):
        (latitude, longitude, elevation) = (str(lat), str(lon), 0)
        ov = LFSMObserver()
        ov.lon = longitude
        ov.lat = latitude
        ov.elev = elevation
        minute = (base_time.minute + t * 10) % 60
        hour = (base_time.hour + (base_time.minute + t * 10) // 60) % 24
        day = base_time.day + (base_time.hour + (base_time.minute + t * 10) // 60) // 24
        ov.date = datetime(2020, 12, day, hour, minute, base_time.second)
        sky = ov.generate(frq)
        sky = hp.pixelfunc.ud_grade(sky, nside)
        print(3, day, hour, minute, np.mean(sky))

        obs_uni = sky[None, :] * eep96_uni_healpix
        obs_norm = sky[None, :] * eep96_norm_healpix
        obs_uni_single = sky[None, :] * EEPs_uni_single_healpix
        obs_norm_single = sky[None, :] * EEPs_norm_single_healpix
        obs61_uni = sky[None, :] * eep61_uni_healpix
        ant_temp_uni = np.sum(obs_uni, axis=1)
        ant_temp_norm = np.sum(obs_norm, axis=1)
        ant_temp_uni_single = np.sum(obs_uni_single, axis=1)
        ant_temp_norm_single = np.sum(obs_norm_single, axis=1)
        ant_temp61_uni = np.sum(obs61_uni, axis=1)
        ants96_temps_uni[t, :] = ant_temp_uni
        ants96_temps_norm[t, :] = ant_temp_norm
        ants_temps_uni_single[t, :] = ant_temp_uni_single
        ants_temps_norm_single[t, :] = ant_temp_norm_single
        ants61_temps_uni[t, :] = ant_temp61_uni

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    times = np.linspace(0, 24 * 3600, 140, endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants96_temps_uni)
    std = np.mean(np.std(ants96_temps_uni, axis=1) / np.mean(ants96_temps_uni, axis=1))
    ax.text(4.5, 8500, f"relative std = {std*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_simulation.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants96_temps_norm)
    std = np.mean(np.std(ants96_temps_norm, axis=1) / np.mean(ants96_temps_norm, axis=1))
    ax.text(4.5, 8500, f"relative std = {std*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_simulation_origin.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants_temps_uni_single)
    std = np.mean(np.std(ants_temps_uni_single, axis=1) / np.mean(ants_temps_uni_single, axis=1))
    ax.text(4.5, 8500, f"relative std = {std*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 isolated antennas with errors (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_errors.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants_temps_norm_single)
    std = np.mean(np.std(ants_temps_norm_single, axis=1) / np.mean(ants_temps_norm_single, axis=1))
    ax.text(4.5, 10000, f"relative std = {std*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 isolated antennas with errors (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_errors_origin.eps', dpi=300, facecolor='w')
    plt.show()

    flags = origin_flags.copy()
    flags[31] = True

    base_fontsize = 22
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times[70:] / 3600, (ants61_temps_uni[70:, :] - ants96_temps_uni[70:, ~flags]) / ants96_temps_uni[70:, ~flags])
    ax.set_title('61 LBA antennas (x polarization)')
    ax.set_xlabel('Time over 12h')
    ax.set_ylabel('Relative difference')
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_simulation_reldiff.eps', dpi=300, facecolor='w')
    plt.show()


def power_diff():
    save_figure = True

    num_grids = 2000

    f_index = 230  # f = 44.92 MHz
    polar = 'X'
    base_time_2020 = '2020-12-02 11:58:39.000'
    base_time_2024 = '2024-09-16 18:08:34.000'

    for mode in ['a', 'b']:
        data_2020, times_2020, origin_flags_2020 = _load_data(1, f_index, polar)
        data_2024, times_2024, origin_flags_2024 = _load_data(2, f_index, polar)

        origin_flags_2024[31] = True  # The data from the antenna 31 in data_2024 are invalid
        times_flags = np.full(num_grids, False, dtype=bool)

        if mode == 'a':
            num_ants_2020 = np.sum(~origin_flags_2020)
            num_ants_2024 = np.sum(~origin_flags_2024)
            data_2020 = data_2020[~origin_flags_2020, :]
            data_2024 = data_2024[~origin_flags_2024, :]
        else:
            either_ants_flags = origin_flags_2020 + origin_flags_2024
            times_flags[:int(num_grids / 2)] = True
            num_ants_2020 = np.sum(~either_ants_flags)
            num_ants_2024 = np.sum(~either_ants_flags)
            data_2020 = data_2020[~either_ants_flags, :]
            data_2024 = data_2024[~either_ants_flags, :]

        times_2020 = Time(base_time_2020, format='iso', scale='utc') + times_2020 * u.second
        times_2024 = Time(base_time_2024, format='iso', scale='utc') + times_2024 * u.second

        location = EarthLocation(lon=11.917778 * u.deg, lat=57.393056 * u.deg)
        times_2020.location = location
        times_2024.location = location

        lst_2020 = times_2020.sidereal_time('mean').hour  # Transform to sidereal time
        lst_2024 = times_2024.sidereal_time('mean').hour  # Transform to sidereal time

        lst_grid = np.linspace(0, 24, num_grids)
        interp_2020 = interp1d(lst_2020, data_2020, kind='linear', fill_value="extrapolate")
        interp_2024 = interp1d(lst_2024, data_2024, kind='linear', fill_value="extrapolate")

        data_interp_2020 = interp_2020(lst_grid)
        data_interp_2024 = interp_2024(lst_grid)
        lst_grid = lst_grid[~times_flags]
        data_interp_2020 = data_interp_2020[:, ~times_flags]
        data_interp_2024 = data_interp_2024[:, ~times_flags]

        base_fontsize = 26
        config = {
            "font.family": 'Times New Roman',  # 设置字体类型
            "font.size": base_fontsize,
            "mathtext.fontset": 'stix',
        }
        rcParams.update(config)

        if mode == 'a':
            ks_2020 = np.mean(data_interp_2020) / np.mean(data_interp_2020, axis=1)
            ks_2024 = np.mean(data_interp_2024) / np.mean(data_interp_2024, axis=1)
            data_interp_2020 = data_interp_2020 * ks_2020[:, None]
            data_interp_2024 = data_interp_2024 * ks_2024[:, None]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(lst_grid, data_interp_2020.T)
            ax.set_title(f'{num_ants_2020} LBA antennas (x polarization)')
            ax.set_xlabel('Time over 24h')
            ax.set_ylabel('Self power')
            if save_figure:
                plt.savefig(f'results/24hautocorr_joint_norm_2020.eps', dpi=300, facecolor='w')
            plt.show()

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(lst_grid, data_interp_2024.T)
            ax.set_title(f'{num_ants_2024} LBA antennas (x polarization)')
            ax.set_xlabel('Time over 24h')
            ax.set_ylabel('Self power')
            if save_figure:
                plt.savefig(f'results/24hautocorr_joint_norm_2024.eps', dpi=300, facecolor='w')
            plt.show()

        else:
            data_interp_total = np.vstack((data_interp_2020, data_interp_2024))
            ks_total = np.mean(data_interp_total) / np.mean(data_interp_total, axis=1)
            data_interp_2020 = data_interp_2020 * ks_total[:num_ants_2020, None]
            data_interp_2024 = data_interp_2024 * ks_total[num_ants_2020:, None]
            delta_data = data_interp_2024 - data_interp_2020

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(lst_grid, delta_data.T / data_interp_2020.T)
            ax.set_title(f'{num_ants_2020} LBA antennas (x polarization)')
            ax.set_xlabel('Time over 12h')
            ax.set_ylabel('Relative power difference')
            if save_figure:
                plt.savefig(f'results/12hautocorr_reldiff.eps', dpi=300, facecolor='w')
            plt.show()

            rescaled_delta_data = delta_data / data_interp_2020
            rescaled_delta_data = rescaled_delta_data - np.mean(rescaled_delta_data, axis=0)[None, :]
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(lst_grid, rescaled_delta_data.T)
            ax.set_title(f'{num_ants_2024} LBA antennas (x polarization)')
            ax.set_xlabel('Time over 12h')
            ax.set_ylabel('Rescaled relative difference')
            if save_figure:
                plt.savefig(f'results/12hautocorr_rescaled_reldiff.eps', dpi=300, facecolor='w')
            plt.show()


if __name__ == '__main__':
    # directivity_phis()
    # eels_phis()
    # directivity_frqs()
    # power_antenna()
    # auto_corr_data()
    # lofar_layout()
    # imp_ants()
    # comp_power()
    power_simulation()
    # power_diff()
