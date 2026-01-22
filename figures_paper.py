import sys

import numpy as np
import matplotlib as mpl
# mpl.use('AGG')

from matplotlib import rcParams
from matplotlib import pyplot as plt
import pandas as pd
from nec2array import (ArrayModel, VoltageSource, FreqSteps, Wire, impedanceRLC,
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
raw_data_path = '../raw_data/'
data_path = '../figures_thesis_materials/'
data_path2 = '../figures_paper_materials/'


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
        times = np.linspace(0, 24 * 3600, np.shape(data)[1], endpoint=False)
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

    wire_radius = 0.001
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
    for i in tqdm(range(nr_samples), desc='Generate the random antennas'):
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


def power_antenna():
    save_figure = True

    # N(antennas*polarizations) * N(timings) * N(frequencies)
    # polarization - even: x, odd:y
    d = np.load(raw_data_path + 'SE607_20240430_093342_spw3_int1_dur60_sst.npy')
    f_index = 230
    mean = np.mean(d, axis=1)
    std = np.std(d, axis=1)
    antennas = np.arange(96)
    print(np.shape(d), np.shape(mean))

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
    x_pol = mean[::2, :]
    mean_x = np.mean(x_pol, axis=0)
    std_x = np.std(x_pol, axis=0)
    relat_stdx = std_x[f_index] / mean_x[f_index]
    ax.text(70, 1.4e7, f'relative std = {format(relat_stdx, ".2%")}', fontsize=base_fontsize)
    ax.set_title('x polarization')
    ax.set_xlabel('No. antenna')
    ax.set_ylabel('Auto-correlated power')
    if save_figure:
        plt.savefig(f'results/power_xpol.pdf', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    mean[1::2, f_index][31] = np.nan
    ax.errorbar(antennas, mean[1::2, f_index], yerr=std[1::2, f_index], fmt='-o', color='b', ecolor='r', elinewidth=2,
                capsize=4, capthick=2, markersize=2, linewidth=1)
    y_pol = np.concatenate((mean[1::2, :][:31], mean[1::2, :][32:]))
    mean_y = np.mean(y_pol, axis=0)
    std_y = np.std(y_pol, axis=0)
    relat_stdy = std_y[f_index] / mean_y[f_index]
    ax.text(70, 1.42e7, f'relative std = {format(relat_stdy, ".2%")}', fontsize=base_fontsize)
    ax.set_title('y polarization')
    ax.set_xlabel('No. antenna')
    ax.set_ylabel('Auto-correlated power')
    if save_figure:
        plt.savefig(f'results/power_ypol.pdf', dpi=300, facecolor='w')
    plt.show()


def auto_corr_data():
    save_figure = True

    f_index = 230  # f = 44.92 MHz
    polar = 'X'
    base_time_2020 = '2020-12-02 11:58:39.000'

    data_2020, times_2020, origin_flags_2020 = _load_data(1, f_index, polar)

    num_ants_2020 = np.sum(~origin_flags_2020)

    data_2020 = data_2020[~origin_flags_2020, :]
    times_2020 = Time(base_time_2020, format='iso', scale='utc') + times_2020 * u.second

    location = EarthLocation(lon=11.917778 * u.deg, lat=57.393056 * u.deg)
    times_2020.location = location

    lst_2020 = times_2020.sidereal_time('mean').hour  # Transform to sidereal time

    min_idx = np.argmin(lst_2020)
    lst_2020_sorted = np.concatenate([lst_2020[min_idx:], lst_2020[:min_idx]])
    data_2020_sorted = np.concatenate([data_2020[:, min_idx:], data_2020[:, :min_idx]], axis=1)

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_2020_sorted, data_2020_sorted.T)
    ax.set_title(str(num_ants_2020) + r' LBA x-pol antennas', fontsize=28)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Auto-correlated power')
    if save_figure:
        plt.savefig(f'results/24hautocorr_raw.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/24hautocorr_raw.png', dpi=300, facecolor='w')
    plt.show()


def lofar_layout():
    save_figure = False

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    arr_origin = np.loadtxt(raw_data_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_name = arr_origin[:, 0]
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0]
    arr_y = arr_pos[:, 1]

    edge_elems = [7, 86, 59, 31, 53, 22, 23, 91, 52, 68, 69, 9, 10, 11, 56, 42, 43, 89, 35, 34, 54, 75, 50]
    inner_elems = [i for i in range(96) if i not in edge_elems]
    arr_x_inner = arr_x.copy()[inner_elems]
    arr_y_inner = arr_y.copy()[inner_elems]
    arr_x_edge = arr_x.copy()[edge_elems]
    arr_y_edge = arr_y.copy()[edge_elems]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(arr_x_inner, arr_y_inner, s=50, c='black')
    ax.scatter(arr_x_edge, arr_y_edge, s=50, c='blue')
    ax.plot([-28, -22], [-28, -22], color='red')
    ax.text(-30, -30, 'x pol', fontsize=12)
    ax.plot([-28, -22], [-22, -28], color='red')
    ax.text(-30, -20, 'y pol', fontsize=12)
    for i, name in enumerate(arr_name):
        if i in edge_elems:
            ax.annotate(
                name, (arr_x[i], arr_y[i]), fontsize=12, color='blue', textcoords="offset points",
                xytext=(0, 5), ha='center'
            )
        else:
            ax.annotate(
                name, (arr_x[i], arr_y[i]), fontsize=12, color='black', textcoords="offset points",
                xytext=(0, 5), ha='center'
            )
    ax.set_title('Layout of the Swedish LOFAR LBA')
    ax.set_xlabel('p Axis (m)')
    ax.set_ylabel('q Axis (m)')
    if save_figure:
        plt.savefig(f'results/SE607_layout.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/SE607_layout.png', dpi=300, facecolor='w')
    plt.show()


def imp_ants():
    save_figure = True

    xpol_100 = np.load(f'{data_path}dual_xpol_96_100_f44.92_s101_numa96_imp.npy')
    xpol = np.load(f'{data_path}dual_xpol_96_f44.92_s101_numa96_imp.npy')
    ypol = np.load(f'{data_path}dual_ypol_96_f44.92_s101_numa96_imp.npy')
    print(xpol_100.shape, xpol.shape, ypol.shape)

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
    # self_ypol = np.concatenate((self_ypol[:31], self_ypol[32:]))
    ax.plot(ants, np.diag(np.real(xpol[0, :, :])), 'y-.', label='x pol')
    ax.plot(ants, np.diag(np.real(ypol[0, :, :])), 'r-.', label='y pol')
    ax.plot(ants, np.diag(np.real(xpol_100[0, :, :])), 'b-.', label='x pol (spacing * 100)')
    text = f'x pol relative std = {format(np.std(self_xpol) / np.mean(self_xpol), ".2%")} \n' \
           f'y pol relative std = {format(np.std(self_ypol) / np.mean(self_ypol), ".2%")}'
    ax.text(36, 27.54, text, fontsize=base_fontsize)
    ax.set_xlabel('No. antennas')
    ax.set_ylabel(r'Impedance ($\Omega$)')
    ax.legend(loc='lower left')
    if save_figure:
        plt.savefig(f'results/lofar_imp.pdf', dpi=300, facecolor='w')
    plt.show()


def comp_power():
    save_figure = False

    edge_elems = [7, 86, 59, 31, 53, 22, 23, 91, 52, 68, 69, 9, 10, 11, 56, 42, 43, 89, 35, 34, 54, 75, 50]
    inner_elems = [i for i in range(96) if i not in edge_elems]

    num_grids = 144
    # frq = 44.92
    frq = 41.
    # frq = 59.
    ds = np.load(raw_data_path + 'SE607_20240916_180834_spw3_int519_dur86400_sst.npz')
    freqs_mhz = ds['frequencies'] / 1e6
    f_index = np.argmin(np.abs(freqs_mhz - frq))
    frq = freqs_mhz[f_index]
    # f_index = 230  # f = 44.92 MHz
    polar = 'X'
    base_time_2020 = '2020-12-02 11:58:39.000'

    data_2020, times_2020, origin_flags_2020 = _load_data(1, f_index, polar)
    times_2020 = Time(base_time_2020, format='iso', scale='utc') + times_2020 * u.second

    location = EarthLocation(lon=11.917778 * u.deg, lat=57.393056 * u.deg)
    times_2020.location = location

    lst_2020 = times_2020.sidereal_time('mean').hour  # Transform to sidereal time
    if num_grids == len(lst_2020):
        loc_start = np.where(lst_2020 == np.min(lst_2020))[0][0]
        lst_grid = np.zeros_like(lst_2020)
        lst_grid[:len(lst_2020)-loc_start] = lst_2020[loc_start:]
        lst_grid[len(lst_2020)-loc_start:] = lst_2020[:loc_start]
    else:
        lst_grid = np.linspace(0, 24, num_grids)
    interp_2020 = interp1d(lst_2020, data_2020, kind='linear', fill_value="extrapolate")
    data_interp_2020 = interp_2020(lst_grid)

    ks_2020 = np.mean(data_interp_2020) / np.mean(data_interp_2020, axis=1)
    data_interp_norm_2020 = data_interp_2020 * ks_2020[:, None]
    # data_interp_2020_edge = data_interp_2020[edge_elems, :]
    # data_interp_2020_inner = data_interp_2020[inner_elems, :]
    data_interp_norm_2020_edge = data_interp_norm_2020[edge_elems, :]
    data_interp_norm_2020_inner = data_interp_norm_2020[inner_elems, :]

    with np.load(f'{data_path2}power_simulation_{f_index}.npz', allow_pickle=True) as power_sim:
        if num_grids == len(lst_2020):
            # times = lst_grid * 3600
            times = power_sim['times']
        else:
            times = power_sim['times']
        times = np.linspace(0, 24 * 3600, len(times), endpoint=False)
        ants_temps_uni_iso = power_sim['ants_temps_uni_iso']
        ants96_temps_norm = power_sim['ants96_temps_norm']
        ants96_temps_uni = power_sim['ants96_temps_uni']
        ants_temps_norm_single = power_sim['ants_temps_norm_single']
        ants_temps_uni_single = power_sim['ants_temps_uni_single']
    ants96_temps_uni_edge = ants96_temps_uni[:, edge_elems]
    ants96_temps_uni_inner = ants96_temps_uni[:, inner_elems]
    ants_temps_uni_single_edge = ants_temps_uni_single[:, edge_elems]
    ants_temps_uni_single_inner = ants_temps_uni_single[:, inner_elems]

    base_fontsize = 30
    legend_fontsize = base_fontsize
    text_fontsize = base_fontsize
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_2020.T)
    std = np.sqrt(np.mean(np.var(data_interp_2020, axis=0) / np.mean(data_interp_2020) ** 2))
    ax.text(0.25, 0.92,
            f"relative std = {std * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Auto-correlated power')
    ax.set_title(r'Case Obs, Raw', fontsize=base_fontsize)
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/24hautocorr.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/24hautocorr.png', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_norm_2020_inner[0, :].T, color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(lst_grid, data_interp_norm_2020_inner[1:, :].T, color="#0072B2", linestyle="--")
    ax.plot(lst_grid, data_interp_norm_2020_edge[0, :].T, color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(lst_grid, data_interp_norm_2020_edge[1:, :].T, color="#E69F00", linestyle="-")
    std_inner = np.sqrt(np.mean(np.var(data_interp_norm_2020_inner, axis=0) / np.mean(data_interp_norm_2020) ** 2))
    std_edge = np.sqrt(np.mean(np.var(data_interp_norm_2020_edge, axis=0) / np.mean(data_interp_norm_2020) ** 2))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Auto-correlated power')
    ax.set_title(r'Case Obs, Normalized', fontsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(1.02, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/24hautocorr_norm_check_edge.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/24hautocorr_norm_check_edge.png', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants96_temps_norm)
    ax.plot(times / 3600, ants_temps_uni_iso, color='black', label='Case AF')
    std = np.sqrt(np.mean(np.var(ants96_temps_norm, axis=1) / np.mean(ants96_temps_norm) ** 2))
    ax.text(0.25, 0.92,
            f"relative std = {std * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    ax.set_title('Case MC, Raw', fontsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(1.02, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_simulation_origin.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/xpol_anttemp_simulation_origin.png', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants96_temps_uni_inner[:, 0], color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(times / 3600, ants96_temps_uni_inner[:, 1:], color="#0072B2", linestyle="--")
    ax.plot(times / 3600, ants96_temps_uni_edge[:, 0], color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(times / 3600, ants96_temps_uni_edge[:, 1:], color="#E69F00", linestyle="-")
    ax.plot(times / 3600, ants_temps_uni_iso, color='black', label='Case AF')
    std_inner = np.sqrt(np.mean(np.var(ants96_temps_uni_inner, axis=1) / np.mean(ants96_temps_uni) ** 2))
    std_edge = np.sqrt(np.mean(np.var(ants96_temps_uni_edge, axis=1) / np.mean(ants96_temps_uni) ** 2))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    ax.set_title('Case MC, Normalized', fontsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(1.02, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_simulation_check_edge.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/xpol_anttemp_simulation_check_edge.png', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants_temps_norm_single)
    ax.plot(times / 3600, ants_temps_uni_iso, color='black', label='Case AF')
    std = np.sqrt(np.mean(np.var(ants_temps_norm_single, axis=1) / np.mean(ants_temps_norm_single) ** 2))
    ax.text(0.25, 0.92,
            f"relative std = {std * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    ax.set_title('Case NI, raw', fontsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(1.02, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_errors_origin.pdf', dpi=300+10, facecolor='w')
        plt.savefig(f'results/xpol_anttemp_errors_origin.png', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants_temps_uni_single_inner[:, 0], color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(times / 3600, ants_temps_uni_single_inner[:, 1:], color="#0072B2", linestyle="--")
    ax.plot(times / 3600, ants_temps_uni_single_edge[:, 0], color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(times / 3600, ants_temps_uni_single_edge[:, 1:], color="#E69F00", linestyle="-")
    ax.plot(times / 3600, ants_temps_uni_iso, color='black', label='Case AF')
    std_inner = np.sqrt(np.mean(np.var(ants_temps_uni_single_inner, axis=1) / np.mean(ants_temps_uni_single) ** 2))
    std_edge = np.sqrt(np.mean(np.var(ants_temps_uni_single_edge, axis=1) / np.mean(ants_temps_uni_single) ** 2))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge * 100:.3g}%",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    ax.set_title('Case NI, Normalized', fontsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(1.02, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_errors_check_edge.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/xpol_anttemp_errors_check_edge.png', dpi=300, facecolor='w')
    plt.show()

    ref = np.mean(ants_temps_uni_iso)

    ratio_data_inner = ref / np.mean(data_interp_norm_2020_inner, axis=1)
    data_interp_cali_2020_inner = data_interp_norm_2020_inner * ratio_data_inner[:, None]
    ratio_data_edge = ref / np.mean(data_interp_norm_2020_edge, axis=1)
    data_interp_cali_2020_edge = data_interp_norm_2020_edge * ratio_data_edge[:, None]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_cali_2020_inner[0, :].T, color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(lst_grid, data_interp_cali_2020_inner[1:, :].T, color="#0072B2", linestyle="--")
    ax.plot(lst_grid, data_interp_cali_2020_edge[0, :].T, color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(lst_grid, data_interp_cali_2020_edge[1:, :].T, color="#E69F00", linestyle="-")
    ax.plot(times / 3600, ants_temps_uni_iso, color='black', label='Case AF')
    std_inner = np.sqrt(np.mean(np.var(data_interp_cali_2020_inner, axis=0)))
    std_edge = np.sqrt(np.mean(np.var(data_interp_cali_2020_edge, axis=0)))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Auto-correlated power')
    ax.set_title(r'Case Obs, Calibrated', fontsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(1.02, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/24hautocorr_norm_calibrated.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/24hautocorr_norm_calibrated.png', dpi=300, facecolor='w')
    plt.show()

    ants96_sim_inner = ref / np.mean(ants96_temps_uni_inner, axis=0)
    ants96_temps_cali_inner = ants96_temps_uni_inner * ants96_sim_inner[None, :]
    ants96_sim_edge = ref / np.mean(ants96_temps_uni_edge, axis=0)
    ants96_temps_cali_edge = ants96_temps_uni_edge * ants96_sim_edge[None, :]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants96_temps_cali_inner[:, 0], color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(times / 3600, ants96_temps_cali_inner[:, 1:], color="#0072B2", linestyle="--")
    ax.plot(times / 3600, ants96_temps_cali_edge[:, 0], color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(times / 3600, ants96_temps_cali_edge[:, 1:], color="#E69F00", linestyle="-")
    ax.plot(times / 3600, ants_temps_uni_iso, color='black', label='Case AF')
    std_inner = np.sqrt(np.mean(np.var(ants96_temps_cali_inner, axis=1)))
    std_edge = np.sqrt(np.mean(np.var(ants96_temps_cali_edge, axis=1)))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    ax.set_title('Case MC, Calibrated', fontsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(1.02, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_simulation_calibrated.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/xpol_anttemp_simulation_calibrated.png', dpi=300, facecolor='w')
    plt.show()

    ants_sim_inner = ref / np.mean(ants_temps_uni_single_inner, axis=0)
    ants_temps_uni_cali_inner = ants_temps_uni_single_inner * ants_sim_inner[None, :]
    ants_sim_edge = ref / np.mean(ants_temps_uni_single_edge, axis=0)
    ants_temps_uni_cali_edge = ants_temps_uni_single_edge * ants_sim_edge[None, :]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants_temps_uni_cali_inner[:, 0], color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(times / 3600, ants_temps_uni_cali_inner[:, 1:], color="#0072B2", linestyle="--")
    ax.plot(times / 3600, ants_temps_uni_cali_edge[:, 0], color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(times / 3600, ants_temps_uni_cali_edge[:, 1:], color="#E69F00", linestyle="-")
    ax.plot(times / 3600, ants_temps_uni_iso, color='black', label='Case AF')
    std_inner = np.sqrt(np.mean(np.var(ants_temps_uni_cali_inner, axis=1)))
    std_edge = np.sqrt(np.mean(np.var(ants_temps_uni_cali_edge, axis=1)))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    ax.set_title('Case NI, Calibrated', fontsize=base_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(1.02, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_errors_calibrated.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/xpol_anttemp_errors_calibrated.png', dpi=300, facecolor='w')
    plt.show()

    data_stack = np.vstack((data_interp_cali_2020_inner, data_interp_cali_2020_edge))
    data_mean = np.mean(data_stack, axis=0)
    resi_data_inner = data_interp_cali_2020_inner - data_mean[None, :]
    resi_data_edge = data_interp_cali_2020_edge - data_mean[None, :]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, resi_data_inner[0, :].T, color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(lst_grid, resi_data_inner[1:, :].T, color="#0072B2", linestyle="--")
    ax.plot(lst_grid, resi_data_edge[0, :].T, color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(lst_grid, resi_data_edge[1:, :].T, color="#E69F00", linestyle="-")
    std_inner = np.sqrt(np.mean(np.var(resi_data_inner, axis=0)))
    std_edge = np.sqrt(np.mean(np.var(resi_data_edge, axis=0)))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Auto-correlated power')
    ax.set_title(r'Residuals of Case Obs, Calibrated', fontsize=base_fontsize)
    ax.legend(loc="lower left", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(0, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/24hautocorr_calibrated_resi.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/24hautocorr_calibrated_resi.png', dpi=300, facecolor='w')
    plt.show()

    ants96_temps_stack = np.hstack((ants96_temps_cali_inner, ants96_temps_cali_edge))
    ants96_temps_mean = np.mean(ants96_temps_stack, axis=1)
    resi_ants96_temps_inner = ants96_temps_cali_inner - ants96_temps_mean[:, None]
    resi_ants96_temps_edge = ants96_temps_cali_edge - ants96_temps_mean[:, None]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, resi_ants96_temps_inner[:, 0], color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(times / 3600, resi_ants96_temps_inner[:, 1:], color="#0072B2", linestyle="--")
    ax.plot(times / 3600, resi_ants96_temps_edge[:, 0], color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(times / 3600, resi_ants96_temps_edge[:, 1:], color="#E69F00", linestyle="-")
    std_inner = np.sqrt(np.mean(np.var(resi_ants96_temps_inner, axis=1)))
    std_edge = np.sqrt(np.mean(np.var(resi_ants96_temps_edge, axis=1)))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    ax.set_title('Residuals of Case MC, Calibrated', fontsize=base_fontsize)
    ax.legend(loc="lower left", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(0, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_simulation_calibrated_resi.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/xpol_anttemp_simulation_calibrated_resi.png', dpi=300, facecolor='w')
    plt.show()

    ants_temps_uni_stack = np.hstack((ants_temps_uni_cali_inner, ants_temps_uni_cali_edge))
    ants_temps_uni_mean = np.mean(ants_temps_uni_stack, axis=1)
    resi_ants_temps_uni_inner = ants_temps_uni_cali_inner - ants_temps_uni_mean[:, None]
    resi_ants_temps_uni_edge = ants_temps_uni_cali_edge - ants_temps_uni_mean[:, None]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, resi_ants_temps_uni_inner[:, 0], color="#0072B2", linestyle="--", label='inner elements')
    ax.plot(times / 3600, resi_ants_temps_uni_inner[:, 1:], color="#0072B2", linestyle="--")
    ax.plot(times / 3600, resi_ants_temps_uni_edge[:, 0], color="#E69F00", linestyle="-", label='edge elements')
    ax.plot(times / 3600, resi_ants_temps_uni_edge[:, 1:], color="#E69F00", linestyle="-")
    std_inner = np.sqrt(np.mean(np.var(resi_ants_temps_uni_inner, axis=1)))
    std_edge = np.sqrt(np.mean(np.var(resi_ants_temps_uni_edge, axis=1)))
    ax.text(0.20, 0.92,
            f"relative std inner = {std_inner:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.text(0.20, 0.85,
            f"relative std inner = {std_edge:.3g} K",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color='blue',
            bbox=dict(facecolor='white', alpha=0.0))
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    ax.set_title('Residuals of Case NI, Calibrated', fontsize=base_fontsize)
    ax.legend(loc="lower left", fontsize=legend_fontsize, framealpha=0, bbox_to_anchor=(0, 0))
    plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    if save_figure:
        plt.savefig(f'results/xpol_anttemp_errors_calibrated_resi.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/xpol_anttemp_errors_calibrated_resi.png', dpi=300, facecolor='w')
    plt.show()


def power_diff():
    save_figure = False

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
        either_ants_flags = origin_flags_2020 + origin_flags_2024
        num_ants = np.sum(~either_ants_flags)
        data_2020 = data_2020[~either_ants_flags, :]
        data_2024 = data_2024[~either_ants_flags, :]
        if mode == 'a':
            pass
        else:
            times_flags[:int(num_grids / 2)] = True

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
        print(mode, data_interp_2020.shape, data_interp_2024.shape)

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
            ax.set_title(str(num_ants) + r' actual LBA x-pol antennas in the $\mathbf{dataset\ I}$ (normalized)',
                         fontsize=22)
            ax.set_xlabel('Time over 24h')
            ax.set_ylabel('Auto-correlated power')
            if save_figure:
                plt.savefig(f'results/24hautocorr_joint_norm_2020.pdf', dpi=300, facecolor='w')
                plt.savefig(f'results/24hautocorr_joint_norm_2020.png', dpi=300, facecolor='w')
            plt.show()

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(lst_grid, data_interp_2024.T)
            ax.set_title(str(num_ants) + r' actual LBA x-pol antennas in the $\mathbf{dataset\ II}$ (normalized)',
                         fontsize=22)
            ax.set_xlabel('Time over 24h')
            ax.set_ylabel('Auto-correlated power')
            if save_figure:
                plt.savefig(f'results/24hautocorr_joint_norm_2024.pdf', dpi=300, facecolor='w')
                plt.savefig(f'results/24hautocorr_joint_norm_2024.png', dpi=300, facecolor='w')
            plt.show()

        else:
            data_interp_total = np.vstack((data_interp_2020, data_interp_2024))
            ks_total = np.mean(data_interp_total) / np.mean(data_interp_total, axis=1)
            data_interp_2020 = data_interp_2020 * ks_total[:num_ants, None]
            data_interp_2024 = data_interp_2024 * ks_total[num_ants:, None]
            delta_data = data_interp_2024 - data_interp_2020

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(lst_grid, delta_data.T / data_interp_2020.T)
            ax.set_title(r'Relative difference between the $\mathbf{dataset\ I}$ and $\mathbf{dataset\ II}$',
                         fontsize=22)
            ax.set_xlabel('Time over 12h')
            ax.set_ylabel('Relative power difference')
            if save_figure:
                plt.savefig(f'results/12hautocorr_reldiff.pdf', dpi=300, facecolor='w')
                plt.savefig(f'results/12hautocorr_reldiff.png', dpi=300, facecolor='w')
            plt.show()

            rescaled_delta_data = delta_data / data_interp_2020
            rescaled_delta_data = rescaled_delta_data - np.mean(rescaled_delta_data, axis=0)[None, :]
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(lst_grid, rescaled_delta_data.T)
            ax.set_title(
                r'Relative difference between the $\mathbf{dataset\ I}$ and $\mathbf{dataset\ II}$ (slope removed)',
                fontsize=22)
            ax.set_ylim([-0.03, 0.03])
            ax.set_xlabel('Time over 12h')
            ax.set_ylabel('Rescaled relative difference')
            if save_figure:
                plt.savefig(f'results/12hautocorr_rescaled_reldiff.pdf', dpi=300, facecolor='w')
                plt.savefig(f'results/12hautocorr_rescaled_reldiff.png', dpi=300, facecolor='w')
            plt.show()


def resi_spectrum():
    save_figure = False

    ds = np.load(raw_data_path + 'SE607_20240916_180834_spw3_int519_dur86400_sst.npz')
    # freq_channel = np.argmin(np.abs(freqs_MHz - 44.92))
    # # files = ds.files()  # heads
    # print(freq_channel)
    freqs_mhz = ds['frequencies'] / 1e6
    ch_low = np.argmin(np.abs(freqs_mhz - 30))
    ch_high = np.argmin(np.abs(freqs_mhz - 80))
    print(ch_low, ch_high, np.argmin(np.abs(freqs_mhz - 45)), np.argmin(np.abs(freqs_mhz - 44.92)))
    freqs_mhz = freqs_mhz[ch_low:ch_high]

    std_inner_spectrum = []
    std_edge_spectrum = []
    vis_mean_spectrum = []
    std_inner_spectrum2 = []
    std_edge_spectrum2 = []
    for f_index in tqdm(range(ch_low, ch_high, 1)):

        edge_elems = [7, 86, 59, 31, 53, 22, 23, 91, 52, 68, 69, 9, 10, 11, 56, 42, 43, 89, 35, 34, 54, 75, 50]
        inner_elems = [i for i in range(96) if i not in edge_elems]

        num_grids = 144
        polar = 'X'
        base_time_2020 = '2020-12-02 11:58:39.000'

        data_2020, times_2020, origin_flags_2020 = _load_data(1, f_index, polar)
        times_2020 = Time(base_time_2020, format='iso', scale='utc') + times_2020 * u.second

        location = EarthLocation(lon=11.917778 * u.deg, lat=57.393056 * u.deg)
        times_2020.location = location

        lst_2020 = times_2020.sidereal_time('mean').hour  # Transform to sidereal time
        if num_grids == len(lst_2020):
            loc_start = np.where(lst_2020 == np.min(lst_2020))[0][0]
            lst_grid = np.zeros_like(lst_2020)
            lst_grid[:len(lst_2020) - loc_start] = lst_2020[loc_start:]
            lst_grid[len(lst_2020) - loc_start:] = lst_2020[:loc_start]
        else:
            lst_grid = np.linspace(0, 24, num_grids)
        interp_2020 = interp1d(lst_2020, data_2020, kind='linear', fill_value="extrapolate")
        data_interp_2020 = interp_2020(lst_grid)

        ks_2020 = np.mean(data_interp_2020) / np.mean(data_interp_2020, axis=1)
        data_interp_norm_2020 = data_interp_2020 * ks_2020[:, None]
        data_interp_norm_2020_edge = data_interp_norm_2020[edge_elems, :]
        data_interp_norm_2020_inner = data_interp_norm_2020[inner_elems, :]

        std_inner = np.sqrt(np.mean(np.var(data_interp_norm_2020_inner, axis=0) / np.mean(data_interp_norm_2020) ** 2))
        std_edge = np.sqrt(np.mean(np.var(data_interp_norm_2020_edge, axis=0) / np.mean(data_interp_norm_2020) ** 2))
        std_inner_spectrum.append(std_inner)
        std_edge_spectrum.append(std_edge)
        vis_mean = np.mean(data_interp_norm_2020)
        vis_mean_spectrum.append(vis_mean)

    std_inner_spectrum = np.array(std_inner_spectrum)
    std_edge_spectrum = np.array(std_edge_spectrum)
    vis_mean_spectrum = np.array(vis_mean_spectrum)

    base_fontsize = 26
    legend_fontsize = base_fontsize
    text_fontsize = base_fontsize
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    print(std_inner_spectrum[230], std_edge_spectrum[230])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(freqs_mhz, std_inner_spectrum * 100, label='Inner')
    ax.plot(freqs_mhz, std_edge_spectrum * 100, label='Edge')
    ax.axvline(x=41)
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Relative Standard Deviation [%]')
    ax.set_ylim([-0.1, 3.1])
    ax.set_title(r'Case Obs', fontsize=base_fontsize)
    # plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    ax.legend()
    if save_figure:
        plt.savefig(f'results/1.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/1.png', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(freqs_mhz, vis_mean_spectrum)
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Mean auto-correlated power')
    # ax.set_ylim([-0.1, 3.1])
    # ax.set_title(r'Case Obs, Raw', fontsize=base_fontsize)
    # plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    ax.legend()
    if save_figure:
        plt.savefig(f'results/2.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/2.png', dpi=300, facecolor='w')
    plt.show()

    # sys.exit()

    freqs_mhz2 = freqs_mhz[::3]

    ants96_std_inner_spectrum = []
    ants96_std_edge_spectrum = []
    ants_std_inner_spectrum = []
    ants_std_edge_spectrum = []
    data_mean_spectrum = []
    for f_index in tqdm(range(ch_low, ch_high, 3)):
        with np.load(f'{data_path2}power_simulation_{f_index}.npz', allow_pickle=True) as power_sim:
            # times = power_sim['times']
            # times = np.linspace(0, 24 * 3600, len(times), endpoint=False)
            # ants_temps_uni_iso = power_sim['ants_temps_uni_iso']
            # ants96_temps_norm = power_sim['ants96_temps_norm']
            ants96_temps_uni = power_sim['ants96_temps_uni']
            ants_temps_norm_single = power_sim['ants_temps_norm_single']
            ants_temps_uni_single = power_sim['ants_temps_uni_single']
        ants96_temps_uni_edge = ants96_temps_uni[:, edge_elems]
        ants96_temps_uni_inner = ants96_temps_uni[:, inner_elems]
        ants_temps_uni_single_edge = ants_temps_uni_single[:, edge_elems]
        ants_temps_uni_single_inner = ants_temps_uni_single[:, inner_elems]

        ants96_std_edge = np.sqrt(np.mean(np.var(ants96_temps_uni_edge, axis=1) / np.mean(ants96_temps_uni) ** 2))
        ants96_std_inner = np.sqrt(np.mean(np.var(ants96_temps_uni_inner, axis=1) / np.mean(ants96_temps_uni) ** 2))
        ants_std_edge = np.sqrt(np.mean(np.var(ants_temps_uni_single_edge, axis=1) / np.mean(ants_temps_uni_single) ** 2))
        ants_std_inner = np.sqrt(np.mean(np.var(ants_temps_uni_single_inner, axis=1) / np.mean(ants_temps_uni_single) ** 2))

        ants96_std_edge_spectrum.append(ants96_std_edge)
        ants96_std_inner_spectrum.append(ants96_std_inner)
        ants_std_edge_spectrum.append(ants_std_edge)
        ants_std_inner_spectrum.append(ants_std_inner)

        data_mean = np.mean(ants96_temps_uni)
        data_mean_spectrum.append(data_mean)

    ants96_std_edge_spectrum = np.array(ants96_std_edge_spectrum)
    ants96_std_inner_spectrum = np.array(ants96_std_inner_spectrum)
    ants_std_edge_spectrum = np.array(ants_std_edge_spectrum)
    ants_std_inner_spectrum = np.array(ants_std_inner_spectrum)
    data_mean_spectrum = np.array(data_mean_spectrum)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(freqs_mhz2, ants96_std_inner_spectrum * 100, label='Inner')
    ax.plot(freqs_mhz2, ants96_std_edge_spectrum * 100, label='Edge')
    ax.axvline(x=41)
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Relative Standard Deviation [%]')
    # ax.set_ylim([-0.1, 3.1])
    ax.set_title(r'Case MC', fontsize=base_fontsize)
    # plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    ax.legend()
    if save_figure:
        plt.savefig(f'results/3.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/3.png', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(freqs_mhz2, ants_std_inner_spectrum * 100, label='Inner')
    ax.plot(freqs_mhz2, ants_std_edge_spectrum * 100, label='Edge')
    ax.axvline(x=41)
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Relative Standard Deviation [%]')
    # ax.set_ylim([-0.1, 3.1])
    ax.set_title(r'Case NI', fontsize=base_fontsize)
    # plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    ax.legend()
    if save_figure:
        plt.savefig(f'results/3.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/3.png', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(freqs_mhz2, data_mean_spectrum)
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Mean auto-correlated power')
    # ax.set_ylim([-0.1, 3.1])
    # ax.set_title(r'Case Obs, Raw', fontsize=base_fontsize)
    # plt.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.12)
    ax.legend()
    if save_figure:
        plt.savefig(f'results/2.pdf', dpi=300, facecolor='w')
        plt.savefig(f'results/2.png', dpi=300, facecolor='w')
    plt.show()


def two_lamhalfdip():
    lamhalf = 1.0
    w_radii = 1e-5*2*lamhalf
    dip_len = lamhalf
    p1 = (0., 0., -dip_len/2)
    p2 = (0., 0., +dip_len/2)
    l12 = (p1, p2)
    twodip = ArrayModel('2dip_sbs')
    twodip['dip']['Z'] = Wire(*l12, w_radii).add_port(0.5,'VS')
    return twodip


def generate_loads():
    """
    Test array with given load

    Same array as in test_Array_2_lamhalfdip_sbys()
    """
    # Use function to build model of lambda half dipole
    twodip = two_lamhalfdip()
    fs = FreqSteps('lin', 30, 80., 4.)  # MHz
    twodip.segmentalize(65, fs.max_freq())
    portname = 'VS'
    ex_port = (portname, VoltageSource(1.0))
    arr_pos = [[0.,0.,0.], [6., 0., 0.]]
    twodip.arrayify(element=['dip'], array_positions=arr_pos)
    rps = RadPatternSpec(nth=1, dth=1., thets=90., phis=0.)
    #rps = None
    eepdat = twodip.excite_1by1(ExecutionBlock(fs, ex_port, rps))

    load_adm = impedanceRLC(fs.aslist(False), 50., None, 1.e-12, 'parallel', False)
    #load_adm_k = impedanceRLC(fs.aslist(False), 1000., 17e-7, None, 'series', False)
    print(fs.aslist(False))
    print(load_adm)

    eepNO = eepdat.transform_to('NO', adm_load=load_adm)
    a_NO = eepNO.get_EELs().area_eff()
    a=np.diagonal(eepdat.get_impedances(),axis1=-2,axis2=-1)[...,0]
    b=1/load_adm

    plt.plot(fs.aslist(), np.real(a), 'b')
    plt.plot(fs.aslist(), np.imag(a), 'r')
    plt.plot(fs.aslist(), np.real(b), 'b.-')
    plt.plot(fs.aslist(), np.imag(b), 'r.-')
    plt.grid()
    plt.show()
    _n = a_NO[1,:].squeeze()
    plt.plot(fs.aslist(), _n,'k')
    print(np.max(_n))
    plt.xlabel('Freq. [MHz]')
    plt.ylabel('Area eff [m^2]')
    plt.title('Thin loaded dipole')
    plt.show()


if __name__ == '__main__':
    # power_antenna()
    # auto_corr_data()
    # lofar_layout()
    # imp_ants()
    # comp_power()
    # power_diff()
    # resi_spectrum()
    generate_loads()
    pass
