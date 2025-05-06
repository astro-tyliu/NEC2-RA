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
root_path = '../data/'


def simulate_lofar(frq_cntr, xpol=True, ypol=True, excite='X', ground=True, special=None):
    """
    Args:
        excite (str): It must be 'X' or 'Y'.
    """
    if excite not in ['X', 'Y']:
        raise ValueError('Invalid param: "polar" must be "X" or "Y".')

    save_necfile = False
    save_imp = True
    save_figure = True
    save_figure_data = True
    save_EEP = True
    seed = 42

    # ------------------------------------------------------------------------------------------------------- #
    f_index = 230
    ds = np.load(root_path + 'SE607_20240916_180834_spw3_int519_dur86400_sst.npz')
    # files = ds.files()  # heads
    times = ds['delta_secs'][:, 0]
    data = np.zeros((192, len(times)))
    print(f'Structure of ds: {ds.files}')
    for i in range(len(times)):
        data[:, i] = ds[f'arr_{i}'][0, f_index, :]
    data = data[0::2, :]
    # bad antennas
    flag = np.min(data, axis=1) < 1.e7
    del ds, times, data, i

    ext_thinwire = True
    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0] * 1
    arr_y = arr_pos[:, 1] * 1
    arr_z = np.zeros(len(arr_x))
    arr_pos = np.vstack((arr_x, arr_y, arr_z)).T
    # arr_pos = arr_pos[[94, 46, 44, 85, 36, 37, 64, 4], :]
    arr_pos = arr_pos[~flag, :]
    num_ants = arr_pos.shape[0]
    # ------------------------------------------------------------------------------------------------------- #

    puck_width = 0.090
    puck_height = 1.6
    ant_arm_len = 1.38  # the length of one stick
    proj_arm_len = ant_arm_len / np.sqrt(2)
    wire_radius = 0.0003 * 1
    sep = 2.5 * wire_radius

    model_name = __file__.replace('.py', '')
    lba_model = ArrayModel(model_name)
    lba_model.set_commentline(lba_model.name)
    lba_model.set_commentline('Author: T. Liu')
    lba_model.set_commentline('Date: 2025-05-02')

    element = []
    if xpol:
        px1 = (-np.cos(np.deg2rad(45)) * (proj_arm_len + puck_width / 2),
               -np.sin(np.deg2rad(45)) * (proj_arm_len + puck_width / 2),
               puck_height - proj_arm_len)
        px2 = (-np.cos(np.deg2rad(45)) * puck_width / 2, -np.sin(np.deg2rad(45)) * puck_width / 2, puck_height)
        px3 = (-px2[0], -px2[1], px2[2])
        px4 = (-px1[0], -px1[1], px1[2])
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
        py1 = (-np.sin(np.deg2rad(45)) * (proj_arm_len + puck_width / 2),
               np.cos(np.deg2rad(45)) * (proj_arm_len + puck_width / 2),
               puck_height - proj_arm_len + sep)
        py2 = (-np.sin(np.deg2rad(45)) * puck_width / 2, np.cos(np.deg2rad(45)) * puck_width / 2, puck_height + sep)
        py3 = (-py2[0], -py2[1], py2[2])
        py4 = (-py1[0], -py1[1], py1[2])
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
    lba_model.arrayify(element=element, array_positions=arr_pos)

    nr_freqs = 1
    nr_thetas = 46
    nr_phis = 180
    segmentalize = 101
    _frq_cntr_step = FreqSteps('lin', nr_freqs, frq_cntr, 2.0)
    lba_model.segmentalize(segmentalize, frq_cntr)
    if ground:
        lba_model.set_ground()
    if xpol:
        if lba_model['ant_X']['puck_X'].nr_seg < 3:
            lba_model['ant_X']['puck_X'].nr_seg = 3
    if ypol:
        if lba_model['ant_Y']['puck_Y'].nr_seg < 3:
            lba_model['ant_Y']['puck_Y'].nr_seg = 3
    nr_ants = len(arr_pos)
    _epl = RadPatternSpec(nth=nr_thetas, dth=2., nph=nr_phis, dph=2.)
    eb_arr = ExecutionBlock(_frq_cntr_step, _port_ex, _epl, ext_thinwire=ext_thinwire)
    diag_segs = lba_model.seglamlens(_frq_cntr_step)
    lba_model.add_executionblock('exec', eb_arr)
    diag_thin = lba_model.segthinness()
    s = time()
    eepSCdat = lba_model.excite_1by1(eb_arr, save_necfile=save_necfile)
    e = time()
    print(e - s)
    eepOCdat = eepSCdat.transform_to('OC')
    eepNOdat = eepSCdat.transform_to('NO', imp_load=np.diag(np.ones(nr_ants, dtype=complex)*(5.6-236.7j)))
    # print(_epl.as_thetaphis())
    eelOCdat = eepOCdat.get_EELs()
    eepdat = eepOCdat
    print("Impedances")
    Z = eepdat.get_impedances()

    if special is None:
        prefix = f'dual_{excite.lower()}pol_{num_ants}'
    else:
        prefix = f'dual_{excite.lower()}pol_{num_ants}_{special}'
    print(11111, prefix)

    if save_imp:
        np.save(f'results/{prefix}_f{frq_cntr}_s{segmentalize}_numa{np.shape(arr_pos)[0]}_imp.npy', Z)

    # N (antennas) * N (frequencies) * N (thetas) * N (phis) * N (polarizations)
    EEP = eepNOdat.get_antspats_arr()
    print(np.shape(EEP))
    if save_EEP:
        np.save(f'results/{prefix}_f{frq_cntr}_s{segmentalize}_numa{np.shape(arr_pos)[0]}_EEP.npy', EEP)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(num_ants), np.diag(np.real(Z[0, :, :])))
    plt.show()
    # # import sys
    # # sys.exit()

    if True:
        eelscdat = eepSCdat.get_EELs()
        eelocdat = eepOCdat.get_EELs()
        eelnodat = eepNOdat.get_EELs()
        Hsc_abs = np.array([])
        hoc_abs = np.array([])
        hno_abs = np.array([])
        for ants in range(nr_ants):
            # zenith
            Hsc_abs_ants = np.sqrt(np.abs(eelscdat.eels[ants].f_tht) ** 2
                                   + np.abs(eelscdat.eels[ants].f_phi) ** 2)
            Hsc_abs = np.append(Hsc_abs, Hsc_abs_ants[0, 0, 0])
            hoc_abs_ants = np.sqrt(np.abs(eelocdat.eels[ants].f_tht) ** 2
                                   + np.abs(eelocdat.eels[ants].f_phi) ** 2)
            hoc_abs = np.append(hoc_abs, hoc_abs_ants[0, 0, 0])
            hno_abs_ants = np.sqrt(np.abs(eelnodat.eels[ants].f_tht) ** 2
                                   + np.abs(eelnodat.eels[ants].f_phi) ** 2)
            hno_abs = np.append(hno_abs, hno_abs_ants[0, 0, 0])

        if save_figure_data:
            np.save(f'results/{prefix}_f{frq_cntr}_s{segmentalize}_numa{np.shape(arr_pos)[0]}_figure_data.npy', np.vstack((Hsc_abs, hoc_abs, hno_abs)))

        #### Plot the results
        ants = np.arange(nr_ants)

        base_fontsize = 12
        config = {
            "font.family": 'Times New Roman',  # 设置字体类型
            "font.size": base_fontsize,
            "mathtext.fontset": 'stix',
        }
        rcParams.update(config)
        fig, ax = plt.subplots(figsize=(12, 8), sharex=True)
        ax.plot(ants, Hsc_abs * 377, 'y-.', label='SC EEL')
        ax.plot(ants, hoc_abs, 'r-.', label='OC (from SC) EEL')
        ax.plot(ants, hno_abs, 'b-.', label='loaded')
        ax.set_title(f'{prefix}_lofar_EEL_antenna')
        ax.set_xlabel('No. antennas')
        ax.set_ylabel('EEL [m]')
        ax.legend(loc='best')
        plt.tick_params(labelsize=base_fontsize)
        if save_figure:
            plt.savefig(f'results/{prefix}_f{frq_cntr}_s{segmentalize}_numa{np.shape(arr_pos)[0]}_lofar_eels.eps', dpi=300, facecolor='w')
        plt.show()

    return Z


def imp_ants():
    save_figure = False

    xpol = np.load('results/dual_xpol_96_f44.9_s20_numa96_imp.npy')
    # ypol = np.load('results/dual_ypol_96_f44.92_s101_numa96_imp.npy')
    xpol_100 = np.load('results/dual_xpol_100_f44.9_s20_numa96_imp.npy')
    # xpol = np.load('results/dual_xpol_f44.9_s20_numa96_imp.npy')
    ypol = np.load('results/dual_ypol_f44.9_s20_numa96_imp.npy')
    # xpol = np.load('results/dual_xpol_f60_s121_numa96.npy')
    # ypol = np.load('results/dual_ypol_f60_s121_numa96.npy')
    print(xpol)
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


def arr_layout(xx_autocor=None):
    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_name = arr_origin[:, 0]
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0]
    arr_y = arr_pos[:, 1]

    if xx_autocor is None:
        file_path = root_path + 'sid20240319T124804_SE607_n61_s285_XX_XY_YX_YY_xst.txt'
        df = pd.read_csv(file_path)
        xx_origin = np.array(df.iloc[6:6+96])
        print(df.iloc[7])
        xx = np.zeros((96, 96), dtype=complex)
        # print(xx_origin[0, 0].replace('(', '').replace(')', ''))
        for i in range(96):
            xxi_list = xx_origin[i, 0].replace('(', '').replace(')', '')
            xx[i, :] = np.array(xxi_list.split(), dtype=complex)
        xx_autocor = np.real(np.diag(xx))

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(arr_x, arr_y, s=50, c=xx_autocor, cmap='Reds')
    ax.plot([-28, -22], [-28, -22], color='blue')
    ax.text(-30, -30, 'x pol', fontsize=12)
    ax.plot([-28, -22], [-22, -28], color='blue')
    ax.text(-30, -20, 'y pol', fontsize=12)
    for i, name in enumerate(arr_name):
        ax.annotate(name, (arr_x[i], arr_y[i]), fontsize=12, textcoords="offset points", xytext=(0, 5), ha='center')
    ax.set_title('Scatter Plot Example')
    ax.set_xlabel('p Axis')
    ax.set_ylabel('q Axis')
    fig.colorbar(scatter, ax=ax)
    plt.show()


def power_antenna():
    save_figure = False

    # N(antennas*polarizations) * N(timings) * N(frequencies)
    # polarization - even: x, odd:y
    d = np.load(root_path + 'SE607_20240430_093342_spw3_int1_dur60_sst.npy')
    f_index = 230
    mean = np.mean(d, axis=1)
    std = np.std(d, axis=1)
    antennas = np.arange(96)
    print(np.shape(d), np.shape(mean))

    x_pol = mean[::2, :]
    mean_x = np.mean(x_pol, axis=0)
    std_x = np.std(x_pol, axis=0)
    mutual_impx = np.load('results/dual_xpol_f60_s121_numa96.npy')
    self_impx = np.real(np.diag(mutual_impx[0, :, :]))
    relat_stdx = std_x[f_index]/mean_x[f_index]
    print(std_x[f_index]/mean_x[f_index], np.std(self_impx)/np.mean(self_impx))

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
    self_impy = np.real(np.diag(mutual_impy[0, :, :]))
    self_impy = np.concatenate((self_impy[:31], self_impy[32:]))
    relat_stdy = std_y[f_index] / mean_y[f_index]
    print(std_y[f_index]/mean_y[f_index], np.std(self_impy)/np.mean(self_impy))

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
        ds = np.load(root_path + 'SE607_20201202_115839_spw3_int600_dur86147_sst.npy')
        data = ds[:, :, f_index]
        if polar == 'X':
            data = data[0::2, :]
        elif polar == 'Y':
            data = data[1::2, :]
        else:
            raise ValueError('Invalid param: "polar" must be "X" or "Y".')
        times = np.linspace(0, 24*3600, np.shape(data)[1], endpoint=False)
    elif data_set == 2:
        ds = np.load(root_path + 'SE607_20240916_180834_spw3_int519_dur86400_sst.npz')
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


def power_time():
    save_figure = False
    show = True

    data_set = 1
    f_index = 230  # f = 44.92 MHz
    polar = 'X'

    data, times, flag = _load_data(data_set, f_index, polar)
    data = data[~flag, :]

    # mean = np.mean(data, axis=0)
    # mean = np.repeat(mean[:, None].T, np.shape(data)[0], axis=0)
    # target_product = np.sum(mean * data[:, :], axis=1)
    # self_product = np.sum(data[:, :] ** 2, axis=1)
    # ks = np.sum(mean * data, axis=1) / np.sum(data * data, axis=1)
    ks = np.mean(data) / np.mean(data, axis=1)

    base_fontsize = 18
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    fig, ax = plt.subplots(figsize=(12, 8))
    normalization = data[:, :].T * ks
    ax.plot(times / 3600, normalization)
    ax.set_xlim(left=0., right=24.)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.set_title('96 LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Self power')
    if save_figure:
        plt.savefig(f'results/24hautocorr_raw.eps', dpi=300, facecolor='w')
    if show:
        plt.show()


def comp_power():
    num_grids = 2000

    f_index = 230  # f = 44.92 MHz
    polar = 'X'
    base_time_2020 = '2020-12-02 11:58:39.000'
    base_time_2024 = '2024-09-16 18:08:34.000'

    data_2020, times_2020, origin_flags_2020 = _load_data(1, f_index, polar)
    data_2024, times_2024, origin_flags_2024 = _load_data(2, f_index, polar)

    either_ants_flags = origin_flags_2020 + origin_flags_2024
    if polar == 'X':
        either_ants_flags[31] = True  # The data from the antenna 31 in data_2024 are invalid
    either_ants_flags = np.full(96, False, dtype=bool)
    times_flags = np.full(num_grids, False, dtype=bool)
    # times_flags[:int(num_grids / 2)] = True
    # times_flags[1500:] = True

    num_ants = np.sum(~either_ants_flags)
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

    data_interp_total = np.vstack((data_interp_2020, data_interp_2024))

    # ks_2020 = np.mean(data_interp_2020) / np.mean(data_interp_2020, axis=1)
    # ks_2024 = np.mean(data_interp_2024) / np.mean(data_interp_2024, axis=1)
    ks_total = np.mean(data_interp_total) / np.mean(data_interp_total, axis=1)
    # ks_total /= ks_total

    # data_interp_2020 = data_interp_2020 * ks_2020[:, None]
    # data_interp_2024 = data_interp_2024 * ks_2024[:, None]
    data_interp_2020 = data_interp_2020 * ks_total[:num_ants, None]
    data_interp_2024 = data_interp_2024 * ks_total[num_ants:, None]
    delta_data = data_interp_2024 - data_interp_2020
    
    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_2020.T)
    ax.set_title(f'{num_ants} LBA antennas (x polarization)')
    std = np.mean(np.std(data_interp_2020, axis=0) / np.mean(data_interp_2020, axis=0))
    ax.text(6, 2.3e7, f"relative std = {std*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Self power')
    # plt.savefig(f'results/24hautocorr.eps', dpi=300, facecolor='w')
    # plt.savefig(f'results/24hautocorr_norm.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, data_interp_2024.T)
    ax.set_title(f'{num_ants} LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Self power')
    # plt.savefig(f'results/24hautocorr_norm2.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, delta_data.T)
    ax.set_title(f'{num_ants} LBA antennas (x polarization)')
    ax.set_xlabel('Time over 12h')
    ax.set_ylabel('Power difference')
    # plt.savefig(f'results/12hautocorr_diff.eps', dpi=300, facecolor='w')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, delta_data.T / data_interp_2020.T)
    ax.set_title(f'{num_ants} LBA antennas (x polarization)')
    ax.set_xlabel('Time over 12h')
    ax.set_ylabel('Relative power difference')
    # plt.savefig(f'results/12hautocorr_reldiff.eps', dpi=300, facecolor='w')
    plt.show()

    rescaled_delta_data = delta_data / data_interp_2020
    rescaled_delta_data = rescaled_delta_data - np.mean(rescaled_delta_data, axis=0)[None, :]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, rescaled_delta_data.T)
    ax.set_title(f'{num_ants} LBA antennas (x polarization)')
    ax.set_xlabel('Time over 12h')
    ax.set_ylabel('Rescaled relative difference')
    # plt.savefig(f'results/12hautocorr_rescaled_reldiff.eps', dpi=300, facecolor='w')
    plt.show()

    ants_valid = ~either_ants_flags.copy()
    ants_broken = either_ants_flags.copy()
    if polar == 'X':
        ants_broken[31] = False  # The data from the antenna 31 in data_2024 are invalid
    index_valid = np.where(ants_valid)[0]
    index_broken = np.where(ants_broken)[0]
    index_invalid = np.array([31])

    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
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

    # rescaled_data = delta_data - np.mean(delta_data, axis=0)[None, :]
    rescaled_data = delta_data
    stds = np.sqrt(np.mean(rescaled_data[:, :] ** 2, axis=1))

    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(arr_xvalid, arr_yvalid, s=50, c=stds, cmap='Reds')
    scatter = ax.scatter(arr_xvalid, arr_yvalid, s=50, c='red')
    ax.scatter(arr_xbroken, arr_ybroken, s=50, c='black', marker='x')
    ax.scatter(arr_xinvalid, arr_yinvalid, s=50, c='blue')
    # ax.scatter(arr_x, arr_y, s=50)
    ax.plot([-28, -22], [-28, -22], color='blue')
    ax.text(-30, -30, 'x pol', fontsize=12)
    ax.plot([-28, -22], [-22, -28], color='blue')
    ax.text(-30, -20, 'y pol', fontsize=12)
    for i, name in enumerate(arr_name):
        ax.annotate(name, (arr_x[i], arr_y[i]), fontsize=12, textcoords="offset points", xytext=(0, 5), ha='center')
    ax.set_title('SE607 layout')
    ax.set_xlabel('p Axis (m)')
    ax.set_ylabel('q Axis (m)')
    # fig.colorbar(scatter, ax=ax)
    # plt.savefig(f'results/SE607_layout.eps', dpi=300, facecolor='w')
    # plt.savefig(f'results/SE607_layout_simple.eps', dpi=300, facecolor='w')
    plt.show()

    min_dists = np.array([])
    for i in range(len(index_valid)):
        dists = np.sqrt((arr_xinvalid - arr_xvalid[i]) ** 2 + (arr_yinvalid - arr_yvalid[i]) ** 2)
        min_dists = np.append(min_dists, np.mean((1 / dists) ** 2))
    from scipy.stats import pearsonr, spearmanr
    r, p_value = spearmanr(min_dists, stds-np.min(stds))

    return


def statistical_analysis():
    # N(antennas*polarizations) * N(timings) * N(frequencies)
    # polarization - even: x, odd:y
    d = np.load(root_path + 'SE607_20240430_093342_spw3_int1_dur60_sst.npy')
    f_index = 230
    mean = np.mean(d, axis=1)
    std = np.std(d, axis=1)
    antennas = np.arange(96)
    d_mean = d - mean[:, None, :]

    xdata_f = d[::2, :, f_index]
    xcorr_matrix = np.corrcoef(xdata_f)
    print("x covariance：")
    print(xcorr_matrix)

    ydata_f = d[1::2, :, f_index]
    ycorr_matrix = np.corrcoef(ydata_f)
    print("y covariance：")
    print(ycorr_matrix)

    refer_antenna = 11

    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_name = arr_origin[:, 0]
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0]
    arr_y = arr_pos[:, 1]
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(arr_x, arr_y, s=50, c=np.abs(xcorr_matrix[:, refer_antenna]), cmap='Reds')
    for i, name in enumerate(arr_name):
        ax.annotate(name, (arr_x[i], arr_y[i]), fontsize=12, textcoords="offset points", xytext=(0, 5), ha='center')
    ax.set_title(f'antenna no.{refer_antenna} as reference, X-polar')
    ax.set_xlabel('p Axis')
    ax.set_ylabel('q Axis')
    fig.colorbar(scatter, ax=ax)
    plt.show()

    distance = np.sqrt((arr_pos[:, 0]-arr_pos[refer_antenna, 0])**2 + (arr_pos[:, 1]-arr_pos[refer_antenna, 1])**2)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(distance, xcorr_matrix[:, refer_antenna])
    ax.set_title(f'antenna no.{refer_antenna} as reference, X-polar')
    ax.set_xlabel('distance')
    ax.set_ylabel('correlation')
    plt.show()

    # refer_antennas = [0, 30]
    # corr =
    # for i in range(0, np.shape(d)[2], 10):
    #     xdata_f = d[::2, :, i]
    #     xcorr_matrix = np.corrcoef(xdata_f)
    #     ydata_f = d[1::2, :, i]
    #     ycorr_matrix = np.corrcoef(ydata_f)
    # nf = np.arange(np.shape(d)[2])
    # fig, ax = fig.subplots(figsize=(12, 8))
    # ax.plot()


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
    ground = False

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
    if ground:
        warnick6.set_ground()
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
        # arr_pos = [[0., 0., 0.], [100000., 0., 0.], [200000., 0., 0.]]
        arr_pos = [[(antnr * d - (nr_ants - 1) / 2 * d) * 1, 0., 0.] for antnr in range(nr_ants)]
        warnick6.arrayify(element=['ant_Z'], array_positions=arr_pos)
        eepSCdat = warnick6.excite_1by1(eb_arr, save_necfile=False)
        # eepSCdat = warnick6.calc_eeps_SC(eb_arr, save_necfile=False)
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

        fig, ax = plt.subplots(figsize=(12, 8))
        imps = eepSCdat.get_impedances()[0, :, :]
        ax.plot(np.arange(nr_ants), np.diag(np.real(eepSCdat.get_impedances()[0, :, :])))
        plt.show()

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


def lofar_directivity_phis():
    save_figure = False

    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0]
    arr_y = arr_pos[:, 1]
    arr_z = np.zeros(len(arr_x))
    arr_pos = np.vstack((arr_x, arr_y, arr_z)).T
    arr_pos = arr_pos[8:16, :]
    print(arr_pos)

    puck_width = 0.090
    puck_height = 1.6
    ant_arm_len = 1.38
    wire_radius = 0.0003
    sep = 2.5 * wire_radius

    model_name = __file__.replace('.py', '')
    lba_model = ArrayModel(model_name)
    lba_model.set_commentline(lba_model.name)
    lba_model.set_commentline('Author: T. Liu')
    lba_model.set_commentline('Date: 2024-11-08')

    xpol = True
    ypol = True
    excite = 'X'
    frq_cntr = 44.9 + 40
    ground = False

    element = []
    if xpol:
        px1 = (-np.cos(np.deg2rad(45))*(ant_arm_len+puck_width)/2, -np.sin(np.deg2rad(45))*(ant_arm_len+puck_width)/2.0,
              puck_height-ant_arm_len/np.sqrt(2))
        px2 = (-np.cos(np.deg2rad(45))*puck_width/2, -np.sin(np.deg2rad(45))*puck_width/2, puck_height)
        px3 = (-px2[0], -px2[1], px2[2])
        px4 = (-px1[0], -px1[1], px1[2])
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
        py1 = (-np.cos(np.deg2rad(45))*(ant_arm_len+puck_width)/2, np.sin(np.deg2rad(45))*(ant_arm_len+puck_width)/2.0,
              puck_height-ant_arm_len/np.sqrt(2)+sep)
        py2 = (-np.cos(np.deg2rad(45))*puck_width/2, np.sin(np.deg2rad(45))*puck_width/2, puck_height+sep)
        py3 = (-py2[0], -py2[1], py2[2])
        py4 = (-py1[0], -py1[1], py1[2])
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
    lba_model.arrayify(element=element, array_positions=arr_pos)

    nr_freqs = 1
    nr_thetas = 46
    nr_phis = 180
    segmentalize = 201
    _frq_cntr_step = FreqSteps('lin', nr_freqs, frq_cntr, 2.0)
    lba_model.segmentalize(segmentalize, frq_cntr)
    if ground:
        lba_model.set_ground()
    if xpol:
        if lba_model['ant_X']['puck_X'].nr_seg < 3:
            lba_model['ant_X']['puck_X'].nr_seg = 3
    if ypol:
        if lba_model['ant_Y']['puck_Y'].nr_seg < 3:
            lba_model['ant_Y']['puck_Y'].nr_seg = 3
    nr_ants = len(arr_pos)
    _epl = RadPatternSpec(nth=nr_thetas, dth=2., nph=nr_phis, dph=2.)
    # _epl = None
    eb_arr = ExecutionBlock(_frq_cntr_step, _port_ex, _epl)
    s = time()
    eepSCdat = lba_model.excite_1by1(eb_arr, save_necfile=False)
    eepOCdat = eepSCdat.transform_to('OC')
    eepNOdat = eepSCdat.transform_to('NO', imp_load=np.diag(np.ones(nr_ants, dtype=complex)*(5.6-236.7j)))
    # print(_epl.as_thetaphis())
    e = time()
    print(e - s)
    eelOCdat = eepOCdat.get_EELs()
    eepdat = eepOCdat
    print("Impedances")
    Z =  eepdat.get_impedances()

    #### Set up array of 6 elements with two different spacings
    #### and reference antenna to nr. 2
    ref_ant = 0
    frq_idx = 0
    pats_OC = []
    pats_SC = []
    pows_act_OC = []
    pows_act_SC = []
    # The two spacings in units of lambda:
    pow_act_SC = eepSCdat.get_pow_arr()[frq_idx, ref_ant, ref_ant]
    pows_act_SC.append(pow_act_SC)
    # Get OC from SC through transformation:
    eepOCdat = eepSCdat.transform_to('OC')
    pow_act_OC = (np.abs(eepOCdat.current_excite) ** 2
                  * np.real(eepOCdat.impedances[frq_idx, ref_ant, ref_ant]) / 2.)
    pows_act_OC.append(pow_act_OC)
    # [n_antennas, n_frequencies, n_thetas, n_phis, n_polarizations]
    _antspats = eepSCdat.get_antspats_arr()
    pats_SC.append(_antspats[:, 0, 45, :, :].squeeze())
    _antspats = eepOCdat.get_antspats_arr()
    pats_OC.append(_antspats[:, 0, 45, :, :].squeeze())

    def dbi(efield, pow):
        eta = 377
        U = (np.abs(efield[:, 0]) ** 2 + np.abs(efield[:, 1]) ** 2) / (2 * eta)
        dbi_ = 10 * np.log10(U / pow * 4 * np.pi)
        return dbi_

    #### Plot the results
    phis = _epl.as_thetaphis()[1]

    base_fontsize = 12
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    fig, ax = plt.subplots(figsize=(12, 8), sharex=True)
    spidx = 0
    print(pows_act_SC)
    pat_SC_dbi = dbi(pats_SC[spidx][ref_ant, :], pows_act_SC[spidx])
    ax.plot(phis, pat_SC_dbi, 'y-.')
    pat_OC_dbi = dbi(pats_OC[spidx][ref_ant, :], pows_act_OC[spidx])
    ax.plot(phis, pat_OC_dbi, 'r-.')
    ax.set_title(f'lofar')
    ax.set_xlabel('phi [deg]')
    ax.set_ylabel('Directivity [dBi]')
    ax.legend(['SC', 'OC (from SC)'], fontsize=base_fontsize)
    plt.tick_params(labelsize=base_fontsize)
    if save_figure:
        plt.savefig(f'results/directivity_phi.eps', dpi=300, facecolor='w')
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


def eels_ants():
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
    frq_cntr = 3.0e8 * lambda_ / 1e6
    frq_cntr_step = FreqSteps('exp', nr_freqs, frq_cntr, incr=1.022)
    warnick6.segmentalize(201, frq_cntr)
    _port_ex = ('the_port', VoltageSource(1.0))
    phis = 90.
    rps = RadPatternSpec(thets=90., nph=1, dph=360, phis=phis)
    eb_arr = ExecutionBlock(frq_cntr_step, _port_ex, rps)

    #### Set up array of 6 elements with two different spacings
    #### and reference antenna to nr. 2
    nr_ants = 24
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
        eelocdat = eepOCdat.get_EELs()
        Hsc_abs = np.array([])
        hoc_abs = np.array([])
        for ants in range(nr_ants):
            aaa = eelscdat.eels
            Hsc_abs_ants = np.sqrt(np.abs(eelscdat.eels[ants].f_tht) ** 2
                                   + np.abs(eelscdat.eels[ants].f_phi) ** 2)
            Hsc_abs = np.append(Hsc_abs, Hsc_abs_ants[0, 0, 0])
            hoc_abs_ants = np.sqrt(np.abs(eelocdat.eels[ants].f_tht) ** 2
                                   + np.abs(eelocdat.eels[ants].f_phi) ** 2)
            hoc_abs = np.append(hoc_abs, hoc_abs_ants[0, 0, 0])
        Hscs_abs.append(Hsc_abs)
        hocs_abs.append(hoc_abs)

    #### Plot the results
    ants = np.arange(nr_ants)

    base_fontsize = 12
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    fig, axs1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for spidx in range(len(spacefacs)):
        linel, = axs1[spidx].plot(ants, Hscs_abs[spidx], 'y-.', label='SC eel')
        axs2 = axs1[spidx].twinx()
        liner, = axs2.plot(ants, hocs_abs[spidx], 'r-.', label='OC (from SC) eel')
        axs1[spidx].set_title(f'd={spacefacs[spidx]}*lambda')
        axs1[spidx].set_xlabel('No. antennas')
        axs1[spidx].set_ylabel('SC eel [m/Ohm]')
        axs2.set_ylabel('OC eel [m]')
        lines = [linel, liner]
        labels = [line.get_label() for line in lines]
        axs1[spidx].legend(lines, labels, loc='upper left')
    plt.tick_params(labelsize=base_fontsize)
    if save_figure:
        plt.savefig(f'results/eels_ants.eps', dpi=300, facecolor='w')
    plt.show()


def spherical_harmonics():
    frq = 58.6

    gsm_2016 = GlobalSkyModel16(freq_unit='MHz')
    temp = gsm_2016.generate(frq)
    gsm_2016.view(logged=True)
    plt.show()

    mean = np.mean(temp)
    std = np.std(temp)
    print(mean, std, np.max(temp))

    from pygdsm import GlobalSkyModel
    gsm = GlobalSkyModel(freq_unit='MHz')
    temp = gsm.generate(frq)
    gsm.view(logged=True)
    plt.show()

    mean = np.mean(temp)
    std = np.std(temp)
    print(mean, std, np.max(temp))


def simulate_EEPs():
    save_necfile = False
    save_EEP = False

    puck_width = 0.090
    puck_height = 1.6
    ant_arm_len = 1.38
    wire_radius = 0.0003
    sep = 2.5 * wire_radius

    p1 = (-np.cos(np.deg2rad(45)) * (ant_arm_len + puck_width) / 2,
          -np.sin(np.deg2rad(45)) * (ant_arm_len + puck_width) / 2.0,
          puck_height - ant_arm_len / np.sqrt(2))
    p2 = (-np.cos(np.deg2rad(45)) * puck_width / 2, -np.sin(np.deg2rad(45)) * puck_width / 2, puck_height)
    p3 = (-p2[0], -p2[1], p2[2])
    p4 = (-p1[0], -p1[1], p1[2])
    l12 = (p1, p2)
    l23 = (p2, p3)
    l34 = (p3, p4)

    py1 = (
    -np.cos(np.deg2rad(45)) * (ant_arm_len + puck_width) / 2, np.sin(np.deg2rad(45)) * (ant_arm_len + puck_width) / 2.0,
    puck_height - ant_arm_len / np.sqrt(2) + sep)
    py2 = (-np.cos(np.deg2rad(45)) * puck_width / 2, np.sin(np.deg2rad(45)) * puck_width / 2, puck_height + sep)
    py3 = (-py2[0], -py2[1], py2[2])
    py4 = (-py1[0], -py1[1], py1[2])
    ly12 = (py1, py2)
    ly23 = (py2, py3)
    ly34 = (py3, py4)

    model_name = __file__.replace('.py', '')
    lba_model = ArrayModel(model_name)
    lba_model.set_commentline(lba_model.name)
    lba_model.set_commentline('Author: T. Liu')
    lba_model.set_commentline('Date: 2024-05-28')

    lba_model['ant_X']['-X'] = Wire(*l12, wire_radius)
    lba_model['puck']['LNA_connect'] = Wire(*l23, wire_radius)
    lba_model['ant_X']['+X'] = Wire(*l34, wire_radius)
    lba_model['puck']['LNA_connect'].add_port(0.5, 'LNA_x', VoltageSource(1.0))
    lba_model['ant_Y']['-Y'] = Wire(*ly12, wire_radius)
    lba_model['pucky']['LNAy_connect'] = Wire(*ly23, wire_radius)
    lba_model['ant_Y']['+Y'] = Wire(*ly34, wire_radius)
    lba_model['pucky']['LNAy_connect'].add_port(0.5, 'LNA_y', VoltageSource(1.0))

    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0]
    arr_y = arr_pos[:, 1]
    arr_z = np.zeros(len(arr_x))
    arr_pos = np.vstack((arr_x, arr_y, arr_z)).T
    arr_pos = arr_pos[:4, :]
    lba_model.arrayify(element=['ant_Y', 'pucky'],
                       array_positions=arr_pos)
    ch_frq_cntr = 300
    frq_cntr = ch_frq_cntr * 100 / 512

    nr_freqs = 1
    nr_thetas = 10
    nr_phis = 10
    segmentalize = 143 + 140
    _frq_cntr_step = FreqSteps('lin', nr_freqs, frq_cntr, 2.0)
    lba_model.segmentalize(segmentalize, frq_cntr)
    _port_ex = ('LNA_y', VoltageSource(1.0))
    nr_ants = len(arr_pos)
    _epl = RadPatternSpec(nth=nr_thetas, dth=180./nr_thetas, nph=nr_phis, dph=360./nr_phis)
    eb_arr = ExecutionBlock(_frq_cntr_step, _port_ex, _epl)
    s = time()
    eepSCdat = lba_model.calc_eeps_SC(eb_arr, save_necfile=save_necfile)
    eepNOdat = eepSCdat.transform_to('NO', imp_load=np.diag(np.ones(nr_ants) * 50.))
    print(_epl.as_thetaphis())
    e = time()
    print(e - s)
    # N (antennas) * N (frequencies) * N (thetas) * N (phis) * N (polarizations)
    EEP = eepNOdat.get_antspats_arr()
    print(np.shape(EEP))
    if save_EEP:
        np.save(f'results/EEP_LOFARy_{ch_frq_cntr}.npy', EEP)


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
    # judge = -1
    # repeat = np.array([], dtype=int)
    gain_fix = np.zeros(12 * n ** 2)
    counts = np.zeros(12 * n ** 2)
    for i in range(gain_array.shape[1]):
        gain_fix[int(gain_array[1, i])] += gain_array[0, i]
        counts[int(gain_array[1, i])] += 1
    gain_fix /= counts
    colatitude_rad, lon_rad = hp.pix2ang(nside, index, nest=False)  # set prime vertical as "equator"
    beam = hp.pixelfunc.get_interp_val(gain_fix, colatitude_rad, lon_rad, nest=False)

    return index, beam


def power_simulation():
    # LOFAR: x to east (several degrees difference), y to north (several degrees difference), z to up
    # healpix: x to up, y to east, z to north
    frq = 44.92
    nside = 256  # at least 256 to avoid repetition of pixels
    lon = 11.917778
    lat = 57.393056
    f_index = 230

    nr_thetas = 46
    nr_phis = 180
    thetas = np.linspace(0, 90, nr_thetas, endpoint=True) * np.pi / 180
    phis = np.linspace(0, 360, nr_phis, endpoint=False) * np.pi / 180

    #TODO 分析一下这八张图（不同厚度的wire，不同segment，是否归一化），把发现告诉Tobia
    # eep96 = np.load('results/dual_xpol_96_EEP_300.npy')[:, 0, :, :, :]
    # eep96 = np.load('results/dual_xpol_96_EEP_300_old.npy')[:, 0, :, :, :]
    # eep96 = np.load('results/dual_xpol_96_EEP_300.npy')[:, 0, :, :, :]
    # eep96 = np.load('results/dual_xpol_96_2_EEP_300.npy')[:, 0, :, :, :]
    # eep96 = np.load('results/dual_xpol_96_f44.92_s21_numa96_EEP.npy')[:, 0, :, :, :]
    # eep96 = np.load('results/dual_xpol_96_thick_wire_f44.92_s21_numa96_EEP.npy')[:, 0, :, :, :]
    # eep96 = np.load('results/dual_xpol_96_f44.92_s101_numa96_EEP.npy')[:, 0, :, :, :]
    eep96 = np.load('results/dual_xpol_96_thick_wire_f44.92_s101_numa96_EEP.npy')[:, 0, :, :, :]

    eep96 = np.abs(eep96[:, :, :, 0]) ** 2 + np.abs(eep96[:, :, :, 1]) ** 2
    eep96_healpix = np.zeros((96, 12 * nside ** 2))
    _, beam960 = _ant_coord_trans(nside, thetas, phis, eep96[0, :, :].T)
    for i in range(96):
        _, beam = _ant_coord_trans(nside, thetas, phis, eep96[i, :, :].T)
        beam /= np.sum(beam)
        # beam /= np.sum(beam960)
        print(i, np.sum(beam))
        # hp.mollview(beam)
        # plt.show()
        eep96_healpix[i, :] = beam

    # EEPs_healpix = np.zeros((2, 96, 12 * nside ** 2), dtype=complex)
    # eep96_healpix = np.zeros((96 - 0, 12 * nside ** 2), dtype=complex)
    # pix = np.arange(0, 12 * nside ** 2, 1)
    # _lambda = 3.0e2 / frq
    # # _, beam0 = _ant_coord_trans(nside, thetas, phis, EEPs[0, :, :].T)
    # for i in tqdm(range(96), desc="Normalizing"):
    #     _, real_EEPs_tht = _ant_coord_trans(nside, thetas, phis, np.real(eep96[i, :, :, 0]).T)
    #     _, imag_EEPs_tht = _ant_coord_trans(nside, thetas, phis, np.imag(eep96[i, :, :, 0]).T)
    #     _, real_EEPs_phi = _ant_coord_trans(nside, thetas, phis, np.real(eep96[i, :, :, 1]).T)
    #     _, imag_EEPs_phi = _ant_coord_trans(nside, thetas, phis, np.imag(eep96[i, :, :, 1]).T)
    #     EEPs_tht = real_EEPs_tht + 1j * imag_EEPs_tht
    #     EEPs_phi = real_EEPs_phi + 1j * imag_EEPs_phi
    #     norm_index = np.sqrt(np.sum(np.abs(EEPs_tht) ** 2 + np.abs(EEPs_phi) ** 2))
    #     real_EEPs_tht /= norm_index
    #     imag_EEPs_tht /= norm_index
    #     real_EEPs_phi /= norm_index
    #     imag_EEPs_phi /= norm_index
    #     # hp.mollview(real_EEPs_phi)
    #     # plt.show()
    #     # hp.mollview(imag_EEPs_phi)
    #     # plt.show()
    #     EEPs_healpix[0, i, :] = real_EEPs_tht + 1j * imag_EEPs_tht
    #     EEPs_healpix[1, i, :] = real_EEPs_phi + 1j * imag_EEPs_phi
    #
    #     if i >= 0:
    #         eep96_healpix[i - 0, :] = EEPs_healpix[0, i - 0, :] * EEPs_healpix[0, i, :].conj() + \
    #                                   EEPs_healpix[1, i - 0, :] * EEPs_healpix[1, i, :].conj()
    #     # hp.mollview(eep96_healpix[i - 0, :])
    #     # plt.show()

    _1, _2, origin_flags = _load_data(2, f_index, 'X')

    index_invalid = np.sum(~origin_flags[:31])
    eep61 = np.load('results/dual_xpol_62_parts_EEP_300.npy')[:, 0, :, :, :]
    eep61 = np.delete(eep61, index_invalid, axis=0)
    eep61 = np.abs(eep61[:, :, :, 0]) ** 2 + np.abs(eep61[:, :, :, 1]) ** 2
    eep61_healpix = np.zeros((61, 12 * nside ** 2))
    # _, beam61 = _ant_coord_trans(nside, thetas, phis, eep61[0, :, :].T)
    for i in range(61):
        _, beam = _ant_coord_trans(nside, thetas, phis, eep61[i, :, :].T)
        beam /= np.sum(beam)
        print(i, np.sum(beam))
        # hp.mollview(beam)
        # plt.show()
        eep61_healpix[i, :] = beam

    times = np.linspace(0, 24*3600, 1000, endpoint=False)
    base_time = '2020-12-02 11:58:39.000'
    times = Time(base_time, format='iso', scale='utc') + times * u.second
    location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    times.location = location
    lst = times.sidereal_time('mean').hour  # Transform to sidereal time
    base_time = times.datetime[np.where(lst == np.min(lst))[0]][0]

    timings = 140
    ants96_temps = np.zeros((timings, 96))
    ants61_temps = np.zeros((timings, 61))
    for t in range(timings):
        (latitude, longitude, elevation) = (str(lat), str(lon), 0)
        ov = LFSMObserver()
        # from pygdsm import GSMObserver
        # ov = GSMObserver()
        ov.lon = longitude
        ov.lat = latitude
        ov.elev = elevation
        minute = (base_time.minute + t * 10) % 60
        hour = (base_time.hour + (base_time.minute + t * 10) // 60) % 24
        day = base_time.day + (base_time.hour + (base_time.minute + t * 10) // 60) // 24
        ov.date = datetime(2020, 12, day, hour, minute, base_time.second)
        # ov.date = datetime(2013, 7, day, hour, minute, 0)
        sky = ov.generate(frq)
        # ov.view(logged=True, show=True)
        # hp.mollview(sky)
        # plt.show()
        # plt.title(f'{day}:{hour}:{minute}')
        # plt.show()
        sky = hp.pixelfunc.ud_grade(sky, nside)
        print(np.mean(sky), sky[100:110])

        obs = sky[None, :] * eep96_healpix
        ant_temp = np.sum(obs, axis=1)
        ants96_temps[t, :] = ant_temp
        # ants96_temps[t, :] = np.mean(sky)
        print(day, hour, minute)
        # mean_temp = np.append(mean_temp, np.mean(sky))
        # mean_temp2 = np.append(mean_temp2, np.mean(sky_match))
        # plt.plot(thetas_ext, eep96[0, :])
        # plt.show()

        obs2 = sky[None, :] * eep61_healpix
        ant_temp2 = np.sum(obs2, axis=1)
        ants61_temps[t, :] = ant_temp2

    flags = origin_flags.copy()
    flags[31] = True
    print(111, np.mean(np.std(ants96_temps[70:, ~flags], axis=1)) / np.mean(ants96_temps[70:, ~flags]))
    print(222, np.mean(np.std(ants61_temps[70:, :] / np.mean(ants61_temps[70:, :]), axis=1)))

    # lst_grid = np.linspace(0, 24, num_grids)
    # interp_2020 = interp1d(lst, ants96_temps.T, kind='linear', fill_value="extrapolate")
    # interp_2024 = interp1d(lst, ants61_temps.T, kind='linear', fill_value="extrapolate")
    #
    # temps_interp_2020 = interp_2020(lst_grid)
    # temps_interp_2024 = interp_2024(lst_grid)
    # flag2 = np.full(num_grids, False, dtype=bool)
    # # flag2[:int(num_grids / 2)] = True
    # lst_grid = lst_grid[~flag2]
    # temps_interp_2020 = temps_interp_2020[:, ~flag2]
    # temps_interp_2024 = temps_interp_2024[:, ~flag2]
    #
    # ants96_temps = temps_interp_2020.T
    # ants61_temps = temps_interp_2024.T
    # times = lst_grid

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    times = np.linspace(0, 24 * 3600, 140, endpoint=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants96_temps)
    std = np.mean(np.std(ants96_temps, axis=1) / np.mean(ants96_temps, axis=1))
    ax.text(4.5, 8500, f"relative std = {std*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    # plt.savefig(f'results/xpol_anttemp_simulation_origin.eps', dpi=300, facecolor='w')
    # plt.savefig(f'results/xpol_anttemp_simulation.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants61_temps)
    ax.set_title('61 LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    # plt.savefig(f'results/xpol_anttemp_simulation2.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times[70:] / 3600, (ants61_temps[70:, :] - ants96_temps[70:, ~flags]) / ants96_temps[70:, ~flags])
    ax.set_title('61 LBA antennas (x polarization)')
    ax.set_xlabel('Time over 12h')
    ax.set_ylabel('Relative difference')
    # plt.savefig(f'results/xpol_anttemp_simulation_reldiff.eps', dpi=300, facecolor='w')
    plt.show()

    ants_valid = ~flags.copy()
    ants_broken = flags.copy()
    ants_broken[31] = False  # The data from the antenna 31 in data_2024 are invalid
    index_valid = np.where(ants_valid)[0]
    index_broken = np.where(ants_broken)[0]
    index_invalid = np.array([31])

    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
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

    stds = np.sqrt(np.mean((ants61_temps[70:, :] - ants96_temps[70:, ~flags]) ** 2, axis=0))

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(arr_xvalid, arr_yvalid, s=50, c=stds, cmap='Reds')
    ax.scatter(arr_xbroken, arr_ybroken, s=50, c='black', marker='x')
    ax.scatter(arr_xinvalid, arr_yinvalid, s=50, c='blue')
    ax.plot([-28, -22], [-28, -22], color='blue')
    ax.text(-30, -30, 'x pol', fontsize=12)
    ax.plot([-28, -22], [-22, -28], color='blue')
    ax.text(-30, -20, 'y pol', fontsize=12)
    for i, name in enumerate(arr_name):
        ax.annotate(name, (arr_x[i], arr_y[i]), fontsize=12, textcoords="offset points", xytext=(0, 5), ha='center')
    ax.set_title('Scatter Plot Example')
    ax.set_xlabel('p Axis')
    ax.set_ylabel('q Axis')
    fig.colorbar(scatter, ax=ax)
    plt.show()

    min_dists = np.array([])
    for i in range(len(index_valid)):
        dists = np.sqrt((arr_xinvalid - arr_xvalid[i]) ** 2 + (arr_yinvalid - arr_yvalid[i]) ** 2)
        min_dists = np.append(min_dists, np.mean((1 / dists) ** 2))
    from scipy.stats import pearsonr, spearmanr
    r, p_value = spearmanr(min_dists, stds)

    return


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

        
def single_antenna():
    save_figure = False

    nside = 256
    lon = 11.917778
    lat = 57.393056
    frq_cntr = 44.92
    xpol = True
    ypol = True
    excite = 'X'
    ground = True

    num_ants = 1

    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0] * 1
    arr_y = arr_pos[:, 1] * 1
    arr_z = np.zeros(len(arr_x))
    arr_pos = np.vstack((arr_x, arr_y, arr_z)).T
    arr_pos = arr_pos[:num_ants, :]

    nr_samples = 96

    EEPs, _, imps = _random_antenna(nr_samples, frq_cntr)
    real_imps = np.real(imps)

    base_fontsize = 18
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    mean_imps = np.mean(real_imps)
    std_imps = np.std(real_imps)
    percent_RMS = std_imps / mean_imps
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axvspan(mean_imps - std_imps, mean_imps + std_imps, color='yellow', alpha=0.3,
               label=f'Percent RMS = {percent_RMS*100:.3g}%')
    ax.hist(real_imps, bins=30, density=False, alpha=0.6, color='b')
    ax.axvline(mean_imps, color='r', linestyle='--', label=rf'Sample Mean (μ = {mean_imps:.3g} $\Omega$)')
    ax.legend()
    # ax.title(f'Distribution of the Impedances (mean={mean_imps:.4f}, relative std={std_imps / mean_imps:.4f})')
    ax.set_xlabel(r'Impedance [$\Omega$]')
    ax.set_ylabel(f'Counts [/{nr_samples}]')
    if save_figure:
        plt.savefig(f'results/imps_distribution.eps', dpi=300)
    plt.show()

    nr_thetas = 46
    nr_phis = 180
    thetas = np.linspace(0, 90, nr_thetas, endpoint=True) * np.pi / 180
    phis = np.linspace(0, 360, nr_phis, endpoint=False) * np.pi / 180
    EEPs = np.abs(EEPs[:, 0, :, :, 0]) ** 2 + np.abs(EEPs[:, 0, :, :, 1]) ** 2
    # EEPs = EEPs + 2.j
    EEPs_healpix = np.zeros((nr_samples, 12 * nside ** 2))
    _, beam0 = _ant_coord_trans(nside, thetas, phis, EEPs[0, :, :].T)
    for i in range(nr_samples):
        _, beam = _ant_coord_trans(nside, thetas, phis, EEPs[i, :, :].T)
        beam /= np.sum(beam)
        # beam /= np.sum(beam0)
        print(i, np.sum(beam))
        # hp.mollview(beam)
        # plt.show()
        EEPs_healpix[i, :] = beam

    times = np.linspace(0, 24 * 3600, 1000, endpoint=False)
    base_time = '2020-12-02 11:58:39.000'
    times = Time(base_time, format='iso', scale='utc') + times * u.second
    location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    times.location = location
    lst = times.sidereal_time('mean').hour  # Transform to sidereal time
    base_time = times.datetime[np.where(lst == np.min(lst))[0]][0]

    timings = 140
    ants_temps = np.zeros((timings, nr_samples))
    for t in range(timings):
        (latitude, longitude, elevation) = (str(lat), str(lon), 0)
        ov = LFSMObserver()
        # ov = GSMObserver()
        ov.lon = longitude
        ov.lat = latitude
        ov.elev = elevation
        minute = (base_time.minute + t * 10) % 60
        hour = (base_time.hour + (base_time.minute + t * 10) // 60) % 24
        day = base_time.day + (base_time.hour + (base_time.minute + t * 10) // 60) // 24
        ov.date = datetime(2020, 12, day, hour, minute, base_time.second)
        # ov.date = datetime(2013, 7, day, hour, minute, 0)
        sky = ov.generate(frq_cntr)
        # ov.view(logged=True, show=True)
        # hp.mollview(sky)
        # plt.show()
        # plt.title(f'{day}:{hour}:{minute}')
        # plt.show()
        sky = hp.pixelfunc.ud_grade(sky, nside)
        print(np.mean(sky), sky[100:110])

        obs = sky[None, :] * EEPs_healpix
        ant_temp = np.sum(obs, axis=1)
        ants_temps[t, :] = ant_temp
        # ants_temps[t, :] = np.mean(sky)
        print(day, hour, minute)
        # mean_temp = np.append(mean_temp, np.mean(sky))
        # mean_temp2 = np.append(mean_temp2, np.mean(sky_match))
        # plt.plot(thetas_ext, eep96[0, :])
        # plt.show()

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    times = np.linspace(0, 24 * 3600, 140, endpoint=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, ants_temps)
    mean = np.mean(ants_temps, axis=1)[:, None]
    # total_mape = np.mean(np.abs(ants_temps - mean) / mean)
    # ax.text(5, 8000, f"Total MAPE = {total_mape*100:.3g}%", fontsize=base_fontsize, color='blue',
    #         bbox=dict(facecolor='white', alpha=0.5))
    std = np.mean(np.std(ants_temps, axis=1) / np.mean(ants_temps, axis=1))
    ax.text(4.5, 8500 + 1500, f"relative std = {std*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 isotropic antennas with errors (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    # plt.savefig(f'results/xpol_anttemp_errors.eps', dpi=300, facecolor='w')
    # plt.savefig(f'results/xpol_anttemp_errors_origin.eps', dpi=300, facecolor='w')
    plt.show()

    ks = np.mean(ants_temps) / np.mean(ants_temps, axis=0)
    ants_temps = ants_temps * ks[None, :]
    times = np.linspace(0, 24 * 3600, 140, endpoint=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    std = np.mean(np.std(ants_temps, axis=1) / np.mean(ants_temps, axis=1))
    ax.plot(times / 3600, ants_temps)
    mean = np.mean(ants_temps, axis=1)[:, None]
    total_mape = np.mean(np.abs(ants_temps - mean) / mean)
    ax.text(5, 8000, f"Total MAPE = {total_mape*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 isotropic antennas with errors (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    # plt.savefig(f'results/xpol_anttemp_simulation.eps', dpi=300, facecolor='w')
    plt.show()
    return ants_temps


def time_test():
    # 示例数据（替换为实际数据）
    times_2020 = Time(['2020-12-02T{:02d}:00:00'.format(i) for i in range(24)], format='isot', scale='utc')
    times_2024 = Time(['2024-09-16T{:02d}:00:00'.format(i) for i in range(24)], format='isot', scale='utc')
    data_2020 = np.arange(24)  # 替换为实际数据
    data_2024 = np.arange(24)  # 替换为实际数据

    # 观测地点
    location = EarthLocation(lon=11.917778 * u.deg, lat=57.393056 * u.deg)
    times_2020.location = location
    times_2024.location = location

    # 转换为本地平恒星时 (LST)
    lst_2020 = times_2020.sidereal_time('mean').hour  # 直接用小时表示
    lst_2024 = times_2024.sidereal_time('mean').hour

    # sorted_indices_2020 = np.argsort(lst_2020)
    # lst_2020 = lst_2020[sorted_indices_2020]
    # data_2020 = data_2020[sorted_indices_2020]
    # sorted_indices_2024 = np.argsort(lst_2024)
    # lst_2024 = lst_2024[sorted_indices_2024]
    # data_2024 = data_2024[sorted_indices_2024]

    # 创建统一的 LST 网格（0 到 24 小时，分为 1000 个点）
    lst_grid = np.linspace(0, 24, 1000)

    # 对数据进行插值到统一网格
    interp_2020 = interp1d(lst_2020, data_2020, kind='linear', fill_value="extrapolate")
    interp_2024 = interp1d(lst_2024, data_2024, kind='linear', fill_value="extrapolate")

    data_interp_2020 = interp_2020(lst_grid)
    data_interp_2024 = interp_2024(lst_grid)

    # 计算差值
    delta_data = data_interp_2020 - data_interp_2024

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(lst_grid, data_interp_2020, label='Interpolated Data 2020', lw=2, color='blue')
    plt.plot(lst_grid, data_interp_2024, label='Interpolated Data 2024', lw=2, color='orange')
    plt.plot(lst_grid, delta_data, label='Difference (2020 - 2024)', lw=2, color='green', linestyle='--')

    plt.xlabel('Local Sidereal Time (hours)')
    plt.ylabel('Data Value')
    plt.title('Comparison of Astronomical Data (Interpolated to LST)')
    plt.legend()
    plt.grid(True)
    plt.show()


def normalization():
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
    data_interp_2020 = interp_2020(lst_grid)  # 96 * 2000
    ks_2020 = np.mean(data_interp_2020) / np.mean(data_interp_2020, axis=1)
    data_interp_2020 = data_interp_2020 * ks_2020[:, None]

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
    mean = np.mean(data_interp_2020, axis=0)[None, :]
    total_mape1 = np.mean(np.abs(data_interp_2020 - mean) / mean)
    ax.text(5, 2.0e7, f"Total MAPE = {total_mape1*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Self power')
    # plt.savefig(f'results/24hautocorr_norm.eps', dpi=300, facecolor='w')
    plt.show()
    # -------------------------------------------------------------------------------------------------------------- #

    # LOFAR: x to east (several degrees difference), y to north (several degrees difference), z to up
    # healpix: x to up, y to east, z to north
    frq = 44.92
    nside = 256  # at least 256 to avoid repetition of pixels
    lon = 11.917778
    lat = 57.393056

    nr_thetas = 46
    nr_phis = 180
    thetas = np.linspace(0, 90, nr_thetas, endpoint=True) * np.pi / 180
    phis = np.linspace(0, 360, nr_phis, endpoint=False) * np.pi / 180

    eep96 = np.load('results/dual_xpol_96_EEP_300.npy')[:, 0, :, :, :]
    eep96 = np.abs(eep96[:, :, :, 0]) ** 2 + np.abs(eep96[:, :, :, 1]) ** 2
    eep96_healpix = np.zeros((96, 12 * nside ** 2))
    _, beam960 = _ant_coord_trans(nside, thetas, phis, eep96[0, :, :].T)
    for i in range(96):
        _, beam = _ant_coord_trans(nside, thetas, phis, eep96[i, :, :].T)
        beam /= np.sum(beam)
        # beam /= np.sum(beam960)
        print(i, np.sum(beam))
        # hp.mollview(beam)
        # plt.show()
        eep96_healpix[i, :] = beam

    times = np.linspace(0, 24*3600, 2000, endpoint=False)
    base_time = '2020-12-02 11:58:39.000'
    times = Time(base_time, format='iso', scale='utc') + times * u.second
    location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    times.location = location
    lst = times.sidereal_time('mean').hour  # Transform to sidereal time
    base_time = times.datetime[np.where(lst == np.min(lst))[0]][0]

    timings = 140
    ants96_temps = np.zeros((timings, 96))
    for t in range(timings):
        (latitude, longitude, elevation) = (str(lat), str(lon), 0)
        ov = GSMObserver()
        ov.lon = longitude
        ov.lat = latitude
        ov.elev = elevation
        minute = (base_time.minute + t * 10) % 60
        hour = (base_time.hour + (base_time.minute + t * 10) // 60) % 24
        day = base_time.day + (base_time.hour + (base_time.minute + t * 10) // 60) // 24
        ov.date = datetime(2020, 12, day, hour, minute, base_time.second)
        # ov.date = datetime(2013, 7, day, hour, minute, 0)
        sky = ov.generate(frq)
        # ov.view(logged=True, show=True)
        # hp.mollview(sky)
        # plt.show()
        # plt.title(f'{day}:{hour}:{minute}')
        # plt.show()
        sky = hp.pixelfunc.ud_grade(sky, nside)
        print(np.mean(sky), sky[100:110])

        obs = sky[None, :] * eep96_healpix
        ant_temp = np.sum(obs, axis=1)
        ants96_temps[t, :] = ant_temp
        # ants96_temps[t, :] = np.mean(sky)
        print(day, hour, minute)
        # mean_temp = np.append(mean_temp, np.mean(sky))
        # mean_temp2 = np.append(mean_temp2, np.mean(sky_match))
        # plt.plot(thetas_ext, eep96[0, :])
        # plt.show()

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    times = np.linspace(0, 24 * 3600, 140, endpoint=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    # ants96_temps = ants96_temps - ()
    mean = np.mean(ants96_temps, axis=1)[:, None]
    total_mape2 = np.mean(np.abs(ants96_temps - mean) / mean)
    ax.plot(times / 3600, ants96_temps)
    ax.text(4.5, 8000, f"Total MAPE = {total_mape2*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 LBA antennas (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    # plt.savefig(f'results/xpol_anttemp_simulation.eps', dpi=300, facecolor='w')
    plt.show()

    # coef = np.mean(data_interp_2020.T, axis=0) / np.mean(ants96_temps, axis=0)
    # obs_temps = data_interp_2020.T / coef[None, :]
    # fig, ax = plt.subplots(figsize=(12, 8))
    # std3 = np.mean(np.std(obs_temps, axis=1) / np.mean(obs_temps, axis=1))
    # ax.plot(lst_grid, obs_temps)
    # ax.set_title('96 LBA antennas (x polarization)')
    # ax.set_xlabel('Time over 24h')
    # ax.set_ylabel('Antenna temperature (K)')
    # # plt.savefig(f'results/xpol_anttemp_simulation.eps', dpi=300, facecolor='w')
    # plt.show()

    print(total_mape1, total_mape2)


def interferometry(rel_std):
    nside = 16
    lon = 11.917778
    lat = 57.393056
    frq_cntr = 44.92

    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0] * 1
    arr_y = arr_pos[:, 1] * 1
    arr_z = np.zeros(len(arr_x))
    arr_pos = np.vstack((arr_x, arr_y, arr_z)).T  # 96 * 3
    arr_pos = arr_pos[:, :]
    # Coordinate Transformation
    new_arr_pos = arr_pos[:, [2, 0, 1]]

    nr_samples = 96
    _, EELs, imps = _random_antenna(nr_samples, frq_cntr, rel_std=rel_std)
    nr_thetas = 46
    nr_phis = 180
    thetas = np.linspace(0, 90, nr_thetas, endpoint=True) * np.pi / 180
    phis = np.linspace(0, 360, nr_phis, endpoint=False) * np.pi / 180
    EELs_healpix = np.zeros((2, nr_samples, 12 * nside ** 2), dtype=complex)
    cross_EELs = np.zeros((nr_samples - 1, 12 * nside ** 2), dtype=complex)
    pix = np.arange(0, 12 * nside ** 2, 1)
    pix2vec = np.array(hp.pix2vec(nside, pix, nest=False))  # 3 * npix
    _lambda = 3.0e2 / frq_cntr
    # _, beam0 = _ant_coord_trans(nside, thetas, phis, EEPs[0, :, :].T)
    norm_index = None
    for i in tqdm(range(nr_samples), desc="Normalizing"):
        _, real_EELs_tht = _ant_coord_trans(nside, thetas, phis, np.real(EELs[i].f_tht[0, :, :]).T)
        _, imag_EELs_tht = _ant_coord_trans(nside, thetas, phis, np.imag(EELs[i].f_tht[0, :, :]).T)
        _, real_EELs_phi = _ant_coord_trans(nside, thetas, phis, np.real(EELs[i].f_phi[0, :, :]).T)
        _, imag_EELs_phi = _ant_coord_trans(nside, thetas, phis, np.imag(EELs[i].f_phi[0, :, :]).T)
        if not i:
            EELs_tht = real_EELs_tht + 1j * imag_EELs_tht
            EELs_phi = real_EELs_phi + 1j * imag_EELs_phi
            norm_index = np.sqrt(np.sum(np.abs(EELs_tht) ** 2 + np.abs(EELs_phi) ** 2))
        real_EELs_tht /= norm_index
        imag_EELs_tht /= norm_index
        real_EELs_phi /= norm_index
        imag_EELs_phi /= norm_index
        # hp.mollview(beam)
        # plt.show()
        EELs_healpix[0, i, :] = real_EELs_tht + 1j * imag_EELs_tht
        EELs_healpix[1, i, :] = real_EELs_phi + 1j * imag_EELs_phi

        if i >= 1:
            vec_baseline = new_arr_pos[i - 1, :] - new_arr_pos[i, :]
            index = np.dot(pix2vec.T, vec_baseline) * 2 * np.pi / _lambda
            cross_EELs[i - 1, :] = np.exp(1j * index) * (EELs_healpix[0, i - 1, :] * EELs_healpix[0, i, :].conj() +
                EELs_healpix[1, i - 1, :] * EELs_healpix[1, i, :].conj())

    times = np.linspace(0, 24 * 3600, 1000, endpoint=False)
    base_time = '2020-12-02 11:58:39.000'
    times = Time(base_time, format='iso', scale='utc') + times * u.second
    location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    times.location = location
    lst = times.sidereal_time('mean').hour  # Transform to sidereal time
    base_time = times.datetime[np.where(lst == np.min(lst))[0]][0]

    timings = 144
    corss_vis = np.zeros((timings, nr_samples - 1), dtype=complex)
    for t in tqdm(range(timings), desc='Observing'):
        (latitude, longitude, elevation) = (str(lat), str(lon), 0)
        # ov = LFSMObserver()
        ov = GSMObserver()
        ov.lon = longitude
        ov.lat = latitude
        ov.elev = elevation
        minute = (base_time.minute + t * 10) % 60
        hour = (base_time.hour + (base_time.minute + t * 10) // 60) % 24
        day = base_time.day + (base_time.hour + (base_time.minute + t * 10) // 60) // 24
        ov.date = datetime(2020, 12, day, hour, minute, base_time.second)
        # ov.date = datetime(2013, 7, day, hour, minute, 0)
        sky = ov.generate(frq_cntr)
        # ov.view(logged=True, show=True)
        # hp.mollview(sky)
        # plt.show()
        # plt.title(f'{day}:{hour}:{minute}')
        # plt.show()
        sky = hp.pixelfunc.ud_grade(sky, nside)

        obs = sky[None, :] * cross_EELs
        ant_temp = np.sum(obs, axis=1)
        corss_vis[t, :] = ant_temp
        # corss_vis[t, :] = np.mean(sky)
        # mean_temp = np.append(mean_temp, np.mean(sky))
        # mean_temp2 = np.append(mean_temp2, np.mean(sky_match))
        # plt.plot(thetas_ext, eep96[0, :])
        # plt.show()
    abs_cross_vis = np.abs(corss_vis)
    angle_cross_vis = np.angle(corss_vis)

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    times = np.linspace(0, 24 * 3600, timings, endpoint=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, abs_cross_vis)
    if rel_std == 0.0:
        ax.set_title('no geometric diff and no mutual coupling (x polarization)')
    elif rel_std == 0.01:
        ax.set_title('geometric diff but no mutual coupling (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('visibility')
    # plt.savefig(f'results/cross_no_geometric_diff_and_no_mutual_coupling_temps.eps', dpi=300, facecolor='w')
    # plt.savefig(f'results/cross_geometric_diff_but_no_mutual_coupling_temps.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, angle_cross_vis)
    if rel_std == 0.0:
        ax.set_title('no geometric diff and no mutual coupling (x polarization)')
    elif rel_std == 0.01:
        ax.set_title('geometric diff but no mutual coupling (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('angle')
    # plt.savefig(f'results/cross_no_geometric_diff_and_no_mutual_coupling_angles.eps', dpi=300, facecolor='w')
    plt.show()
    return abs_cross_vis


def vis_simu():
    # LOFAR: x to east (several degrees difference), y to north (several degrees difference), z to up
    # healpix: x to up, y to east, z to north
    frq_cntr = 44.92
    nside = 16  # at least 256 to avoid repetition of pixels
    lon = 11.917778
    lat = 57.393056
    nr_ants = 96

    nr_thetas = 46
    nr_phis = 180
    thetas = np.linspace(0, 90, nr_thetas, endpoint=True) * np.pi / 180
    phis = np.linspace(0, 360, nr_phis, endpoint=False) * np.pi / 180

    arr_origin = np.loadtxt(root_path + 'Pos_LBA_SE607_local.txt', dtype=str)
    arr_pos = arr_origin[:, 1:3].astype(float)
    arr_x = arr_pos[:, 0] * 1
    arr_y = arr_pos[:, 1] * 1
    arr_z = np.zeros(len(arr_x))
    arr_pos = np.vstack((arr_x, arr_y, arr_z)).T  # 96 * 3
    arr_pos = arr_pos[:, :]
    # Coordinate Transformation
    new_arr_pos = arr_pos[:, [2, 0, 1]]

    eep96 = np.load('results/dual_xpol_96_EEP_300.npy')[:, 0, :, :, :]  # 96*46*180*2
    # eep96 = np.abs(eep96[:, :, :, 0]) ** 2 + np.abs(eep96[:, :, :, 1]) ** 2
    eep96_healpix = np.zeros((nr_ants, 12 * nside ** 2, 2))

    EEPs_healpix = np.zeros((2, nr_ants, 12 * nside ** 2), dtype=complex)
    cross_EEPs = np.zeros((nr_ants - 1, 12 * nside ** 2), dtype=complex)
    pix = np.arange(0, 12 * nside ** 2, 1)
    pix2vec = np.array(hp.pix2vec(nside, pix, nest=False))  # 3 * npix
    _lambda = 3.0e2 / frq_cntr
    norm_index = None
    # _, beam0 = _ant_coord_trans(nside, thetas, phis, EEPs[0, :, :].T)
    for i in tqdm(range(nr_ants), desc="Normalizing"):
        _, real_EEPs_tht = _ant_coord_trans(nside, thetas, phis, np.real(eep96[i, :, :, 0]).T)
        _, imag_EEPs_tht = _ant_coord_trans(nside, thetas, phis, np.imag(eep96[i, :, :, 0]).T)
        _, real_EEPs_phi = _ant_coord_trans(nside, thetas, phis, np.real(eep96[i, :, :, 1]).T)
        _, imag_EEPs_phi = _ant_coord_trans(nside, thetas, phis, np.imag(eep96[i, :, :, 1]).T)
        if not i:
            EEPs_tht = real_EEPs_tht + 1j * imag_EEPs_tht
            EEPs_phi = real_EEPs_phi + 1j * imag_EEPs_phi
            norm_index = np.sqrt(np.sum(np.abs(EEPs_tht) ** 2 + np.abs(EEPs_phi) ** 2))
        real_EEPs_tht /= norm_index
        imag_EEPs_tht /= norm_index
        real_EEPs_phi /= norm_index
        imag_EEPs_phi /= norm_index
        # hp.mollview(beam)
        # plt.show()
        EEPs_healpix[0, i, :] = real_EEPs_tht + 1j * imag_EEPs_tht
        EEPs_healpix[1, i, :] = real_EEPs_phi + 1j * imag_EEPs_phi

        if i >= 1:
            vec_baseline = new_arr_pos[i - 0, :] - new_arr_pos[i, :]
            index = np.dot(pix2vec.T, vec_baseline) * 2 * np.pi / _lambda
            cross_EEPs[i - 1, :] = np.exp(1j * index) * (EEPs_healpix[0, i - 1, :] * EEPs_healpix[0, i, :].conj() +
                EEPs_healpix[1, i - 1, :] * EEPs_healpix[1, i, :].conj())

    times = np.linspace(0, 24 * 3600, 1000, endpoint=False)
    base_time = '2020-12-02 11:58:39.000'
    times = Time(base_time, format='iso', scale='utc') + times * u.second
    location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    times.location = location
    lst = times.sidereal_time('mean').hour  # Transform to sidereal time
    base_time = times.datetime[np.where(lst == np.min(lst))[0]][0]

    timings = 144
    corss_vis = np.zeros((timings, nr_ants - 1), dtype=complex)
    for t in tqdm(range(timings), desc='Observing'):
        (latitude, longitude, elevation) = (str(lat), str(lon), 0)
        # ov = LFSMObserver()
        ov = GSMObserver()
        ov.lon = longitude
        ov.lat = latitude
        ov.elev = elevation
        minute = (base_time.minute + t * 10) % 60
        hour = (base_time.hour + (base_time.minute + t * 10) // 60) % 24
        day = base_time.day + (base_time.hour + (base_time.minute + t * 10) // 60) // 24
        ov.date = datetime(2020, 12, day, hour, minute, base_time.second)
        sky = ov.generate(frq_cntr)
        # ov.view(logged=True, show=True)
        # hp.mollview(sky)
        # plt.show()
        # plt.title(f'{day}:{hour}:{minute}')
        # plt.show()
        sky = hp.pixelfunc.ud_grade(sky, nside)

        obs = sky[None, :] * cross_EEPs
        ant_temp = np.sum(obs, axis=1)
        corss_vis[t, :] = ant_temp
        # corss_vis[t, :] = np.mean(sky)
        # mean_temp = np.append(mean_temp, np.mean(sky))
        # mean_temp2 = np.append(mean_temp2, np.mean(sky_match))
        # plt.plot(thetas_ext, eep96[0, :])
        # plt.show()
    abs_cross_vis = np.abs(corss_vis)
    angle_cross_vis = np.angle(corss_vis)

    base_fontsize = 26
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    times = np.linspace(0, 24 * 3600, timings, endpoint=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(times / 3600, abs_cross_vis)
    ax.set_title('no geometric diff but mutual_coupling_temps (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('visibility')
    plt.savefig(f'results/cross_no_geometric_diff_but_mutual_coupling_temps.eps', dpi=300, facecolor='w')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    std = np.mean(np.std(angle_cross_vis, axis=1) / np.mean(angle_cross_vis, axis=1))
    ax.plot(times / 3600, angle_cross_vis)
    mean = np.mean(angle_cross_vis, axis=1)[:, None]
    total_mape = np.mean(np.abs(angle_cross_vis - mean) / mean)
    ax.text(5, 8000, f"Total MAPE = {total_mape*100:.3g}%", fontsize=base_fontsize, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('96 isotropic antennas with errors (x polarization)')
    ax.set_xlabel('Time over 24h')
    ax.set_ylabel('Antenna temperature (K)')
    # plt.savefig(f'results/xpol_anttemp_simulation.eps', dpi=300, facecolor='w')
    plt.show()
    return abs_cross_vis


def vis_data():
    num_grids = 144

    ds = np.load(root_path + 'SE607_20201202_115700_spw3_sb230_dur512_xst.npz')
    arrs = []
    for f in range(len(ds['delta_secs'])):
        arrs.append(ds['arr_' + str(f)])
    arrs = np.asarray(arrs)
    start_time = ds['start_datetime']
    times = np.linspace(0, 24 * 3600, 166, endpoint=False)

    off_diag = np.zeros((166, 95))
    for ant in range(95):
        off_diag[:, ant] = np.abs(arrs[:, 0, 0, ant, ant + 1])

    # num_ants = np.sum(~either_ants_flags)
    # data_2020 = data_2020[~either_ants_flags, :]
    # data_2024 = data_2024[~either_ants_flags, :]
    times = Time(start_time, scale='utc') + times * u.second

    location = EarthLocation(lon=11.917778 * u.deg, lat=57.393056 * u.deg)
    times.location = location

    lst = times.sidereal_time('mean').hour  # Transform to sidereal time

    lst_grid = np.linspace(0, 24, num_grids)
    interp = interp1d(lst, off_diag.T, kind='linear', fill_value="extrapolate")
    off_diag = interp(lst_grid).T

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(lst_grid, off_diag)
    plt.show()

    # times = np.linspace(0, 24 * 3600, 1000, endpoint=False)
    # # base_time = '2020-12-02 11:58:39.000'
    # times = Time(start_time, format='iso', scale='utc') + times * u.second
    # location = EarthLocation(lon=lon * u.deg, lat=lat * u.deg)
    # times.location = location
    # lst = times.sidereal_time('mean').hour  # Transform to sidereal time
    # start_time = times.datetime[np.where(lst == np.min(lst))[0]][0]
    return off_diag


def comp_vis():
    ref = interferometry(rel_std=0.0)
    tol = interferometry(rel_std=0.01)
    mutual = vis_simu()
    data = vis_data()
    np.savez("results/vis.npz", arr1=ref, arr2=tol, arr3=mutual, arr4=data)

    ds = np.load("results/vis.npz")

    ref = ds['arr1']
    tol = ds['arr2']
    mutual = ds['arr3']
    data = ds['arr4']

    tol *= np.mean(ref) / np.mean(tol)
    mutual *= np.mean(ref) / np.mean(mutual)
    data *= np.mean(ref) / np.mean(data)

    times = np.linspace(0, 24 * 3600, 144, endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    mean = np.mean(np.std(tol - ref, axis=0))
    ax.text(20000, -100, f"mean std = {mean:.3g}", color='blue', bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('tol - ref')
    ax.plot(times, tol - ref)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    mean = np.mean(np.std(mutual - ref, axis=0))
    ax.text(20000, -2000, f"mean std = {mean:.3g}", color='blue', bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('mutual - ref')
    ax.plot(times, mutual - ref)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 8))
    mean = np.mean(np.std(data - ref, axis=0))
    ax.text(20000, -1000, f"mean std = {mean:.3g}", color='blue', bbox=dict(facecolor='white', alpha=0.5))
    ax.set_title('data - ref')
    ax.plot(times, data - ref)
    plt.show()


if __name__ == '__main__':
    st = time()
    # arr_layout()
    # simulate_lofar(frq_cntr=44.92, xpol=True, ypol=True, excite='X', ground=True, special='broken')
    # imp_ants()
    # power_antenna()
    # power_time()
    # comp_power()
    # statistical_analysis()
    # directivity_phis()
    # lofar_directivity_phis()
    # directivity_frqs()
    # eels_phis()
    # eels_ants()
    # spherical_harmonics()
    # simulate_EEPs()
    power_simulation()
    # aaa = interferometry()
    # bbb = single_antenna()
    # time_test()
    # normalization()
    # vis_simu()
    # vis_data()
    # comp_vis()

    et = time()
    print("time: %.2f seconds" % (et - st))
