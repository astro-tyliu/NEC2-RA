import numpy as np

from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm, Normalize
from matplotlib.cm import ScalarMappable

from scipy.ndimage import median_filter

np.set_printoptions(precision=4, linewidth=80)


RAW_DIR = '../raw_data/'
DATA_PATH = '../general_materials/'
DATA_PATH_PAPER = '../paper_materials/'


def _mask_out_of_band(raw_data, start_idx, end_idx):
    """
    Mask out data outside the specified frequency channel range.

    Parameters:
    -----------
    raw_data : numpy.ndarray
        2D array of raw visibility data (Time x Frequency).
    start_idx : int
        The starting channel index of the valid band (inclusive).
    end_idx : int
        The ending channel index of the valid band (exclusive).

    Returns:
    --------
    numpy.ma.MaskedArray
        Data with out-of-band channels masked.
    """
    # Initialize a masked array with no mask initially
    masked_data = np.ma.masked_array(raw_data, mask=False)

    # Mask channels before the start_idx
    if start_idx > 0:
        masked_data[:, :start_idx] = np.ma.masked

    # Mask channels after the end_idx
    # We use end_idx because we want to mask the end_idx channel
    if end_idx < raw_data.shape[1]:
        masked_data[:, end_idx:] = np.ma.masked

    return masked_data


def _remove_rfi_time_axis(masked_data, time_window_bins=5, threshold_sigma=7.0):
    """
    Robust RFI removal using a sliding median filter and Median Absolute Deviation (MAD).
    Processes data channel by channel to account for bandpass shape.

Parameters:
    -----------
    masked_data : numpy.ma.MaskedArray
        2D array of visibility data (Time x Frequency) with initial band masks applied.
    time_window_bins : int, optional
        Number of time bins used for the sliding median filter. Represents the temporal scale
        of the background model. Default is 5.
    threshold_sigma : float, optional
        The clipping threshold based on equivalent Gaussian standard deviations (derived from MAD).
        Default is 7.0.

    Returns:
    --------
    numpy.ma.MaskedArray
        The visibility data with newly generated masks for wideband burst RFI (e.g., lightning).
    """
    # Copy data to avoid modifying the original array
    cleaned_data = masked_data.copy()
    n_times, n_freqs = cleaned_data.shape

    full_background = np.zeros_like(cleaned_data.data)
    full_residual = np.zeros_like(cleaned_data.data)

    for f in range(n_freqs):
        # Skip if the entire frequency channel is already masked (e.g., out of band)
        if cleaned_data[:, f].mask.all():
            continue

        channel_data = cleaned_data[:, f].data

        # 1. Background modeling: Sliding median filter along the time axis
        # This isolates the slow-varying sky and instrument background
        background = median_filter(channel_data, size=time_window_bins)
        full_background[:, f] = background

        # 2. Flattening: Subtract the background to get zero-mean residuals
        residual = channel_data - background
        full_residual[:, f] = residual

        # 3. Robust noise estimation: 使用一阶差分法 (First-order Difference)
        # 相邻点相减可以彻底抵消缓慢的基线起伏，提取出纯粹的热噪声
        diffs = np.diff(channel_data)

        median_diff = np.median(diffs)
        mad_diff = np.median(np.abs(diffs - median_diff))

        # 这种情况下极难出现 mad_diff == 0，除非数据全是毫无噪声的数字阶梯
        if mad_diff == 0:
            print(f"mad_diff is zero in the channel {f}")
            cleaned_data.mask[:, f] = True
            continue

        # 4. Define threshold: 将差分的 MAD 转换为原始数据的等效标准差
        # 物理规律：由于差分是两个独立噪声变量相减，其方差会扩大 2 倍 (σ_diff^2 = 2 * σ^2)
        # 因此在还原为原始数据的标准差时，必须除以根号 2 (np.sqrt(2))
        sigma_equiv = (1.4826 * mad_diff) / np.sqrt(2)
        threshold = threshold_sigma * sigma_equiv

        # 5. Flag RFI: Mark pixels exceeding the robust threshold
        rfi_mask = np.abs(residual) > threshold
        cleaned_data.mask[:, f] |= rfi_mask

    return cleaned_data, full_background, full_residual


def _remove_rfi_freq_axis(masked_data, freq_window_bins=15, threshold_sigma=7.0):
    """
    Removes narrowband continuous RFI by flattening the bandpass and scanning the frequency axis.
    Must be executed AFTER the time-axis broadband RFI mitigation to avoid cross-contamination.

    Parameters:
    -----------
    masked_data : numpy.ma.MaskedArray
        2D array of visibility data that has ALREADY been cleaned of wideband bursts
        (i.e., the output from the time-axis mitigation step).
    freq_window_bins : int, optional
        Number of frequency channels used for the sliding median filter to account for
        residual bandpass ripples. Default is 15.
    threshold_sigma : float, optional
        The clipping threshold based on equivalent Gaussian standard deviations. Default is 7.0.

    Returns:
    --------
    numpy.ma.MaskedArray
        The visibility data with additional masks covering narrowband continuous wave RFI.
    """
    cleaned_data = masked_data.copy()
    n_times, n_freqs = cleaned_data.shape

    full_interpolated_spectrum = np.zeros_like(cleaned_data.data)
    full_local_background = np.zeros_like(cleaned_data.data)

    # 1. Isolate the static instrument response and sky spectrum (The Fingerprint)
    # Using np.ma.median ensures previously flagged broadband RFI does not bias the bandpass model
    bandpass_model = np.ma.median(cleaned_data, axis=0)

    for t in range(n_times):
        if cleaned_data[t, :].mask.all():
            continue

        time_slice = cleaned_data[t, :].data
        slice_mask = cleaned_data[t, :].mask

        # 2. Flatten the frequency cliff to create a zero-mean baseline
        flattened_spectrum = time_slice - bandpass_model.data

        # 3. Temporarily patch the "pits" (flagged broadband RFI) before spatial filtering
        # This prevents scipy.ndimage from leaking RFI energy into the local background model
        valid_idx = ~slice_mask
        if np.sum(valid_idx) < 2:
            continue

        freq_indices = np.arange(n_freqs)
        interpolated_spectrum = np.interp(
            freq_indices,
            freq_indices[valid_idx],
            flattened_spectrum[valid_idx]
        )

        full_interpolated_spectrum[t, :] = interpolated_spectrum

        # 4. Extract local baseline to account for any residual bandpass ripples
        local_background = median_filter(interpolated_spectrum, size=freq_window_bins, mode='reflect')
        residual = interpolated_spectrum - local_background

        full_local_background[t, :] = local_background

        # 5. Robust noise estimation strictly on physically valid pixels
        median_res = np.median(residual[valid_idx])
        mad = np.median(np.abs(residual[valid_idx] - median_res))

        if mad == 0:
            continue

        # 6. Isolate and flag narrowband spikes
        sigma_equiv = 1.4826 * mad
        threshold = threshold_sigma * sigma_equiv
        rfi_mask = np.abs(residual) > threshold

        cleaned_data.mask[t, :] |= rfi_mask

    return cleaned_data, full_interpolated_spectrum, full_local_background


# def _plot_waterfall(
#         data, x_axis, y_axis, save_figure=None, show_figure=None, title=None, file_dir='../results/waterfalls/',
#         filename=None
# ):
#     """
#     Plots and optionally saves a waterfall visualization of the provided 2D array.
#     """
#     # Hardcoded font and plot configurations
#     base_fontsize = 30
#     config = {
#         "font.family": 'Times New Roman',
#         "font.size": base_fontsize,
#         "mathtext.fontset": 'stix',
#     }
#     rcParams.update(config)
#
#     # Calculate image extent based on axis arrays
#     plot_extent = [x_axis.min(), x_axis.max(), y_axis.max(), y_axis.min()]
#
#     fig, ax = plt.subplots(figsize=(14, 8))
#
#     # Plotting data in log10 scale
#     im = ax.imshow(np.log10(data), aspect='auto', extent=plot_extent, cmap='viridis')
#
#     # Colorbar configuration
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(r'$\log_{10}(\mathrm{Intensity})$')
#
#     # Hardcoded ticks and labels
#     ax.set_xticks([40, 50, 60, 70])
#     ax.set_yticks([0, 4, 8, 12, 16, 20])
#     ax.set_xlabel('Frequency (MHz)')
#     ax.set_ylabel('Time (Hours)')
#
#     # Optional Title
#     if title is not None:
#         ax.set_title(title)
#
#     fig.tight_layout()
#
#     # Optional Save
#     if save_figure and filename is not None:
#         fig.savefig(file_dir + filename, bbox_inches='tight', dpi=300)
#
#     # Optional Show
#     if show_figure:
#         plt.show()
#
#     # Free memory
#     plt.close(fig)


def _plot_waterfall(data, x_axis, y_axis, use_symlog=False, linthresh=10000000.0, save_figure=None, show_figure=None,
                   title=None, file_dir='../results/waterfalls/', filename=None):
    """
    Plots and optionally saves a waterfall visualization.
    Supports standard Log10 scale (for raw data) and SymLog scale (for residual/flattened data containing negative values).

    Parameters:
    data, x_axis, y_axis: Required data arrays.
    use_symlog (bool): If True, uses SymLogNorm. If False, uses standard LogNorm.
    linthresh (float): The range within which the plot is linear (only used if use_symlog=True).
                       Set this roughly to the standard deviation (1-sigma) of your background noise.
    """
    base_fontsize = 30
    config = {
        "font.family": 'Times New Roman',
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    plot_extent = [x_axis.min(), x_axis.max(), y_axis.max(), y_axis.min()]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Apply appropriate normalization based on whether data contains negative values
    if use_symlog:
        # SymLogNorm handles positive and negative values, linear near zero
        norm = SymLogNorm(linthresh=linthresh, base=10, vmin=np.ma.min(data), vmax=np.ma.max(data))
        cbar_label = r'$\mathrm{SymLog}_{10}(\mathrm{Intensity})$'
        cmap = 'seismic'
    else:
        # Standard LogNorm for strictly positive data (like Raw Visibility)
        # Avoids manual np.log10() to keep colorbar ticks in original data scale
        # Use a small vmin to avoid log(<=0) errors if standard data has minor artifacts
        valid_min = np.ma.min(data[data > 0]) if np.any(data > 0) else 1e-5
        norm = LogNorm(vmin=valid_min, vmax=np.ma.max(data))
        cbar_label = r'$\log_{10}(\mathrm{Intensity})$'
        cmap = 'viridis'

    # Plot using the configured norm instead of manual np.log10()
    im = ax.imshow(data, aspect='auto', extent=plot_extent, cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks([40, 50, 60, 70])
    ax.set_yticks([0, 4, 8, 12, 16, 20])
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Time (Hours)')

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    if save_figure and filename is not None:
        fig.savefig(filename, bbox_inches='tight', dpi=300)

    if show_figure:
        plt.show()

    plt.close(fig)


def _plot_spectra(freqs, data, show_figure=False, filename=None, title=None, alpha=0.3):
    """
    Plots multiple 1D frequency spectra on a single plot.
    Lines are color-coded based on their time index to visualize temporal evolution.

    Parameters:
    freqs (numpy.ndarray): 1D array of frequencies (length: 256).
    data (numpy.ndarray): 2D array of intensity data (Time x Frequency, e.g., 144 x 256).
    save_figure (bool): Whether to save the plot to a file.
    show_figure (bool): Whether to display the plot.
    title (str): Optional title for the plot.
    filename (str): Optional file path to save the plot.
    """
    # 1. Apply formatting configurations
    base_fontsize = 30
    config = {
        "font.family": 'Times New Roman',
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    # 2. Setup figure
    fig, ax = plt.subplots(figsize=(14, 8))

    n_times = data.shape[0]

    # 3. Setup colormap to differentiate time slices
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n_times))

    # 4. Plot each time slice as a separate line
    # alpha=0.3 makes overlapping regions darker and dense, showing the distribution
    for t in range(n_times):
        ax.plot(freqs, data[t, :], color=colors[t], alpha=alpha, linewidth=1.5)

    # 5. Add colorbar to represent the Time Index
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=n_times - 1))
    sm.set_array([])  # Required for ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Time Index')

    # 6. Set labels and axes limits
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Intensity')
    ax.set_xlim(freqs.min(), freqs.max())

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    # 7. Save and show
    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=300)

    if show_figure:
        plt.show()

    # Free memory
    plt.close(fig)


def _plot_time_series(times, data, use_log=False, show_figure=False, filename=None, title=None, alpha=0.3):
    """
    Plots multiple 1D time series on a single plot.
    Lines are color-coded based on their frequency index to visualize spectral differences over time.

    Parameters:
    times (numpy.ndarray): 1D array of time bins (length: 144).
    data (numpy.ndarray): 2D array of intensity data (Time x Frequency, e.g., 144 x 256).
    use_log (bool): If True, sets the Y-axis to logarithmic scale. Default is False.
    show_figure (bool): Whether to display the plot.
    filename (str): Optional file path to save the plot. Acts as the save flag.
    title (str): Optional title for the plot.
    """
    # 1. Apply formatting configurations
    base_fontsize = 30
    config = {
        "font.family": 'Times New Roman',
        "font.size": base_fontsize,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)

    # 2. Setup figure
    fig, ax = plt.subplots(figsize=(14, 8))

    n_times, n_freqs = data.shape

    # 3. Setup colormap to differentiate frequency slices
    # Using 'plasma' (purple-orange-yellow) to contrast with 'viridis' used in spectra plots
    cmap = plt.get_cmap('plasma')  # plasma
    colors = cmap(np.linspace(0, 1, n_freqs))

    # 4. Plot each frequency slice as a separate line
    # data[:, f] extracts the time series for the f-th frequency channel
    for f in range(n_freqs):
        ax.plot(times, data[:, f], color=colors[f], alpha=alpha, linewidth=1.5)

    # 5. Add colorbar to represent the Frequency Index
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=n_freqs - 1))
    sm.set_array([])  # Required for ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Frequency Index')

    # 6. Set labels and axes limits
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Intensity')
    ax.set_xlim(times.min(), times.max())

    if use_log:
        ax.set_yscale('log')

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    # 7. Save and show (Using the 'Single Source of Truth' paradigm)
    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=300)

    if show_figure:
        plt.show()

    # Free memory
    plt.close(fig)


def rfi_flagging(polar='X'):
    freqs_cut = (30.0, 80.0)

    raw_data = np.load(RAW_DIR + 'SE607_20201202_115839_spw3_int600_dur86147_sst.npy')
    if polar == 'X':
        raw_data = raw_data[0::2, :, :]
    elif polar == 'Y':
        raw_data = raw_data[1::2, :, :]
    else:
        raise ValueError('Invalid param: "polar" must be "X" or "Y".')
    # print('The shape of raw_data:', raw_data.shape)

    ds = np.load(RAW_DIR + 'SE607_20240916_180834_spw3_int519_dur86400_sst.npz')
    freqs_mhz = ds['frequencies'] / 1e6

    times_2020 = np.linspace(0, 24, np.shape(raw_data)[1], endpoint=False)

    valid_indices = np.where((freqs_mhz >= freqs_cut[0]) & (freqs_mhz <= freqs_cut[1]))[0]
    start_idx = valid_indices[0]  # Included
    end_idx = valid_indices[-1] + 1  # Excluded
    # plot_freqs = freqs_mhz[start_idx:end_idx]

    for antenna_idx in range(1):
        raw_data_ant = raw_data[antenna_idx, :, :]

        mask_out_of_data = _mask_out_of_band(raw_data_ant, start_idx, end_idx)
        print(np.std(mask_out_of_data.compressed()))
        for s in range(5, 0, -1):
            time_axis_clean_data, time_axis_background, time_axis_residual = _remove_rfi_time_axis(
                mask_out_of_data, time_window_bins=5, threshold_sigma=s
            )
            masked_residual = np.ma.masked_array(time_axis_residual, mask=time_axis_clean_data.mask)
            # print(np.sum(time_axis_clean_data.mask[:, start_idx:end_idx]) / (len(times_2020) * (end_idx - start_idx)), np.std(time_axis_clean_data.compressed()))
            print(np.sum(time_axis_clean_data.mask[:, start_idx:end_idx]) / (
                        len(times_2020) * (end_idx - start_idx)), np.std(masked_residual.compressed()))
        all_axis_clean_data, all_axis_interp_spectrum, all_axis_background = _remove_rfi_freq_axis(
            time_axis_clean_data, freq_window_bins=5, threshold_sigma=15.0
        )

        # _plot_time_series(times_2020, raw_data_ant[:, start_idx:end_idx], show_figure=True)
        # _plot_time_series(times_2020, time_axis_background[:, start_idx:end_idx], show_figure=True)
        # _plot_time_series(times_2020, time_axis_clean_data[:, start_idx:end_idx], show_figure=True)

        # _plot_spectra(
        #     freqs_mhz[start_idx:end_idx], mask_out_of_data[:, start_idx:end_idx], show_figure=True, filename=None,
        #     title=None, alpha=0.3
        # )
        # _plot_spectra(
        #     freqs_mhz[start_idx:end_idx], time_axis_background[:, start_idx:end_idx], show_figure=True, filename=None,
        #     title=None, alpha=0.3
        # )
        _plot_spectra(
            freqs_mhz[start_idx:end_idx], time_axis_clean_data[:, start_idx:end_idx], show_figure=True, filename=None,
            title=None, alpha=0.3
        )

        # _plot_waterfall(mask_out_of_data[:, start_idx:end_idx], freqs_mhz[start_idx:end_idx],
        #                 times_2020, show_figure=True)
        # _plot_waterfall(time_axis_background[:, start_idx:end_idx], freqs_mhz[start_idx:end_idx],
        #                 times_2020, show_figure=True)
        _plot_waterfall(time_axis_clean_data[:, start_idx:end_idx], freqs_mhz[start_idx:end_idx],
                        times_2020, show_figure=True)

        # plot_raw_data = raw_data[antenna_idx, :, start_idx:end_idx]
        # print('The shape of plot_raw_data:', plot_raw_data.shape)

        # mask_out_of_data = _mask_out_of_band(raw_data[antenna_idx, :, :], freqs_mhz)
        # time_axis_clean_data, time_axis_background, time_axis_residual = _remove_rfi_time_axis(
        #     mask_out_of_data, time_window_bins=15, threshold_sigma=15.0
        # )
        # all_axis_clean_data, all_axis_interp_spectrum, all_axis_background = _remove_rfi_freq_axis(
        #     time_axis_clean_data, freq_window_bins=5, threshold_sigma=15.0
        # )

        # plot_data = all_axis_clean_data[:, start_idx:end_idx]

        # print('The shape of plot_all_axis_clean:', plot_all_axis_clean.shape)

        # _plot_spectra(freqs_mhz[start_idx:end_idx], raw_data_ant[:, start_idx:end_idx], show_figure=True)

        #
        # # _plot_spectra(freqs_mhz[start_idx:end_idx], mask_out_of_data[:, start_idx:end_idx], show_figure=True)
        #
        # # _plot_time_series(times_2020, time_axis_background[:, start_idx:end_idx], show_figure=True)
        #
        # # _plot_spectra(plot_freqs, time_axis_residual[:, start_idx:end_idx], show_figure=True)
        #
        # # _plot_time_series(times_2020, time_axis_residual[:, start_idx:end_idx], show_figure=True)
        #
        # # _plot_spectra(plot_freqs, plot_time_axis_clean, show_figure=True)
        #
        # _plot_time_series(times_2020, time_axis_clean_data[:, start_idx+1:end_idx:10], show_figure=True)

        # step = 10
        # for j in range(step):
        #     _plot_time_series(times_2020, raw_data_ant[:, start_idx+j:end_idx:10], show_figure=False,
        #                       filename=f'../results/raw_data_vis_time/raw_x_antenna{antenna_idx:02}_group{j}.png',
        #                       title=f'raw data, channel group {j}, antenna {antenna_idx:02}, X pol', alpha=1.0
        #                       )
        #     _plot_time_series(times_2020, time_axis_clean_data[:, start_idx+j:end_idx:step], show_figure=False,
        #                       filename=f'../results/step1_flagging_vis_time/step1_x_antenna{antenna_idx:02}_group{j}.png',
        #                       title=f'Step 1, channel group {j}, antenna {antenna_idx:02}, X pol', alpha=1.0
        #                       )

        # _plot_spectra(plot_freqs, plot_all_axis_clean, show_figure=True)
        #
        # _plot_time_series(times_2020, plot_all_axis_clean, show_figure=True)

        # ---------------------------------------------------------------

        # _plot_waterfall(
        #     plot_raw_data, plot_freqs, times_2020, show_figure=True,
        #     title=f'Raw Data - Antenna {antenna_idx}, X Pol'
        # )
        #
        # # _plot_waterfall(
        # #     plot_time_axis_background, plot_freqs, times_2020, show_figure=True,
        # #     title=f'Background along Time Axis - Antenna {antenna_idx}, X Pol'
        # # )
        #
        # _plot_waterfall(
        #     plot_time_axis_clean, plot_freqs, times_2020, show_figure=True,
        #     title=f'Cleaned Data along Time Axis - Antenna {antenna_idx}, X Pol'
        # )
        #
        # # _plot_waterfall(
        # #     plot_all_axis_interp_spec, plot_freqs, times_2020, show_figure=True, use_symlog=True,
        # #     title=f'Interpolated Spectrum - Antenna {antenna_idx}, X Pol'
        # # )
        # #
        # # _plot_waterfall(
        # #     plot_all_axis_background, plot_freqs, times_2020, show_figure=True, use_symlog=True,
        # #     title=f'Background - Antenna {antenna_idx}, X Pol'
        # # )
        #
        # _plot_waterfall(
        #     plot_all_axis_clean, plot_freqs, times_2020, show_figure=True,
        #     title=f'Final Cleaned Data - Antenna {antenna_idx}, X Pol'
        # )


if __name__ == '__main__':
    rfi_flagging()
