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


def mask_out_of_band(raw_data, start_idx, end_idx):
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


def flag_bad_channels_mad_of_mads(masked_data, ch_low, ch_high, freq_window_bins=15, threshold_sigma_freq=5.0):
    """
    Cross-frequency validation (MAD-of-MADs) applied only to the valid sub-band [ch_low, ch_high).
    Preserves the original full shape of the data and outputs.

    Parameters:
    -----------
    masked_data : numpy.ma.MaskedArray
        2D array of visibility data (Time x Frequency) in its full shape.
    ch_low : int
        The starting index (inclusive) of the valid frequency band.
    ch_high : int
        The ending index (exclusive) of the valid frequency band.
    freq_window_bins : int, optional
        Number of frequency channels used for the sliding median filter. Default is 15.
    threshold_sigma_freq : float, optional
        The clipping threshold for the frequency channel MADs. Default is 5.0.

    Returns:
    --------
    cleaned_data : numpy.ma.MaskedArray
        Full shape data with heavily contaminated channels newly masked in the valid band.
    full_mad_array : numpy.ndarray
        1D array (full n_freqs length). NaNs outside [ch_low, ch_high).
    full_smooth_base : numpy.ndarray
        1D array (full n_freqs length). NaNs outside [ch_low, ch_high).
    """
    # Copy to avoid modifying the input array directly
    cleaned_data = masked_data.copy()
    n_times, n_freqs = cleaned_data.shape

    # Initialize full-shape arrays with NaNs
    full_mad_array = np.full(n_freqs, np.nan)
    full_smooth_base = np.full(n_freqs, np.nan)

    # ---------------------------------------------------------
    # 1. Isolate the active sub-band for calculation
    # ---------------------------------------------------------
    num_active_channels = ch_high - ch_low
    sub_mad_array = np.full(num_active_channels, np.nan)

    for i, f in enumerate(range(ch_low, ch_high)):
        # Skip if the channel is already entirely masked
        if cleaned_data[:, f].mask.all():
            continue

        channel_data = cleaned_data[:, f].data
        diffs = np.diff(channel_data)

        median_diff = np.median(diffs)
        mad_diff = np.median(np.abs(diffs - median_diff))
        sub_mad_array[i] = mad_diff

    # ---------------------------------------------------------
    # 2. Handle missing channels within the sub-band
    # ---------------------------------------------------------
    valid_idx = ~np.isnan(sub_mad_array)
    if not np.any(valid_idx):
        print("Warning: All channels in the sub-band are fully masked.")
        return cleaned_data, full_mad_array, full_smooth_base

    interp_mad_diff = np.copy(sub_mad_array)
    interp_mad_diff[~valid_idx] = np.interp(
        np.flatnonzero(~valid_idx),
        np.flatnonzero(valid_idx),
        sub_mad_array[valid_idx]
    )

    # ---------------------------------------------------------
    # 3. Frequency-domain baseline modeling (only on valid sub-band)
    # ---------------------------------------------------------
    sub_smooth_base = median_filter(interp_mad_diff, size=freq_window_bins)

    # Calculate residuals for valid channels
    residuals = sub_mad_array - sub_smooth_base
    valid_residuals = residuals[valid_idx]

    median_residual = np.median(valid_residuals)
    mad_residual = np.median(np.abs(valid_residuals - median_residual))

    sigma_equiv_freq = 1.4826 * mad_residual
    threshold = median_residual + threshold_sigma_freq * sigma_equiv_freq

    # Flag positive outliers within the sub-band
    bad_sub_channels = (residuals > threshold) & valid_idx

    # ---------------------------------------------------------
    # 4. Map results back to the full-shape arrays
    # ---------------------------------------------------------
    num_bad_channels = np.sum(bad_sub_channels)
    if num_bad_channels > 0:
        print(f"MAD-of-MADs flagged {num_bad_channels} heavily contaminated channels in [{ch_low}, {ch_high}).")
        # Convert local sub-band indices to global full-array indices
        bad_global_indices = np.where(bad_sub_channels)[0] + ch_low
        cleaned_data.mask[:, bad_global_indices] = True

    # Fill the active regions of the full arrays
    full_mad_array[ch_low:ch_high] = sub_mad_array
    full_smooth_base[ch_low:ch_high] = sub_smooth_base

    return cleaned_data, full_mad_array, full_smooth_base


def remove_rfi_time_axis(masked_data, time_window_bins=5, threshold_sigma=7.0,
                         mask_threshold_freq=0.5, mask_threshold_time=0.5):
    """
    Robust RFI removal using a sliding median filter and Median Absolute Deviation (MAD).
    Processes data channel by channel to account for bandpass shape.
    Includes isolated valid point filter and global broad-band/persistent RFI flagging.

    Parameters:
    - masked_data: Input NumPy masked array (time, frequency).
    - time_window_bins: Window size for the median filter.
    - threshold_sigma: Sigma multiplier for transient RFI detection.
    - mask_threshold_freq: Fraction (0 to 1) of newly masked frequencies within the
                           VALID band to trigger masking the entire time step.
    - mask_threshold_time: Fraction (0 to 1) of masked times at a given frequency
                           channel to trigger masking the entire frequency channel.
    """
    n_times, n_freqs = masked_data.shape

    # --- Sanity Check Start ---
    # 1. Check if each channel is either completely masked or completely unmasked initially
    channel_all_masked = masked_data.mask.all(axis=0)
    channel_none_masked = (~masked_data.mask).all(axis=0)
    if not (channel_all_masked | channel_none_masked).all():
        raise ValueError("Sanity Check Failed: Input mask is inconsistent along the time axis. "
                         "Channels must be either 100% masked or 100% unmasked initially.")
    valid_channels = np.where(channel_none_masked)[0]
    first_valid = valid_channels[0]
    last_valid = valid_channels[-1]
    if len(valid_channels) != (last_valid - first_valid + 1):
        raise ValueError("Sanity Check Failed: Unmasked channels must be continuous.")

    cleaned_data = masked_data.copy()

    full_background = np.zeros_like(cleaned_data.data)
    full_residual = np.zeros_like(cleaned_data.data)

    # --- Pass 1: Channel-by-channel processing ---
    for f in range(first_valid, last_valid + 1):
        channel_data = cleaned_data[:, f].data

        # 1. Background modeling (using 'wrap' for 24-hour diurnal continuity)
        background = median_filter(channel_data, size=time_window_bins, mode='wrap')
        full_background[:, f] = background

        # 2. Flattening
        residual = channel_data - background
        full_residual[:, f] = residual

        # 3. The biggest difference from the smooth background
        bg_diffs = np.diff(background)
        max_physical_rate = np.percentile(np.abs(bg_diffs), 95)

        # 4. Define threshold using equivalent sigma
        threshold = threshold_sigma * max_physical_rate

        # 5. Flag transient RFI
        rfi_mask = np.abs(residual) > threshold
        cleaned_data.mask[:, f] |= rfi_mask

        # 6. Isolated point detection (including boundaries)
        current_mask = cleaned_data.mask[:, f]
        padded_mask = np.pad(current_mask, (1, 1), constant_values=True)
        isolated_points = padded_mask[:-2] & padded_mask[2:] & ~current_mask
        cleaned_data.mask[:, f] |= isolated_points

    # --- Pass 2: Global threshold-based flagging ---

    # 7. Broadband RFI Flagging: Focus ONLY on the valid frequency band
    # Extract the middle slice and calculate the fraction of flagged pixels
    valid_band_mask = cleaned_data.mask[:, first_valid: last_valid + 1]
    fraction_masked_freqs = valid_band_mask.mean(axis=1)
    broadband_rfi_mask = fraction_masked_freqs > mask_threshold_freq
    # Mask ALL frequencies (including ends, to be safe) for the time steps that exceed the threshold
    cleaned_data.mask[broadband_rfi_mask, :] = True

    # 8. Persistent RFI Flagging: Focus ONLY on the valid frequency band
    # Calculate fraction along the time axis for the valid channels
    fraction_masked_times = cleaned_data.mask[:, first_valid: last_valid + 1].mean(axis=0)
    persistent_rfi_mask = fraction_masked_times > mask_threshold_time
    # Find the absolute indices of the persistent RFI channels and mask them
    persistent_global_indices = np.where(persistent_rfi_mask)[0] + first_valid
    cleaned_data.mask[:, persistent_global_indices] = True

    return cleaned_data, full_background, full_residual


def remove_rfi_freq_axis(masked_data, freq_window_bins=15, threshold_sigma=7.0):
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


def plot_waterfall(data, x_axis, y_axis, use_symlog=False, linthresh=10000000.0, save_figure=None, show_figure=True,
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
        cbar_label = 'Intensity'
        cmap = 'seismic'
    else:
        # Standard LogNorm for strictly positive data (like Raw Visibility)
        # Avoids manual np.log10() to keep colorbar ticks in original data scale
        # Use a small vmin to avoid log(<=0) errors if standard data has minor artifacts
        valid_min = np.ma.min(data[data > 0]) if np.any(data > 0) else 1e-5
        norm = LogNorm(vmin=valid_min, vmax=np.ma.max(data))
        cbar_label = 'Intensity'
        cmap = 'viridis'

    # Plot using the configured norm instead of manual np.log10()
    im = ax.imshow(data, aspect='auto', extent=plot_extent, cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # ax.set_xticks([40, 50, 60, 70])
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


def plot_spectra(freqs, data, show_figure=False, filename=None, title=None, alpha=0.3):
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


def plot_time_series(times, data, use_log=False, show_figure=False, filename=None, title=None, alpha=0.3):
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
    ax.set_ylim(4.5e6, 9.0e6)
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
    print('The shape of raw_data:', raw_data.shape)

    ds = np.load(RAW_DIR + 'SE607_20240916_180834_spw3_int519_dur86400_sst.npz')
    freqs_mhz = ds['frequencies'] / 1e6

    times_2020 = np.linspace(0, 24, np.shape(raw_data)[1], endpoint=False)

    valid_indices = np.where((freqs_mhz >= freqs_cut[0]) & (freqs_mhz <= freqs_cut[1]))[0]
    start_idx = valid_indices[0]  # Included
    end_idx = valid_indices[-1] + 1  # Excluded
    # plot_freqs = freqs_mhz[start_idx:end_idx]

    for antenna_idx in range(90, 96):
        raw_data_ant = raw_data[antenna_idx, :, :]

        mask_out_of_data = mask_out_of_band(raw_data_ant, start_idx, end_idx)
        print(antenna_idx, np.std(mask_out_of_data.compressed()))
        # for s in range(5, 0, -1):
        #     time_axis_clean_data, time_axis_background, time_axis_residual = remove_rfi_time_axis(
        #         mask_out_of_data, time_window_bins=5, threshold_sigma=s
        #     )
        #     masked_residual = np.ma.masked_array(time_axis_residual, mask=time_axis_clean_data.mask)
        #     # print(np.sum(time_axis_clean_data.mask[:, start_idx:end_idx]) / (len(times_2020) * (end_idx - start_idx)), np.std(time_axis_clean_data.compressed()))
        #     print(np.sum(time_axis_clean_data.mask[:, start_idx:end_idx]) / (
        #                 len(times_2020) * (end_idx - start_idx)), np.std(masked_residual.compressed()))
        time_axis_clean_data, time_axis_background, time_axis_residual = remove_rfi_time_axis(
            mask_out_of_data, time_window_bins=7, threshold_sigma=2.0
        )
        print(np.shape(raw_data), np.shape(time_axis_clean_data))
        # all_axis_clean_data, all_axis_interp_spectrum, all_axis_background = remove_rfi_freq_axis(
        #     time_axis_clean_data, freq_window_bins=5, threshold_sigma=15.0
        # )

        diffs_frq = (time_axis_clean_data[:, 1:] - time_axis_clean_data[:, :-1]) / time_axis_clean_data[:, :-1]
        plot_spectra(
            freqs_mhz[start_idx:end_idx], diffs_frq[:, start_idx:end_idx], show_figure=True, filename=None,
            title=None, alpha=0.3
        )
        # plot_waterfall(time_axis_clean_data[:, start_idx:end_idx], freqs_mhz[start_idx:end_idx],
        #                 times_2020, show_figure=True)
        # # _plot_waterfall(time_axis_background[:, start_idx:end_idx], freqs_mhz[start_idx:end_idx],
        # #                 times_2020, show_figure=True)
        # plot_waterfall(time_axis_clean_data[:, start_idx:end_idx], freqs_mhz[start_idx:end_idx],
        #                times_2020, show_figure=False)

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

        # num_groups = 10
        # step = int(np.ceil((end_idx - start_idx) / num_groups))
        # for j in range(num_groups):
        #     start_jth = start_idx + j * step
        #     end_jth = min(start_idx + (j + 1) * step, end_idx)
        #     # plot_time_series(
        #     #     times_2020, raw_data_ant[:, start_jth:end_jth], show_figure=False,
        #     #     filename=f'../results/raw_data_vis_time/raw_x_antenna{antenna_idx:02}_group{j}.png',
        #     #     title=f'raw data, channel group {j}, antenna {antenna_idx:02}, X pol', alpha=1.0
        #     # )
        #     plot_time_series(
        #         times_2020, time_axis_clean_data[:, start_jth:end_jth], show_figure=False,
        #         filename=f'../results/step1_flagging_vis_time/step1_x_antenna{antenna_idx:02}_group{j}.png',
        #         title=f'Step 1, channel group {j}, antenna {antenna_idx:02}, X pol', alpha=1.0
        #     )

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
