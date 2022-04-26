# Signal processing for NIRS.
# Created by Weiwei Jiang on 20201105. 
# 

import numpy as np
from scipy.signal import savgol_filter, detrend, decimate, resample_poly

# Utilities.
def get_argmax(data, *, tol):
    """ Find the index of final occurance of max value """
    idx_max = 0
    max_value = data[0]
    for idx, item in enumerate(data):
        
        if (data[idx] >= max_value) or (max_value - data[idx] < tol):
            idx_max = idx
            max_value = data[idx]
            
    return idx_max

def normalize_array(input_arr, revert=True, low_percentile=None, high_percentile=None):
    """ Robust normalization of an array. """
    # Create a copy of input to prevent manipulation.
    input_arr = np.array(np.copy(input_arr))
    
    if low_percentile is None:
        low_percentile = 0
    if high_percentile is None:
        high_percentile = 100
    
    # Find percentiles if specified.
    min_edge, max_edge = np.percentile(input_arr, [low_percentile, high_percentile])
    
    # Set min-max to percentiles if specified.
    input_arr[input_arr < min_edge] = min_edge
    input_arr[input_arr > max_edge] = max_edge
    
    # Find min max values.
    min_value = np.min(input_arr)
    max_value = np.max(input_arr)
    
    arr_normalized = (input_arr - min_value) / (max_value - min_value)
    
    if revert:
        arr_normalized = 1.0 - arr_normalized # Revert information pixel to 1. 
    
    return arr_normalized

def invalid_to_nearest(signal, copy=True):
    """ Repleace invalid values (nan/inf) to nearest valid values. """
    # Init.
    idx_left_valid = None
    idx_right_valid = None
    normal_state = np.isfinite(signal[0])
    
    if copy:
        new_signal = np.copy(signal)
    else:
        new_signal = signal
    
    for idx, s in enumerate(signal):
        # Looking for nan/inf.
        if normal_state & (~np.isfinite(s)):
            # Nan segment starts.
            normal_state = False
            idx_left_valid = idx - 1
        elif (~normal_state) & np.isfinite(s):
            # Nan segment ends.
            normal_state = True
            idx_right_valid = idx
            
            # Fill-in non-valid segment.
            if idx_left_valid is None:
                # Head-nan process.
                new_signal[:idx_right_valid] = signal[idx_right_valid]
            else:
                # Find mid-point.
                idx_mid = int(np.ceil((idx_right_valid + idx_left_valid) / 2))
                
                # Fill-in, mid-point = right_valid
                new_signal[idx_left_valid+1:idx_mid] = signal[idx_left_valid]
                new_signal[idx_mid:idx_right_valid] = signal[idx_right_valid]
            
    # Tail process. 
    if not normal_state:
        new_signal[idx_left_valid+1:] = signal[idx_left_valid]
    
    return new_signal


def moving_average(signal, N):
    from scipy.ndimage.filters import uniform_filter1d
    """ Moving average, nearest-padding, left-and-right. """
    return uniform_filter1d(signal, size=N, mode="reflect")
    
    
def estimate_snr(received, signal=None, N=20):
    """ Estimating SNR. If signal unknown then use moving average to estimate.
    """
    
    # Estimate signal levels using moving average.
    if signal is None:
        signal = moving_average(received, N)
    
    power_est = np.mean(np.abs(signal))
    noise_est = (np.sqrt(np.sum((received - signal) ** 2) / (len(received) - 1))) / np.sqrt(2)
    snr_est = power_est / noise_est
    
    return snr_est


def construct_scan_data(scan_data):
    """Get intensity and reference spectra from 2D scan data."""
    all_intensity_data = []
    all_reference_data = []
    for data_1d in scan_data:
        
        intensity_data_1d = []
        reference_data_1d = []

        for data in data_1d:
            
            intensity_data_1d.append(data["intensity"])
            reference_data_1d.append(data["reference"])
        
        all_intensity_data.append(intensity_data_1d)
        all_reference_data.append(intensity_data_1d)
    
    return np.array(all_intensity_data, dtype=np.float), np.array(all_reference_data, dtype=np.float)


def process_signal(wavelength_list, raw_intensity, raw_reference, reference_spectrum, *,
                   selected_indexes=None, savgol_window=11, savgol_polyorder=3,
                   moving_average_window=11, decimate_factor=8, absorbance_mode=True):
    """Processing NIRS spectra. """
    
    # Convert to numpy arrays.
    raw_intensity = np.array(raw_intensity)
    raw_reference = np.array(raw_reference)
    reference_spectrum = np.array(reference_spectrum)
    
    # Remove low SNR wavelengths.
    wavelength_list_processed = wavelength_list
    if selected_indexes is not None:
        wavelength_list_processed = wavelength_list[selected_indexes]
        raw_intensity = raw_intensity[selected_indexes]
        raw_reference = raw_reference[selected_indexes]
        reference_spectrum = reference_spectrum[selected_indexes]
    
    # Pre-smoothing.
    processed = savgol_filter(raw_intensity, window_length=savgol_window, polyorder=savgol_polyorder)

    # Convert to absorbance.
    if absorbance_mode:
        processed = -np.log10(processed / raw_reference) - reference_spectrum

        # Fill-in non-valid values.
        processed = invalid_to_nearest(processed)

    # Moving average and decimal.
    processed = moving_average(processed, N=moving_average_window)
    processed = processed[::decimate_factor]
    
    return np.array(wavelength_list_processed[::decimate_factor]), np.array(processed)