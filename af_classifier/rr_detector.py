"""
rr_detector.py - Provides functions for detecting QRS complexes and RR intervals in ECG signals.

Copyright (C) 2017  Patrick Schwab, ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy.signal import lfilter


def filter_ecg_signal(ecg):
    """
    Bandpass filters ECG signals.

    :param ecg: A single ECG signal (1d-array)
    :return: The bandpass filtered ECG signal
    """
    # Low-pass filter.
    num_lp = np.zeros(13)
    num_lp[12] = 1
    num_lp[0] = 1
    num_lp[6] = -2

    den_lp = np.zeros(3)
    den_lp[0] = 1
    den_lp[1] = -2
    den_lp[2] = 1

    ecg_lp = lfilter(num_lp, den_lp, ecg)

    # High-pass filter.
    num_hp = np.zeros(33)
    num_hp[32] = 1 / 32.
    num_hp[17] = -1
    num_hp[16] = 1
    num_hp[0] = -1 / 32.

    den_hp = np.zeros(2)
    den_hp[0] = 1
    den_hp[1] = -1

    ecg_hp = lfilter(num_hp, den_hp, ecg_lp)

    # Derivative filter.
    num_der = np.zeros(5)
    num_der[0] = 2
    num_der[1] = 1
    num_der[3] = -1
    num_der[4] = -2
    num_der *= 0.1

    return lfilter(num_der, 1, ecg_hp)


def get_qrs(ecg, sampling_rate=300, normalized_distances=False, with_qrs=True):
    """
    Performs QRS detection on ECG signals using an adaptive Pan-Tompkins based QRS algorithm.

    :param ecg: An ECG signal.
    :param sampling_rate: Sampling rate of the ECG signal in Hz.
    :param normalized_distances: Whether or not to return the distances between peaks in normalised form.
    :param with_qrs: Whether or not to return windows around detected RR peaks that capture the whole QRS complex.
    :return: A tuple of 2 elements, with the first element being the indices of detected R-peaks in the ECG signal if
            normalized_distances is False. If normalized_distances is True then the first element corresponds to the
            normalised distance between consecutive R peaks (dRR-Interval). The second element is an array of
            equal-sized windows around detected R-peaks, if with_qrs is True, or None if with_qrs is False.
    """
    # This algorithm is parametrised on a set of assumptions about the ECG signal.
    assumed_max_heart_rate_in_bpm = 130
    assumed_max_pt_duration_in_ms = 0.67
    assumed_standard_qrs_duration_in_ms = 0.15

    ecg_filtered = filter_ecg_signal(ecg)

    # Square the signal.
    ecg_s = np.square(ecg_filtered)

    # QRS complex detection via moving averages.
    n_mov = int(sampling_rate * assumed_standard_qrs_duration_in_ms)
    num_mov = 1./n_mov * np.ones(n_mov)
    ecg_mov = lfilter(num_mov, 1, ecg_s)

    # @PS: changed to moving threshold to be more robust to local noise.
    n_th = int(sampling_rate)
    num_th = 1./n_th * np.ones(n_th)
    threshold = lfilter(num_th, 1, ecg_s)

    # Window lengths must correspond roughly to QRS complex durations.
    # We remove smaller window RR peak candidates from the index set.
    window_below_threshold = np.less(ecg_mov, threshold).astype(int)
    window_below_threshold_entry_and_exit = np.diff(window_below_threshold)
    idx_below_threshold_entry = np.where(np.equal(window_below_threshold_entry_and_exit, 1))[0]
    idx_below_threshold_exit = np.where(np.equal(window_below_threshold_entry_and_exit, -1))[0]

    if idx_below_threshold_entry[0] > idx_below_threshold_exit[0]:
        # Sequence starts with an entry.
        idx_below_threshold_exit = idx_below_threshold_exit[1:]

    if len(idx_below_threshold_exit) < len(idx_below_threshold_entry):
        idx_below_threshold_exit = np.hstack((idx_below_threshold_exit, [len(ecg)]))

    window_length = idx_below_threshold_exit - idx_below_threshold_entry

    # Only consider indices with minimal QS duration.
    idx = idx_below_threshold_entry[np.greater(window_length, assumed_standard_qrs_duration_in_ms * sampling_rate)]

    # Window size depending on assumed max PT duration.
    win = int(assumed_max_pt_duration_in_ms*sampling_rate)

    # Must have at least one bound margin in the ECG signal,
    # because the moving averages need time to adapt to the signal structure.
    bounds = win * 1.5
    idx = idx[np.greater(idx, bounds)]
    idx = idx[np.less(idx, len(ecg) - bounds)]

    # The tightest QRS margin window length depends on the assumed max heart rate we will encounter.
    qrs_win = int(60./assumed_max_heart_rate_in_bpm * sampling_rate)
    qrs_idx = idx - qrs_win + np.argmax([ecg[i - qrs_win:i] for i in idx], axis=-1)

    # Return equal sized intervals around the RR peaks as extracted QRS complexes.
    if with_qrs:
        qrs = np.vstack([ecg[i - win/2:i + win/2] for i in qrs_idx])
    else:
        qrs = None

    if normalized_distances:
        index_array = np.diff(qrs_idx) / float(sampling_rate)
    else:
        index_array = qrs_idx

    return index_array, qrs

