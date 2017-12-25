"""
feature_extractor.py - Methods for extracting features from ECG signals.

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
from pywt import wavedec
from scipy.signal import butter, lfilter, hann
from scipy.signal import decimate, periodogram
from af_classifier.model_factory import ModelFactory
from af_classifier.rr_detector import get_qrs


class FeatureVectorExtractor(object):
    def __init__(self, extractors):
        self.extractors = extractors

    def extract_features(self, row):
        vector = None
        for extractor in self.extractors:
            values = extractor.extract(row)
            if not isinstance(values, np.ndarray):
                values = np.array([values])

            if vector is None:
                vector = values
            else:
                vector = np.concatenate((vector, values))

        return vector

    def extract_all(self, x):
        data = []
        previous_len = None
        same_len = True
        for x_row in x:
            next_row = self.extract_features(x_row)
            data.append(next_row)

            if (previous_len is not None) and \
               ((isinstance(next_row, np.ndarray) and next_row.size != previous_len) or
               (isinstance(next_row, list) and len(next_row) != previous_len)):
                same_len = False

            previous_len = len(next_row) if isinstance(next_row, list) else next_row.size

        if same_len:
            return np.vstack(data)
        else:
            return data


class FeatureExtractor(object):
    KEY = "root"

    def __init__(self):
        pass

    def extract(self, x):
        pass

    @staticmethod
    def get_extractor_for_key(key):
        for subclass in FeatureExtractor.__subclasses__():
            if subclass.KEY == key:
                return subclass
        return None

    def get_params(self):
        return np.asarray([], dtype=np.dtype(object))

    def __repr__(self):
        return self.__class__.KEY + repr(self.get_params())


class AverageFeatureExtractor(FeatureExtractor):
    KEY = "avg"

    def extract(self, x):
        return np.average(x, axis=-1)


class StandardDeviationFeatureExtractor(FeatureExtractor):
    KEY = "std"

    def extract(self, x):
        return np.std(x, axis=-1)


class AbsoluteAverageDeviationFeatureExtractor(FeatureExtractor):
    KEY = "aad"

    def extract(self, x):
        return np.average(np.absolute(x - np.average(x)))


class HistogramFeatureExtractor(FeatureExtractor):
    KEY = "hist"

    def __init__(self, num_bins=10):
        super(HistogramFeatureExtractor, self).__init__()
        self.num_bins = num_bins

    def get_params(self):
        return np.asarray([self.num_bins], dtype=np.dtype(object))

    def extract(self, x):
        return np.histogram(x - np.average(x), bins=self.num_bins, density=True)[0]


class SpectralDensityFeatureExtractor(FeatureExtractor):
    KEY = "psd"

    def __init__(self, sampling_frequency):
        super(SpectralDensityFeatureExtractor, self).__init__()
        self.sampling_frequency = sampling_frequency

    def get_params(self):
        return np.asarray([self.sampling_frequency], dtype=np.dtype(object))

    def extract(self, x):
        return periodogram(x, fs=self.sampling_frequency, window='hann')[1]


class BandPowerFeatureExtractor(FeatureExtractor):
    KEY = "bp"

    def __init__(self, sampling_frequency, bands=None):
        super(BandPowerFeatureExtractor, self).__init__()
        self.sampling_frequency = sampling_frequency
        self.bands = bands

    def get_params(self):
        return np.asarray([self.sampling_frequency, self.bands], dtype=np.dtype(object))

    def extract(self, x):
        return BandPowerFeatureExtractor.compute_band_powers(x, self.sampling_frequency, self.bands)

    @staticmethod
    def compute_band_powers(x, sampling_frequency, bands):
        f, pxx = periodogram(x, fs=sampling_frequency, window='hann')

        if bands is None:
            bands = (0, sampling_frequency / 2.)

        if not isinstance(bands, list):
            bands = [bands]

        ret_val = np.zeros((len(bands), x.shape[0]))
        for i, band in enumerate(bands):
            ret_val[i] = BandPowerFeatureExtractor.compute_band_power(band, pxx, f)

        if len(bands) == 1:
            ret_val = ret_val[0]
        else:
            ret_val = ret_val.T

        return ret_val

    @staticmethod
    def compute_band_power(band, pxx, f):
        idx_start = np.argmax(np.greater_equal(f, band[0]))
        idx_end = len(f) - np.argmax(np.less_equal(f[::-1], band[1]))

        # Using rectangle integration to approximate band power.
        rectangle_widths = np.diff(f.T[idx_start:idx_end])
        return np.dot(np.hstack((rectangle_widths, 0)), pxx.T[idx_start:idx_end])


class WaveletCoefficientFeatureExtractor(FeatureExtractor):
    KEY = "wco"

    def __init__(self, level=3, indices=None):
        super(WaveletCoefficientFeatureExtractor, self).__init__()
        self.level = level
        self.indices = indices

    def get_params(self):
        return np.asarray([self.level, self.indices], dtype=np.dtype(object))

    def extract(self, x):
        return WaveletCoefficientFeatureExtractor.get_wavelet_coefficient_vector(x, self.level, self.indices)

    @staticmethod
    def get_wavelet_coefficient_vector(x, level, indices):
        base = WaveletCoefficientFeatureExtractor.compute_wavelet_coefficients(x, level)
        if indices is not None:
            base = [base[i] for i in indices]
        return np.hstack(base)

    @staticmethod
    def compute_wavelet_coefficients(x, level):
        if x.ndim == 1:
            return wavedec(hann(x.shape[-1]) * x, 'db4', level=level)
        else:
            return map(np.asarray, np.asarray([wavedec(hann(x.shape[-1]) * x[i], 'db4', level=level)
                                               for i in range(x.shape[0])]).T.tolist())


class RelativeWaveletEnergiesFeatureExtractor(FeatureExtractor):
    KEY = "wle"

    def __init__(self, level=4, with_total=True):
        super(RelativeWaveletEnergiesFeatureExtractor, self).__init__()
        self.level = level
        self.with_total = with_total

    def get_params(self):
        return np.asarray([self.level, self.with_total], dtype=np.dtype(object))

    def extract(self, x):
        return RelativeWaveletEnergiesFeatureExtractor\
            .compute_relative_wavelet_energies(x, self.level, self.with_total)

    @staticmethod
    def compute_relative_wavelet_energies(x, level, with_total):
        coefficients = WaveletCoefficientFeatureExtractor.compute_wavelet_coefficients(x, level)
        energies = [np.sum(np.square(coefficients[i]), axis=-1)
                    for i in range(len(coefficients))]
        energies = np.asarray(energies).T
        total_energy = np.sum(energies, axis=-1)

        if len(total_energy.shape) == 0:
            total_energy = np.asarray([total_energy])

        total_energy = total_energy[:, np.newaxis]
        relative_energies = energies / total_energy
        if with_total:
            relative_energies = np.concatenate((relative_energies, total_energy / 100.), axis=-1)
        return relative_energies


class WaveletEntropyFeatureExtractor(FeatureExtractor):
    KEY = "went"

    def __init__(self, level=4):
        super(WaveletEntropyFeatureExtractor, self).__init__()
        self.level = level

    def get_params(self):
        return np.asarray([self.level], dtype=np.dtype(object))

    def extract(self, x):
        return WaveletEntropyFeatureExtractor.compute_wavelet_entropy(x, self.level)

    @staticmethod
    def compute_wavelet_entropy(x, level):
        relative_energies = RelativeWaveletEnergiesFeatureExtractor.compute_relative_wavelet_energies(x, level,
                                                                                                      with_total=False)
        return -np.sum(relative_energies * np.log(relative_energies), axis=-1)


class LengthFeatureExtractor(FeatureExtractor):
    KEY = "len"

    def __init__(self):
        super(LengthFeatureExtractor, self).__init__()

    def extract(self, x):
        return len(x)


class DownsampleFeatureExtractor(FeatureExtractor):
    KEY = "down"

    def __init__(self, downsample_factor=2):
        super(DownsampleFeatureExtractor, self).__init__()
        self.downsample_factor = downsample_factor

    def get_params(self):
        return np.asarray([self.downsample_factor], dtype=np.dtype(object))

    def extract(self, x):
        return decimate(x, self.downsample_factor, zero_phase=False, axis=-1)


class EncoderFeatureExtractor(FeatureExtractor):
    KEY = "enc"

    def __init__(self, encoder, output_dim):
        super(EncoderFeatureExtractor, self).__init__()
        self.encoder = encoder
        self.output_dim = output_dim

    def get_params(self):
        return np.asarray([self.output_dim], dtype=np.dtype(object))

    def extract(self, x):
        return EncoderFeatureExtractor.get_encoder_output(self.encoder, x, self.output_dim)

    @staticmethod
    def get_encoder_output(encoder, x, output_dim):
        if x.ndim == 1:
            x = np.asarray([x])

        dim0 = x.shape[0]
        vals_enc = encoder.predict(x)
        return vals_enc.reshape((dim0, output_dim))


class StackedEncoderFeatureExtractor(FeatureExtractor):
    KEY = "stenc"

    def __init__(self, model, dimension):
        super(StackedEncoderFeatureExtractor, self).__init__()
        if "," in dimension:
            self.dimension = map(int, dimension.split(","))
        else:
            self.dimension = [int(dimension)]

        if "," in model:
            self.model = model.split(",")
        else:
            self.model = [model]

        loaded_models = []
        for model_name in self.model:
            loaded_models.append(ModelFactory.get_model(model_name))
        self.model = loaded_models

    def get_params(self):
        return np.asarray([self.model, self.dimension], dtype=np.dtype(object))

    def extract(self, x):
        x_next = x
        for output_dimension, model in zip(self.dimension, self.model):
            x_next = EncoderFeatureExtractor.get_encoder_output(model,
                                                                x_next,
                                                                output_dimension)
        return x_next


class ButterworthFeatureExtractor(FeatureExtractor):
    KEY = "bpass"

    def __init__(self, freq, sampling_frequency):
        super(ButterworthFeatureExtractor, self).__init__()
        self.cutoff_of_band_frequency = freq
        self.sampling_frequency = sampling_frequency

    def get_params(self):
        return np.asarray([self.cutoff_of_band_frequency, self.sampling_frequency], dtype=np.dtype(object))

    def extract(self, x):
        return ButterworthFeatureExtractor.filter(x, self.cutoff_of_band_frequency, self.sampling_frequency)

    @staticmethod
    def filter(x, cutoff_or_band_frequency, sampling_frequency):
        max_freq = sampling_frequency / 2.
        if not isinstance(cutoff_or_band_frequency, (list, tuple, np.ndarray)):
            btype = 'lowpass'
        else:
            btype = 'bandpass'
        b, a = butter(5, np.asarray(cutoff_or_band_frequency) / float(max_freq), btype=btype)
        return lfilter(b, a, x, axis=0)


class DeltaRRFeatureExtractor(FeatureExtractor):
    KEY = "drr"

    def __init__(self, sampling_frequency=300, normalized_distances=False):
        super(DeltaRRFeatureExtractor, self).__init__()
        self.sampling_frequency = sampling_frequency
        self.normalized_distances = normalized_distances

    def get_params(self):
        return np.asarray([self.sampling_frequency, self.normalized_distances], dtype=np.dtype(object))

    def extract(self, x):
        return get_qrs(x, self.sampling_frequency, self.normalized_distances)[0]


class QRSFeatureExtractor(FeatureExtractor):
    KEY = "qrs"

    def __init__(self, sampling_frequency=300):
        super(QRSFeatureExtractor, self).__init__()
        self.sampling_frequency = sampling_frequency

    def get_params(self):
        return np.asarray([self.sampling_frequency], dtype=np.dtype(object))

    def extract(self, x):
        return get_qrs(x, self.sampling_frequency)[1]


def make_drr_pair(x):
    x1 = np.append(x, 0)
    x2 = np.append(0, x)
    return np.concatenate(([x1], [x2]))[:, 1:-1].T


class DualDeltaRRFeatureExtractor(FeatureExtractor):
    KEY = "ddrr"

    def __init__(self, sampling_frequency=300, normalized_distances=False):
        super(DualDeltaRRFeatureExtractor, self).__init__()
        self.sampling_frequency = sampling_frequency
        self.normalized_distances = normalized_distances

    def get_params(self):
        return np.asarray([self.sampling_frequency, self.normalized_distances], dtype=np.dtype(object))

    def extract(self, x):
        return make_drr_pair(get_qrs(x, self.sampling_frequency, self.normalized_distances)[0])


class QRSDurationFeatureExtractor(FeatureExtractor):
    KEY = "qrsd"

    def __init__(self, sampling_frequency=300):
        super(QRSDurationFeatureExtractor, self).__init__()
        self.sampling_frequency = sampling_frequency

    def get_params(self):
        return np.asarray([self.sampling_frequency], dtype=np.dtype(object))

    def extract(self, x):
        return QRSDurationFeatureExtractor.get_qrs_duration(x, self.sampling_frequency)

    @staticmethod
    def get_qrs_duration(x, sampling_frequency):
        qrs_idx = x.shape[-1] / 2
        qs_win = int(0.125*sampling_frequency)

        q_idx = qrs_idx - qs_win + np.argmin(x[..., qrs_idx - qs_win:qrs_idx], axis=-1)
        s_idx = qrs_idx + np.argmin(x[..., qrs_idx:qrs_idx + qs_win], axis=-1)

        return (s_idx - q_idx) / float(sampling_frequency)


class RAmplitudeFeatureExtractor(FeatureExtractor):
    KEY = "ramp"

    def __init__(self):
        super(RAmplitudeFeatureExtractor, self).__init__()

    def extract(self, x):
        return RAmplitudeFeatureExtractor.get_r_amplitude(x)

    @staticmethod
    def get_r_amplitude(x):
        qrs_idx = x.shape[-1] / 2
        return np.abs(x[..., qrs_idx] - np.median(x, axis=-1))


class QAmplitudeFeatureExtractor(FeatureExtractor):
    KEY = "qamp"

    def __init__(self, sampling_frequency=300, normalized=True):
        super(QAmplitudeFeatureExtractor, self).__init__()
        self.sampling_frequency = sampling_frequency
        self.normalized = normalized

    def get_params(self):
        return np.asarray([self.sampling_frequency, self.normalized], dtype=np.dtype(object))

    def extract(self, x):
        return QAmplitudeFeatureExtractor.get_q_amplitude(x, self.sampling_frequency, self.normalized)

    @staticmethod
    def get_q_amplitude(x, sampling_frequency, normalized):
        qrs_idx = x.shape[-1] / 2
        qs_win = int(0.125*sampling_frequency)

        q_idx = qrs_idx - qs_win + np.argmin(x[..., qrs_idx - qs_win:qrs_idx], axis=-1)

        if not isinstance(q_idx, np.ndarray):
            q_idx = np.asarray([q_idx])

        if x.ndim == 1:
            x = np.asarray([x])

        q_idx_vals = [x[i, q_idx[i]] for i in range(len(q_idx))]
        ret_val = np.abs(q_idx_vals, np.median(x, axis=-1))

        if normalized:
            clip_val = 3
            # Normalized in percentage of R amplitude.
            ret_val = np.min(np.vstack([ret_val / RAmplitudeFeatureExtractor.get_r_amplitude(x),
                                       [clip_val] * x.shape[0]]), axis=0)

        return ret_val


class PCAFeatureExtractor(FeatureExtractor):
    KEY = "pca"

    def __init__(self, pca):
        super(PCAFeatureExtractor, self).__init__()
        self.pca = pca

    def get_params(self):
        is_kernel_pca = hasattr(self.pca, 'alpha')
        if is_kernel_pca:
            return np.asarray([self.pca.alphas_, self.pca.lambdas_],
                              dtype=np.dtype(object))

        return np.asarray([self.pca.whiten, self.pca.components_],
                          dtype=np.dtype(object))

    def extract(self, x):
        dim0 = 1 if x.ndim == 1 else x.shape[0]
        return self.pca.transform(x.reshape(dim0, -1))
