"""
pipeline_feature.py - Provides functions to extract features as part of the data processing pipeline.

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
from __future__ import print_function

import sys
import os.path
from af_classifier.feature_extractor import *
from af_classifier.model_factory import ModelFactory

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class PipelineFeatureNormalizationFactory(object):
    PARAMS = {}
    MEAN_POSTFIX = "-mean"
    STD_POSTFIX = "-std"

    @staticmethod
    def normalize_feature(feature, data_set):
        mean_name = repr(feature) + PipelineFeatureNormalizationFactory.MEAN_POSTFIX
        std_name = repr(feature) + PipelineFeatureNormalizationFactory.STD_POSTFIX
        cached_mean = PipelineFeatureNormalizationFactory.PARAMS.get(mean_name)
        if cached_mean is None:
            print("WARNING: Normalizing on the fly, because norm factors were not available. ", file=sys.stderr)
            mean, stddev, _, _ = data_set.normalize()
            PipelineFeatureNormalizationFactory.PARAMS[mean_name] = mean
            PipelineFeatureNormalizationFactory.PARAMS[std_name] = stddev
        else:
            mean = cached_mean
            stddev = PipelineFeatureNormalizationFactory.PARAMS[std_name]
            data_set.normalize(mean, stddev)

    @staticmethod
    def load_params(file_name):
        if os.path.isfile(file_name):
            np_archive = np.load(file_name)
            PipelineFeatureNormalizationFactory.PARAMS = \
                dict([(name, np_archive[name]) for name in np_archive])
        else:
            print("WARNING: Tried to load normalization params file that doesnt exist: ",
                  file_name, file=sys.stderr)

    @staticmethod
    def save_params(file_name):
        np.savez(file_name, **PipelineFeatureNormalizationFactory.PARAMS)


class PipelineFeature(object):
    NORMALIZE_KEY = "norm"

    def __init__(self, normalized):
        self.normalized = normalized

    @staticmethod
    def has_normalize(feature_string):
        return PipelineFeature.feature_list_contains(feature_string, PipelineFeature.NORMALIZE_KEY)

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        pass

    @staticmethod
    def remove_modifier(feature_name):
        idx = feature_name.find("(")
        if idx == -1:
            return feature_name
        else:
            return feature_name[:idx]

    @staticmethod
    def get_modifier_list(feature):
        idx_start = feature.find("(")
        if idx_start == -1:
            return False
        else:
            idx_end = feature.find(")")
            return feature[idx_start + 1:idx_end]

    @staticmethod
    def has_modifier(feature, name):
        return PipelineFeature.feature_list_contains(PipelineFeature.get_modifier_list(feature), name)

    @staticmethod
    def feature_list_contains(feature_string, name):
        features = feature_string.split(",")
        return any(name == PipelineFeature.remove_modifier(f) for f in features)

    def get_normalized_feature_set(self, data_set):
        feature_set = self.get_extracted_feature_set(data_set)
        if self.normalized:
            PipelineFeatureNormalizationFactory.normalize_feature(self.__class__, feature_set)
        return feature_set

    def get_extracted_feature_set(self, data_set):
        pass


class DeltaRRPipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency):
        super(DeltaRRPipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, DeltaRRFeatureExtractor.KEY):
            return DeltaRRPipelineFeature(PipelineFeature.has_normalize(feature_string),
                                          context["adjusted_sampling_frequency"])

    def get_extracted_feature_set(self, data_set):
        feature_set = data_set.extract([DeltaRRFeatureExtractor(self.adjusted_sampling_frequency, True)])

        if len(data_set.x) == 1:
            feature_set.x = [feature_set.x[0]]

        return feature_set


class WaveletEnergyPipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency, wavelet_level=4):
        super(WaveletEnergyPipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency
        self.wavelet_level = wavelet_level

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, RelativeWaveletEnergiesFeatureExtractor.KEY):
            return WaveletEnergyPipelineFeature(PipelineFeature.has_normalize(feature_string),
                                                context["adjusted_sampling_frequency"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        data_energy = data_qrs.extract([RelativeWaveletEnergiesFeatureExtractor(self.wavelet_level)])

        if isinstance(data_energy.x, list) and len(data_set.x) == 1:
            data_energy.x = np.asarray([data_energy.x])

        return data_energy


class RAmplitudePipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency):
        super(RAmplitudePipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, RAmplitudeFeatureExtractor.KEY):
            return RAmplitudePipelineFeature(PipelineFeature.has_normalize(feature_string),
                                             context["adjusted_sampling_frequency"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        data_ramp = data_qrs.extract([RAmplitudeFeatureExtractor()])
        return data_ramp


class QAmplitudePipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency):
        super(QAmplitudePipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, QAmplitudeFeatureExtractor.KEY):
            return QAmplitudePipelineFeature(PipelineFeature.has_normalize(feature_string),
                                             context["adjusted_sampling_frequency"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        data_qamp = data_qrs.extract([QAmplitudeFeatureExtractor(self.adjusted_sampling_frequency,
                                                                 normalized=True)])
        return data_qamp


class QRSDurationPipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency):
        super(QRSDurationPipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, QRSDurationFeatureExtractor.KEY):
            return QRSDurationPipelineFeature(PipelineFeature.has_normalize(feature_string),
                                              context["adjusted_sampling_frequency"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        return data_qrs.extract([QRSDurationFeatureExtractor(self.adjusted_sampling_frequency)])


class WaveletCoefficientPipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency, wavelet_level=4, wavelet_indices=list([0, 1, 2])):
        super(WaveletCoefficientPipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency
        self.wavelet_level = wavelet_level
        self.wavelet_indices = wavelet_indices

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, WaveletCoefficientFeatureExtractor.KEY):
            return WaveletCoefficientPipelineFeature(PipelineFeature.has_normalize(feature_string),
                                                     context["adjusted_sampling_frequency"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        return data_qrs.extract([WaveletCoefficientFeatureExtractor(level=self.wavelet_level,
                                                                    indices=self.wavelet_indices)])


class EncoderPipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency, encoder_model, encoder_dim,
                 wavelet_level=4, wavelet_indices=list([0, 1, 2])):
        super(EncoderPipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency
        self.wavelet_level = wavelet_level
        self.wavelet_indices = wavelet_indices
        self.encoder_model = encoder_model
        self.encoder_dim = encoder_dim

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, EncoderFeatureExtractor.KEY):
            return EncoderPipelineFeature(PipelineFeature.has_normalize(feature_string),
                                          context["adjusted_sampling_frequency"],
                                          context["encoder_model"],
                                          context["encoder_dim"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        data_wco = data_qrs.extract([WaveletCoefficientFeatureExtractor(level=self.wavelet_level,
                                                                        indices=self.wavelet_indices)])
        data_enc = data_wco.extract([EncoderFeatureExtractor(ModelFactory.get_model(self.encoder_model),
                                                             self.encoder_dim)])
        return data_enc


class QRSPipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency):
        super(QRSPipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, QRSFeatureExtractor.KEY):
            return QRSPipelineFeature(PipelineFeature.has_normalize(feature_string),
                                      context["adjusted_sampling_frequency"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        return data_qrs


class WaveletEntropyPipelineFeature(PipelineFeature):
    def __init__(self, normalized, adjusted_sampling_frequency, wavelet_level=4):
        super(WaveletEntropyPipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency
        self.wavelet_level = wavelet_level

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, WaveletEntropyFeatureExtractor.KEY):
            return WaveletEntropyPipelineFeature(PipelineFeature.has_normalize(feature_string),
                                                 context["adjusted_sampling_frequency"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        data_went = data_qrs.extract([WaveletEntropyFeatureExtractor(self.wavelet_level)])
        return data_went


class StackedEncoderPipelineFeature(PipelineFeature):
    KPCA_CACHE = {}

    def __init__(self, normalized, adjusted_sampling_frequency, morph_encoder_model, morph_encoder_dimension,
                 with_kpca, kpca_file):
        super(StackedEncoderPipelineFeature, self).__init__(normalized)
        self.adjusted_sampling_frequency = adjusted_sampling_frequency
        self.morph_encoder_model = morph_encoder_model
        self.morph_encoder_dimension = morph_encoder_dimension
        self.with_kpca = with_kpca
        self.kpca_file = kpca_file

    @staticmethod
    def configure_pipeline_feature(feature_string, context):
        if PipelineFeature.feature_list_contains(feature_string, StackedEncoderFeatureExtractor.KEY) or \
           PipelineFeature.feature_list_contains(feature_string, "morphenc"):
            return StackedEncoderPipelineFeature(PipelineFeature.has_normalize(feature_string),
                                                 context["adjusted_sampling_frequency"],
                                                 context["morph_encoder_model"],
                                                 context["morph_encoder_dimension"],
                                                 PipelineFeature.feature_list_contains(feature_string, "kpca"),
                                                 context["kpca_file"])

    def get_extracted_feature_set(self, data_set):
        data_qrs = data_set.extract([QRSFeatureExtractor(self.adjusted_sampling_frequency)])
        next_data_set = data_qrs.extract([StackedEncoderFeatureExtractor(self.morph_encoder_model,
                                                                         self.morph_encoder_dimension)])

        if self.with_kpca:
            kpca = StackedEncoderPipelineFeature.KPCA_CACHE.get(self.kpca_file)
            if kpca is None:
                print("INFO: Loading KPCA-file: ", self.kpca_file, file=sys.stderr)
                kpca = pickle.load(open(self.kpca_file, "r"))
                StackedEncoderPipelineFeature.KPCA_CACHE[self.kpca_file] = kpca
            next_data_set = next_data_set.extract([PCAFeatureExtractor(kpca)])

        return next_data_set


class PipelineFeatureFactory(object):
    ORDERED_FEATURE_BLOCKS = [
        DeltaRRPipelineFeature,
        WaveletEnergyPipelineFeature,
        RAmplitudePipelineFeature,
        QAmplitudePipelineFeature,
        QRSDurationPipelineFeature,
        WaveletCoefficientPipelineFeature,
        EncoderPipelineFeature,
        QRSPipelineFeature,
        WaveletEntropyPipelineFeature,
        StackedEncoderPipelineFeature
    ]

    @staticmethod
    def get_blocks_for_feature_string(feature_string, context):
        responders = []
        for cls in PipelineFeatureFactory.ORDERED_FEATURE_BLOCKS:
            response = cls.configure_pipeline_feature(feature_string, context)
            if response is not None:
                responders.append(response)
        return responders
