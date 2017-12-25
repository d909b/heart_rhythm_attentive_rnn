"""
pipeline.py - Provides functions to set up a data processing pipeline for ECG analysis.

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
from os.path import join, basename
from keras.preprocessing.sequence import pad_sequences
from af_classifier.data_source import PhysioNet2017DataSource
from af_classifier.feature_extractor import *
from af_classifier.pipeline_feature import PipelineFeatureFactory

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


def select_subset_of_data(args, raw_sequence_data):
    fraction_of_data_set = float(args["fraction_of_data_set"])
    if fraction_of_data_set < 1.0:
        print("INFO: Using fraction ", fraction_of_data_set, " of data set.", file=sys.stderr)
        raw_sequence_data, _, _ = raw_sequence_data.randomized_split(fraction_of_data_set,
                                                                     # Prevent shuffling to preserve reproducibility.
                                                                     permutation_indices=range(len(raw_sequence_data.x)))
    return raw_sequence_data


def configure_classes(args, raw_sequence_data):
    if args["one_vs_k"]:
        selected_class = int(args["selected_class"])
        if args["selected_class_counter"] is not None:
            selected_class_counter = int(args["selected_class_counter"])
            new_balance = [0]*raw_sequence_data.get_num_classes()
            new_balance[selected_class] = None
            new_balance[selected_class_counter] = None
            raw_sequence_data.set_class_balance(new_balance)

        raw_sequence_data = raw_sequence_data.to_one_vs_k(int(args["selected_class"]))
        num_outputs = 2
    else:
        num_outputs = 4

    return num_outputs, raw_sequence_data


def equalize_data_set(args, raw_sequence_data, smote_class=0):
    """
    Equalizes class misbalances in the data set through various means.

    :param args: The program arguments.
    :param raw_sequence_data: The ECG sequence data set.
    :param smote_class: The class to oversample using SMOTE.
    :return: A data set with equalized class balance.
    """
    if args["equalize"]:
        equalize_mode = args["equalize_mode"]
        if equalize_mode == "augmax" \
           or equalize_mode == "augmin" \
           or equalize_mode == "augsmote" \
           or equalize_mode == "aug":
            from af_classifier.external_data import load_external_data

            external_data_set = load_external_data()
            _, external_data_set = configure_classes(args, external_data_set)
            raw_sequence_data = raw_sequence_data.merged(external_data_set)

            if equalize_mode == "aug":
                return raw_sequence_data

            if equalize_mode == "augmax":
                equalize_mode = "max"
            elif equalize_mode == "augmin":
                equalize_mode = "min"
            else:  # augsmote
                equalize_mode = "smote"

        if equalize_mode == "max":
            class_size = max(raw_sequence_data.get_class_balance())
        elif equalize_mode == "smote":
            raw_sequence_data = create_synthetic_balanced_data_set(args, raw_sequence_data, smote_class)
            class_size = None
        else:
            class_size = min(raw_sequence_data.get_class_balance())

        if class_size is not None:
            balance = [class_size]*raw_sequence_data.get_num_classes()
            raw_sequence_data.set_class_balance(balance)
    return raw_sequence_data


def downsample(args, raw_sequence_data):
    downsampling_factor = int(args["downsample"])
    if downsampling_factor > 1:
        print("INFO: Downsampling by factor ", downsampling_factor, file=sys.stderr)
        raw_sequence_data = raw_sequence_data.extract([DownsampleFeatureExtractor()])
    return raw_sequence_data, downsampling_factor


def correct_feature_form(data_entry, is_drr, with_drr):
    assert data_entry.ndim <= 2, "Data entry must not be more than 2-dimensional."

    # Delta RR drops one beat in length due to taking the first difference.
    # Therefore all the other features must too be adjusted in size to match.
    if with_drr and not is_drr:
        if data_entry.ndim == 2:
            return data_entry[1:]
        else:
            return data_entry[1:, np.newaxis]
    else:
        if data_entry.ndim == 2:
            return data_entry[:]
        else:
            return data_entry[:, np.newaxis]


def extract_features(args, data_set, downsampling_factor):
    features = args["features"].split(",")
    with_drr = any(DeltaRRFeatureExtractor.KEY == f for f in features)

    num_inputs = 0
    feature_list = []

    context = {"adjusted_sampling_frequency": args["sampling_frequency"] / float(downsampling_factor),
               "encoder_model": args["encoder_model"],
               "encoder_dim": args["encoding_dim"],
               "morph_encoder_model": args["morph_encoder_model"],
               "morph_encoder_dimension": args["morph_encoder_dimension"],
               "kpca_file": args["tpca_params_file"]}
    blocks = PipelineFeatureFactory.get_blocks_for_feature_string(args["features"], context)

    for block in blocks:
        feature_data_set = block.get_normalized_feature_set(data_set)

        # Ensure all feature data sets have the same number of dimensions.
        if isinstance(feature_data_set.x, np.ndarray) and feature_data_set.x.ndim == 2:
            feature_data_set.x = np.asarray([feature_data_set.x])

        if isinstance(feature_data_set.x[0], np.ndarray) and feature_data_set.x[0].ndim == 1:
            feature_len = 1
        else:
            feature_len = feature_data_set.x[0].shape[-1]

        feature_list.append(feature_data_set)
        num_inputs += feature_len

    # Concatenate all features to a single feature vector.
    feature_data_x = [np.concatenate([correct_feature_form(feature_list[j].x[i],
                                                           is_drr=j == 0,
                                                           with_drr=with_drr)
                                      for j in range(len(feature_list))],
                                     axis=-1)
                      for i in range(len(data_set.x))]

    return data_set.__class__(feature_data_x, data_set.y), num_inputs


def preprocess_data_set(args, raw_sequence_data, zero_pad=True):
    seq_train, seq_test = split_and_normalize_data_set(args, raw_sequence_data)

    if seq_train is not None:
        seq_train, _, _ = configure_and_extract_from_data_set(args, seq_train, is_training_set=True, zero_pad=zero_pad)
    seq_test, num_inputs, num_outputs = configure_and_extract_from_data_set(args, seq_test, zero_pad=zero_pad)

    return seq_train, seq_test, num_inputs, num_outputs


def split_and_normalize_data_set(args, data_set):
    using_train_set = not np.isclose([args["test_set_fraction"]], [1.0])

    if args["load_meta_params"]:
        print("INFO: Loading meta params from file: ", args["load_meta_params"], file=sys.stderr)
        split_indices, normalization_params = pickle.load(open(args["load_meta_params"], "r"))
    else:
        print("INFO: Not using meta params.", file=sys.stderr)
        split_indices, normalization_params = None, None

    if using_train_set:
        train_set_fraction = 1 - args["test_set_fraction"]
        seq_train, seq_test, split_indices \
            = data_set.randomized_split(train_set_fraction, permutation_indices=split_indices)

        if normalization_params is None:
            # Prevent information leak from test to train set by using train set only normalisation factors.
            normalization_params = seq_train.normalize()
        else:
            seq_train.normalize(*normalization_params)
    else:
        seq_train = None
        seq_test = data_set
        split_indices = None

    seq_test.normalize(*normalization_params)

    if using_train_set:
        seq_train = select_subset_of_data(args, seq_train)
    seq_test = select_subset_of_data(args, seq_test)

    if args["meta_params_file"]:
        # Save split indices and normalization params used for this run.
        pickle.dump((split_indices, normalization_params),
                    open(join(args["output_directory"], args["meta_params_file"]), "w"),
                    pickle.HIGHEST_PROTOCOL)

    return seq_train, seq_test


def configure_and_extract_from_data_set(args, data_set, is_training_set=False, zero_pad=True):
    num_outputs, data_set = configure_classes(args, data_set)
    if is_training_set:
        data_set = equalize_data_set(args, data_set)

    # print("INFO: Target class balance is: ", data_set.get_class_balance(), file=sys.stderr)

    data_set, downsampling_factor = downsample(args, data_set)

    extracted, num_inputs = extract_features(args, data_set, downsampling_factor)

    if zero_pad:
        # Ensure equal beat length. The zero-padding will be masked by RNNs.
        max_num_beats = max(map(len, extracted.x))
        extracted.x = pad_sequences(extracted.x,
                                    maxlen=max_num_beats,
                                    padding="post",
                                    truncating="post",
                                    dtype="float32")

        extracted = expand_dim_if_necessary(extracted)

    return extracted, num_inputs, num_outputs


def expand_dim_if_necessary(data_set):
    if isinstance(data_set.x, np.ndarray) and data_set.x.ndim == 2:
        data_set.x = np.expand_dims(data_set.x, axis=2)
    elif isinstance(data_set.x, list) and (isinstance(data_set.x[0], np.ndarray) and len(data_set.x[0].shape) == 2):
        for idx in range(len(data_set.x)):
            data_set.x[idx] = np.expand_dims(data_set.x[idx], axis=1)
            data_set.x[idx] = np.expand_dims(data_set.x[idx], axis=0)
    return data_set


def expand_dims_if_necessary(train_set, test_set):
    return expand_dim_if_necessary(train_set), expand_dim_if_necessary(test_set)


def do_single_prediction(args, model, file_name):
    raw_sequence_data = PhysioNet2017DataSource()
    raw_sequence_data.load_single(file_name)

    # No need to split into test / train set.
    args["test_set_fraction"] = 1.
    # No need to save meta params for single predictions.
    args["meta_params_file"] = None
    raw_sequence_data, _, _, _ = preprocess_data_set(args, raw_sequence_data)

    raw_sequence_data.x = np.array(raw_sequence_data.x)

    if isinstance(raw_sequence_data.x, np.ndarray) and raw_sequence_data.x.ndim == 2:
        raw_sequence_data.x = np.expand_dims(raw_sequence_data.x, axis=0)

    y = model.predict(raw_sequence_data.x)
    y_idx = np.argmax(y, axis=1)[0]

    return basename(file_name) + "," + PhysioNet2017DataSource.CODE_MAP[y_idx] + "\n"


def load_sequence_data(args):
    print("Loading data set: ", args["dataset"], file=sys.stderr)

    raw_sequence_data = PhysioNet2017DataSource()
    raw_sequence_data.load(args["dataset"])
    raw_sequence_data.to_categorical(4)
    return raw_sequence_data


def prepare_autoencoder_data_set(data_set):
    """
    Prepares a data set for training an auto-encoder,
    i.e. the temporal dimension is collapsed and
    the input is set to be the target value.

    :param data_set: The data set to prepare.
    :return: The prepared data set.
    """
    data_set.x = np.vstack(data_set.x)
    data_set.y = data_set.x
    return data_set


def create_synthetic_balanced_data_set(args, data_set, selected_class, ratio='auto'):
    """
    Creates a balanced data set by adding synthetic samples to underrepresented classes using SMOTE upsampling.

    :param args: Program arguments.
    :param data_set: The data set to balance.
    :param selected_class: The class to be balanced with synthetic samples.
    :param ratio: Upsampling ratio.
    :return: A data set with the selected class balanced using synthetic samples.
    """
    from imblearn.combine import SMOTEENN

    num_classes = data_set.get_num_classes()
    seq_copy = data_set.to_one_vs_k(selected_class)

    # X must be padded and y must be binarized to work with the SMOTE implementation.
    padded_x = pad_sequences(seq_copy.x,
                             maxlen=args["max_sequence_length"],
                             padding="post",
                             truncating="post",
                             dtype="float32")

    binary_y = np.argmax(seq_copy.y, axis=-1)

    sm = SMOTEENN(n_jobs=4, ratio=ratio)
    new_x, new_y = sm.fit_sample(padded_x, binary_y)

    # Transform the data back to the application format.
    synthetic_data_set = data_set.__class__(new_x, new_y)
    synthetic_data_set.to_categorical(num_classes)
    synthetic_data_set = synthetic_data_set.single_class_data_set(0)
    synthetic_data_set.y = np.ones((len(synthetic_data_set.x), 1)) * selected_class
    synthetic_data_set.to_categorical(num_classes)

    synthetic_data_set.x = map(lambda x: x, synthetic_data_set.x)
    synthetic_data_set.y = synthetic_data_set.y.tolist()

    # Remove the samples used to generate synthetic samples.
    balance = data_set.get_class_balance()
    balance[selected_class] = 0
    data_set.set_class_balance(balance)

    data_set.x = data_set.x.tolist()
    data_set.y = data_set.y.tolist()

    # Merge sets.
    return_set = data_set.merged(synthetic_data_set)
    return_set.x = np.asarray(return_set.x)
    return_set.y = np.asarray(return_set.y)

    return return_set
