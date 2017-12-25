"""
util.py - Utility functions.

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

import argparse
import json
import os
import sys

import theano

from af_classifier.feature_extractor import *
from af_classifier.pipeline_feature import PipelineFeatureNormalizationFactory

theano.config.openmp = True


class ReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{} is not a readable dir".format(prospective_dir))


def parse_args():
    parser = argparse.ArgumentParser(description='An AF Classifier.')

    parser.add_argument("--dataset", action=ReadableDir, required=True,
                        help="Data set to be loaded.")
    parser.add_argument("--level2_dataset", action=ReadableDir,
                        help="Data set to be loaded as level 2 test set.")
    parser.add_argument("--single_file", required=False, default="",
                        help="Data set to be loaded.")
    parser.add_argument("--test_set_fraction", type=int, default=0.2,
                        help="Fraction of data set to be held out for testing.")
    parser.add_argument("--seed", type=int, default=999,
                        help="Random seed to initialise the pseudo-random number generator (PRNG) with.")
    parser.add_argument("--num_bags", type=int, default=0,
                        help="Number of bags to create from the training data set.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Size of batches to use during training phases.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs to train for.")
    parser.add_argument("--max_sequence_length", type=int, default=9000,
                        help="Maximum number of data points in a time series. Excess data points are truncated.")
    parser.add_argument("--sampling_frequency", type=int, default=300,
                        help="Sampling frequency of the time series.")
    parser.add_argument("--output_directory", default="./models",
                        help="Base directory of all output files.")
    parser.add_argument("--model_name", default="af_classifier-{epoch:02d}.h5",
                        help="Saved model file name.")
    parser.add_argument("--load_existing", default="",
                        help="File name of model to initialise training with.")
    parser.add_argument("--do_train", default=True,
                        help="Whether or not to include the training phase.")
    parser.add_argument("--downsample", default=1,
                        help="Downsampling level of original signal.")
    parser.add_argument("--selected_class", default=0,
                        help="Class selected for 1-vs-k training.")
    parser.add_argument("--selected_class_counter", default=None,
                        help="Counter class selected for 1-vs-1 training.")
    parser.add_argument("--u_dropout", default=0.45,
                        help="Amount of recurrent dropout applied for regularization [0, 1).")
    parser.add_argument("--dropout", default=0.45,
                        help="Amount of forward dropout applied for regularization [0, 1).")
    parser.add_argument("--noise", default=0.,
                        help="Amount of gaussian noise applied at the input level.")
    parser.add_argument("--num_units", default=256,
                        help="Number of hidden units per (forward) layer.")
    parser.add_argument("--num_recurrent_units", default=128,
                        help="Number of hidden units per recurrent layer.")
    parser.add_argument("--num_recurrent_layers", default=2,
                        help="Number of recurrent layers.")
    parser.add_argument("--fraction_of_data_set", default=1,
                        help="Fraction of data set to use.")
    parser.add_argument('--one_vs_k', dest='one_vs_k', action='store_true',
                        help="Whether to train a one vs k model.")
    parser.add_argument('--multi_class', dest='one_vs_k', action='store_false',
                        help="Whether to train a multi-class model.")
    parser.set_defaults(one_vs_k=True)
    parser.add_argument('--equalize', dest='equalize', action='store_true',
                        help="Whether to equalize class balance(s) in the training data.")
    parser.add_argument('--equalize_mode', default='min',
                        help="Type of class balance equalisation to apply. (min, max, aug, augmin, augmax)")
    parser.set_defaults(equalize=False)
    parser.add_argument("--meta_params_file", default="meta-params.pickle",
                        help="File to store learning meta parameters in "
                             "(test set split indices and normalisation params).")
    parser.add_argument("--meta_params_file2", default="meta-params2.pickle",
                        help="File to store level 2 learning meta parameters in "
                             "(test set split indices).")
    parser.add_argument("--load_meta_params", default="./models/meta-params.pickle",
                        help="To load meta parameters from an existing file for the level 1 training set.")
    parser.add_argument("--load_meta_params2", default="./models/meta-params2.pickle",
                        help="To load meta parameters from an existing file for the level 2 training set.")
    parser.add_argument("--features", default="drr,wle",
                        help="Which features to extract per PQRST window.")
    parser.add_argument("--encoder_model", default="./models/ae_encoder.h5",
                        help="Encoder to use for the 'enc' feature extactor.")
    parser.add_argument("--ensemble", default="./models/ensemble.json",
                        help="Ensemble model description file to load.")
    parser.add_argument("--precompile", dest='precompile', action='store_true',
                        help="Whether or not to save the precompiled model.")
    parser.set_defaults(precompile=False)
    parser.add_argument("--class_weighted", dest='class_weighted', action='store_true',
                        help="Whether or not to weight classes by their relative sizes during training.")
    parser.set_defaults(class_weighted=False)
    parser.add_argument("--stacked", dest='stacked', action='store_true',
                        help="Whether or not to train autoencoders by stacking.")
    parser.set_defaults(stacked=False)
    parser.add_argument("--encoding_dim", default=16,
                        help="Dimension of latent feature space when building an autoencoder.")
    parser.add_argument("--morph_encoder_model",
                        default="./models/stae_3l_p0_gleveled_dim24-ae-75-0.0528_encoder.h5,"
                                "./models/stae_3l_p0_gleveled_dim24-ae-301-0.0434_1_encoder.h5,"
                                "./models/stae_3l_p0_gleveled_dim24-ae-300-0.0811_2_encoder.h5",
                        help="Morphology encoders to use for the 'morphenc' feature extractor.")
    parser.add_argument("--morph_encoder_dimension", default="96,48,24",
                        help="Dimension of the morphology encoder used for the 'morphenc' feature extractor.")
    parser.add_argument("--morph_decoder_model",
                        default="./models/stae_sc0.00001_pca_p0.1_3l_d24-ae-208-0.0453_2_decoder.h5",
                        help="Morphology decoder to use for autoencoder reconstruction visualisations.")
    parser.add_argument("--morph_decoder_dimension", default="200",
                        help="Dimension of the morphology encoder used for the 'morphenc' feature extractor.")
    parser.add_argument("--learning_rate", default=0.001,
                        help="Learning rate used for training models.")
    parser.add_argument("--pca_params_file",
                        default="./models/pca.pickle",
                        help="PCA params to use for PCA transform of raw heart beats.")
    parser.add_argument("--tpca_params_file",
                        default="./models/kpca.pickle",
                        help="t-PCA params to use for t-PCA transform of raw heart beats.")
    parser.add_argument("--normalization_params_file",
                        default="./models/norm.npz",
                        help="Normalization params to use for feature blocks.")
    parser.add_argument("--inference", default="vi",
                        help="Which inference method to use for HMM training. One of (gibbs, meanfield, sgd).")
    parser.add_argument("--hmm_models",
                        default="./models/cont80_hmm_gibbs_nu64-rnn-50-1834514.1669_0.pickle,"
                                "./models/cont80_hmm_gibbs_nu64-rnn-80-1360198.9181_1.pickle,"
                                "./models/cont80_hmm_gibbs_nu64-rnn-80-1698033.8764_2.pickle,"
                                "./models/cont80_hmm_gibbs_nu64-rnn-10-556057.7259_3.pickle",
                        help="Hidden Markvo Models (HMMs) to use for class likelihood estimation.")
    parser.add_argument("--likelihood_normalization_params",
                        default="./models/logp_norm.npz",
                        help="Normalization params to use for HMM likelihoods.")
    parser.set_defaults(do_hyperopt=False)
    parser.add_argument("--do_hyperopt", dest='do_hyperopt', action='store_true',
                        help="Whether or not to optimise hyper parameters.")
    parser.set_defaults(cache_level1=False)
    parser.add_argument("--cache_level1", dest='cache_level1', action='store_true',
                        help="Whether or not to cache level 1 data.")
    parser.add_argument("--cache_file",
                        default="./models/data_cache.pickle",
                        help="Cache file to store data in.")
    parser.set_defaults(early_gc_disable=False)
    parser.add_argument("--early_gc_disable", dest='early_gc_disable', action='store_true',
                        help="Whether or not to disable the garbage collector at the start of the app.")

    return vars(parser.parse_args())


def initialise_app():
    args = parse_args()
    print("Arguments:", args, file=sys.stderr)

    # Fix random seed for reproducibility
    np.random.seed(args["seed"])

    # Higher recursion limit is required to be able to pickle keras models.
    sys.setrecursionlimit(20000)

    PipelineFeatureNormalizationFactory.load_params(args["normalization_params_file"])

    return args


def read_ensemble_json(file_name):
    with open(file_name) as f:
        return json.load(f)


def equal_class_weights(data_set):
    from sklearn.utils.class_weight import compute_class_weight

    classes, y = np.unique(np.argmax(data_set.y, axis=-1), return_inverse=True)
    return compute_class_weight("balanced", classes, y)


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    From https://github.com/fchollet/keras/issues/341

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)
