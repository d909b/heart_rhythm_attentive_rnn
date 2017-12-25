"""
hmm_model.py - Wrapper class for Hidden Markov models (HMMs).

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

import gc
import sys
import numpy as np

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class HMMModel(object):
    def __init__(self, models, normalising_factors):
        assert len(models) == len(normalising_factors)

        self.models = models
        self.normalising_factors = normalising_factors

    def normalise_likelihoods(self, l):
        return (l - self.normalising_factors[:, 0]) / (self.normalising_factors[:, 1] - self.normalising_factors[:, 0])

    def predict(self, x):
        def process_row(model):  # Currying the model variable.
            def process_row_with_model(row):
                # Always return to the same random state to make evaluation runs independent
                # of the number of log-likelihoods that have been computed.
                state = np.random.get_state()
                likelihood = model.log_likelihood(row)
                np.random.set_state(state)
                return likelihood
            return process_row_with_model

        # Must remove the zero padding that is added for RNN processing as it would bias the likelihoods.
        x_without_zero_padding = map(lambda s: s[~np.all(s == 0, axis=-1)], x)

        # Get likelihoods for each item and model.
        y = map(lambda model: map(process_row(model), x_without_zero_padding), self.models)
        y = np.asarray(y).T

        # Return normalised likelihoods in range [0,1].
        return self.normalise_likelihoods(y)


def load_hmm_models(models, overwrite_with_clean=False):
    if "," in models:
        model_list = models.split(",")
    else:
        model_list = [models]

    return map(lambda model: load_hmm_model(model, overwrite_with_clean), model_list)


def load_hmm_model(file_name, overwrite_with_clean=False):
    # Disable GC for calls to pickle to increase de-serialisation speed.
    before = gc.isenabled()
    gc.disable()

    print('INFO: Loading', file_name, file=sys.stderr)
    model = pickle.load(open(file_name, "r"))

    if overwrite_with_clean:
        model.states_list[:] = []
        pickle.dump(model, open(file_name, "w"), pickle.HIGHEST_PROTOCOL)

    if before:
        gc.enable()

    return model


def get_normalising_factors(models, data_set):
    factors = np.zeros((len(models), 2))
    for i, model in enumerate(models):
        # Get likelihoods for each time series in the data set.
        log_p = map(lambda row: model.log_likelihood(row), data_set.x)

        # Store the respective maximum and minimum as normalising factors.
        factors[i][0], factors[i][1] = np.min(log_p), np.max(log_p)
    return factors
