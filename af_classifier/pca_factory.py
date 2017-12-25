"""
pca_factory.py - Provides a factory for pre-computed principal component analysis (PCA) transformations.

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
from sklearn.decomposition import PCA

if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class PCAFactory(object):
    LOADED_PCAS = {}

    @staticmethod
    def from_tuple(tuple):
        # Create PCA object
        components, explained_variance, mean, whiten = tuple
        pca = PCA(whiten=whiten)
        pca.components_ = components
        pca.explained_variance_ = explained_variance
        pca.mean_ = mean
        return pca

    @staticmethod
    def get(file_name):
        # Check cache first.
        tuple = PCAFactory.get_from_cache(file_name)
        if tuple is not None:
            print("INFO: Loading ", file_name, " from cache.", file=sys.stderr)
            return PCAFactory.from_tuple(tuple)

        print("INFO: Loading ", file_name, file=sys.stderr)
        tuple = pickle.load(open(file_name, "r"))

        # Cache newly loaded models.
        PCAFactory.cache(file_name, tuple)

        return PCAFactory.from_tuple(tuple)

    @staticmethod
    def cache(file_name, tuple):
        PCAFactory.LOADED_PCAS[file_name] = tuple

    @staticmethod
    def get_from_cache(file_name):
        return PCAFactory.LOADED_PCAS.get(file_name)

    @staticmethod
    def save_all_models_precompiled():
        for name, tuple in PCAFactory.LOADED_PCAS.items():
            pickle.dump(tuple, open(name, "w"), pickle.HIGHEST_PROTOCOL)
