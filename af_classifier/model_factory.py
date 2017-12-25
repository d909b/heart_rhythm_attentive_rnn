"""
model_factory.py - Factory for loading pre-trained models.

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
import keras.backend as K
from os.path import splitext
from keras import initializers
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.engine.topology import Layer


if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


def bounded_relu(x):
    # Bounded to prevent NaNs.
    return K.relu(x, max_value=1)


class AttentionWithContext(Layer):
    """
    Temporal attention layer for recurrent neural networks.

    Following the method from:
    Yang, Z., Yang, D., Dyer, C., He, X., Smola, A. J., & Hovy, E. H. (2016).
    Hierarchical Attention Networks for Document Classification. In HLT-NAACL (pp. 1480-1489).
    """

    def __init__(self, bias=True, **kwargs):
        self.supports_masking = True

        initializer_type = "glorot_uniform"
        self.initial_weights_w = initializers.get(initializer_type)
        self.initial_weights_u = initializers.get(initializer_type)
        self.with_bias = bias
        self.w = None
        self.u = None
        self.b = None

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.initial_weights_w,
                                 name='{}_w'.format(self.name))
        if self.with_bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name))

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.initial_weights_u,
                                 name='{}_u'.format(self.name))

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None  # Any masking is removed at this layer.

    def call(self, input, input_mask=None):
        u_it = K.dot(input, self.w)
        if self.with_bias:
            u_it += self.b

        attention_weights = K.exp(K.dot(K.tanh(u_it), self.u))

        if input_mask is not None:
            attention_weights *= K.cast(input_mask, K.floatx())

        # Add a small value to avoid division by zero if sum of weights is very small.
        attention_weights /= K.cast(K.sum(attention_weights, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attention_weights = K.expand_dims(attention_weights)

        # Weight initial input by attention.
        weighted_input = input * attention_weights

        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        # The temporal dimension is collapsed via attention.
        return input_shape[0], input_shape[-1]

    def compute_output_shape(self, input_shape):
        return self.get_output_shape_for(input_shape)


class ModelFactory(object):
    LOADED_MODELS = {}

    @staticmethod
    def get_model(file_name):
        # Disable GC for calls to pickle to increase de-serialisation speed.
        before = gc.isenabled()
        gc.disable()

        # Check cache first.
        model = ModelFactory.get_from_cache(file_name)
        if model is not None:
            print("INFO: Loading ", file_name, " from cache.", file=sys.stderr)
            return model

        print("INFO: Loading ", file_name, file=sys.stderr)
        root, ext = splitext(file_name)

        if ext == ".h5":
            # Any custom objects _must_ be in scope for loading.
            with CustomObjectScope({"<lambda>": bounded_relu,
                                   "AttentionWithContext": AttentionWithContext}):
                model = load_model(file_name)
        elif ext == ".pickle":
            model = pickle.load(open(file_name, "r"))
        else:
            raise NameError("Model must be either stored in .h5 or .pickle format. Aborting.")

        # Cache newly loaded models.
        ModelFactory.cache_model(file_name, model)

        if before:
            gc.enable()

        return model

    @staticmethod
    def cache_model(file_name, model):
        ModelFactory.LOADED_MODELS[file_name] = model

    @staticmethod
    def get_from_cache(file_name):
        return ModelFactory.LOADED_MODELS.get(file_name)

    @staticmethod
    def save_all_models_precompiled():
        for name, model in ModelFactory.LOADED_MODELS.items():
            pickle.dump(model, open(name + ".pickle", "w"), pickle.HIGHEST_PROTOCOL)
