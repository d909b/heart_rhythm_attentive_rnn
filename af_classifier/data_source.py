"""
data_source.py - Utility methods for working with physionet data sets.

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
import os
import hashlib
import numpy as np
from scipy import io
from pandas import read_csv
from keras.utils.np_utils import to_categorical
from af_classifier.feature_extractor import FeatureVectorExtractor
from af_classifier.feature_extraction_cache import FeatureExtractionCache


class DataSource(object):
    def __init__(self, x=np.asarray([]), y=np.asarray([])):
        self.x = x
        self.y = y

    def copy(self):
        return self.__class__(np.copy(self.x), np.copy(self.y))

    def load(self, directory):
        pass

    def normalize(self, mean=None, stddev=None, min=None, max=None):
        if isinstance(self.x, np.ndarray) and self.x.dtype != np.dtype('object'):
            x_base = self.x
        else:
            x_base = np.hstack(self.x) if self.x[0].ndim == 1 else np.vstack(self.x)

        if mean is None:
            mean = np.mean(x_base, axis=0)

        if stddev is None:
            stddev = np.std(x_base, axis=0)

        if isinstance(self.x, list) or \
           (isinstance(self.x, np.ndarray) and self.x.dtype == np.dtype('object')):
            self.x = [(row - mean) / stddev for row in self.x]
        else:
            self.x = (self.x - mean) / stddev

        return mean, stddev, 0, 0

    @staticmethod
    def denormalize(data, mean, stddev, min, max):
        return data * stddev + mean

    def create_bag(self, fraction=1/3.):
        """
        With (intentional) duplicates.
        """
        idx = np.random.randint(low=0, high=len(self.x)-1, size=int(len(self.x)*fraction))
        if isinstance(self.x, np.ndarray):
            x_new = np.take(self.x, idx, axis=0)
        else:
            x_new = [self.x[i] for i in idx]

        y_new = np.take(self.y, idx, axis=0)

        return self.__class__(x_new, y_new)

    def create_bags(self, num_bags=3):
        bags = []
        for i in range(num_bags):
            bags.append(self.create_bag(1./num_bags))

        return bags

    def randomized_split(self, fraction=0.8, permutation_indices=None, normalize=False):
        fraction = max(min(fraction, 1), 0)
        num_rows = len(self.x)

        if permutation_indices is None:
            new_idx = np.random.permutation(num_rows)
        else:
            new_idx = permutation_indices

        # Randomly shuffle the data set before splitting.
        if isinstance(self.x, np.ndarray):
            new_x = np.take(self.x, new_idx, axis=0)
        else:
            new_x = [self.x[idx] for idx in new_idx]

        new_y = np.take(self.y, new_idx, axis=0)

        cut_idx = int(num_rows * fraction)
        x_train = new_x[:cut_idx]
        x_test = new_x[cut_idx:]
        y_train = new_y[:cut_idx]
        y_test = new_y[cut_idx:]
        train_set = self.__class__(x_train, y_train)
        test_set = self.__class__(x_test, y_test)

        if normalize:
            normalization_params = train_set.normalize()
            test_set.normalize(*normalization_params)

        return train_set, test_set, new_idx

    def to_categorical(self, num_classes):
        self.y = to_categorical(self.y, num_classes)

    def to_indexed(self):
        self.y = np.argmax(self.y, axis=-1)

    def to_one_vs_k(self, selected_class):
        indices = np.where(np.argmax(self.y, axis=1) == int(selected_class))[0]
        # 0 is the selected class, 1 is all others.
        new_y = np.asarray([1] * len(self.y))
        new_y[indices] = 0
        return self.__class__(np.copy(self.x), to_categorical(new_y, num_classes=2))

    def get_num_classes(self):
        return self.y.shape[-1]

    def get_num_samples_in_class(self, selected_class):
        if self.y.ndim == 1:
            return 0
        return np.sum(np.argmax(self.y, axis=1) == int(selected_class))

    def get_class_balance(self):
        num_classes = self.get_num_classes()
        balance = np.zeros(num_classes)
        for i in range(num_classes):
            balance[i] = self.get_num_samples_in_class(i)
        return balance

    def set_class_balance(self, balance):
        assert len(balance) == self.get_num_classes()

        balanced_classes = [self.take_from_class(i, balance[i], with_y=True) for i in range(len(balance))]
        balanced_x = map(lambda x: x[0], balanced_classes)
        balanced_y = map(lambda x: x[1], balanced_classes)

        if isinstance(self.x, np.ndarray):
            self.x = np.concatenate(balanced_x, axis=0)
        else:
            self.x = sum(balanced_x, [])

        self.y = np.concatenate(balanced_y, axis=0)

    def take_from_class(self, selected_class, num_samples=None, with_y=None):
        """
        If __num_samples__ is bigger than the number of samples available, duplicate samples will be provided.
        """
        indices = np.where(np.argmax(self.y, axis=1) == int(selected_class))[0]
        if num_samples is not None:
            num_samples = int(num_samples)
            if num_samples <= len(indices):
                indices = indices[:num_samples]
            else:
                num_duplicates = num_samples - len(indices)
                # Add duplicates to fill the remaining number of samples.
                dup_indices = np.random.randint(0, len(indices), size=num_duplicates)
                # Shuffle the indices.
                indices = np.random.permutation(np.hstack((indices, np.take(indices, dup_indices))))

        if isinstance(self.x, np.ndarray):
            ret_val = self.x[indices.astype(int)]
        else:
            ret_val = [self.x[i] for i in indices.astype(int)]

        if with_y is None:
            return ret_val
        else:
            return ret_val, self.y[indices.astype(int)]

    def limit_size(self, size):
        self.x = self.x[:size]
        self.y = self.y[:size]
        return self

    def single_class_data_set(self, selected_class):
        indices = np.where(np.argmax(self.y, axis=1) == int(selected_class))[0]
        if isinstance(self.x, np.ndarray):
            x_vals = np.copy(self.x[indices])
        else:
            x_vals = [self.x[idx] for idx in indices]
        return self.__class__(x_vals, np.copy(self.y[indices]))

    def extract(self, extractors):
        cached_data_set = FeatureExtractionCache.get_from_cache(self, extractors)
        if cached_data_set is None:
            extractor = FeatureVectorExtractor(extractors)
            return_value = self.__class__(extractor.extract_all(self.x), np.copy(self.y))
            FeatureExtractionCache.cache_feature(self, extractors, return_value)
            return return_value
        else:
            return cached_data_set

    def ensure_no_leftover_batches(self, batch_size):
        leftover = len(self.y) % batch_size
        if leftover != 0:
            self.y = self.y[leftover:]
            self.x = self.x[leftover:]
            if len(self.y) == 0:
                print("ERROR: Reduced data set size to 0.")

    @staticmethod
    def merge_arrays(a1, a2):
        if isinstance(a1, np.ndarray) and \
           isinstance(a2, np.ndarray) and \
           a1.ndim == a2.ndim:
            return np.concatenate((a1, a2), axis=0)
        else:
            return a1 + a2

    def merged(self, other_data_set):
        return self.__class__(DataSource.merge_arrays(self.x, other_data_set.x),
                              DataSource.merge_arrays(self.y, other_data_set.y))

    def __hash__(self):
        def hash_np_array(array):
            if isinstance(array, np.ndarray):
                hash_base = array.view(np.uint8)
            else:
                hash_base = array
            return int(hashlib.sha1(hash_base).hexdigest(), 16)

        def hash_list_of_np_arrays(list_of_arrays):
            if len(list_of_arrays) == 0:
                x_hashed = 0
            else:
                max_hash_depth = 100
                x_hashed = hash_np_array(list_of_arrays[0]) ^ len(list_of_arrays)
                for i in range(1, min(len(list_of_arrays), max_hash_depth)):
                    x_hashed ^= hash_np_array(list_of_arrays[i])
            return x_hashed

        def hash_array(array):
            return hash_list_of_np_arrays(array)

        return hash_array(self.x) ^ hash_array(self.y)

    def __eq__(self, other):
        return np.array_equal(self.x, other.x) and np.array_equal(self.y, other.y)

    def __ne__(self, other):
        return not (self == other)


class PhysioNet2017DataSource(DataSource):
    MAPPING_NAME_NORMAL = "N"
    MAPPING_NAME_AF = "A"
    MAPPING_NAME_OTHER = "O"
    MAPPING_NAME_NOISY = "~"

    MAPPING_CODE_NORMAL = 0
    MAPPING_CODE_AF = 1
    MAPPING_CODE_OTHER = 2
    MAPPING_CODE_NOISY = 3

    # Define a mapping from class names to integer indices.
    NAME_MAP = {
        MAPPING_NAME_NORMAL: MAPPING_CODE_NORMAL,
        MAPPING_NAME_AF: MAPPING_CODE_AF,
        MAPPING_NAME_OTHER: MAPPING_CODE_OTHER,
        MAPPING_NAME_NOISY: MAPPING_CODE_NOISY
    }

    # Define a mapping from integer codes to class names.
    CODE_MAP = {
        MAPPING_CODE_NORMAL: MAPPING_NAME_NORMAL,
        MAPPING_CODE_AF: MAPPING_NAME_AF,
        MAPPING_CODE_OTHER: MAPPING_NAME_OTHER,
        MAPPING_CODE_NOISY: MAPPING_NAME_NOISY
    }

    ECG_GAIN = 1000.0

    def __init__(self, x=np.asarray([]), y=np.asarray([])):
        super(PhysioNet2017DataSource, self).__init__(x, y)

    def load(self, directory):
        file_list, self.y = PhysioNet2017DataSource.read_reference(directory)
        self.x = PhysioNet2017DataSource.read_all_in_list(directory, file_list)

    def load_single(self, file_path):
        self.y = np.array([0])
        self.x = [PhysioNet2017DataSource.read_single(file_path)]

    @staticmethod
    def read_single(file_name):
        return io.loadmat(file_name)['val'][0] / PhysioNet2017DataSource.ECG_GAIN

    @staticmethod
    def read_all_in_list(directory, file_list):
        sequence_data = []
        for file_name in file_list:
            file_path = os.path.join(directory, file_name + ".mat")
            raw_data = PhysioNet2017DataSource.read_single(file_path)
            sequence_data.append(raw_data)

        return sequence_data

    @staticmethod
    def read_reference(dir_name):
        # Define a converter mapping the class name column to the correct integer indices.
        # NOTE: Any class not in the __mapping__ is indexed with __MAPPING_CODE_OTHER__
        converters = {1: lambda val: PhysioNet2017DataSource.NAME_MAP[val[0]]}

        data = read_csv(
            os.path.join(dir_name, "REFERENCE.csv"),
            header=None,
            skipinitialspace=True,
            delimiter=",",
            lineterminator="\n",
            encoding="utf8",
            converters=converters,
            engine="c"
        )

        return data[0].values, data[1].values
