"""
feature_extraction_cache.py - Cache to reload already extracted features.

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


class FeatureExtractionCache(object):
    CACHE = {}

    @staticmethod
    def cache_feature(data_set, extractors, feature_set):
        if FeatureExtractionCache.CACHE.get(data_set, None) is None:
            FeatureExtractionCache.CACHE[data_set] = {repr(extractors): feature_set.copy()}
        else:
            FeatureExtractionCache.CACHE[data_set][repr(extractors)] = feature_set.copy()

    @staticmethod
    def get_from_cache(data_set, extractors):
        data_set_cache = FeatureExtractionCache.CACHE.get(data_set, None)
        if data_set_cache is not None:
            feature_set = data_set_cache.get(repr(extractors), None)
            if feature_set is not None:
                return feature_set.copy()
        return None
