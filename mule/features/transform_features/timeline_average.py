# coding=utf-8
# Copyright 2021 Pandora Media, LLC.
#
# Licensed under the GNU GPL License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Python standard library imports
# None.

# Third party imports
import numpy as np

# Local imports
from .transform_feature import TransformFeature


class TimelineAverage(TransformFeature):
    """
    A feature that averages over data segments extracted from the input feature.
    """

    _feature_count = 0

    #
    # Methods
    #
    def _extract(self, source_feature, start_time, chunk_size):
        """
        Extracts an average of feature segments from the input feature over the provided time range.

        Args:
            source_feature: mule.features.AudioWaveform - The feature to transform into a mel spectrogram

            start_time: int - The index in the feature at which to start extracting / transforming.

            chunk_size: int - The length of the chunk following the `start_time` to extract
            the feature from.
        """
        data = np.squeeze(super()._extract(source_feature, start_time, chunk_size))

        self._feature_count += data.shape[1]

        # TODO [matt.c.mccallum 12.15.22]: Handle overflow here for large features
        data = np.sum(data, axis=1, keepdims=True)

        return data

    def add_data(self, data):
        """
        Adds data to previously accumulated data via summation.

        Arg:
            data: np.ndarray - The data to be appended to the object's summation.
        """
        if not hasattr(self, '_data') or self._data is None:
            self._data = np.zeros(data.shape)

        self._data = self._data + data
        
    @property
    def data(self):
        """
        The average of feature segments seen so far.
        """
        return self._data/self._feature_count
