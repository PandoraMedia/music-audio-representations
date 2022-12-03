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
from scooch import Param

# Local imports
from ...extractors import Extractor
from ..feature import Feature


class TransformFeature(Feature):
    """
    Base class for all features that are transforms of other features.
    """

    #
    # SCOOCH Configuration
    #
    _extractor = Param(
        Extractor,
        doc="An object defining how data will be extracted from the input feature and provided to the transformation of this feature."
    )

    # The size in time of each chunk that this feature will process at any one time.
    _CHUNK_SIZE = 44100*60*15

    #
    # Methods
    #
    def from_feature(self, source_feature):
        """
        Populates this features data as a transform of the provided input feature.

        Args:
            source_feature: mule.features.Feature - A feature from which this feature will
            be created as a transformation thereof.
        """
        boundaries = list(range(0, len(source_feature), self._CHUNK_SIZE)) + [len(source_feature)]
        chunks = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
        for start_time, end_time in chunks:
            data = self._extract(source_feature, start_time, end_time-start_time)
            if data is not None and len(data):
                self.add_data(data)

    def _extract(self, source_feature, start_time, chunk_size):
        """
        Extracts feature data as a transformation of a given source feature for a given
        time-chunk.

        Args:
            source_feature: mule.features.Feature - The feature to transform.

            start_time: int - The index in the feature at which to start extracting / transforming.

            chunk_size: int - The length of the chunk following the `start_time` to extract
            the feature from.
        """
        end_time = start_time + chunk_size
        return self._extractor.extract_range(source_feature, start_time, end_time)

    def __len__(self):
        if hasattr(self, '_data'):
            return self._data.shape[1]
        else:
            return 0
        