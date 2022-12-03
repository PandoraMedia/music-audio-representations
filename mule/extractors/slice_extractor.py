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
import numpy as np

# Local imports
from .extractor import Extractor


class SliceExtractor(Extractor):
    """
    An extractor that extracts slices of feature data each surrounding an
    index in the provided input feature.
    """

    #
    # SCOOCH Configuration
    #
    _hop = Param(
        int,
        default=150,
        doc="How often to extract a slice of data from the input when extracting over a time range, in indices."
    )
    _look_forward = Param(
        int,
        default=150,
        doc="How many indices ahead of the index to be extracted from, to include in a given slice."
    )
    _look_backward = Param(
        int,
        default=150,
        doc="How many indices behind the index to be extract from, to include in a given slice"
    )
    _standard_normalize = Param(
        bool,
        default=True,
        doc="Whether to standard normalize each slice of data that extracted from the input feature."
    )

    # 
    # Methods
    #
    def extract_range(self, feature, start_index, end_index):
        """
        Extracts data over a given index range from a single feature. This will
        extract regularly spaced 2D slices of data from the input feature. Note
        that first feature will be extracted at an integer multiple of the extractor's
        configured `hop` parameter from the beginning of the feature.

        Args:
            feature: mule.feature.Feature - A feature to extract data from.

            start_index: int - The first index (inclusive) at which to start extracting
            slices.

            end_index: int - The last index (exclusive) at which to return data.

        Return:
            numpy.ndarray - The extracted feature data. Time on the first axis, features
            on the remaining axes.
        """
        indices = [time for time in range(0, end_index, self._hop) if time > start_index]
        features = [feature]*len(indices)
        if len(features)==0:
            return np.empty((0,0))
        return self.extract_batch(features, indices)
        
    def extract_batch(self, features, indices):
        """
        Extracts a batch of slices of data from a range of features, centered on specific 
        indices.

        Args:
            features: list(mule.features.Feature) - A list of features from which to extract 
            data from.

            indices: list(int) - A list of indices, the same size as `features`. Each element
            provides an index at which to extract data from the coressponding element in the
            `features` argument.

        Return:
            np.ndarray - A batch of features, with features on the batch dimension on the first
            axis and feature data on the remaining axes.
        """
        indices = [idx if idx >= self._look_backward else self._look_backward for idx in indices]
        indices = [idx if idx <= len(feat) - self._look_forward else len(feat) - self._look_forward for idx, feat in zip(indices, features)]
        samples = [feature.data[:, (idx-self._look_backward):(idx+self._look_forward)] for idx, feature in zip(indices, features)]
        samples = [x.reshape((1, *x.shape, 1)) for x in samples] # Add batch and channel dimensions
        samples = np.vstack(samples)

        if self._standard_normalize:
            samples -= np.mean(samples, axis=(1,2,3), keepdims=True)
            all_vars = np.std(samples, axis=(1,2,3), keepdims=True)
            all_vars = np.nan_to_num(all_vars, nan=1.0, posinf=1.0, neginf=1.0)
            all_vars = np.maximum(all_vars, 0.01)
            samples /= all_vars 

        return samples
