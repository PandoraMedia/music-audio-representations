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
from scooch import Configurable

# Local imports
# None.


class Extractor(Configurable):
    """
    Base class for classes that are responsible for extracting data
    from mule.features.Feature classes.
    """

    # 
    # Methods
    #
    def extract_range(self, feature, start_index, end_index):
        """
        Extracts data over a given index range from a single feature.

        Args:
            feature: mule.feature.Feature - A feature to extract data from.

            start_index: int - The first index (inclusive) at which to return data.

            end_index: int - The last index (exclusive) at which to return data.

        Return:
            numpy.ndarray - The extracted feature data. Features on first axis, time on
            second axis.
        """
        raise NotImplementedError(f"The {self.__class__.__name__} class has no `extract_range` method.")

    def extract_batch(self, features, indices):
        """
        Extracts a batch of features from potentially multiple features, each potentially
        at distinct indices.

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
        raise NotImplementedError(f"The {self.__class__.__name__} class has no `extract_batch` method.")
