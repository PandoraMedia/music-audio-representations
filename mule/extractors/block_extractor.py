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
# None.

# Local imports
from .extractor import Extractor


class BlockExtractor(Extractor):
    """
    Extracts a single block of data from the provided feature, over the provided time range.
    """

    # 
    # Methods
    #
    def extract_range(self, feature, start_index, end_index):
        """
        Extract all feature indices over a given index range.

        Args:
            feature: mule.feature.Feature - A feature to extract data from.

            start_index: int - The first index (inclusive) at which to return data.

            end_index: int - The last index (exclusive) at which to return data.

        Return:
            numpy.ndarray - The extracted feature data. Features on first axis, time on
            second axis.
        """
        return feature.data[:, start_index:end_index]
