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
from scooch import Configurable
from scooch import ConfigList

# Local imports
from .features import SourceFeature
from .features import TransformFeature


class Analysis(Configurable):
    """
    A class encapsulating analysis of a single input file.
    """

    #
    # SCOOCH Configuration
    #
    _source_feature = Param(
        SourceFeature,
        doc="The feature used to decode the provided raw file data."
    )
    _feature_transforms = Param(
        ConfigList(TransformFeature),
        doc="Feature transformations to apply, in order, to the source feature generated from the input file."
    )

    #
    # Methods
    #
    def analyze(self, fname):
        """
        Analyze features for a single filepath.

        Args:
            fname: str - The filename path, from which to generate features.

        Return:
            mule.features.Feature - The feature resulting from the configured feature
            transformations.
        """
        self._source_feature.from_file(fname)
        input_feature = self._source_feature
        for feature in self._feature_transforms:
            feature.from_feature(input_feature)
            input_feature = feature

        return input_feature
