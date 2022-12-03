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
from ...source_files import SourceFile

# Local imports
from .. import Feature


class SourceFeature(Feature):
    """
    A feature that is derived directly from raw data, e.g., a data file.
    """

    #
    # SCOOCH Configuration
    #
    _input_file = Param(
        SourceFile,
        doc="The file object defining the parameters of the raw data that this feature is constructed from."
    )

    _CHUNK_SIZE = 44100*60*15

    # 
    # Methods
    #
    def from_file(self, fname):
        """
        Takes a file and processes its data in chunks to form a feature.

        Args:
            fname: str - The path to the input file from which this feature is constructed.
        """
        # Load file
        self._input_file.load(fname)
        
        # Read samples into data
        processed_input_frames = 0
        while processed_input_frames<len(self._input_file):
            data = self._extract(self._input_file, processed_input_frames, self._CHUNK_SIZE)
            processed_input_frames += self._CHUNK_SIZE
            self.add_data(data)

    def __len__(self):
        """
        Returns the number of bytes / samples / indices in the input data file.
        """
        return len(self._input_file)
