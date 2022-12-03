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
import tempfile

# Third party imports
from scooch import Configurable
import numpy as np

# Local imports
# None.


class Feature(Configurable):
    """
    The base class for all feature types.
    """

    def __del__(self):
        """
        **Destructor**
        """
        # NOTE [matt.c.mccallum 09.20.22]: This destructor is not guaranteed to be called on interpreter exit..
        if hasattr(self, '_data_file'):
            self._data_file.close()

    def add_data(self, data):
        """
        Adds data to extend the object's current data via concatenation along the time axis.
        This is useful for populating data in chunks, where populating it all at once would
        cause excessive memory usage.

        Arg:
            data: np.ndarray - The data to be appended to the object's memmapped numpy array.
        """
        if not hasattr(self, '_data_file'):
            self._data_file = tempfile.NamedTemporaryFile(mode='w+b')

        if not hasattr(self, '_data') or self._data is None:
            original_data_size = 0
        else:
            original_data_size = self._data.shape[1]
        final_size = original_data_size + data.shape[1]

        filename = self._data_file.name

        # NOTE [matt.c.mccallum 06.22.20]: By using the np.memmap method instead of numpy.lib.format.open_memmap, 
        #                                  we can incrementally change the size of the array on disk, but we aren't
        #                                  able to save the shape and data type metadata.
        self._data = np.memmap(filename, dtype='float32', mode='r+',shape=(data.shape[0],final_size), order='F')
        self._data[:,original_data_size:] = data

    def _extract(self, source, length):
        """
        Extracts feature data file or other feature a given time-chunk.

        Args:
            source_feature: mule.features.Feature - The feature to transform.

            start_time: int - The index in the input at which to start extracting / transforming.

            chunk_size: int - The length of the chunk following the `start_time` to extract
            the feature from.
        """
        raise NotImplementedError(f"The {self.__name__.__class__} has no feature extraction method")

    def save(self, path):
        """
        Save the feature data blob to disk.

        Args:
            path: str - The path to save the data to.
        """
        np.save(path, self._data)

    @property
    def data(self):
        """
        The feature data blob itself.
        """
        return self._data
