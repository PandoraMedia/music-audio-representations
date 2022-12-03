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


class SourceFile(Configurable):
    """
    Base class for SCOOCH configurable file readers.
    """

    def load(self, fname):
        """
        Any preprocessing steps to load a file prior to reading it.

        Args:
            fname: file-like - A file like object to be loaded.
        """
        raise NotImplementedError(f"The class, {self.__class__.__name__}, has no method for loading files")

    def read(self, n):
        """
        Reads an amount of data from the file.

        Args:
            n: int - A size parameter indicating the amount of data to read.

        Return:
            object - The decoded data read and in memory.
        """
        raise NotImplementedError(f"The class, {self.__class__.__name__}, has no method for reading files")

    def __len__(self):
        raise NotImplementedError(f"The class, {self.__class__.__name__} has no method for determining file data length")
        