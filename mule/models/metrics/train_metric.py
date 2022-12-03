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


# Local imports
# None.

# Third party imports
from scooch import Configurable

# Python standard library imports
# None.


class TrainMetric(Configurable):
    """
    A base class for keras or custom metrics.
    """

    def init(self):
        """
        Compile the metric function. Separated from constructor so that it may be called
        outside of the SCOOCH class heirarchy instantiation.
        """
        self.initialize_base()

