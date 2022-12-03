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
import tensorflow as tf

# Python standard library imports
# None.


class Loss(Configurable):
    """
    A base class for keras or custome loss functions.
    """

    def init(self):
        """
        Initialize the Tensorflow Loss class that this class inherits from. This is separated from the class's
        constructor, so that it can be initialized at any time, under the desired Tensorflow scopes (e.g., a
        multi-GPU distribution scope), rather than only at construction time.
        """
        self.initialize_base()

    @classmethod
    def from_config(cls, config):
        if 'cfg' not in config.keys():
            cfg = {cls._NAME: None}
        else:
            cfg = config.pop('cfg')
        inst = cls(cfg=cfg, **config)
        inst.init()
        return inst
