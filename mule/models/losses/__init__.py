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
import inspect
import sys

# Third party imports
import tensorflow as tf
from scooch import configurize

# Local imports
from .loss import Loss
from .sim_clr_loss import SimClrLoss


# Augment all Tensorflow loss functions with the Configurable class
clsmembers = inspect.getmembers(sys.modules[tf.keras.losses.__name__], inspect.isclass)
for mem in clsmembers:
    if mem[0] != 'Loss':
        tf_class = configurize(mem[1], Loss, init_base_on_construction=False)
        exec(mem[0] + ' = tf_class')
        