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
from .lr_schedule import LRSchedule
from .warmup_schedule import WarmupSchedule


# Augment all schedules with the Configurable class
sched_clsmembers = inspect.getmembers(
    sys.modules[tf.keras.optimizers.schedules.__name__], inspect.isclass
)
for mem in sched_clsmembers:
    if mem[0] != "LearningRateSchedule":
        tf_class = configurize(mem[1], LRSchedule, init_base_on_construction=False)
        exec(mem[0] + ' = tf_class')
