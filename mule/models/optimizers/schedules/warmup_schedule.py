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
# None

# Third party imports
import tensorflow as tf
from scooch import Param

# Local imports
from .lr_schedule import LRSchedule


class WarmupSchedule(LRSchedule, tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Adds a linear warmup to a learning rate schedule.
    """

    _cold_learning_rate = Param(
        float,
        default=0.0,
        doc="Learning rate to start the warmup with"
    )
    _warm_learning_rate = Param(
        float,
        doc="Desired learning rate after the warmup"
    )
    _warmup_steps = Param(
        int,
        doc="Number of warmup steps"
    )
    _schedule = Param(
        LRSchedule,
        doc="Original learning rate schedule"
    )
    _initial_step = Param(
        int,
        default=0,
        doc="The step to start the schedule out - higher integers will start training at steps later in the schedule."
    )

    def init(self):
        self._warmup = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self._cold_learning_rate,
            decay_steps=self._warmup_steps,
            end_learning_rate=self._warm_learning_rate,
        )
        self._schedule.init()

    def __call__(self, step):
        """
        Get the learning rate for a step.

        Args:
            step: TensorFlow Tensor - the current training step.
        """
        # NOTE [matt.c.mccallum 04.26.22]: Keras's model.fit() always passes 0 on the first step to the learning
        #      rate schedule, regardless of the `initial_epoch` argument. We add an offset here so that we can
        #      customize where the learning rate schedule begins.
        step = step + self._initial_step
        return tf.cond(
            step < self._warmup_steps,
            lambda: self._warmup(step),
            lambda: self._schedule(step - self._warmup_steps)
        )

    def get_config(self):
        return self.cfg

    @classmethod
    def from_config(cls, cfg):
        x = cls(cfg)
        x.init()
        return x
        