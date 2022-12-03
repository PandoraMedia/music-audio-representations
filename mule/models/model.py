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
from abc import abstractmethod

# Third party imports
from scooch import Configurable
from scooch import ConfigList
from scooch import Param
import tensorflow as tf

# Local imports
from .losses import Loss
from .optimizers import Optimizer
from .metrics import TrainMetric


class Model(Configurable):
    """
    A class defining the architecture of a model along with the training parameters such as loss function and optimizer.
    """

    # 
    # SCOOCH Configuration
    #
    _optimizer = Param(
        Optimizer,
        doc="A configuration for a Tensorflow optimizer to use for model training."
    )
    _loss = Param(
        Loss,
        default=None,
        doc="A SCOOCH configuration / dictionary specifying the loss function used during training."
    )
    _metrics = Param(
        ConfigList(TrainMetric),
        doc=" - A list of metric configurations to be compiled into the model, to observe whilst training."
    )

    _MODEL_CLASS = tf.keras.Model

    # 
    # Methods
    #
    def __init__(self, cfg):
        """
        **Constructor:**

        Args:
            cfg - A SCOOCH Config object defining the configuration parameters for this class.
        """
        super().__init__(cfg)
        self._model = None
        self._input_shapes = None
        self._output_shapes = None

    def init(self, input_shapes, output_shapes):
        """
        Peform any actions immediately after the model is defined. For example, specify IO shapes which
        may affect model architecture, define the loss and compile the model.

        Args:
            input_shapes: <list(list(list(int)))> - A list containing lists of dimensions of all input
            tensors.

            output_shapes: <list(list(list(int)))> - A list containing lists of dimensions of all output
            tensors.
        """
        self._input_shapes = input_shapes
        self._output_shapes = output_shapes

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():

            self._model = self._make_model()

            # Initialize these here, now that we're in the distribution strategy scope
            self._loss.init()
            self._optimizer.init()
            if self._metrics is not None:
                for mt in self._metrics:
                    if isinstance(mt, TrainMetric):
                        mt.init()

            # Compile the model
            self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
    
    @abstractmethod
    def _make_model(self):
        """
        The method where the actual model definition occurs. This is to be overridden by each of the
        model classes, defining what the model architecture is.

        Return:
            <tf.keras.Model> - The model instance that this class configurizes and encapsulates, for training,
            inference, or otherwise.
        """
        raise NotImplementedError(f'This model, {self.__class__.__name__}, has not been implemented')

    @property
    def model(self):
        """
        Type: <tf.keras.Model>
            The keras model instance that is encapsulated within this configurable model wrapper.
        """
        return self._model
