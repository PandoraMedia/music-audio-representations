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
from .model import Model

# Third party imports
import tensorflow as tf
from scooch import Param

# Python standard library imports
# None.


class MLP(Model):
    """
    A model where all layers form dense transformations interleaved with non-linearities.
    """

    #
    # SCOOCH Configuration
    #
    _layer_widths = Param(
        list,
        doc="The number of neurons in each and every layer of the dense net"
    )
    _activation = Param(
        str,
        default='relu',
        doc="A string specifying the Tensorflow activation function at the output of each and every neuron at each and every layer."
    )
    _l2_regularization = Param(
        float,
        default=None,
        doc="The parameter applied to l2 regularization of dense net weights. If None, no regularization will be applied."
    )
    _input_dropout = Param(
        float,
        default=None,
        doc="Dropout probability applied directly to the input features of the model (None for no dropout)"
    )
    _layer_dropouts = Param(
        list,
        default=None,
        doc="Dropout probability for a layer added after the activation is applied at each hidden layer (None, or the same size as `layer_widths`)"
    )
    _output_dropout = Param(
        float,
        default=None,
        doc="Dropout probability applied directly to the output of the model, after any transformations (None for no dropout)"
    )

    # 
    # Methods
    #
    def _make_model(self):
        """
        Defines a model with an arbitrary number of dense layers with an arbitrary number
        of neurons, and an arbitrary activation function.

        Return:
            tf.keras.Model - A trainable Keras model.
        """
        input = tf.keras.Input(shape=self._input_shapes[0][1:]) # NOTE [matt.c.mccallum 11.29.21]: Assume the first index is the input

        output_size = self._output_shapes[0][1]

        layers = []

        if self._input_dropout:
            layers += [tf.keras.layers.Dropout(self._input_dropout)]

        for idx, layer_width in enumerate(self._layer_widths):
            layers += [tf.keras.layers.Dense(layer_width, activation=self._activation, kernel_regularizer=self._get_regularizer())]
            if self._layer_dropouts and self._layer_dropouts[idx]:
                layers += [tf.keras.layers.Dropout(self._layer_dropouts[idx])]
                
        layers += [tf.keras.layers.Dense(output_size, activation='linear', kernel_regularizer=self._get_regularizer())]
        if self._output_dropout:
            layers += [tf.keras.layers.Dropout(self._output_dropout)]

        output = input
        for lyr in layers:
            output = lyr(output)

        return self._MODEL_CLASS([input], [output])

    def _get_regularizer(self):
        """
        Returns the regularizer according to the object's configuration.

        Return:
            <tf.keras.regularizers.Regularizer or None> - The regularizer to use in model layers.
        """
        if self._l2_regularization:
            return tf.keras.regularizers.L2(l2=self._l2_regularization)
        else:
            return None
