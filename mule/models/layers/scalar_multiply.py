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
import tensorflow as tf

# Local imports
# None.


class ScalarMultiply(tf.keras.layers.Layer):
    """
    Implements a very simple layer that multiplies a tensor by a scalar. This scalar
    can be constant or learnable.
    """

    def __init__(self, init_gain, learnable=False, **kwargs):
        """
        **Constructor**

        Args:
            init_gain: <float> - The initial scalar gain that this layer will apply.

            learnable: <bool> - Whether to update the gain during training.
        """
        self._init_gain = init_gain
        self._learnable = learnable

        if self._learnable:
            self.gain = tf.Variable(self._init_gain, trainable=True)
        else:
            self.gain = tf.constant(self._init_gain)

        super(ScalarMultiply, self).__init__(**kwargs)

    def call(self, input):
        """
        Apply the layer to an input.

        Args:
            inputs: <tf.Tensor> - The input tensor to apply the scalar to.

        Return:
            <tf.Tensor> - The output tensor.
        """
        return self.gain*input

    def compute_output_shape(self, input_shape):
        """
        Get the output shape - identical to input.

        Args:
            input_shape: <tf.Tensor> - The input shape.

        Return:
            <tf.Tensor> - The output shape.
        """
        return input_shape.get_shape()

    def get_config(self):
        """
        Get the parameters of the layer for serialization, so that it can be reconstructed
        / deserialized.

        Return:
            <dict(str, object)> - A dictionary containing the properties required to
            reconstruct the object.
        """
        return {'init_gain': self._init_gain, 'learnable': self._learnable}

    @classmethod
    def from_config(cls, cfg):
        """
        Construct the object from previously serialized parameters (does not load weights).

        Args:
            cfg: <dict(str, object)> - The parameters from a serialized layer.

        Return:
            <ScalarMultiply> - The reconstructed ScalarMultiply layer.
        """
        return cls(**cfg)
