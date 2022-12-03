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


class StochDepth(tf.keras.layers.Layer):
    """
    Stochastic Depth layer. Implements Stochastic Depth as described in
    [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382), 
    to randomly drop residual branches in residual architectures.
    """

    def __init__(self, survival_probability: float = 0.5, scale_during_test: bool = False, **kwargs):
        """
        **Constructor**

        Args:
            survival_probability: <float> - The probability of the residual branch being kept.

            scale_during_test: <bool> - Whether to apply scaling to residual branch activations during test 
            time, after training, to compensate for the fact that branches are ranodmly dropped during training.
        """
        super().__init__(**kwargs)

        self.survival_probability = survival_probability
        self.scale_during_test = scale_during_test

    def call(self, x, training=None):
        """
        Apply the layer to the inputs, thereby combining the residual and skip layers via addition,
        with a random probability of dropping the resiudal component from the addition.

        Args:
            inputs: <list(tf.Tensor)> - A two element list containing a `shortcut` and `residual`
            tensor. Both tensors must be of the same shape.

            training: <tf.Tensor> - A boolean tensor defining whether it is train or test time. 

        Return:
            <tf.Tensor> - The tensor of the combined residual and skip connections.
        """
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("input must be a list of length 2.")

        shortcut, residual = x

        # Random bernoulli variable indicating whether the branch should be kept or not or not
        b_l = tf.keras.backend.random_bernoulli(
            tf.shape(x[1])[:1], p=self.survival_probability, dtype=self._compute_dtype_object
        )
        # NOTE [matt.c.mccallum 03.30.22]: Assume that this is being used on a 2D convolutional layer here.
        b_l = b_l[:, tf.newaxis, tf.newaxis, tf.newaxis]

        def _call_train():
            return shortcut + b_l * residual

        def _call_test_scale():
            return shortcut + self.survival_probability * residual

        def _call_test_no_scale():
            return shortcut + residual

        if self.scale_during_test:
            return tf.keras.backend.in_train_phase(
                _call_train, _call_test_scale, training=training
            )
        else:
            return tf.keras.backend.in_train_phase(
                _call_train, _call_test_no_scale, training=training
            )

    def compute_output_shape(self, input_shapes):
        """
        The size of the output, identical to both input.

        Args:
            input_shapes: <list(tf.Tensor)> - The sizes of each of the inputs.

        Return:
            <tf.Tensor> - The size of the output.
        """
        return input_shapes[0]

    def get_config(self):
        """
        Get the model parameters so that it may be reconstructed after serialization.

        Return:
            <dict(str, obj)> - Just the parameters of the constructor which will be
            automatically applied at deserialization.
        """
        base_config = super().get_config()

        config = {
            "survival_probability": self.survival_probability,
            "scale_during_test": self.scale_during_test
        }

        return {**base_config, **config}
