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
import warnings

# Local imports
# None.


class WeightStandardization(tf.keras.layers.Wrapper):
    """
    Applies weight scaled normalization [1] to a convolutional or dense layer, which is very similar
    to centered (zero-mean) weight-norm layers, however, has no learnable magnitude, forcing unit
    variance and zero mean in the weights.

    This code is largely inspied by the WeightNorm layer found in the tensorflow_addons package:

        https://www.tensorflow.org/addons/tutorials/layers_weightnormalization

    and that in the tensorflow probability package:

        https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm

    Example:

        ```python
        net = WeightStandardization(tf.keras.layers.Conv2D(2, 2, activation='relu'), input_shape=(32, 32, 3))(x)
        net = WeightStandardization(tf.keras.layers.Dense(120, activation='relu'))(net)
        net = WeightStandardization(tf.keras.layers.Dense(num_classes))(net)
        ```

    References:
    
        [1] Qiao, Siyuan, Huiyu Wang, Chenxi Liu, Wei Shen, and Alan Yuille. "Micro-batch training with batch-channel 
        normalization and weight standardization." arXiv preprint arXiv:1903.10520 (2019).
    """

    def __init__(self, layer, **kwargs):
        """
        **Constructor**
        
        Args:

          layer: <tf.keras.layers.Layer> - Supported layer types are `Dense`, `Conv2D`, and 
          `Conv2DTranspose`. Layers with multiple inputs are not supported.

          **kwargs: <dict(str, object)> Additional keyword args passed to `tf.keras.layers.Wrapper`.
        """
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a `tf.keras.layers.Layer` '
                'instance. You passed: {input}'.format(input=layer))

        layer_type = type(layer).__name__
        # TODO [matt.c.mccallum 03.12.22]: Support Conv2DTranspose layers, by supporting channels
        #      on the first dimension.
        if layer_type not in ['Dense', 'Conv2D']:
            warnings.warn('`WeightNorm` is tested only for `Dense`, and `Conv2D` layers.'
                          'You passed a layer of type `{}`'
                          .format(layer_type))

        super(WeightStandardization, self).__init__(layer, **kwargs)

        self.filter_axis = -1

        self._track_trackable(layer, name='layer')

    def _compute_weights(self):
        """
        Generate weights with standardization.
        """
        
        weight = self.v

        mean = tf.math.reduce_mean(weight, axis=(0, 1, 2), keepdims=True)
        var = tf.math.reduce_variance(weight, axis=(0, 1, 2), keepdims=True)
        fan_in = tf.cast(tf.math.reduce_prod(weight.get_shape()[:-1]), tf.float32) # Batch dimension?

        # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
        eps = tf.constant(0.0001)
        scale = tf.math.rsqrt(tf.math.maximum(var * fan_in, eps)) * self.gain
        shift = mean * scale
        self.layer.kernel = weight * scale - shift

    def build(self, input_shape=None):
        """
        Build `Layer`.

        Args:

          input_shape: The shape of the input to `self.layer`.

        Raises:

          ValueError: If `Layer` does not contain a `kernel` of weights
        """

        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[0] = None
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('`WeightStandardization` must wrap a layer that'
                                ' contains a `kernel` for weights')

            kernel_norm_axes = list(range(self.layer.kernel.shape.rank))
            kernel_norm_axes.pop(self.filter_axis)
            # Convert `kernel_norm_axes` from a list to a constant Tensor to allow
            # TF checkpoint saving.
            self.kernel_norm_axes = tf.constant(kernel_norm_axes)

            # to avoid a duplicate `kernel` variable after `build` is called
            self.v = self.layer.kernel
            self.layer.kernel = None

            # Get gain
            # NOTE [matt.c.mccallum 03.12.22]: No need to broadcast this, it should
            #      match the var dimensions
            self.gain = tf.Variable(initial_value=tf.ones(self.v.get_shape()[-1], dtype=self.v.dtype), trainable=True)

        super(WeightStandardization, self).build()

    def call(self, inputs):
        """
        Call `Layer`.
        
        Args:
            inputs: <tf.Tensor> - The input tensor to apply the weight standardized layer to.

        Return:
            <tf.Tensor> - The tensor at the output of the layer.
        """
        self._compute_weights()
        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape after the application of `Layer`.

        Args:
            input_shape: <tf.Tensor> - The shape of the input to this layer.

        Return:
            <tf.Tensor> - The shape after the application of this layer.
        """
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
