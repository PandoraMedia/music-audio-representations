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
import itertools

# Third party imports
import tensorflow as tf
import numpy as np
from scooch import Param

# Local imports
from .layers import WeightStandardization
from .layers import ScalarMultiply
from .layers import StochDepth
from .model import Model


def _scaled_activation(activation_name):
    """
    Apply a scaled activation function according to [1, 2].

    Args:
        activation_name: str - The name of the scaled activation function to apply.

        input: tf.Tensor - The tensor to apply the activation to.

    Return:
        tf.Tensor - The input with the activation applied.

    References:
    
        [1] Arpit, Devansh, Yingbo Zhou, Bhargava Kota, and Venu Govindaraju. "Normalization propagation: A 
            parametric technique for removing internal covariate shift in deep networks." In International 
            Conference on Machine Learning, pp. 1168-1176. PMLR, 2016.

        [2] Brock, A., De, S., & Smith, S. L. (2021). Characterizing signal propagation to close the performance 
            gap in unnormalized resnets. arXiv preprint arXiv:2101.08692.
    """
    activations = {
        'gelu': lambda x: tf.nn.gelu(x) * 1.7015043497085571,
        # NOTE [matt.c.mccallum 03.22.22]: The scalar below is taken directly from DeepMind reference code,
        #      it is similar but not equal to the value suggested in the literature of: np.sqrt(2.0/(1-(1/np.pi)))
        'relu': lambda x: tf.nn.relu(x) * 1.7139588594436646 
    }
    return activations[activation_name]


class SfNfNetF0(Model):
    """
    This model implements the SF-NFNet-F0 architecture described in [1].
    It consists of two series' of NFNet-F0 stages [2], one with high time
    resolution and low channel count, another with low time resolution and
    high channel count, based on the effectiveness of this for audio, previously
    demonstrated in [3].

    Much of this code (including the NFNet) blocks was adapted from the architecture
    written in Haiku by DeepMind:

        https://github.com/deepmind/deepmind-research/blob/1642ae3499c8d1135ec6fe620a68911091dd25ef/nfnets/nfnet.py

    Reference code for specfic NFNet blocks such as Squeeze and Excitation blocks, 
    Weight Standardized Convolutions and Stochastic Depth layers can be found here:

        https://github.com/deepmind/deepmind-research/blob/1642ae3499c8d1135ec6fe620a68911091dd25ef/nfnets/base.py

    References:

        [1] Wang, Luyu, Pauline Luc, Yan Wu, Adria Recasens, Lucas Smaira, Andrew Brock, 
        Andrew Jaegle et al. "Towards Learning Universal Audio Representations." arXiv 
        preprint arXiv:2111.12124 (2021).

        [2] Brock, Andy, Soham De, Samuel L. Smith, and Karen Simonyan. "High-performance 
        large-scale image recognition without normalization." In International Conference 
        on Machine Learning, pp. 1059-1071. PMLR, 2021.

        [3] Kazakos, Evangelos, Arsha Nagrani, Andrew Zisserman, and Dima Damen. 
        "Slow-fast auditory streams for audio recognition." In ICASSP 2021-2021 IEEE 
        International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
        pp. 855-859. IEEE, 2021.
    """

    # TODO [matt.c.mccallum 03.18.22]: Currently this network is a huge class with several private
    #      `_make_<module_name>_module` and `_apply_<module_name>_module` methods. Each of these pairs
    #      of layers forming modules should be should in-turn, also be layers.

    # 
    # SCOOCH Configuration
    #
    _projector_activation = Param(
        str,
        default='relu',
        doc="Activation function used in any projection layers prior to the network output."
    )
    _projector_layers = Param(
        list,
        doc=")> - Number of nuerons in each dense projector layers between the convolutional layers, and the output. Also defines the number of dense layers there. Can be an empty list."
    )
    _include_fc = Param(
        bool,
        default=True,
        doc="Include a dense layer between the convolutional layers, projection, and loss function"
    )
    _alpha = Param(
        float,
        default=0.2,
        doc="The variance scaling factor to be applied at the end of each residual connection"
    )
    _scaled_activation_type = Param(
        str,
        default='gelu',
        doc="The activation function to use when applying scaled activations"
    )
    _f_value = Param(
        int,
        default=0,
        doc="The 'F' value for the NFNet architecture. An F-Value > 0 multiplies the number of NFNet blocks in each stage by (F+1)"
    )

    # 
    # Methods
    #
    def _make_model(self):
        """
        Constructs the model object.

        Return:
            tf.keras.Model - A keras model object.
        """
        inputs = [tf.keras.Input(shape=self._input_shapes[0][1:])]

        self._make_layers(self._output_shapes[0][1])

        # Add activation for classification
        self._output_layers += [tf.keras.layers.Activation(tf.keras.activations.sigmoid)]

        outputs = [self._apply_layers(inp) for inp in inputs]

        model = self._MODEL_CLASS(inputs, outputs)

        return model

    def get_conv(self, channels, kernel, strides, groups, activation=None):
        """
        Creates the variance preserving weight standardized convolutional layer as described in [1].
        This is provided as a one or two element list, depending on whether an activation is provided.
        If an activation function is provided, it will be implemented as a separate tf.keras.layers.Activation
        layer.

        Args:

            channels: <int> - The number of channels at the output of the layer.

            kernel: <list(int)> - The dimensions of the 2D convolutional kernel.

            strides: <list(int)> - The convolutional stride across each direction in the 2D features.

            groups: <int> - The number of channel groups the input will be split into for a grouped 
            convolution operation.

            activation: <None or function or str> - The activation function applied after the convolution,
            This should be implemented according to the variance scaling rules specified in [2].

        References:

            [1] Brock, Andy, Soham De, Samuel L. Smith, and Karen Simonyan. "High-performance 
            large-scale image recognition without normalization." In International Conference 
            on Machine Learning, pp. 1059-1071. PMLR, 2021.

            [2] Brock, A., De, S., & Smith, S. L. (2021). Characterizing signal propagation to close the performance 
            gap in unnormalized resnets. arXiv preprint arXiv:2101.08692.
        """

        # NOTE [matt.c.mccallum 04.01.22]: The DeepMind reference code written in Haiku applies Variance scaling initialization
        #      to the unnormalized kernel, they then normalize the kernel. So the initial value of the kernel regardless is
        #      1.0 averaged across the fan_in dimension, regardless of the variance scaling initialization. The weight scaled 
        #      normalization we have implemented in Keras takes the same approach.

        layers = [
            WeightStandardization(tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=kernel,
                strides=strides,
                padding="same",
                groups=groups,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=1.0,
                    mode='fan_in',
                    distribution='untruncated_normal'
                )
            ))
        ]

        if activation is not None:
            return layers + [tf.keras.layers.Activation(activation)]
        else:
            return layers

    def _make_layers(self, output_shape):
        """
        Makes the layers in the model and saves them as Model class attributes,
        without applying the layers to any variables.
        """

        #
        # Make Stems
        #
        self._slow_stem = self._make_stem_module(
                kernels=[
                    [3, 1],
                    [3, 1],
                    [3, 1],
                    [3, 3]
                ],
                channels=[16, 32, 64, 128],
                strides=[
                    [2, 8], # Integrated data striding layer into first convolution
                    [1, 1],
                    [1, 1],
                    [2, 2]
                ]
            )

        self._fast_stem = self._make_stem_module(
                kernels=[
                    [3, 3],
                    [3, 3],
                    [3, 3],
                    [3, 3]
                ],
                channels=[2, 4, 8, 16],
                strides=[
                    [2, 2], # Integrated data striding layer into first convolution
                    [1, 1],
                    [1, 1],
                    [2, 2]
                ]
            )

        #
        # Initialize parameters for NFNet Blocks
        #
        self._nfnet_stage_depths = [x*(self._f_value+1) for x in (1,2,6,3)] # Computes the number of NFNet blocks in each stage based on the provided F-Value
        cumulative_stage_depths = np.concatenate(([0],np.cumsum(self._nfnet_stage_depths)))
        self._stoch_depth_survival_probs = 0.1*np.arange(cumulative_stage_depths[-1])/(cumulative_stage_depths[-1])
        self._stoch_depth_survival_probs = [
            self._stoch_depth_survival_probs[st:end] for st, end in zip(cumulative_stage_depths[:-1], cumulative_stage_depths[1:])
        ]
        self._stage_expected_vars = [1.0] + [(1.0+self._alpha**2)**0.5]*3 # In the reference code, an initial beta of 1.0 is applied at the first stage, and betas after a single update at the latter stages.
        self._stage_downsamples = [1] + [2]*3

        #
        # Construct NFNet stages for the slow path
        #
        slow_nfnet_kernels = [
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]]
        ]
        slow_nfnet_input_sizes = [256, 512, 1024, 2560]
        slow_nfnet_output_sizes = [256, 512, 1536, 1536]

        self._slow_layers = [
            self._make_nfnet_stage(
                kernels=k,
                freq_downsample=f,
                input_channels=i,
                output_channels=o, 
                group_size=128, 
                alpha=self._alpha, 
                input_expected_var=e,
                stoch_depths=s,
                num_blocks=n
            ) for k, f, i, o, e, s, n in zip(
                slow_nfnet_kernels,
                self._stage_downsamples,
                slow_nfnet_input_sizes,
                slow_nfnet_output_sizes,
                self._stage_expected_vars,
                self._stoch_depth_survival_probs,
                self._nfnet_stage_depths
            )
        ]

        #
        # Construct NFNet stages for the fast path
        #
        fast_nfnet_kernels = [
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]],
            [[1, 1],[1, 3],[3, 1],[1, 1]]
        ]
        fast_nfnet_input_sizes = [16, 32, 64, 192]
        fast_nfnet_output_sizes = [32, 64, 192, 192]

        self._fast_layers = [
            self._make_nfnet_stage(
                kernels=k,
                freq_downsample=f,
                input_channels=i,
                output_channels=o, 
                group_size=16, 
                alpha=self._alpha, 
                input_expected_var=e,
                stoch_depths=s,
                num_blocks=n
            ) for k, f, i, o, e, s, n in zip(
                fast_nfnet_kernels,
                self._stage_downsamples,
                fast_nfnet_input_sizes,
                fast_nfnet_output_sizes,
                self._stage_expected_vars,
                self._stoch_depth_survival_probs,
                self._nfnet_stage_depths
            )
        ]

        #
        # Construct fast-to-slow fusion layers
        #
        # NOTE [matt.c.mccallum 04.01.22]: Conversation with the authors of the SF-NFNet-F0 stated that a 1x1 conv layer
        #      in fast to slow fusion expands the feature dimension by [2,4,8,16] times. They also mentioned that the inputs
        #      to each of the NFNet stages had [256,512,1024,2560] channels. These two facts do not add up as the expansion by
        #      the stated factors causes the stages to have inputs of [256,512,1024,4608] channels. We go with the stated factors
        #      of [2,4,8,16] because it brings the model's total parameter count closer to the stated value in the paper (62M 
        #      parameters here, 63M parameters mentioned in the paper).
        self._fusion_layers = [
            self._make_fast_to_slow_fusion(time_kernel_length=7, time_stride=4, input_channels=32, output_channels=128),
            self._make_fast_to_slow_fusion(time_kernel_length=7, time_stride=4, input_channels=32, output_channels=256),
            self._make_fast_to_slow_fusion(time_kernel_length=7, time_stride=4, input_channels=64, output_channels=512),
            self._make_fast_to_slow_fusion(time_kernel_length=7, time_stride=4, input_channels=192, output_channels=3072)
        ]

        #
        # Construct summarization and aggregation layers at the output
        #
        self._output_layers = [
            tf.keras.layers.GlobalAveragePooling2D(),   # Slow path
            tf.keras.layers.GlobalAveragePooling2D(),   # Fast path
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Activation(_scaled_activation(self._scaled_activation_type))
        ]

        #
        # Any projector / dense net to be attached to the residual network output pre-loss function
        #
        # TODO [matt.c.mccallum 03.22.22]: It is not clear from the literature and reference code wherther projectors use
        #      scaled or unscaled activations. For now these are unscaled, but we should clarify this as more informaton
        #      becomes available.
        # TODO [matt.c.mccallum 04.08.22]: It is also not clear whether these projectors have batch normalization layers
        #      interspersed, as they do in Google's SimCLR work. Because this is a normalizer free network, we leave thes
        #      batch norm layers out for now.
        self._projectors = [
            tf.keras.layers.Dense(num_nuerons, activation=self._projector_activation, name=f'projector_{idx}') 
            for idx, num_nuerons in enumerate(self._projector_layers)
        ]

        # TODO [matt.c.mccallum 11.22.22]: Make the use of bias configurable here.
        if self._include_fc:
            self._projectors += [tf.keras.layers.Dense(
                output_shape, 
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), 
                name='fc',
                use_bias=False # NOTE [matt.c.mccallum 04.08.22]: This shouldn't have a bias according to the SimCLR reference code here: https://github.com/google-research/simclr/blob/master/model_util.py#L141-L177
            )]

    def _apply_layers(self, input_tensor):
        """
        Take the models layers that have been constructed and apply them to an input tensor.

        Args:
            input_tensor: <tf.Tensor> - The input tensor that the model's layers will be applied to.

        Return:
            <tf.Tensor> - The output, after applying the model's layers.
        """

        # Apply stem modules
        slow = self._apply_stem(self._slow_stem, input_tensor)
        fast = self._apply_stem(self._fast_stem, input_tensor)

        # For each nfnet_transition module
        for fuse, slw_lyr, fst_lyr in zip(self._fusion_layers, self._slow_layers, self._fast_layers):
            slow = self._apply_fast_to_slow_fusion(fuse, slow, fast)
            slow = self._apply_nfnet_stage(slw_lyr, slow)
            fast = self._apply_nfnet_stage(fst_lyr, fast)

        # Apply global average pool and concat
        slow_out = self._output_layers[0](slow)
        fast_out = self._output_layers[1](fast)
        output = self._output_layers[2]([slow_out, fast_out])
        output = self._output_layers[3](output)

        # Apply projector layers
        for proj in self._projectors:
            output = proj(output)

        return output

    def _make_stem_module(self, kernels, channels, strides):
        """
        Create the stem module. This is a series of convolutional layers that are applied on
        the input, prior to any residual stages.

        Args:
            kernels: <list(list(int))> - Size of the 2D convolutional kernels, at each stem layer.

            channels: <list(int)> - Number of channels at the output of each convolutional layer.

            strides: <list(list(int))> - The stride of each 2D convolutional kernel, at each stem layer.

        Return:
            <list(tf.keras.layers.Layer)> - The constructed layers for the stem module.
        """
        activations = [_scaled_activation(self._scaled_activation_type)]*(len(kernels)-1) + [None]
        layers = [
            self.get_conv(
                channels=c,
                kernel=k,
                strides=s,
                groups=1,   # No groups in stem layers
                activation=a
            ) for c, k, s, a in zip(channels, kernels, strides, activations)
        ]
        return list(itertools.chain(*layers))

    def _apply_stem(self, stem_module, input):
        """
        Applies a stem module to an input.

        Args:
            stem_module: <list(tf.keras.layers.Layer)> - Layers constructed by the `_make_stem_module` method.

            input: tf.Tensor - The input tensor to apply the layers to.

        Return:
            <tf.Tensor> - The output, after applying the stem module layers.
        """
        for lyr in stem_module:
            input = lyr(input)
        return input

    def _make_nfnet_stage(self, kernels, freq_downsample, input_channels, output_channels, group_size, alpha, input_expected_var, stoch_depths, num_blocks):
        """
        Constructs an NFNet stage, made up of multiple NFNet blocks.

        Args:
            kernels: <list(list(int))> - The size of each of the 2D convolutional kernels, identical 
            in each NFNet block.

            freq_downsample: <int> - The factor to downsample the frequency dimension by using the 
            second convolutional layer of the first NFNet block.

            input_channels: <int> - The number of channels at the input to the first NFNet block 
            in this stage.

            output_channels: <int> - The number of channels at the output of every NFNet block in 
            this stage.

            group_size: <int> - The number of inputs to include in each group in the grouped 
            convolutional stages.

            alpha: <float> - The alpha value, used to scale tensor variances for initialization, 
            to ensure unit variance between each block.

            input_expected_var: <float> - The expected variance at the input to the first NFNet 
            block of this stage. Important for scaling for initialization.

            stoch_depths: <list(float)> - The stochastic depth skip probability for each of the 
            NFNet blocks in this stage.

            num_blocks: <int> - The number of NFNet blocks to include in this stage.

        Return:
            list(list(list(tf.keras.layers.Layer)))) - All layers in the NFNet stage. Each item
            in the top level list as an NFNet block. Each item in the second level list includes
            the components of that NFNet stage separated into input layers, residual layers, skip
            layers, and output layers.
        """
        # NFNet transition block first
        blocks = [self._make_nfnet_block(
            kernels, 
            freq_downsample, 
            input_channels, 
            output_channels,
            group_size, 
            alpha,
            1.0/input_expected_var,
            float(stoch_depths[0]),
            is_transition=True
        )]

        # NFNet non-transition blocks
        expected_std = (input_expected_var**2.0 + alpha**2.0)**0.5
        for idx in range(1,num_blocks):
            blocks += [self._make_nfnet_block(
                kernels,
                1, 
                output_channels, 
                output_channels,
                group_size, 
                alpha, 
                1.0/expected_std,
                float(stoch_depths[idx])
            )]
            expected_std = (expected_std**2.0 + alpha**2.0)**0.5
        return blocks

    def _apply_nfnet_stage(self, blocks, input):
        """
        Applies all blocks in an NFNet stage to an input tensor.

        Args:
            blocks: list(list(list(tf.Tensor))) - All layers comprising the NFNet stage, separated into
            NFNet blocks, and in turn into NFBlock components (i.e., input, residual, skip, output).

        Return:
            <tf.Tensor> - The tensor after applying the NFNet stage layers.
        """
        output = input
        for block in blocks:
            output = self._apply_nfnet_block(block, output)
        return output
        
    def _make_nfnet_block(self, kernels, freq_downsample, input_channels, output_channels, group_size, alpha, beta, stoch_depth, is_transition=False):
        """
        Make a single NFNet block, comprised of input layers, residual layers, skip layers and output layers.

        Args:
            kernels: <list(list(int))> - The size of each of the 2D convolutional kernels in this block.

            freq_downsample: <int> - The factor to downsample the frequency dimension by increasing the
            stride of the second convolutional layer.

            input_channels: <int> - The number of channels at the input to this NFNet block.

            output_channels: <int> - The number of channels at the output of this NFNet block.

            group_size: <int> - The number of input channels to include in each group in the grouped 
            convolutional stages.

            alpha: <float> - The alpha value, applied as a scalar multiplier at the input. Used to 
            scale tensor variances for initialization, to ensure unit variance between each block.

            beta: <float> - The beta value, used to scale tensor variances for initialization, 
            to ensure unit variance between each block.

            stoch_depth: <float> - The stochastic depth skip probability for the residual path
            in this stage.

            is_transition: <bool> - Force this block to be configured as a NFNet transition block.

        Return:
            list(list(tf.layers.Layer)) - A four element list comprised of:
                [
                    input_layers, <= Layers to be applied directly to the input
                    residual_layers, <= Layers applied after input layers, forming the residual path
                    skip_layers,  <= Layers applied after input layers, forming the skip path
                    output_layers <= Layers to aggregate the residual and skip path, to apply prior to the output
                ]
        """

        # Any difference in dimensions (either across frequency, or across the channel dimension),
        # means that this is an NFNet transition block. There is also the option to force this block
        # to be configured as an NFNet transition block, with matching i/o dimension, which is required
        # in the first nfnet block on the slow path in [1], which has matching i/o channels (due to 
        # concatenation with the fast path), and no frequency downsampling.
        is_transition_block = (freq_downsample > 1) or (input_channels != output_channels) or is_transition

        #
        # Input layers + initialize the residual path depending on whether this is a transition block.
        #
        if is_transition_block:
            input_layers = [
                tf.keras.layers.Activation(_scaled_activation(self._scaled_activation_type)),
                ScalarMultiply(beta)
            ]
            residual_path = []
        else:
            input_layers = []
            residual_path = [
                tf.keras.layers.Activation(_scaled_activation(self._scaled_activation_type)),
                ScalarMultiply(beta)
            ]

        #
        # Path 1 (residual)
        # 
        strides = [[1,1], [freq_downsample,1], [1,1], [1,1]] # NOTE [matt.c.mccallum 03.11.22]: Should this frequency downsample be on the frequency convolution layer, or the first (time convoluation) layer? The original NFNet paper talks about striding on the first 3 x 3 conv layer, but in this case, freq convs don't happen until the second non-1x1 conv layer.
        per_layer_out_chans = [output_channels//2]*3 + [output_channels]
        groups = [1] + [output_channels//2//group_size]*2 + [1]
        activations = [_scaled_activation(self._scaled_activation_type)]*(len(kernels)-1) + [None]
        residual_path += list(itertools.chain(*[
            self.get_conv(
                channels=c,
                kernel=k,
                strides=s,
                groups=g,
                activation=a
            ) for c, k, s, g, a in zip(per_layer_out_chans, kernels, strides, groups, activations)
        ])) + [
            self._make_squeeze_and_excite(per_layer_out_chans[-1]),
            ScalarMultiply(0.0, learnable=True),
            ScalarMultiply(alpha)
        ]

        #
        # Path 2 (skip)
        # 
        skip_path = []
        if freq_downsample > 1: # <= Only include average pooling when we are downsampling
            skip_path += [
                tf.keras.layers.AveragePooling2D(
                    pool_size=[freq_downsample, 1], # Should this be 1 or 2 on the time dimension? Seeing we only subsample the frequency dimension...
                    strides=[freq_downsample, 1],
                    padding='same'
                )
            ]
        if is_transition_block:
            skip_path += [
                WeightStandardization(tf.keras.layers.Conv2D(
                    filters=output_channels,
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    padding='same',
                    groups=1,
                    activation='linear', # NOTE [matt.c.mccallum 03.11.22]: No activation on shortcut path
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=1.0,
                        mode='fan_in',
                        distribution='untruncated_normal'
                    )
                ))
            ]

        #
        # Concat / Add to provide the output
        #
        output_layers = [
            StochDepth(survival_probability=1-stoch_depth, scale_during_test=False)
        ]

        return [input_layers, residual_path, skip_path, output_layers]
    
    def _apply_nfnet_block(self, nfnet_block, input):
        """
        Apply a single NFNet block to the provided input tensor.

        Args:
            nfnet_block: <list(list(tf.keras.layers.Layer))> - Layers constructed by the 
            `_make_nfnet_block` method. This should include four lists of layers, comprising
            the input layers, residual layers, skip layers, and output layers.

            input: tf.Tensor - The input tensor to apply the layers to.

        Return:
            <tf.Tensor> - The output, after applying the stem module layers.
        """
        # Input path
        for lyr in nfnet_block[0]:
            input = lyr(input)

        # Residual path
        conv = input
        for idx, lyr in enumerate(nfnet_block[1]):
            # NOTE [matt.c.mccallum 03.18.22]: Squeeze and excite layer is always third 
            # to last on the conv path, and requires custom application. Once we move these
            # modules to their own layer class, with a call method, this can be removed.
            if len(nfnet_block[1])-idx == 3:
                conv = self._apply_squeeze_and_excite(lyr, conv)
            else:
                conv = lyr(conv)

        # Skip path
        skip = input
        for lyr in nfnet_block[2]:
            skip = lyr(skip)

        # Output
        output = nfnet_block[3][0]([skip, conv])
        return output

    def _make_squeeze_and_excite(self, output_channels):
        """
        Create a squeeze and excite module [1].

        Args:
            output_channels: <int> - The number of channels at the output of the squeeze and excitation block.

        Return:
            <list(tf.keras.layers.Layer)> - The layers comprising the squeeze and excite module.

        References:
            [1] Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In Proceedings of the 
            IEEE conference on computer vision and pattern recognition, pp. 7132-7141. 2018.
        """
        # NOTE [matt.c.mccallum 03.22.22]: To mirror closely the NFNet reference code produced by DeepMind,
        #      we apply a regular relu and sigmoid here as activation, rather than the scaled activation
        #      employed throughout the rest of the NFNet block.
        return [
            tf.keras.layers.GlobalAveragePooling2D(keepdims=False),
            tf.keras.layers.Dense(units=output_channels//2, activation='relu', use_bias=True), # Squeeze excite layers usually have half the number of hidden units to output units, but we should make this configurable
            tf.keras.layers.Dense(units=output_channels, activation='sigmoid', use_bias=True)
            # NOTE [matt.c.mccallum 03.12.22]: Broadcasting of the output is done when this layer is applied
        ]

    def _apply_squeeze_and_excite(self, se_lyrs, input):
        """
        Apply a squeeze and excite module to an input.

        Args:
            se_lyrs: <list(tf.keras.layers.Layer)> - The layers comprising the squeeze and 
            excite module

            input: <tf.Tensor> - The input tensor to apply the layers to.

        Return:
            <tf.Tensor> - The output after the module has been applied to the input.
        """
        output = input

        for lyr in se_lyrs:
            output = lyr(output)

        # NOTE [matt.c.mccallum 03.12.22]: Re-insert time and frequency dimensions below.
        output = tf.expand_dims(tf.expand_dims(output*2.0, axis=1), axis=1)*input
        return output

    def _make_fast_to_slow_fusion(self, time_kernel_length, time_stride, input_channels, output_channels):
        """
        Make layers that comprise the operations in order to fuse the fast path of NFNet stages to the
        slow path. This simply applies a Tx1 time-strided convolution to the fast path, to match the 
        dimensionality of the slow path, then concatenates with the slow path.

        Args:
            time_kernel_length: <int> - The length of the convolutional kernel in time.

            time_stride: <int> - The size of the stride along the time dimension.

            output_channels: <int> - The number of channels of the fast path at the output
            of the convolutional layer, prior to concatenation with the slow path.

        Return:
            list(tf.keras.layers.Layer) - The layers in the fast to slow fusion module.
        """

        return [
            WeightStandardization(tf.keras.layers.Conv2D(
                filters=input_channels,
                kernel_size=[1, time_kernel_length],
                strides=[1, time_stride],
                padding='same',
                activation='linear', # NOTE [matt.c.mccallum 03.11.22]: No activation on shortcut path
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=1.0,
                    mode='fan_in',
                    distribution='untruncated_normal'
                )
            )),
            WeightStandardization(tf.keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same',
                activation='linear', # NOTE [matt.c.mccallum 03.11.22]: No activation on shortcut path
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=1.0,
                    mode='fan_in',
                    distribution='untruncated_normal'
                )
            )),
            tf.keras.layers.Concatenate(axis=-1), # Concatenates on the last (channel) dimension by default
        ]

    def _apply_fast_to_slow_fusion(self, fusion_module, slow, fast):
        """
        Apply the fast-to-slow fusion module to create the next (fused) input for
        the slow path.

        Args:
            fusion_module: <list(tf.keras.layers.Layer)> - The layers comprising the fusion module.

            slow: <tf.Tensor> - The output of the previous stage form the slow path, to be fused
            with the fast path.

            fast: <tf.Tensor> - The output of the previous stage of the fast path, to be fused into
            the slow path.

        Return:
            <tf.Tensor> - The output of the fast-to-slow fusion, to be used as input to the next
            stage in the slow path.
        """
        fast = fusion_module[0](fast)
        fast = fusion_module[1](fast)
        return fusion_module[2]([slow, fast])
