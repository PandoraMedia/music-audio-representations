# coding=utf-8
# Copyright 2024 Pandora Media, LLC.
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

# Python standard library Imports
# None.

# Third party imports
import tensorflow as tf

# Local imports
# None


__all__ = [
    'get_scaled_activation'
]


def _scaled_gelu_activation(x):
    return tf.nn.gelu(x) * 1.7015043497085571

def _scaled_relu_activation(x):
    # NOTE [matt.c.mccallum 03.22.22]: The scalar below is taken directly from DeepMind reference code,
    #      it is similar but not equal to the value suggested in the literature of: np.sqrt(2.0/(1-(1/np.pi)))
    return tf.nn.relu(x) * 1.7139588594436646 


tf.keras.utils.get_custom_objects().update({'_scaled_gelu_activation':_scaled_gelu_activation})
tf.keras.utils.get_custom_objects().update({'_scaled_relu_activation':_scaled_relu_activation})


def get_scaled_activation(activation_name):
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
        'gelu': '_scaled_gelu_activation',
        'relu': '_scaled_relu_activation'
    }
    return activations[activation_name]
