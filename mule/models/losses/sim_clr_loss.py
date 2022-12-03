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
from scooch import Param

# Local imports
from .loss import Loss


class SimClrLoss(Loss, tf.keras.losses.Loss):
    """
    Implements the loss function described in [1]. Intended for unsupervised learning via
    augmented data pairs.

    [1] - Ting, C., et. al "A Simple Framework for Contrastive Learning of Visual Representations",
        arXiv preprint arXiv:2002.05709, 2020.

    """

    #
    # SCOOCH Configuration
    #
    _temperature = Param(
        float,
        default=0.1,
        doc="The SimCLR loss temperature. (See [1] in class docstring)."
    )

    _LARGE_NUMBER = 1e10

    # 
    # Methods
    #
    def __init__(self, cfg, *args, **kwargs):
        """
        Constructor.
 
        Args:
            cfg: Config - The SCOOCH loss's configuration.
        """
        super().__init__(cfg, *args, **kwargs)

        # NOTE [matt.c.mccallum 02.10.22]: Remove any keras based reduction / normalization across GPUs, we handle this ourselves
        self._star_kwargs.update({
            'reduction': tf.keras.losses.Reduction.SUM
        })

    @staticmethod
    def _cross_replica_concat(tensor, strategy=None):
        """Reduce a concatenation of the `tensor` across devices.
        Args:
            tensor: tensor to concatenate.
            strategy: A `tf.distribute.Strategy`. If not set, CPU execution is assumed.
        Returns:
            Tensor of the same rank as `tensor` with first dimension `num_replicas`
            times larger.

        Taken from:
            https://github.com/google-research/simclr/blob/master/tf2/objective.py
        """
        if strategy is None or strategy.num_replicas_in_sync <= 1:
            return tensor

        num_replicas = strategy.num_replicas_in_sync

        replica_context = tf.distribute.get_replica_context()
        with tf.name_scope('cross_replica_concat'):
            # This creates a tensor that is like the input tensor but has an added
            # replica dimension as the outermost dimension. On each replica it will
            # contain the local values and zeros for all other values that need to be
            # fetched from other replicas.
            ext_tensor = tf.scatter_nd(
                indices=[[replica_context.replica_id_in_sync_group]],
                updates=[tensor],
                shape=tf.concat([[num_replicas], tf.shape(tensor)], axis=0)
            )

            # As every value is only present on one replica and 0 in all others, adding
            # them all together will result in the full tensor on all replicas.
            ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, ext_tensor)

            # Flatten the replica dimension.
            # The first dimension size will be: tensor.shape[0] * num_replicas
            # Using [-1] trick to support also scalar input.
            return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])

    def call(self, y_true, y_pred):
        """
        Compute the loss function.

        Args:
            y_true: <tf.Tensor> - Unused. This is an unsupervised SimCLR loss and so uses no target data.

            y_pred: list(<tf.Tensor>) - Two model outputs in the order: anchor_output, augmented_output.
        """
        # Collect examples from all GPUs
        strat = tf.distribute.get_strategy()

        # Assume the output layer of the network is already l2 normalized... No need for the line below 
        y_pred = tf.keras.backend.l2_normalize(y_pred, axis=-2)
        per_device_batch_size = tf.shape(y_pred)[0]
        
        all_examples = self._cross_replica_concat(y_pred, strat)
        raw, augmented = all_examples[:,:,0], all_examples[:,:,1]
        total_batch_size = tf.shape(raw)[0]
        collected_examples = tf.concat([raw, augmented], axis=0)
        collected_examples = tf.squeeze(collected_examples)

        # Compute logits
        all_sims = tf.tensordot(collected_examples, tf.transpose(collected_examples), axes=1)/self._temperature
        all_sims -= tf.linalg.diag(tf.ones(tf.shape(all_sims)[0]))*self._LARGE_NUMBER
        labels = tf.linalg.diag(tf.ones([total_batch_size]))
        empty = tf.zeros([total_batch_size, total_batch_size])
        labels = tf.concat([tf.concat([empty, labels], axis=1), tf.concat([labels, empty], axis=1)], axis=0)

        # Get index of device
        replica_context = tf.distribute.get_replica_context()
        replica_id = tf.cast(
            tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
        start_idx = replica_id * per_device_batch_size
        end_idx = (replica_id+1) * per_device_batch_size

        # Just get the losses that are relevant to the current device
        # TODO [matt.c.mccallum 01.23.22]: More efficient to do this before computing the logits above,
        #      rather than computing all logits on every device. But it's a drop in the ocean compared
        #      to the networks we're training.
        this_all_a_sims = all_sims[start_idx : end_idx]
        this_a_labels = labels[start_idx : end_idx]
        this_all_b_sims = all_sims[start_idx + total_batch_size : end_idx + total_batch_size]
        this_b_labels = labels[start_idx + total_batch_size : end_idx + total_batch_size]

        loss_a = tf.nn.softmax_cross_entropy_with_logits(this_a_labels, this_all_a_sims)
        loss_b = tf.nn.softmax_cross_entropy_with_logits(this_b_labels, this_all_b_sims)
        loss = tf.reduce_mean(loss_a + loss_b)

        num_replicas = strat.num_replicas_in_sync
        loss = loss/num_replicas

        return loss

