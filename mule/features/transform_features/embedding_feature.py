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
import urllib.parse as urlparse

# Third party imports
import tensorflow as tf
from scipy.special import logsumexp
import numpy as np
from scooch import Param

# Local imports
from .transform_feature import TransformFeature


class EmbeddingFeature(TransformFeature):
    """
    A feature that applies an ML model to input data to create an embedding.
    """

    #
    # SCOOCH Configuration
    #
    _model_location = Param(
        str,
        doc="Location of the model to load and analyze feature data with."
    )
    _apply_softmax = Param(
        bool,
        default=False,
        doc="Whether to apply a softmax to the model output feature."
    )

    _model = None

    _CHUNK_SIZE = 8000

    # 
    # Methods
    #
    def __init__(self, cfg):
        """
        *Constructor*

        Args:
            cfg: scooch.Config - The scooch configuration for this object.
        """
        super().__init__(cfg)
        self.load_model()

    def _extract(self, source_feature, start_time, chunk_size):
        """
        Extracts feature data by applying a model to samples of an input feature, over
        a given time period.

        Args:
            source_feature: mule.features.Feature - The feature to transform.

            start_time: int - The index in the feature at which to start extracting / transforming.

            chunk_size: int - The length of the chunk following the `start_time` to extract
            the feature from.
        """
        data = super()._extract(source_feature, start_time, chunk_size)

        if len(data):
            data = self._model.predict(
                    [data],
                    callbacks=None
                )
            data = data.T
            if self._apply_softmax:
                data = np.nan_to_num(np.exp(data - logsumexp(data, axis=0)))
            return data
        else:
            return None

    def load_model(self):
        """
        Loads the model that this feature uses from a configured location.
        """
        # If GS use google-cloud-storage...
        parsed_loc = urlparse.urlparse(self._model_location)
        if parsed_loc.scheme == 'gs':
            # TODO [matt.c.mccallum 11.21.22]: Write google storage download code here.
            pass
        # Otherwise assume local
        self._model = tf.keras.models.load_model(self._model_location, compile=False)
