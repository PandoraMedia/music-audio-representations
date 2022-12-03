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
import numpy as np
from scooch import Param
import librosa

# Local imports
from .transform_feature import TransformFeature


class MelSpectrogram(TransformFeature):
    """
    A feature that is a mel spectrogram transformation of its input.
    """

    #
    # SCOOCH Configuration
    #
    _n_fft = Param(
        int,
        default=1024,
        doc="Size of the FFT operation, prior to mel window summarization."
    )
    _hop_length = Param(
        int,
        default=256,
        doc="The number of samples between the start of any two consecutive analysis windows inclusive of the sample at the start of the first analysis window."
    )
    _win_length = Param(
        int,
        default=512,
        doc="Number of audio samples in each window"
    )
    _window = Param(
        str,
        default='hamming',
        doc="A string selecting the window type"
    )
    _n_mels = Param(
        int,
        default=96,
        doc="Number of mel windows each spectrogram window is summarized into across frequency"
    )
    _fmin = Param(
        float,
        default=40.0,
        doc="The frequency in Hz of the lowest frequency mel band"
    )
    _fmax = Param(
        float,
        default=8000.0,
        doc="The frequency in Hz of the highest frequeny mel band"
    )
    _norm = Param(
        float,
        default=2.0,
        doc="The order of normalization (e.g., 2.0 is power normalization)"
    )
    _mag_compression = Param(
        str,
        default='linear',
        doc="The type of compression to be applied to the magnitude mel values"
    )
    _mag_range = Param(
        float,
        default=9999999999.0,
        doc="The range below the maximum magnitude to limit the magnitudes to. If set to None, the value range will not be limited."
    )
    _power = Param(
        float,
        default=2.0,
        doc="The exponent to be used to compute the power spectrogram"
    )
    _htk = Param(
        bool,
        default=False,
        doc="Whether or not to use the HTK formula for mel frequencies (see librosa docs)"
    )
    _pad_mode = Param(
        str,
        default='reflect',
        doc="How to pad the signal in the beginning and at the end (see np.pad)"
    )

    _COMPRESSION_FUNCS = {
        '10log10': lambda x: 10*np.log10(x),
        'log10_nonneg': lambda x: np.log10(10000.0*x + 1.0),
        'log': lambda x: np.log(x),
        'linear': lambda x: x,
        None: lambda x: x,
    }

    _SMALLEST_MAGNITUDE = -9999999999.0

    #
    # Methods
    #
    def _extract(self, source_feature, start_time, chunk_size):
        """
        Extracts a mel spectrogram of the input feature over the provided time range.

        Args:
            source_feature: mule.features.AudioWaveform - The feature to transform into a mel spectrogram

            start_time: int - The index in the feature at which to start extracting / transforming.

            chunk_size: int - The length of the chunk following the `start_time` to extract
            the feature from.
        """
        # TODO [matt.c.mccallum 11.19.22]: Better handling of boundary conditions by zero padding only start and end of input, not of each chunk.

        data = np.squeeze(super()._extract(source_feature, start_time, chunk_size))
        
        mel_data = librosa.feature.melspectrogram(
            y=data, 
            sr=source_feature.sample_rate, 
            n_fft=self._n_fft, 
            hop_length=self._hop_length, 
            win_length=self._win_length, 
            window=self._window, 
            center=True, 
            pad_mode=self._pad_mode, 
            power=self._power,
            n_mels=self._n_mels,
            fmin=self._fmin,
            fmax=self._fmax,
            norm=self._norm,
            htk=self._htk,
            dtype=np.float32
        )

        mel_data = self._COMPRESSION_FUNCS[self._mag_compression](mel_data)
        mel_data = np.nan_to_num(mel_data, nan=self._SMALLEST_MAGNITUDE, posinf=self._SMALLEST_MAGNITUDE, neginf=self._SMALLEST_MAGNITUDE)

        if self._mag_range is not None:
            max_val = np.amax(mel_data)
            mel_data = mel_data - max_val
            mel_data = np.maximum(mel_data, -self._mag_range)
        
        return mel_data