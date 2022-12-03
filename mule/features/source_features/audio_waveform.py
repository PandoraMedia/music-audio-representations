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
import librosa
from scooch import Param

# Local imports
from .source_feature import SourceFeature


class AudioWaveform(SourceFeature):
    """
    A feature representing an audio waveform, extracted from an audio file.
    """

    #
    # SCOOCH Configuration
    #
    _sample_rate = Param(
        int,
        default=16000,
        doc="The sample rate of the output waveform. Input audio files will be resampled to this."
    )
    # TODO [matt.c.mccallum 11.18.22]: SCOOCH does not support virtual params at the moment. For now
    #      it is up to the user to ensure the input type is an AudioFile, but once virtual parameter
    #      support is added to SCOOCH, we should override the `_input_file` from the SourceFeature
    #      base class parameter here, to ensure that is of an AudioFile type.

    #
    # Methods
    #
    def _extract(self, source, start_time, chunk_size):
        """
        Extracts data from the audio waveform from a source file, for a given number of samples.

        Args:
            source: mule.source_files.AudioFile - An audio file to extract data from.

            start_time: int - The sample index at which to start extracting an audio chunk from.
            Currently this is unused, and audio extraction will continue from where it last left
            off.

            chunk_size: int - The number of samples to extract in this call.

        Return:
            np.ndarray - The extracted audio samples, summed to mono. Sample index is on the second
            axis.
        """
        # TODO [matt.c.mccallum 11.19.22]: Make this function work for non-sequential reads, i.e., respect the start_time argument.
        # Read from file
        data = source.read(chunk_size)
        # Sum to mono
        samples = np.sum(data, axis=0)
        # Resample
        if source.sample_rate != self._sample_rate:
            samples = librosa.core.resample(samples, source.sample_rate, self._sample_rate)
        # TODO [matt.c.mccallum 11.18.22]: Improve handling at chunk boundaries
        return samples.reshape((1, -1))

    # 
    # Properties
    #
    @property
    def sample_rate(self):
        """
        The sample rate of the waveform feature itself. (Any audio file will be resampled
        to this).
        """
        return self._sample_rate
