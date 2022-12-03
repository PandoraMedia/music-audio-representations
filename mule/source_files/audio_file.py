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
import tempfile
import io
import subprocess
import struct
import wave
import os

# Third party imports
import numpy as np
from scooch import Param

# Local imports
from .source_file import SourceFile


class AudioFile(SourceFile):
    """
    A class for reading audio files.
    """

    _sample_rate = Param(
        int,
        default=44100,
        doc="The sample rate that an input audio file is interpreted as"
    )
    # TODO [matt.c.mccallum 11.18.22]: Handle sampling rates here

    _wav_file = None
    _use_local_file = False
    _max_len = 60*30 # 30 minutes
    _bit_depth = 16
    _samp_rate = 44100
    _wav_ffmpeg_fmt = 'pcm_s' + str(_bit_depth) + 'le'

    # Map file extensions to ffmpeg formats here
    _FORMATS = {
        '.mp3': 'mp3',
        '.wav': 'wav',
        '.wave': 'wav'
    }

    def __del__(self):
        """
        Destructor.
        """
        if self._wav_file is not None:
            self._wav_file.close()
        
    def load(self, file):
        """
        Converts the file to a temporary wav file. Once converted this wav file will stick around as long as this object
        exists.
        Note: Depending on whether use_local_file is true, this will either read directly from the system pipes for output
        (false), or save the result to a temporary file on disk, transcode, then read the temporary file (true).

        Args:
            file: file - A file object containing mp3 data to decode.
        """
        ext = os.path.splitext(file)[1]
        if ext in self._FORMATS:
            format = self._FORMATS[ext]
        else:
            raise IOError(f"Unsupported audio file extension: {ext}, supplied to audio file reader")

        if self._use_local_file:
            self._tempf = tempfile.NamedTemporaryFile(mode='r+b', suffix='.wav')
            cmd = f"ffmpeg -y -loglevel panic -t {self._max_len} -f {format} -i {file} -map_metadata -1 " \
                f"-vn -acodec {self._wav_ffmpeg_fmt} -ar {self._samp_rate} -f wav {self._tempf.name}"
            subprocess.run(cmd.split(" "))

                # Create a wav object containing the decoded data.
            self._tempf.seek(0)
            self.wav_file = self._tempf
        else:
            with open(file, 'rb') as f:
                cmd = f"ffmpeg -loglevel panic -t {str(self._max_len)} -f {format} -i pipe:0 -map_metadata -1 -vn " \
                    f"-acodec {self._wav_ffmpeg_fmt} -ar {self._samp_rate} -f wav pipe:1"
                x = subprocess.run(cmd.split(" "), input=f.read(), capture_output=True)
                in_mem_file = io.BytesIO(x.stdout)

            # Fix file size as ffmpeg output via std stream doesn't include a file size.
            in_mem_file.seek(0)
            file_length = in_mem_file.seek(0, 2)
            in_mem_file.seek(4)
            in_mem_file.write(struct.pack('i', file_length - 8))
            in_mem_file.seek(0)
            test_data = in_mem_file.read(10000)
            data_start = test_data.find(b'data')
            in_mem_file.seek(data_start + 4)
            in_mem_file.write(struct.pack('i', file_length - data_start - 8))
            in_mem_file.seek(0)

            self.wav_file = in_mem_file

    def _packing_string( self, num_frames ):
        """
        Get the string for packing or unpacking a given number of frames using the struct module.

        Args:
            num_frames: int - The number of frames to cover in the packing string

        Return:
            str - The string to be used with the struct module for unpacking, packing the given number of frames for the
            current audio format described in this object.
        """
        unpack_fmt = '<%i' % ( num_frames * self._n_channels )
        if self._bit_depth == 16:
            unpack_fmt += 'h'
        elif self._bit_depth == 32:
            unpack_fmt += 'i'
        else:
            raise Exception('Unsupporeted bit depth format for packing data.')
        return unpack_fmt
    
    def _read_samples_interleaved_int(self, n=None):
        """
        Reads all samples from the wav file as integers in an interleaved list.
        This replaces any previous data read from the wav file.

        Args:
            n: int - The number of frames (samples per channel) to read.

        Return:
            list[int] - A list of interleaved samples from the audio file.

            int - The number of samples successully read from file. May be less than
            the number of requested samples due to EOF.
        """
        # NOTE [matt.c.mccallum 03.04.19]: We don't use the wave file context-manager below because of problems using it in Python 2.7.10
        if n is None: n = self._num_frames
        data = self._wav_file.readframes( n )
        n_read_frames = int(len(data)*8/self._bit_depth/self._n_channels)
        if (n_read_frames < n): n = n_read_frames

        data = struct.unpack( self._packing_string( n ), data )

        data = list(data)

        return data, n

    def read(self, n=None):
        """
        Reads all samples from the wav file as floats in the range -1.0 <= sample <= 1.0.
        This replaces any previous data read from the wav file.

        Args:
            n: int - The number of samples to read.

        Return:
            np.ndarray - An array of dimensions (num_channels, num_frames) containing float valued audio samples.
        """
        data, n = self._read_samples_interleaved_int( n )
        if n == 0:
            return None

        return_array = np.zeros( ( self._n_channels, n ), dtype=np.float32 )
        for channel in range( self._n_channels ):
            return_array[channel,:] = np.array( data[channel::self._n_channels] )/( 2.0**self._bit_depth )

        data = return_array

        return data

    def __len__(self):
        return self._num_frames

    @property
    def wav_file(self):
        return self._wav_file
    @wav_file.setter
    def wav_file(self, file):
        self._raw_file = file
        self._wav_file = wave.open(file, 'rb')
        self._samp_rate = self._wav_file.getframerate()
        self._n_channels = self._wav_file.getnchannels()
        self._bit_depth = self._wav_file.getsampwidth() * 8
        self._num_frames = min(self._wav_file.getnframes(), int(self._samp_rate*self._max_len))

    @property
    def sample_rate(self):
        return self._sample_rate
