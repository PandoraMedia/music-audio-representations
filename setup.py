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


# Local imports
# None.

# Third party imports
# None.

# Python standard library imports
import setuptools


long_description = """
# MULE

The Musicset Unsupervised Large Embedding (MULE) module is your 
music-audio workhorse!

This module contains [SCOOCH](https://github.com/PandoraMedia/scooch) configurable code to run a simple 
analysis pipeline to extract audio embeddings from audio files which
may then be used for downstream music understanding purposes.

This module requires FFMpeg to read audio files, which may be 
downloaded [here](https://ffmpeg.org/download.html).

In order to create MULE embeddings, you will need a SCOOCH configuration
describing the pipeline, and the model weights. Both are licensed under 
the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license, and can be found in this [module's github repository](https://github.com/PandoraMedia/music-audio-representations).

To create embeddings for a single audio file, e.g., `test.wav` in the current
directory, you can use this module in conjunction with the provided configuration
and model weights:

```
pip install sxmp-mule
git clone https://github.com/PandoraMedia/music-audio-representations.git
cd ./music-audio-representations
mule analyze --config ./supporting_data/configs/mule_embedding.yml -i ../test.wav -o ./embedding.npy
```

For more information on this module, please check out the publication:

[*Supervised and Unsupervised Learning of Audio Representations for Music Understanding*](https://arxiv.org/abs/2210.03799), **M. C. McCallum**, F. Korzeniowski, S. Oramas, F. Gouyon, A. F. Ehmann.

"""


REQUIRED_PACKAGES = [
    'numpy',
    'librosa',
    'click==8.0.0a1',
    'scooch>=1.0.0',
    'tensorflow==2.9.1'
]


setuptools.setup(
    name='sxmp-mule',
    version='1.0.1',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PandoraMedia/music-audio-representations",
    author="Matt C. McCallum",
    author_email="mmccallum@pandora.com",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
    ],
    project_urls={
        'Documentation': 'https://github.com/PandoraMedia/music-audio-representations',
        'Bug Reports': 'https://github.com/PandoraMedia/music-audio-representations/issues',
        'Source': 'https://github.com/PandoraMedia/music-audio-representations',
    },
    license='GNU GPL 3.0',
    keywords='mule audio music embeddings machine learning',

    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    include_package_data=True,

    # CLI
    entry_points = {
        'console_scripts': ['mule=mule.cli:main']
    }
)
