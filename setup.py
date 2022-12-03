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
import distutils.cmd


long_description = """
# MULE

Musicset Unsupervised Large Embeddings are your music-audio workhorse!

"""


REQUIRED_PACKAGES = [
    'numpy',
    'librosa',
    'click==8.0.0a1',
    'scooch',
    'tensorflow'
]


setuptools.setup(
    name='sxmp-mule',
    version='1.0.0',
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Matt C. McCallum",
    author_email="mmccallum@pandora.com",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    project_urls={
        'Documentation': '',
        'Bug Reports': '',
        'Source': '',
    },
    license='',
    keywords='',

    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    include_package_data=True,

    # CLI
    entry_points = {
        'console_scripts': ['mule=mule.cli:main']
    }
)
