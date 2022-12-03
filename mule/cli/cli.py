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
from ..analysis import Analysis

# Third party imports
import click
from scooch import Config
from . import options

# Python standard library imports
# None.


@click.group()
def main():
    pass


@main.command("analyze")
@options.config
@options.input_file
@options.output_file
def run(config, input_file, output_file):
    """
    Constructs the analysis pipeline, adds options to it, and runs it.
    """
    cfg = Config(config)
    analysis = Analysis(cfg)
    feat = analysis.analyze(input_file)
    feat.save(output_file)


if __name__=='__main__':
    main()
