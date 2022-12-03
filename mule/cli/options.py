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
import click

# Local imports
# None.


def partial_option(*args, **kwargs):
    option = click.option(*args, **kwargs)

    def option_decorator(command=None):
        if command:
            # We were used as a decorator without arguments, and now we're being
            # called on the command function.
            return option(command)
        else:
            # We are being called with empty parens to construct a decorator.
            return option

    return option_decorator

#
# Options below
#
config = partial_option(
        "--config", 
        "-c",
        help="Path to yaml scooch file defining an analysis pipeline"
    )

input_file = partial_option(
        "--input-file", 
        "-i",
        help="Path to input audio file to be analyzed"
)

output_file = partial_option(
        "--output-file", 
        "-o",
        help="Path to output file location to place resulting analyzed features"
    )
