# Copyright 2015-2018 Franz Knuth, Fawzi Mohamed, Wael Chibani, Ankit Kariryaa, Lauri Himanen, Danio Brambila
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
This is a setup script for installing the parser locally on python path with
all the required dependencies. Used mainly for local testing.
"""
from setuptools import setup, find_packages


def main():
    # Start package setup
    setup(
        name='fhiaimsparser',
        version='0.1',
        description='NOMAD parser implementation for FHI-aims.',
        package_dir={'': './'},
        packages=find_packages(),
        install_requires=[
            'numpy',
            'nomadcore'
        ],
        entry_points='''
            [console_scripts]
            nomad-fhiaims-parser=fhiaimsparser.parser:main
        '''
    )


# Run main function by default
if __name__ == "__main__":
    main()
