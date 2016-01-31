# FHI-aims Parser
This is the parser for [FHI-aims](https://aimsclub.fhi-berlin.mpg.de).
It is part of the [NOMAD Laboratory](http://nomad-lab.eu).
The official version lives at:

    git@gitlab.mpcdf.mpg.de:nomad-lab/parser-fhi-aims.git

You can browse it at:

    https://gitlab.rzg.mpg.de/nomad-lab/parser-fhi-aims

It relies on having the nomad-meta-info and the python-common repositories one level higher.
The simplest way to have this is to check out nomad-lab-base recursively:

    git clone --recursive git@gitlab.mpcdf.mpg.de:nomad-lab/nomad-lab-base.git

This parser will be in the directory parsers/fhi-aims of this repository.

# Running and Testing the Parser
## Requirements
The required python packages can be installed with (see [python-common](https://gitlab.rzg.mpg.de/nomad-lab/python-common)):

    pip install -r nomad-lab-base/python-common/requirements.txt

## Usage
FHI-aims output files can be parsed with:

    python FhiAimsParser.py [path/toFile]

Yon can accessed the help for the available command line arguments with:

    python FhiAimsParser.py --help

## Test Files
Example output files of FHI-aims can be found in the directory test/examples.
More details about the calculations and files are explained in a README in this directory.

# Documentation of Code
The [google style guide](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments) provides a good template on how the code should be documented. This makes it easier to follow the logic of the parser.
