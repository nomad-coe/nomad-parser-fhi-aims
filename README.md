This is the main repository of the [NOMAD](http://nomad-lab.eu) parser for
[FHI-aims](https://aimsclub.fhi-berlin.mpg.de).

# Example
```python
    from fhiaimsparser import FHIaimsParser
    import matplotlib.pyplot as mpl

    # 1. Initialize a parser by giving a path to the FHI-aims output file and a list of
    # default units
    path = "path/to/main.file"
    default_units = ["eV"]
    parser = FHIaimsParser(path, default_units=default_units)

    # 2. Parse
    results = parser.parse()

    # 3. Query the results with using the id's created specifically for NOMAD.
    scf_energies = results["energy_total_scf_iteration"]
    mpl.plot(scf_energies)
    mpl.show()
```

# Installation
The code is python>=2.7 and python>=3.4 compatible. First download and install
the nomadcore package:

```sh
git clone https://gitlab.mpcdf.mpg.de/nomad-lab/python-common.git
cd python-common
pip install -r requirements.txt
pip install -e .
```

Then download the metainfo definitions to the same folder where the
'python-common' folder was cloned:

```sh
git clone https://gitlab.mpcdf.mpg.de/nomad-lab/nomad-meta-info.git
```

Finally download and install the parser:

```sh
git clone https://gitlab.mpcdf.mpg.de/nomad-lab/parser-cp2k.git
cd parser-cp2k
pip install -e .
```

# Advanced
## Command-line Based Usage
FHI-aims output files can be parsed with:

    python FhiAimsParser.py [path/toFile]

Yon can accessed the help for the available command line arguments with:

    python FhiAimsParser.py --help

## Test Files
Example output files of FHI-aims can be found in the directory test/examples.
More details about the calculations and files are explained in a README in this directory.

## Documentation of Code
The [google style
guide](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments)
provides a good template on how the code should be documented. This makes it
easier to follow the logic of the parser.
