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

from __future__ import absolute_import
from builtins import object
try:
    import setup_paths
except ImportError:
    pass
import numpy as np
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import mainFunction
from nomadcore.simple_parser import SimpleMatcher as SM
from nomadcore.unit_conversion.unit_conversion import convert_unit
from fhiaimsparser.FhiAimsCommon import get_metaInfo
import logging, os, re, sys

############################################################
# This is the parser for the DOS file of FHI-aims.
############################################################

logger = logging.getLogger("nomad.FhiAimsDosParser")

class FhiAimsDosParserContext(object):
    """Context for parsing FHI-aims DOS file (total, atom projected, species projected).

    Attributes:
        dos_energies: Stores parsed energies.
        dos_values: Stores parsed DOS values.

    The onClose_ functions allow processing and writing of cached values after a section is closed.
    They take the following arguments:
        backend: Class that takes care of writing and caching of metadata.
        gIndex: Index of the section that is closed.
        section: The cached values and sections that were found in the section that is closed.
    """
    def __init__(self, writeMetaData = True):
        """Args:
            writeMetaData: Deteremines if metadata is written or stored in class attributes.
        """
        self.writeMetaData = writeMetaData
        # unit cell volume is necessary for unit-conversion compliant with
        # NOMAD-metaInfo
        # unit_cell_volume and self.fermi_energy is set by the 'main' FhiAimsParser before calling us
        self.unit_cell_volume = None
        self.fermi_energy = None

    def startedParsing(self, fInName, parser):
        """Function is called when the parsing starts and the compiled parser is obtained.

        Args:
            fInName: The file name on which the current parser is running.
            parser: The compiled parser. Is an object of the class SimpleParser in nomadcore.simple_parser.py.
        """
        self.parser = parser
        # allows to reset values if the same superContext is used to parse different files
        self.dos_energies = None
        self.dos_values = None

    def onClose_section_dos(self, backend, gIndex, section):
        """Trigger called when section_dos is closed.

        Store the parsed values and write them if writeMetaData is True.
        """
        # extract energies
        dos_energies = section['x_fhi_aims_dos_energy']
        if dos_energies is not None:
            self.dos_energies = np.asarray(dos_energies)
        # extract dos values
        dos_values = []
        if section['x_fhi_aims_dos_value_string'] is not None:
            for string in section['x_fhi_aims_dos_value_string']:
                strings = string.split()
                dos_values.append([float(x) for x in strings])
        if dos_values:
            # need to transpose array since its shape is [number_of_spin_channels,n_dos_values] in the metadata
            self.dos_values = np.transpose(np.asarray(dos_values))
            # FHI-AIMS provides DOS in units of 1/(eV*unit_cell_volume)
            # http://materials-mine.com/tutorials/AIMS/FHI-aims.071914.pdf, page 239
            # NOMAD units are 1/(J*cell)
            #   convert to 1/J
            self.dos_values = convert_unit(self.dos_values, '1/eV', '1/J')
            #   convert to 1/cell
            self.dos_values *= self.unit_cell_volume
        # write metadata only if values were found for both quantities
        if self.writeMetaData:
            if dos_energies is not None and dos_values:
                backend.addArrayValues('dos_energies', self.dos_energies)
                backend.addArrayValues('dos_values', self.dos_values)

def build_FhiAimsDosFileSimpleMatcher():
    """Builds the SimpleMatcher to parse the DOS file of FHI-aims.

    SimpleMatchers are called with 'SM (' as this string has length 4,
    which allows nice formating of nested SimpleMatchers in python.

    Returns:
       SimpleMatcher that parses DOS file of FHI-aims.
    """
    return SM (name = 'Root1',
        startReStr = "",
        forwardMatch = True,
        weak = True,
        subMatchers = [
        SM (name = 'Root2',
            startReStr = r"#.*density.*FHI-aims.*",
            sections = ['section_run'],
            subMatchers = [
            SM (name = 'Root3',
                startReStr = r"# The energy reference for this output is the vacuum level",
                sections = ['section_single_configuration_calculation'],
                subMatchers = [
                SM (name = 'Root4',
                    startReStr = r"#\s*Energy \(eV\)\s*.*",
                    sections = ['section_dos'],
                    subMatchers = [
                    SM (r"\s*(?P<x_fhi_aims_dos_energy__eV>[-+0-9.]+)\s+(?P<x_fhi_aims_dos_value_string>[-+0-9.\s]+)", repeats = True)
                    ])
                ])
            ])
        ])

def get_cachingLevelForMetaName(metaInfoEnv, CachingLvl):
    """Sets the caching level for the metadata.

    Args:
        metaInfoEnv: metadata which is an object of the class InfoKindEnv in nomadcore.local_meta_info.py.
        CachingLvl: Sets the CachingLevel for the sections dos, run, and single_configuration_calculation.
            This allows to run the parser without opening new sections.

    Returns:
        Dictionary with metaname as key and caching level as value.
    """
    # manually adjust caching of metadata
    cachingLevelForMetaName = {
                               'section_dos': CachingLvl,
                               'section_run': CachingLvl,
                               'section_single_configuration_calculation': CachingLvl,
                              }
    # Set all dos metadata to Cache as they need post-processsing.
    for name in metaInfoEnv.infoKinds:
        if name.startswith('x_fhi_aims_dos_'):
            cachingLevelForMetaName[name] = CachingLevel.Cache
    return cachingLevelForMetaName

def main(CachingLvl):
    """Main function.

    Set up everything for the parsing of the FHI-aims DOS file and run the parsing.

    Args:
        CachingLvl: Sets the CachingLevel for the sections dos, run, and single_configuration_calculation.
            This allows to run the parser without opening new sections.
    """
    # get dos file description
    FhiAimsDosSimpleMatcher = build_FhiAimsDosFileSimpleMatcher()
    # loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json
    metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
    metaInfoEnv = get_metaInfo(metaInfoPath)
    # set parser info
    parserInfo = {'name':'fhi-aims-dos-parser', 'version': '1.0'}
    # get caching level for metadata
    cachingLevelForMetaName = get_cachingLevelForMetaName(metaInfoEnv, CachingLvl)
    # start parsing
    mainFunction(mainFileDescription = FhiAimsDosSimpleMatcher,
                 metaInfoEnv = metaInfoEnv,
                 parserInfo = parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = FhiAimsDosParserContext())

if __name__ == "__main__":
    main(CachingLevel.Forward)
