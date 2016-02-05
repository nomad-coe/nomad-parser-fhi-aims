import setup_paths
import numpy as np
import nomadcore.ActivateLogging
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import mainFunction
from nomadcore.simple_parser import SimpleMatcher as SM
from nomadcore.unit_conversion.unit_conversion import convert_unit_function
from FhiAimsCommon import get_metaInfo
import logging, os, re, sys

############################################################
# This is the parser for the band.out file of FHI-aims.
############################################################

logger = logging.getLogger("nomad.FhiAimsBandParser")

class FhiAimsBandParserContext(object):
    """Context for parsing FHI-aims band.out file.

    Attributes:
        band_energies: Stores parsed eigenvalues.
        band_k_points Stores parsed k-points.
        band_occupation: Stores parsed occupations.

    The onClose_ functions allow processing and writing of cached values after a section is closed.
    They take the following arguments:
        backend: Class that takes care of wrting and caching of metadata.
        gIndex: Index of the section that is closed.
        section: The cached values and sections that were found in the section that is closed.
    """
    def __init__(self, writeMetaData = True):
        """Args:
            writeMetaData: Deteremines if metadata is written or stored in class attributes.
        """
        self.writeMetaData = writeMetaData

    def startedParsing(self, fInName, parser):
        """Function is called when the parsing starts and the compiled parser is obtained.

        Args:
            fInName: The file name on which the current parser is running.
            parser: The compiled parser. Is an object of the class SimpleParser in nomadcore.simple_parser.py.
        """
        self.parser = parser
        # get unit from metadata for band energies
        unit = parser.parserBuilder.metaInfoEnv.infoKinds['band_energies'].units
        # get function for unit conversion, aims energies are in eV
        self.converter = convert_unit_function('eV', unit)
        # allows to reset values if the same superContext is used to parse different files
        self.band_energies = None
        self.band_k_points = None
        self.band_occupation = None

    def onClose_section_k_band(self, backend, gIndex, section):
        """Trigger called when section_k_band is closed.

        Store the parsed values and write them if writeMetaData is True.
        """
        # extract k-points
        band_k = []
        for i in ['1', '2', '3']:
            ki = section['fhi_aims_band_k' + i]
            if ki is not None:
                band_k.append(ki)
        if band_k:
            # need to transpose array since its shape is given by [n_k_points_per_segment,3] in the metadata
            self.band_k_points = np.transpose(np.asarray(band_k))
        # extract occupations and eigenvalues
        band_occupation = []
        band_energies = []
        if section['fhi_aims_band_occupation_eigenvalue_string'] is not None:
            for string in section['fhi_aims_band_occupation_eigenvalue_string']:
                strings = string.split()
                # first number is occupation and then every second one
                band_occupation.append(map(float, strings[0::2]))
                # second number is eigenvalue and then every second one
                # convert units
                band_energies.append(map(self.converter, map(float, strings[1::2])))

        if band_occupation:
            # do not need to transpose array since its shape is given by [n_k_points,n_eigen_values] in the metadata
            self.band_occupation = np.asarray(band_occupation)
        if band_energies:
            # do not need to transpose array since its shape is given by [n_k_points,n_eigen_values] in the metadata
            self.band_energies = np.asarray(band_energies)
        # write metadata only if values were found for all three quantities
        if self.writeMetaData:
            if band_k and band_occupation and band_energies:
                # need to add dimension for number_of_k_point_segments
                backend.addArrayValues('band_k_points', np.asarray([self.band_k_points]))
                # need to add dimension for number_of_k_point_segments and max_spin_channel
                backend.addArrayValues('band_occupation', np.asarray([[self.band_occupation]]))
                # need to add dimension for number_of_k_point_segments and max_spin_channel
                backend.addArrayValues('band_energies', np.asarray([[self.band_energies]]))

def build_FhiAimsBandFileSimpleMatcher():
    """Builds the SimpleMatcher to parse the band.out file of FHI-aims.

    SimpleMatchers are called with 'SM (' as this string has length 4,
    which allows nice formating of nested SimpleMatchers in python.

    Returns:
       SimpleMatcher that parses band.out file of FHI-aims. 
    """
    return SM (name = 'Root1',
        startReStr = "",
        sections = ['section_run'],
        forwardMatch = True,
        weak = True,
        subMatchers = [
        SM (name = 'Root2',
            startReStr = "",
            sections = ['section_single_configuration_calculation'],
            forwardMatch = True,
            weak = True,
            subMatchers = [
            SM (name = 'Root3',
                startReStr = "",
                sections = ['section_k_band'],
                forwardMatch = True,
                weak = True,
                subMatchers = [
                SM (r"\s*[0-9]+\s+(?P<fhi_aims_band_k1>[-+0-9.]+)\s+(?P<fhi_aims_band_k2>[-+0-9.]+)\s+(?P<fhi_aims_band_k3>[-+0-9.]+)\s+(?P<fhi_aims_band_occupation_eigenvalue_string>[-+0-9.\s]+)", repeats = True)
                ])
            ])
        ])

def get_cachingLevelForMetaName(metaInfoEnv, CachingLvl):
    """Sets the caching level for the metadata.

    Args:
        metaInfoEnv: metadata which is an object of the class InfoKindEnv in nomadcore.local_meta_info.py.
        CachingLvl: Sets the CachingLevel for the sections k_band, run, and single_configuration_calculation.
            This allows to run the parser without opening new sections.

    Returns:
        Dictionary with metaname as key and caching level as value. 
    """
    # manually adjust caching of metadata
    cachingLevelForMetaName = {
                               'section_k_band': CachingLvl,
                               'section_run': CachingLvl,
                               'section_single_configuration_calculation': CachingLvl,
                              }
    # Set all band metadata to Cache as they need post-processsing.
    for name in metaInfoEnv.infoKinds:
        if name.startswith('fhi_aims_band_'):
            cachingLevelForMetaName[name] = CachingLevel.Cache
    return cachingLevelForMetaName

def main(CachingLvl):
    """Main function.

    Set up everything for the parsing of the FHI-aims band.out file and run the parsing.

    Args:
        CachingLvl: Sets the CachingLevel for the sections k_band, run, and single_configuration_calculation.
            This allows to run the parser without opening new sections.
    """
    # get band.out file description
    FhiAimsBandSimpleMatcher = build_FhiAimsBandFileSimpleMatcher()
    # loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json
    metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
    metaInfoEnv = get_metaInfo(metaInfoPath)
    # set parser info
    parserInfo = {'name':'fhi-aims-band-out-parser', 'version': '1.0'}
    # get caching level for metadata
    cachingLevelForMetaName = get_cachingLevelForMetaName(metaInfoEnv, CachingLvl)
    # start parsing
    mainFunction(mainFileDescription = FhiAimsBandSimpleMatcher,
                 metaInfoEnv = metaInfoEnv,
                 parserInfo = parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = FhiAimsBandParserContext())

if __name__ == "__main__":
    main(CachingLevel.Forward)

