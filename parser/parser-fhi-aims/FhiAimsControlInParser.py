import setup_paths
import numpy as np
import nomadcore.ActivateLogging
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import mainFunction
from nomadcore.simple_parser import SimpleMatcher as SM
from FhiAimsCommon import get_metaInfo, write_controlIn, write_k_grid, write_xc_functional
import logging, os, re, sys

############################################################
# This is the parser for the control.in file of FHI-aims.
############################################################

logger = logging.getLogger("nomad.FhiAimsControlInParser")

class FhiAimsControlInParserContext(object):
    """Context for parsing FHI-aims control.in file.

    Attributes:
        sectionRun: Stores the parsed value/sections found in section_run.

    The onClose_ functions allow processing and writing of cached values after a section is closed.
    They take the following arguments:
        backend: Class that takes care of wrting and caching of metadata.
        gIndex: Index of the section that is closed.
        section: The cached values and sections that were found in the section that is closed.
    """
    def __init__(self, writeSectionRun = True):
        """Args:
            writeSectionRun: Deteremines if metadata is written on close of section_run
                or stored in sectionRun.
        """
        self.writeSectionRun = writeSectionRun

    def startedParsing(self, fInName, parser):
        """Function is called when the parsing starts and the compiled parser is obtained.

        Args:
            fInName: The file name on which the current parser is running.
            parser: The compiled parser. Is an object of the class SimpleParser in nomadcore.simple_parser.py.
        """
        self.parser = parser
        # save metadata
        self.metaInfoEnv = parser.parserBuilder.metaInfoEnv
        # allows to reset values if the same superContext is used to parse different files
        self.sectionRun = None

    def onClose_section_run(self, backend, gIndex, section):
        """Trigger called when section_run is closed.

        Write the keywords from control.in, which belong to settings_run, or store the section.
        """
        if self.writeSectionRun:
            write_controlIn(backend = backend,
                metaInfoEnv = self.metaInfoEnv,
                valuesDict = section.simpleValues,
                writeXC = False,
                location = 'control.in',
                logger = logger)
        else:
            self.sectionRun = section

    def onClose_section_method(self, backend, gIndex, section):
        """Trigger called when fhi_aims_section_controlIn_file is closed.

        Write the keywords from control.in, which belong to section_method.
        """
        write_controlIn(backend = backend,
            metaInfoEnv = self.metaInfoEnv,
            valuesDict = section.simpleValues,
            writeXC = True,
            location = 'control.in',
            logger = logger)

def build_FhiAimsControlInKeywordsSimpleMatchers():
    """Builds the list of SimpleMatchers to parse the control.in keywords of FHI-aims.

    SimpleMatchers are called with 'SM (' as this string has length 4,
    which allows nice formating of nested SimpleMatchers in python.

    Returns:
       List of SimpleMatchers that parses control.in keywords of FHI-aims. 
    """
    # Now follows the list to match the keywords from the control.in.
    # Explicitly add ^ to ensure that the keyword is not within a comment.
    # Repating occurrences of the same keywords are captured.
    # List the matchers in alphabetical order according to keyword name.
    #
    return [
        SM (r"^\s*charge\s+(?P<fhi_aims_controlIn_charge>[-+0-9.eEdD]+)", repeats = True),
        # only the first character is important for aims
        SM (r"^\s*hse_unit\s+(?P<fhi_aims_controlIn_hse_unit>[a-zA-Z])[-_a-zA-Z0-9]+", repeats = True),
        SM (r"^\s*MD_time_step\s+(?P<fhi_aims_controlIn_MD_time_step__ps>[-+0-9.eEdD]+)", repeats = True),
        SM (r"^\s*k_grid\s+(?P<fhi_aims_controlIn_k1>[0-9]+)\s+(?P<fhi_aims_controlIn_k2>[0-9]+)\s+(?P<fhi_aims_controlIn_k3>[0-9]+)", repeats = True),
        # need to distinguish different cases
        SM (r"^\s*occupation_type\s+",
            forwardMatch = True,
            repeats = True,
            subMatchers = [
            SM (r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)\s+(?P<fhi_aims_controlIn_occupation_order>[0-9]+)"),
            SM (r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)")
            ]),
        SM (r"^\s*override_relativity\s+\.?(?P<fhi_aims_controlIn_override_relativity>[-_a-zA-Z]+)\.?", repeats = True),
        # need to distinguish different cases
        SM (r"^\s*relativistic\s+",
            forwardMatch = True,
            repeats = True,
            subMatchers = [
            SM (r"^\s*relativistic\s+(?P<fhi_aims_controlIn_relativistic>[-_a-zA-Z]+\s+[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_relativistic_threshold>[-+0-9.eEdD]+)"),
            SM (r"^\s*relativistic\s+(?P<fhi_aims_controlIn_relativistic>[-_a-zA-Z]+)")
            ]),
        SM (r"^\s*sc_accuracy_rho\s+(?P<fhi_aims_controlIn_sc_accuracy_rho>[-+0-9.eEdD]+)", repeats = True),
        SM (r"^\s*sc_accuracy_eev\s+(?P<fhi_aims_controlIn_sc_accuracy_eev>[-+0-9.eEdD]+)", repeats = True),
        SM (r"^\s*sc_accuracy_etot\s+(?P<fhi_aims_controlIn_sc_accuracy_etot>[-+0-9.eEdD]+)", repeats = True),
        SM (r"^\s*sc_accuracy_forces\s+(?P<fhi_aims_controlIn_sc_accuracy_forces>[-+0-9.eEdD]+)", repeats = True),
        SM (r"^\s*sc_accuracy_stress\s+(?P<fhi_aims_controlIn_sc_accuracy_stress>[-+0-9.eEdD]+)", repeats = True),
        SM (r"^\s*sc_iter_limit\s+(?P<fhi_aims_controlIn_sc_iter_limit>[0-9]+)", repeats = True),
        SM (r"^\s*spin\s+(?P<fhi_aims_controlIn_spin>[-_a-zA-Z]+)", repeats = True),
        SM (r"^\s*verbatim_writeout\s+\.?(?P<fhi_aims_controlIn_verbatim_writeout>[a-zA-Z]+)\.?", repeats = True),
        # need to distinguish two cases: just the name of the xc functional or name plus number (e.g. for HSE functional)
        SM (r"^\s*xc\s+",
            forwardMatch = True,
            repeats = True,
            subMatchers = [
            SM (r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)\s+(?P<fhi_aims_controlIn_hse_omega>[-+0-9.eEdD]+)"),
            SM (r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)")
            ]),
        ]

def build_FhiAimsControlInFileSimpleMatcher():
    """Builds the SimpleMatcher to parse the control.in file of FHI-aims.

    SimpleMatchers are called with 'SM (' as this string has length 4,
    which allows nice formating of nested SimpleMatchers in python.

    Returns:
       SimpleMatcher that parses control.in file of FHI-aims. 
    """
    return SM (name = 'Root1',
        startReStr = "",
        sections = ['section_run'],
        forwardMatch = True,
        weak = True,
        subMatchers = [
        SM (name = 'Root2',
            startReStr = "",
            sections = ['section_method'],
            forwardMatch = True,
            weak = True,
            # The search is done unordered since the keywords do not appear in a specific order.
            subFlags = SM.SubFlags.Unordered,
            subMatchers = build_FhiAimsControlInKeywordsSimpleMatchers()
            )
        ])

def get_cachingLevelForMetaName(metaInfoEnv, CachingLvl):
    """Sets the caching level for the metadata.

    Args:
        metaInfoEnv: metadata which is an object of the class InfoKindEnv in nomadcore.local_meta_info.py.
        CachingLvl: Sets the CachingLevel for the sections method and run. This allows to run the parser
            without opening new sections.

    Returns:
        Dictionary with metaname as key and caching level as value. 
    """
    # manually adjust caching of metadata
    cachingLevelForMetaName = {
                               'section_method': CachingLvl,
                               'section_run': CachingLvl,
                              }
    # Set all controlIn metadata to Cache to capture multiple occurrences of keywords and
    # their last value is then written by the onClose routine in the FhiAimsControlInParserContext.
    for name in metaInfoEnv.infoKinds:
        if name.startswith('fhi_aims_controlIn_'):
            cachingLevelForMetaName[name] = CachingLevel.Cache
    return cachingLevelForMetaName

def main(CachingLvl):
    """Main function.

    Set up everything for the parsing of the FHI-aims control.in file and run the parsing.

    Args:
        CachingLvl: Sets the CachingLevel for the sections method and run. This allows to run the parser
            without opening new sections
    """
    # get control.in file description
    FhiAimsControlInSimpleMatcher = build_FhiAimsControlInFileSimpleMatcher()
    # loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json
    metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
    metaInfoEnv = get_metaInfo(metaInfoPath)
    # set parser info
    parserInfo = {'name':'fhi-aims-control-in-parser', 'version': '1.0'}
    # get caching level for metadata
    cachingLevelForMetaName = get_cachingLevelForMetaName(metaInfoEnv, CachingLvl)
    # start parsing
    mainFunction(mainFileDescription = FhiAimsControlInSimpleMatcher,
                 metaInfoEnv = metaInfoEnv,
                 parserInfo = parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = FhiAimsControlInParserContext())

if __name__ == "__main__":
    main(CachingLevel.Forward)
