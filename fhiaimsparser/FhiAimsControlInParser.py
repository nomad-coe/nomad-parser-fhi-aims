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
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import mainFunction
from nomadcore.simple_parser import SimpleMatcher as SM
from fhiaimsparser.FhiAimsCommon import get_metaInfo, write_controlIn, write_k_grid, write_xc_functional
import logging
import os
import re
import sys

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
        backend: Class that takes care of writing and caching of metadata.
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

    def onClose_x_fhi_aims_section_controlIn_basis_set(self, backend, gIndex, section):
        """doc"""
        #logger.warning("Free-atom basis for %s: basis_func_type: %s n = %s l = %s radius = %s", section["x_fhi_aims_controlIn_species_name"], section["x_fhi_aims_controlIn_basis_func_type"], section["x_fhi_aims_controlIn_basis_func_n"], section["x_fhi_aims_controlIn_basis_func_l"], section["x_fhi_aims_controlIn_basis_func_radius"])

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
        SM (r"^\s*charge\s+(?P<x_fhi_aims_controlIn_charge>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
        # only the first character is important for aims
        SM (r"^\s*hse_unit\s+(?P<x_fhi_aims_controlIn_hse_unit>[a-zA-Z])[-_a-zA-Z0-9]+", repeats = True),
        SM (r"^\s*hybrid_xc_coeff\s+(?P<x_fhi_aims_controlIn_hybrid_xc_coeff>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
        SM (r"^\s*MD_time_step\s+(?P<x_fhi_aims_controlIn_MD_time_step__ps>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
        SM (r"^\s*k_grid\s+(?P<x_fhi_aims_controlIn_k1>[0-9]+)\s+(?P<x_fhi_aims_controlIn_k2>[0-9]+)\s+(?P<x_fhi_aims_controlIn_k3>[0-9]+)", repeats = True),
        # need to distinguish different cases
        SM (r"^\s*occupation_type\s+",
            forwardMatch = True,
            repeats = True,
            subMatchers = [
            SM (r"^\s*occupation_type\s+(?P<x_fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<x_fhi_aims_controlIn_occupation_width>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_controlIn_occupation_order>[0-9]+)"),
            SM (r"^\s*occupation_type\s+(?P<x_fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<x_fhi_aims_controlIn_occupation_width>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)")
            ]),
        SM (r"^\s*override_relativity\s+\.?(?P<x_fhi_aims_controlIn_override_relativity>[-_a-zA-Z]+)\.?", repeats = True),
        # need to distinguish different cases
        SM (r"^\s*relativistic\s+",
            forwardMatch = True,
            repeats = True,
            subMatchers = [
            SM (r"^\s*relativistic\s+(?P<x_fhi_aims_controlIn_relativistic>[-_a-zA-Z]+\s+[-_a-zA-Z]+)\s+(?P<x_fhi_aims_controlIn_relativistic_threshold>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"),
            SM (r"^\s*relativistic\s+(?P<x_fhi_aims_controlIn_relativistic>[-_a-zA-Z]+)")
            ]),
        SM (r"^\s*sc_accuracy_rho\s+(?P<x_fhi_aims_controlIn_sc_accuracy_rho>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
        SM (r"^\s*sc_accuracy_eev\s+(?P<x_fhi_aims_controlIn_sc_accuracy_eev>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
        SM (r"^\s*sc_accuracy_etot\s+(?P<x_fhi_aims_controlIn_sc_accuracy_etot>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
        SM (r"^\s*sc_accuracy_forces\s+(?P<x_fhi_aims_controlIn_sc_accuracy_forces>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
        SM (r"^\s*sc_accuracy_stress\s+(?P<x_fhi_aims_controlIn_sc_accuracy_stress>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
        SM (r"^\s*sc_iter_limit\s+(?P<x_fhi_aims_controlIn_sc_iter_limit>[0-9]+)", repeats = True),
        SM (r"^\s*spin\s+(?P<x_fhi_aims_controlIn_spin>[-_a-zA-Z]+)", repeats = True),
        SM (r"^\s*verbatim_writeout\s+\.?(?P<x_fhi_aims_controlIn_verbatim_writeout>[a-zA-Z]+)\.?", repeats = True),
        # need to distinguish two cases: just the name of the xc functional or name plus number (e.g. for HSE functional)
        SM (r"^\s*xc\s+",
            forwardMatch = True,
            repeats = True,
            subMatchers = [
            SM (r"^\s*xc\s+(?P<x_fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)\s+(?P<x_fhi_aims_controlIn_hse_omega>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"),
            SM (r"^\s*xc\s+(?P<x_fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)")
            ]),


 #       SM (r"\s*\#\s*Definition of \"minimal\" basis", repeats=True,
        SM (r"^\s*species\s*(?P<x_fhi_aims_controlIn_species_name>[a-zA-Z]+)",
        #SM (r"^\s*division", repeats=True,
            sections = ["x_fhi_aims_section_controlIn_basis_set"],
            repeats=True,
            subFlags = SM.SubFlags.Unordered,
            subMatchers = [
                SM(r"\s*nucleus\s+(?P<x_fhi_aims_controlIn_nucleus>[0-9.]+)\s*"),
                SM(r"\s*mass\s+(?P<x_fhi_aims_controlIn_mass>[0-9.]+)\s*"),
                SM(r"\s*l_hartree\s+(?P<x_fhi_aims_controlIn_l_hartree>[0-9]+)\s*"),
                SM(r"\s*cut_pot\s+(?P<x_fhi_aims_controlIn_cut_pot1>[0-9.]+)\s+(?P<x_fhi_aims_controlIn_cut_pot2>[0-9.]+)\s+(?P<x_fhi_aims_controlIn_cut_pot3>[0-9.]+)\s*"),
                SM(r"\s*basis_dep_cutoff\s+(?P<x_fhi_aims_controlIn_basis_dep_cutoff>[-+0-9.dDeE]+)\s"),
                SM(r"\s*radial_base\s+(?P<x_fhi_aims_controlIn_radial_base1>[0-9]+)\s+(?P<x_fhi_aims_controlIn_radial_base2>[-+0-9.dDeE]+)\s*"),
                SM(r"\s*radial_multiplier\s+(?P<x_fhi_aims_controlIn_radial_multiplier>[0-9]+)\s*"),
                SM(name = "angular_grids",
                   startReStr = r"\s*angular_grids\s+(?P<x_fhi_aims_controlIn_angular_grids_method>specified|auto)\s*",
                   endReStr = r"\s*outer_grid\s+\s*(?P<x_fhi_aims_controlIn_outer_grid>[0-9]+)\s*",
                   subMatchers = [
                       SM(r"\s*division\s*(?P<x_fhi_aims_controlIn_division1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*(?P<x_fhi_aims_controlIn_division2>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*", repeats = True),
                   ]),
	        SM (r"^\s*(?P<x_fhi_aims_controlIn_basis_func_type>gaussian|hydro|valence|ion_occ|ionic|confined)"
                    "\s*(?P<x_fhi_aims_controlIn_basis_func_n>[0-9]+)"
                    "\s+(?P<x_fhi_aims_controlIn_basis_func_l>[spdefghijklm])"
                    "\s+(?P<x_fhi_aims_controlIn_basis_func_radius>[.0-9]+)",
                    repeats = True,
                    sections = ["x_fhi_aims_section_controlIn_basis_func"]
                ),
	        SM (r"^\s*(?P<x_fhi_aims_controlIn_basis_func_type>gaussian|hydro|valence|ion_occ|ionic|confined)"
                    "\s*(?P<x_fhi_aims_controlIn_basis_func_n>[0-9]+)"
                    "\s*(?P<x_fhi_aims_controlIn_basis_func_l>[spdefghijklm])\s*auto",
                    repeats = True,
                    sections = ["x_fhi_aims_section_controlIn_basis_func"])
            ])
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
        metaInfo = metaInfoEnv.infoKinds[name]
        if (name.startswith('x_fhi_aims_controlIn_') and
            metaInfo.kindStr == "type_document_content" and
            ("x_fhi_aims_controlIn_method" in metaInfo.superNames or "x_fhi_aims_controlIn_run" in metaInfo.superNames)):
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
