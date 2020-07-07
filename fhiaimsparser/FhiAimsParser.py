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
from builtins import map
from builtins import range
from builtins import object
try:
    import setup_paths
except ImportError:
    pass

import numpy as np
from nomadcore.caching_backend import CachingLevel
from nomadcore.simple_parser import AncillaryParser, mainFunction
from nomadcore.simple_parser import SimpleMatcher as SM
from fhiaimsparser.FhiAimsCommon import get_metaInfo, write_controlIn, write_k_grid, write_xc_functional
from fhiaimsparser import FhiAimsControlInParser
from fhiaimsparser import FhiAimsBandParser
from fhiaimsparser import FhiAimsDosParser
import logging
import os
import re
import sys

from nomad.units import ureg


############################################################
# This is the parser for the main file of FHI-aims.
#
# REMARKS
#
# Energies are parsed in eV to get consistent values
# since not all energies are given in hartree.
############################################################

logger = logging.getLogger('nomad.FhiAimsParser')


class FhiAimsParserContext(object):
    """Context for parsing FHI-aims main file.

    This class keeps tracks of several aims settings to adjust the parsing to them.
    The onClose_ functions allow processing and writing of cached values after a section is closed.
    They take the following arguments:
        backend: Class that takes care of writing and caching of metadata.
        gIndex: Index of the section that is closed.
        section: The cached values and sections that were found in the section that is closed.
    """
    def __init__(self):
        # dictionary of energy values, which are tracked between SCF iterations and written after convergence
        self.totalEnergyList = {
                                'energy_sum_eigenvalues': None,
                                'energy_XC_potential': None,
                                'energy_correction_hartree': None,
                                'energy_correction_entropy': None,
                                'electronic_kinetic_energy': None,
                                'energy_electrostatic': None,
                                'energy_hartree_error': None,
                                'energy_sum_eigenvalues_per_atom': None,
                                'energy_total_T0_per_atom': None,
                                'energy_free_per_atom': None,
                               }
        # dictionary for conversion of relativistic treatment in aims to metadata format
        self.relativisticDict = {
                                 'Non-relativistic': '',
                                 'ZORA': 'scalar_relativistic',
                                 'on-site free-atom approximation to ZORA': 'scalar_relativistic_atomic_ZORA',
                                }

    def initialize_values(self):
        """Initializes the values of certain variables.

        This allows a consistent setting and resetting of the variables,
        when the parsing starts and when a section_run closes.
        """
        self.secMethodIndex = None
        self.secSystemDescriptionIndex = None
        self.inputMethodIndex = None
        self.mainMethodIndex = None
        self.mainCalcIndex = None
        self.vdWMethodIndex = None
        self.maxSpinChannel = 0
        self.scalarZORA = False
        self.periodicCalc = False
        self.MD = False
        self.MDUnitCell = None
        self.unit_cell_volume = None
        self.fermi_energy = None
        self.band_segm_start_end = None
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        self.singleConfCalcs = []
        self.scfConvergence = False
        self.geoConvergence = None
        self.parsedControlInFile = False
        self.controlInSuperContext = None
        self.eigenvalues_occupation = []
        self.eigenvalues_values = []
        self.eigenvalues_kpoints = []
        self.forces_raw = []
        self.dosSuperContext = None
        self.dosParser = None
        self.dosRefSingleConfigurationCalculation = None
        self.dosFound = False
        self.dos_energies = None
        self.dos_values = None
        self.lastCalculationGIndex = None

    def startedParsing(self, fInName, parser):
        """Function is called when the parsing starts.

        Get compiled parser, filename and metadata.

        Args:
            fInName: The file name on which the current parser is running.
            parser: The compiled parser. Is an object of the class SimpleParser in nomadcore.simple_parser.py.
        """
        self.parser = parser
        self.fName = fInName
        # save metadata
        self.metaInfoEnv = self.parser.parserBuilder.metaInfoEnv
        # allows to reset values if the same superContext is used to parse different files
        self.initialize_values()

    def update_eigenvalues(self, section, addStr):
        """Update eigenvalues and occupations if they were found in the given section.

        addStr allows to use this function for the eigenvalues of scalar ZORA
        by extending the metadata with addStr.

        Args:
            section: Contains the cached sections and values.
            addStr: String that is appended to the metadata names.
        """
        # get cached fhi_aims_section_eigenvalues_group
        sec_ev_group = section['x_fhi_aims_section_eigenvalues_group%s' % addStr]
        if sec_ev_group is not None:
            # check if only one group was found
            if len(sec_ev_group) != 1:
                logger.warning("Found %d instead of 1 group of eigenvalues. Used last occurance." % len(sec_ev_group))
            # get cached fhi_aims_section_eigenvalues_group
            sec_ev_spins = sec_ev_group[-1]['x_fhi_aims_section_eigenvalues_spin%s' % addStr]
            for sec_ev_spin in sec_ev_spins:
                occs = []
                evs = []
                kpoints = []
                # get cached fhi_aims_section_eigenvalues_list
                sec_ev_lists = sec_ev_spin['x_fhi_aims_section_eigenvalues_list%s' % addStr]
                # extract occupations and eigenvalues
                for sec_ev_list in sec_ev_lists:
                    occ = sec_ev_list['x_fhi_aims_eigenvalue_occupation%s' % addStr]
                    if occ is not None:
                        occs.append(occ)
                    ev = sec_ev_list['x_fhi_aims_eigenvalue_eigenvalue%s' % addStr]
                    if ev is not None:
                        evs.append(ev)
                # extract kpoints
                for i in ['1', '2', '3']:
                    ki = sec_ev_spin['x_fhi_aims_eigenvalue_kpoint%s%s' % (i, addStr)]
                    if ki is not None:
                        kpoints.append(ki)
                # append values for each spin channel
                self.eigenvalues_occupation.append(occs)
                self.eigenvalues_values.append(evs)
                if kpoints:
                    # transpose list
                    kpoints = list(map(lambda *x: list(x), *kpoints))
                    self.eigenvalues_kpoints.append(kpoints)

    def compile_dos_parser(self):
        """Instantiate superContext and construct parser for DOS file.
        """
        self.dosSuperContext = FhiAimsDosParser.FhiAimsDosParserContext(False)
        self.dosParser = AncillaryParser(
            fileDescription = FhiAimsDosParser.build_FhiAimsDosFileSimpleMatcher(),
            parser = self.parser,
            cachingLevelForMetaName = FhiAimsDosParser.get_cachingLevelForMetaName(self.metaInfoEnv, CachingLevel.Ignore),
            superContext = self.dosSuperContext)

    def onClose_section_run(self, backend, gIndex, section):
        """Trigger called when section_run is closed.

        Write convergence of geometry optimization.
        Write the keywords from control.in and the aims output from the parsed control.in, which belong to settings_run.
        Write the last occurrence of a keyword/setting, i.e. [-1], since aims uses the last occurrence of a keyword.
        Variables are reset to ensure clean start for new run.

        ATTENTION
        backend.superBackend is used here instead of only the backend to write the JSON values,
        since this allows to bybass the caching setting which was used to collect the values for processing.
        However, this also bypasses the checking of validity of the metadata name by the backend.
        The scala part will check the validity nevertheless.
        """
        # write dos
        # The explanation why we write the dos not unil section_run closes is given in onClose_section_dos.
        if self.dos_energies is not None and self.dos_values is not None:

            gIndexTmp = backend.superBackend.openSection('section_dos', parent_index=self.dosRefSingleConfigurationCalculation)
            backend.superBackend.addArrayValues('dos_energies', self.dos_energies)
            backend.superBackend.addArrayValues('dos_values', self.dos_values)
            backend.superBackend.closeSection('section_dos', gIndexTmp)

        # write geometry optimization convergence
        if self.geoConvergence is not None:
            backend.addValue('x_fhi_aims_geometry_optimization_converged', self.geoConvergence)
        # use values of control.in which was parsed in section_method

        if self.parsedControlInFile:
            if self.controlInSuperContext.sectionRun is not None:
                valuesDict = self.controlInSuperContext.sectionRun.simpleValues
            else:
                valuesDict = {}
            location = 'control.in',
        # otherwise use values of the verbatim writeout of control.in
        else:
            valuesDict = section.simpleValues
            location = 'verbatim writeout of control.in',
        # write settings of control.in
        write_controlIn(backend = backend,
            metaInfoEnv = self.metaInfoEnv,
            valuesDict = valuesDict,
            writeXC = False,
            location = location,
            logger = FhiAimsControlInParser.logger)
        # write settings of aims output from the parsed control.in
        for k,v in section.simpleValues.items():
            if k.startswith('x_fhi_aims_controlInOut_'):
                backend.superBackend.addValue(k, v[-1])
        # reset all variables
        self.initialize_values()
        # check for geometry optimization convergence
        if section['x_fhi_aims_geometry_optimization_converged'] is not None:
            if section['x_fhi_aims_geometry_optimization_converged'][-1] == 'is converged':
                self.geoConvergence = True
            else:
                self.geoConvergence = False
        if self.geoConvergence:
            sampling_method = "geometry_optimization"
        elif len(self.singleConfCalcs) > 1:
            pass # to do
        else:
            return
        samplingGIndex = backend.openSection("section_sampling_method")
        backend.addValue("sampling_method", sampling_method)
        backend.closeSection("section_sampling_method", samplingGIndex)
        frameSequenceGIndex = backend.openSection("section_frame_sequence")
        backend.addValue("frame_sequence_to_sampling_ref", samplingGIndex)
        backend.addArrayValues("frame_sequence_local_frames_ref", np.asarray(self.singleConfCalcs))
        backend.closeSection("section_frame_sequence", frameSequenceGIndex)

    def onClose_x_fhi_aims_section_MD_detect(self, backend, gIndex, section):
        """Trigger called when fhi_aims_section_MD_detect is closed.

        Detect if a MD run was performed.
        """
        self.MD = True

    def onOpen_section_method(self, backend, gIndex, section):
        # keep track of the latest method section
        self.secMethodIndex = gIndex
        if self.inputMethodIndex is None:
            self.inputMethodIndex = gIndex
        if self.mainMethodIndex is None:
            self.mainMethodIndex = gIndex

    def onClose_section_method(self, backend, gIndex, section):
        """Trigger called when section_method is closed.
        """
        ## input method
        if gIndex == self.inputMethodIndex:
            self.closingInputMethodSection(backend, gIndex, section)
        if section["electronic_structure_method"]:
            m = section["electronic_structure_method"][0]
            if m == "DFT":
                backend.openNonOverlappingSection("section_method_to_method_refs")
                backend.addValue("method_to_method_kind", "core_settings")
                backend.addValue("method_to_method_ref", self.inputMethodIndex)
                backend.closeNonOverlappingSection("section_method_to_method_refs")
                self.closingDFTMethodSection(backend, gIndex, section)
            elif m == "G0W0":
                backend.openNonOverlappingSection("section_method_to_method_refs")
                backend.addValue("method_to_method_kind", "starting_point")
                backend.addValue("method_to_method_ref", self.inputMethodIndex)
                backend.closeNonOverlappingSection("section_method_to_method_refs")
                self.closingG0W0MethodSection(backend, gIndex, section)
            elif m == "scGW":
                backend.openNonOverlappingSection("section_method_to_method_refs")
                backend.addValue("method_to_method_kind", "starting_point")
                backend.addValue("method_to_method_ref", self.inputMethodIndex)
                backend.closeNonOverlappingSection("section_method_to_method_refs")
                self.closingScGWMethodSection(backend, gIndex, section)
            else:
                backend.pwarn("unexpected electronic structure method value %s" % m)
        elif section["van_der_Waals_method"]:
            self.closingVdWMethodSection(backend, gIndex, section)
        else:
            backend.pwarn("unexpected section_method %s" % section)

    def closingDFTMethodSection(self, backend, gIndex, section):
        """Called when section_method that contains the DFT calculation is closed."""
        self.mainMethodIndex = gIndex
        self.vdWMethodIndex = None

    def closingG0W0MethodSection(self, backend, gIndex, section):
        """Called when section_method that should contain the G0W0 is closed."""
        pass

    def closingScGWMethodSection(self, backend, gIndex, section):
        """Called when section_method that should contain the scGW is closed."""
        pass

    def closingVdWMethodSection(self, backend, gIndex, section):
        """Called when section_method that should contain the vdW is closed."""
        self.vdWMethodIndex = gIndex

    def closingInputMethodSection(self, backend, gIndex, section):
        """Called when section_method that should contain the main input is closed.

        Write the keywords from control.in and the aims output from the parsed control.in, which belong to section_method.
        Write the last occurrence of a keyword/setting, i.e. [-1], since aims uses the last occurrence of a keyword.
        Detect MD and scalar ZORA here.
        Write metadata for XC-functional and relativity_method using the dictionaries xcDict and relativisticDict.

        ATTENTION
        backend.superBackend is used here instead of only the backend to write the JSON values,
        since this allows to bybass the caching setting which was used to collect the values for processing.
        However, this also bypasses the checking of validity of the metadata name by the backend.
        The scala part will check the validity nevertheless.
        """
        # check if control.in keywords were found or verbatim_writeout is false
        verbatim_writeout = True
        counter = 0
        for k,v in section.simpleValues.items():
            if k == 'x_fhi_aims_controlIn_verbatim_writeout':
                # only the first letter is important for aims
                if v[-1][0] in ['f', 'F']:
                    verbatim_writeout = False
                    break
            if k.startswith('x_fhi_aims_controlIn_'):
                counter += 1
        # write settings of verbatim writeout of control.in if it was found
        if counter != 0 and verbatim_writeout:
            write_controlIn(backend = backend,
                metaInfoEnv = self.metaInfoEnv,
                valuesDict = section.simpleValues,
                writeXC = True,
                location = 'verbatim writeout of control.in',
                logger = logger)
        # otherwise parse control.in file
        else:
            logger.warning("Found no verbatim writeout of control.in. I will try to parse the control.in file directly.")
            fName = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(self.fName)), "control.in"))
            try:
                with open(fName) as fIn:
                    # construct parser for control.in file
                    # metadata belonging to section_run is not written but stored in the superContext
                    self.controlInSuperContext = FhiAimsControlInParser.FhiAimsControlInParserContext(False)
                    self.controlInParser = AncillaryParser(
                        fileDescription = FhiAimsControlInParser.build_FhiAimsControlInFileSimpleMatcher(),
                        parser = self.parser,
                        cachingLevelForMetaName = FhiAimsControlInParser.get_cachingLevelForMetaName(self.metaInfoEnv, CachingLevel.Ignore),
                        superContext = self.controlInSuperContext)
                    # parse control.in file
                    self.controlInParser.parseFile(fIn)
                    # set flag so that metadata of control.in file is also written on close of section_run
                    self.parsedControlInFile = True
            except IOError:
                logger.warning("Could not find control.in file in directory '%s'. No metadata for fhi_aims_controlIn was written." % os.path.dirname(os.path.abspath(self.fName)))
        # list of excluded metadata for writeout
        # k_grid is written with k1 so that k2 and k2 are not needed.
        # The xc setting have to be handeled separatly since having more than one gives undefined behavior.
        # hse_omega is only written if HSE was used and converted according to hse_unit which is not written since not needed.
        # hybrid_xc_coeff is only written for hybrid functionals.
        exclude_list = [
                        'x_fhi_aims_controlInOut_k2',
                        'x_fhi_aims_controlInOut_k3',
                        'x_fhi_aims_controlInOut_xc',
                        'x_fhi_aims_controlInOut_hse_omega',
                        'x_fhi_aims_controlInOut_hse_unit',
                        'x_fhi_aims_controlInOut_hybrid_xc_coeff',
                       ]
        # band releated data does not change the calculation and will be processed separatly
        for name in self.metaInfoEnv.infoKinds:
            if name.startswith('x_fhi_aims_controlInOut_band_'):
                exclude_list.append(name)
        # write settings of aims output from the parsed control.in
        for k,v in section.simpleValues.items():
            if k.startswith('x_fhi_aims_controlInOut_'):
                if k in exclude_list:
                    continue
                # write k_krid
                elif k == 'x_fhi_aims_controlInOut_k1':
                    write_k_grid(backend, 'x_fhi_aims_controlInOut_k', section.simpleValues)
                elif k == 'x_fhi_aims_controlInOut_relativistic_threshold':
                    # write threshold only for scalar ZORA
                    if section['x_fhi_aims_controlInOut_relativistic'] is not None:
                        if section['x_fhi_aims_controlInOut_relativistic'][-1] == 'ZORA':
                            backend.superBackend.addValue(k, v[-1])
                # default writeout
                else:
                    backend.superBackend.addValue(k, v[-1])
        # detect scalar ZORA
        if section['x_fhi_aims_controlInOut_relativistic'] is not None:
            if section['x_fhi_aims_controlInOut_relativistic'][-1] == 'ZORA':
                self.scalarZORA = True
        # get number of spin channels
        if section['x_fhi_aims_controlInOut_number_of_spin_channels'] is not None:
            self.maxSpinChannel = section['x_fhi_aims_controlInOut_number_of_spin_channels'][-1]
        # convert relativistic setting to metadata string
        InOut_relativistic = section['x_fhi_aims_controlInOut_relativistic']
        if InOut_relativistic is not None:
            relativistic = self.relativisticDict.get(InOut_relativistic[-1])
            if relativistic is not None:
                backend.addValue('relativity_method', relativistic)
            else:
                logger.warning("The relativistic setting '%s' could not be converted to the required string for the metadata 'relativity_method'. Please add it to the dictionary relativisticDict." % InOut_relativistic[-1])
        # handling of xc functional
        write_xc_functional(backend = backend,
            metaInfoEnv = self.metaInfoEnv,
            metaNameStart = 'x_fhi_aims_controlInOut',
            valuesDict = section.simpleValues,
            location = 'FHI-aims output from the parsed control.in',
            logger = logger)
        # handle start end end points of band segments
        start_k = []
        end_k = []
        for i in ['1', '2', '3']:
            ski = section['x_fhi_aims_controlInOut_band_segment_start' + i]
            eki = section['x_fhi_aims_controlInOut_band_segment_end' + i]
            if ski is not None and eki is not None:
                start_k.append(ski)
                end_k.append(eki)
        if start_k and end_k:
            # need to transpose arrays since the innermost dimension is 3 according to the metadata
            band_segm_start = np.transpose(np.asarray(start_k))
            band_segm_end = np.transpose(np.asarray(end_k))
            # check if start and end have the same dimensions
            if band_segm_start.shape == band_segm_end.shape:
                # construct array that has the dimensions [number_of_k_point_segments,2,3] according to the metadata
                shape = band_segm_start.shape
                self.band_segm_start_end = np.empty([shape[0], 2, shape[1]])
                self.band_segm_start_end[:, 0, :] = band_segm_start
                self.band_segm_start_end[:, 1, :] = band_segm_end
            else:
                logger.error("The shape %s of array band_segm_start and the shape %s of array band_segm_end are inconsistent." % (band_segm_start.shape, band_segm_end.shape))

    def onOpen_section_system(self, backend, gIndex, section):
        # keep track of the latest system description section
        self.secSystemDescriptionIndex = gIndex

    def onClose_section_system(self, backend, gIndex, section):
        """Trigger called when section_system is closed.

        Writes atomic positions, atom labels and lattice vectors.
        """
        # Write atomic geometry in the case of MD only if there has been SCF iterations
        # because the atomic geometry together with the velocities are repeated after the finished SCF cycle.
        if not self.MD or self.scfIterNr > -1:
            # write atomic positions
            atom_pos = []
            for i in ['x', 'y', 'z']:
                api = section['x_fhi_aims_geometry_atom_positions_' + i]
                if api is not None:
                    atom_pos.append(api)
            if atom_pos:
                # need to transpose array since its shape is [number_of_atoms,3] in the metadata
                backend.addArrayValues('atom_positions', np.transpose(np.asarray(atom_pos)))
            # write atom labels
            atom_labels = section['x_fhi_aims_geometry_atom_labels']
            if atom_labels is not None:
                backend.addArrayValues('atom_labels', np.asarray(atom_labels))
            # write atomic velocities in the case of MD
            if self.MD:
                atom_vel = []
                for i in ['x', 'y', 'z']:
                    avi = section['x_fhi_aims_geometry_atom_velocity_' + i]
                    if avi is not None:
                        atom_vel.append(avi)
                if atom_vel:
                    # need to transpose array since its shape is [number_of_atoms,3] in the metadata
                    backend.addArrayValues('atom_velocities', np.transpose(np.asarray(atom_vel)))
        # For MD, the coordinates of the unit cell are not repeated.
        # Therefore, we have to store the unit cell, which was read in the beginning, i.e. scfIterNr == -1.
        if not self.MD or self.scfIterNr == -1:
            # write/store unit cell if present and set flag self.periodicCalc
            unit_cell = []
            for i in ['x', 'y', 'z']:
                uci = section['x_fhi_aims_geometry_lattice_vector_' + i]
                if uci is not None:
                    unit_cell.append(uci)
            if unit_cell:
                unit_cell = np.transpose(unit_cell)
                # from metadata: "The first index is x,y,z and the second index the lattice vector."
                # => unit_cell has already the right format
                if self.MD:
                    self.MDUnitCell = unit_cell
                else:
                    backend.addArrayValues('simulation_cell', unit_cell)
                    backend.addArrayValues('configuration_periodic_dimensions', np.asarray([True, True, True]))
                self.periodicCalc = True
                # save the unit cell volume for DOS normalization
                self.unit_cell_volume = np.abs(np.linalg.det(unit_cell))
            # If unit cell information was not set, then this calculation is non-periodic.
            else:
                backend.addArrayValues('configuration_periodic_dimensions', np.asarray([False, False, False]))
                self.periodicCalc = False

        # write stored unit cell in case of MD
        if self.MD and self.scfIterNr > -1:
            if self.periodicCalc:
                backend.addArrayValues('simulation_cell', self.MDUnitCell)
                backend.addArrayValues('configuration_periodic_dimensions', np.asarray([True, True, True]))
            else:
                backend.addArrayValues('configuration_periodic_dimensions', np.asarray([False, False, False]))

    def onOpen_section_single_configuration_calculation(self, backend, gIndex, section):
        # write the references to section_method and section_system
        backend.addValue('single_configuration_to_calculation_method_ref', self.secMethodIndex)
        backend.addValue('single_configuration_calculation_to_system_ref', self.secSystemDescriptionIndex)
        self.singleConfCalcs.append(gIndex)

    def onClose_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is closed.

        Write number of SCF iterations and convergence.
        Check for convergence of geometry optimization.
        Write eigenvalues.
        Write converged energy values (not for scalar ZORA as these values are extracted separatly).
        Write reference to section_method and section_system
        """
        # write number of SCF iterations
        backend.addValue('number_of_scf_iterations', self.scfIterNr)
        # write SCF convergence and reset
        backend.addValue('single_configuration_calculation_converged', self.scfConvergence)
        self.scfConvergence = False
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        # write eigenvalues if found
        if self.eigenvalues_occupation and self.eigenvalues_values:

            occ = np.asarray(self.eigenvalues_occupation)
            ev = np.asarray(self.eigenvalues_values)
            # check if there is the same number of spin channels
            if len(occ) == len(ev):
                kpt = None
                if self.eigenvalues_kpoints:
                    kpt = np.asarray(self.eigenvalues_kpoints)
                # check if there is the same number of spin channels for the periodic case
                if kpt is None or len(kpt) == len(ev):
                    gIndexTmp = backend.openSection('section_eigenvalues')
                    backend.addArrayValues('eigenvalues_occupation', occ)
                    backend.addArrayValues('eigenvalues_values', ev)
                    if kpt is not None:
                        for ispin in range(1,kpt.shape[0]):
                            if not (abs(kpt[ispin]-kpt[0])<1.0e-6).all():
                                raise Exception("k point coordinates of various spin channels differ")
                        backend.addArrayValues('eigenvalues_kpoints', kpt[0])
                    backend.closeSection('section_eigenvalues', gIndexTmp)
                else:
                    logger.warning("Found %d spin channels for eigenvalue kpoints but %d for eigenvalues in single configuration calculation %d." % (len(kpt), len(ev), gIndex))
            else:
                logger.warning("Found %d spin channels for eigenvalue occupation but %d for eigenvalues in single configuration calculation %d." % (len(occ), len(ev), gIndex))
        # write converged energy values
        # With scalar ZORA, the correctly scaled energy values are given in a separate post-processing step, which are read there.
        # Therefore, we do not write the energy values for scalar ZORA.
        if not self.scalarZORA:
            for k,v in self.totalEnergyList.items():
                if v is not None:
                    backend.addValue(k, v[-1]) # v is a list
        # write forces
        forces_free = []
        for i in ['x', 'y', 'z']:
            fi = section['x_fhi_aims_atom_forces_free_' + i]
            if fi is not None:
                forces_free.append(fi)
        if forces_free:
            # need to transpose array since its shape is [number_of_atoms,3] in the metadata
            backend.addArrayValues('atom_forces_free', np.transpose(np.asarray(forces_free)))
        if self.forces_raw:
            # need to transpose array since its shape is [number_of_atoms,3] in the metadata
            backend.addArrayValues('atom_forces_free_raw', np.transpose(np.asarray(self.forces_raw)))
        # get reference to current section_single_configuration_calculation if DOS was found in there
        if self.dosFound:
            self.dosRefSingleConfigurationCalculation = gIndex
            self.dosFound = False

        # Save the Fermi energy from the last converged SCF step
        try:
            scf_iterations = section["section_scf_iteration"]
            last_iteration = scf_iterations[-1]
            self.fermi_energy = (last_iteration['x_fhi_aims_energy_reference_fermi'][0] * ureg.eV).to(ureg.J).m
            backend.addArrayValues('energy_reference_fermi', [self.fermi_energy, self.fermi_energy])
        except Exception:
            pass

        self.lastCalculationGIndex = gIndex

    def onClose_x_fhi_aims_section_eigenvalues_ZORA(self, backend, gIndex, section):
        """Trigger called when fhi_aims_section_eigenvalues_ZORA is closed.

        Eigenvalues are extracted.
        """
        # reset eigenvalues
        self.eigenvalues_occupation = []
        self.eigenvalues_values = []
        self.eigenvalues_kpoints = []
        self.update_eigenvalues(section, '_ZORA')

    def onClose_section_scf_iteration(self, backend, gIndex, section):
        """Trigger called when section_scf_iteration is closed.

        Count number of SCF iterations and check for convergence.
        Energy values are tracked (not for scalar ZORA as these values are extracted separatly).
        Eigenvalues are tracked (not for scalar ZORA, see onClose_x_fhi_aims_section_eigenvalues_ZORA).

        The quantitties that are tracked during the SCF cycle are reset to default.
        I.e. scalar values are allowed to be set None, and lists are set to empty.
        This ensures that after convergence only the values associated with the last SCF iteration are written
        and not some values that appeared in earlier SCF iterations.
        """
        # count number of SCF iterations
        self.scfIterNr += 1
        # check for SCF convergence
        if section['x_fhi_aims_single_configuration_calculation_converged'] is not None:
            self.scfConvergence = True
        # keep track of energies
        # With scalar ZORA, the correctly scaled energy values and eigenvalues are given in a separate post-processing step,
        # which are read there. Therefore, we do not track the energy values for scalar ZORA.
        if not self.scalarZORA:
            for k in self.totalEnergyList:
                self.totalEnergyList[k] = section[k + '_scf_iteration']
            # reset eigenvalues
            self.eigenvalues_occupation = []
            self.eigenvalues_values = []
            self.eigenvalues_kpoints = []
            self.update_eigenvalues(section, '')
        # keep track of raw forces
        self.forces_raw = []
        for i in ['x', 'y', 'z']:
            fi = section['x_fhi_aims_atom_forces_raw_' + i]
            if fi is not None:
                self.forces_raw.append(fi)

    def onClose_section_dos(self, backend, gIndex, section):
        """Trigger called when section_dos is closed.

        DOS is parsed from external file but the result is only stored and written later in fhi_aims_section_dos
        in section_run. This is done because aims writes the non-perturbative DOS after every relaxation/MD step.
        I.e., we will encounter the DOS output statement for every step (single configuration calculation). However,
        there is only one file, which corresponds to the last step. Since we cannot figure out if we are in the last
        step while we are in section_single_configuration_calculation, we write the stored DOS in onClose_section_run
        with a reference to the last section_single_configuration_calculation where a DOS was found. For the
        perturpative DOS (aims keyword dos_kgrid_factors) this is not necessary, but is still done to be consistent.
        """
        # reset dos
        self.dos_energies = None
        self.dos_values = None
        # construct file name
        dirName = os.path.dirname(os.path.abspath(self.fName))
        dFile = 'KS_DOS_total_raw.dat'
        fName = os.path.normpath(os.path.join(dirName, dFile))
        try:
            with open(fName) as fIn:
                # construct parser for DOS file if not present
                if self.dosSuperContext is None or self.dosParser is None:
                    self.compile_dos_parser()
                # forward the unit cell volume and fermi energy to the DOS parser for
                # normalization
                self.dosSuperContext.fermi_energy = self.fermi_energy
                self.dosSuperContext.unit_cell_volume = self.unit_cell_volume
                # parse DOS file
                self.dosParser.parseFile(fIn)
                # set flag that DOS was parsed successfully and store values
                if self.dosSuperContext.dos_energies is not None and self.dosSuperContext.dos_values is not None:
                    self.dosFound = True
                    self.dos_energies = self.dosSuperContext.dos_energies
                    self.dos_values = self.dosSuperContext.dos_values
                else:
                    logger.error("DOS parsing unsuccessful. Parsing of file %s in directory '%s' did not yield energies or values for DOS." % (dFile, dirName))
        except IOError:
            logger.error("DOS parsing unsuccessful. Could not find %s file in directory '%s'." % (dFile, dirName))

    def onClose_section_species_projected_dos(self, backend, gIndex, section):
        """Trigger called when section_species_projected_dos is closed.

        The extracted file names for the species projected DOS are parsed.
        Values are written according to metadata.
        """
        files = section['x_fhi_aims_species_projected_dos_file']
        labels = section['x_fhi_aims_species_projected_dos_species_label']
        if files is not None and labels is not None:
            returnVal = self.parse_projected_dos(backend, files, 'species')
            # write species label if parsing of files was successful
            # The species label is repeated maxSpinChannel times in the list of parsed species labels.
            # Therefore, take only every maxSpinChannel-th element from the list.
            metaName = 'species_projected_dos_species_label'
            if returnVal == 0:
                backend.addArrayValues('species_projected_dos_species_label', np.asarray(labels[::self.maxSpinChannel]))

    def onClose_section_atom_projected_dos(self, backend, gIndex, section):
        """Trigger called when section_atom_projected_dos is closed.

        The extracted file names for the atom projected DOS are parsed.
        Values are written according to metadata.
        """
        files = section['x_fhi_aims_atom_projected_dos_file']
        if files is not None:
            self.parse_projected_dos(backend, files, 'atom')

    def parse_projected_dos(self, backend, files, kind):
        """Parses projected DOS from files and writes values.

        This function can be used to parse the atom and species projected DOS.

        Args:
            backend: Class that takes care of writing and caching of metadata.
            files: List of files that contain projected DOS.
            kind: Specifies if 'atom' or species' are parsed.

        Returns:
             0 if parsing was successful.
            -1 if parsing was unsuccessful.
        """
        # kind for small and capital letter at the beginning
        kind = kind.lower()
        Kind = kind.capitalize()
        dos_energies = None
        dos_total = []
        dos_l = []
        n_l = []
        max_n_l = 0
        # get directiory of currently parsed file
        dirName = os.path.dirname(os.path.abspath(self.fName))
        # check if number of files is divisible by maxSpinChannel
        if len(files) % self.maxSpinChannel == 0:
            # construct parser for DOS file if not present
            if self.dosSuperContext is None or self.dosParser is None:
                self.compile_dos_parser()
            # loop over files with step length maxSpinChannel
            for i in range(0, len(files), self.maxSpinChannel):
                # we create list of files that are of the same kind but have a different spin channel
                files_spin = files[i:i + self.maxSpinChannel]
                dos_total_spin = []
                dos_l_spin = []
                n_l_spin = []
                # loop over spin channels
                for dFile in files_spin:
                    # construct file name
                    fName = os.path.normpath(os.path.join(dirName, dFile))
                    try:
                        with open(fName) as fIn:
                            # forward the unit cell volume to the DOS parser
                            # for normalization
                            self.dosSuperContext.unit_cell_volume = self.unit_cell_volume
                            # parse DOS file
                            self.dosParser.parseFile(fIn)
                            # extract values
                            if self.dosSuperContext.dos_energies is not None and self.dosSuperContext.dos_values is not None:
                                # check if energies are consistent for different files
                                if dos_energies is None:
                                    dos_energies = self.dosSuperContext.dos_energies
                                elif not np.array_equal(dos_energies, self.dosSuperContext.dos_energies):
                                    logger.error("%s projected DOS parsing unsuccessful. The energies in %s are not consistent with other files in directory '%s'." % (Kind, dFile, dirName))
                                    return -1
                                # check that at least two columns for dos_values were found
                                val_length = len(self.dosSuperContext.dos_values)
                                if val_length > 1:
                                    # first column is total DOS
                                    dos_total_spin.append(self.dosSuperContext.dos_values[0])
                                    # the column afterwards are the l values
                                    dos_l_spin.append(self.dosSuperContext.dos_values[1:])
                                    # save number of l values and the maximal value
                                    # -1 because first column is total DOS
                                    n_l_spin.append(val_length - 1)
                                    if val_length - 1 > max_n_l:
                                        max_n_l = val_length - 1
                                else:
                                    logger.error("%s projected DOS parsing unsuccessful. Parsing of file %s in directory '%s' did not yield values for l-specific DOS." % (Kind, dFile, dirName))
                                    return -1
                            else:
                                logger.error("%s projected DOS parsing unsuccessful. Parsing of file %s in directory '%s' did not yield values for energies or DOS." % (Kind, dFile, dirName))
                                return -1
                    except IOError:
                        logger.error("%s projected DOS parsing unsuccessful. Could not find %s file in directory '%s'." % (Kind, dFile, dirName))
                        return -1
                # append values for spin channels to list
                # no further error checking needed because we already exited the functon if we found an error for one of the spin channels
                dos_total.append(dos_total_spin)
                dos_l.append(dos_l_spin)
                n_l.append(n_l_spin)
            # add array of zeros to those projected DOS that have less l values than max_n_l
            for projected_dos, l in zip(dos_l, n_l):
                for i in range(len(projected_dos)):
                    if l[i] < max_n_l:
                        # The dimensions of the added zero array are:
                        # first dimension:  difference between maximum number of l values (max_n_l) and current number of l values (l[i])
                        # second dimension: same as second dimension of the current DOS (projected_dos[i])
                        projected_dos[i] = np.vstack((projected_dos[i], np.zeros((max_n_l - l[i] , projected_dos[i].shape[1]))))
            # write values
            backend.addArrayValues( kind + '_projected_dos_energies', dos_energies)
            # need to swap axis 0 (number_of_*) and axis 1 (number_of_spin_channels)
            # since its shape is [number_of_spin_channels,number_of_*,n_*_projected_dos_values] in the metadata
            backend.addArrayValues(kind + '_projected_dos_values_total', np.swapaxes(np.asarray(dos_total), 0, 1))
            # in aims there is no projected DOS for per m value
            backend.addValue(kind + '_projected_dos_m_kind', 'integrated')
            # we create an array of the form [[0,0], [1,0], ..., [max_n_l-1,0]] to specify the l values for the projected DOS
            backend.addArrayValues(kind + '_projected_dos_lm', np.column_stack((np.arange(max_n_l), np.zeros(max_n_l, dtype=np.int))))
            # need to swap axis 0 (number_of_*) and axis 2 (number_of_lm_*_projected_dos)
            # since its shape is [number_of_lm_*_projected_dos,number_of_spin_channels,number_of_*,n_*_projected_dos_values] in the metadata
            backend.addArrayValues(kind + '_projected_dos_values_lm', np.swapaxes(np.asarray(dos_l), 0, 2))
            return 0
        else:
            logger.error("%s projected DOS parsing unsuccessful. The number of files (%d) for the %s projected DOS in directory '%s' must be divisible by the number of spin channels (%d)." % (Kind, len(files), kind, dirName, self.maxSpinChannel))
            return -1

    def onClose_section_k_band(self, backend, gIndex, section):
        """Trigger called when section_k_band is closed.

        Band structure is parsed from external band.out files.
        """
        # check if start/end of segements was found in controlInOut
        if self.band_segm_start_end is not None:
            # check if band segemnts were found
            if section['x_fhi_aims_band_segment'] is not None:
                # construct parser for band.out file
                bandSuperContext = FhiAimsBandParser.FhiAimsBandParserContext(False)
                bandParser = AncillaryParser(
                    fileDescription = FhiAimsBandParser.build_FhiAimsBandFileSimpleMatcher(),
                    parser = self.parser,
                    cachingLevelForMetaName = FhiAimsBandParser.get_cachingLevelForMetaName(self.metaInfoEnv, CachingLevel.Ignore),
                    superContext = bandSuperContext)
                band_k_points = []
                band_energies = []
                band_occupations = []
                parsed_segments = []
                # get directiory of currently parsed file
                dirName = os.path.dirname(os.path.abspath(self.fName))
                # loop over found band segements
                for seg in section['x_fhi_aims_band_segment']:
                    band_k_points_seg = None
                    band_energies_spin = []
                    band_occupations_spin = []
                    # loop over spin channels
                    for spin in range(1, self.maxSpinChannel + 1):
                        # construct file name
                        bFile = "band%d%03d.out" % (spin, seg)
                        fName = os.path.normpath(os.path.join(dirName, bFile))
                        try:
                            with open(fName) as fIn:
                                # parse band.out file
                                bandParser.parseFile(fIn)
                                # extract values
                                if all(x is not None for x in [bandSuperContext.band_energies, bandSuperContext.band_k_points, bandSuperContext.band_occupations]):
                                    # check if k-points are the same for the spin channels
                                    if band_k_points_seg is None:
                                        band_k_points_seg = bandSuperContext.band_k_points
                                    elif not np.array_equal(band_k_points_seg, bandSuperContext.band_k_points):
                                        band_k_points_seg = None
                                        logger.warning("The k-points of spin channel 1 in file band1%03d.out and spin channel %d in file %s are not equal in directory '%s'." % (seg, spin, bFile, dirName))
                                    band_energies_spin.append(bandSuperContext.band_energies)
                                    band_occupations_spin.append(bandSuperContext.band_occupations)
                                else:
                                    logger.warning("Parsing of band structure file %s in directory '%s' did not yield values for k-points, energies, or occupations." % (bFile, dirName))
                        except IOError:
                            logger.warning("Could not find %s file in directory '%s'." % (bFile, dirName))
                    # append values for spin channels to list and save which segment was parsed successfully
                    if band_k_points_seg is not None and len(band_energies_spin) == self.maxSpinChannel and len(band_occupations_spin) == self.maxSpinChannel:
                        parsed_segments.append(seg - 1)
                        band_k_points.append(band_k_points_seg)
                        band_energies.append(band_energies_spin)
                        band_occupations.append(band_occupations_spin)
                    else:
                        logger.warning("Band segement %d could not be parsed correctly. Band structure parsing incomplete." % seg)
                # write values if band segments were parsed successfully
                if parsed_segments:
                    for isegment in range(len(parsed_segments)):
                        segmentGIndex = backend.openSection("section_k_band_segment")
                        backend.addArrayValues('band_energies', np.asarray(band_energies[isegment]))
                        backend.addArrayValues('band_k_points', np.asarray(band_k_points[isegment]))
                        backend.addArrayValues('band_occupations', np.asarray(band_occupations[isegment]))
                        # write only the start/end values of successfully parsed band segments
                        backend.addArrayValues('band_segm_start_end', self.band_segm_start_end[parsed_segments[isegment]])
                        backend.closeSection("section_k_band_segment", segmentGIndex)
                else:
                    logger.error("Band structure parsing unsuccessful. Found band structure calculation in main file, but none of the corresponding bandXYYY.out files could be parsed successfully.")

    def setStartingPointCalculation(self, parser):
        backend = parser.backend
        backend.openSection('section_calculation_to_calculation_refs')
        if self.lastCalculationGIndex:
            backend.addValue('calculation_to_calculation_ref', self.lastCalculationGIndex)
        backend.addValue('calculation_to_calculation_kind', 'pertubative GW')
#        backend.closeSection('section_calculation_to_calculation_refs')
        return None

def build_FhiAimsMainFileSimpleMatcher():
    """Builds the SimpleMatcher to parse the main file of FHI-aims.

    First, several subMatchers are defined, which are then used to piece together
    the final SimpleMatcher.
    SimpleMatchers are called with 'SM (' as this string has length 4,
    which allows nice formating of nested SimpleMatchers in python.

    Returns:
       SimpleMatcher that parses main file of FHI-aims.
    """
    ########################################
    # submatcher for control.in
    controlInSubMatcher = SM (name = 'ControlIn',
        startReStr = r"\s*The contents of control\.in will be repeated verbatim below",
        endReStr = r"\s*Completed first pass over input file control\.in \.",
        subMatchers = [
        SM (r"\s*unless switched off by setting 'verbatim_writeout \.false\.' \."),
        SM (r"\s*in the first line of control\.in \."),
        SM (name = 'ControlInKeywords',
            startReStr = r"\s*-{20}-*",
            weak = True,
            # The search is done unordered since the keywords do not appear in a specific order.
            subFlags = SM.SubFlags.Unordered,
            # get control.in subMatcher from FhiAimsControlInParser.py
            subMatchers = FhiAimsControlInParser.build_FhiAimsControlInKeywordsSimpleMatchers()
            ), # END ControlInKeywords
        SM (r"\s*-{20}-*", weak = True)
        ])
    ########################################
    # submatcher for aims output from the parsed control.in
    controlInOutSubMatcher = SM (name = 'ControlInOut',
        startReStr = r"\s*Reading file control\.in\.",
        subMatchers = [
        SM (name = 'ControlInOutLines',
            startReStr = r"\s*-{20}-*",
            sections = ['section_topology'],
            weak = True,
            subFlags = SM.SubFlags.Unordered,
            subMatchers = [
            # Now follows the list to match the aims output from the parsed control.in.
            # The search is done unordered since the output is not in a specific order.
            # Repating occurrences of the same keywords are captured.
            # List the matchers in alphabetical order according to metadata name.
            #
            SM (name = 'BandSegment',
                startReStr = r"\s*Plot band\s*[0-9]+",
                repeats = True,
                subMatchers = [
                SM (r"\s*\|\s*begin\s*(?P<x_fhi_aims_controlInOut_band_segment_start1>[-+0-9.]+)\s+(?P<x_fhi_aims_controlInOut_band_segment_start2>[-+0-9.]+)\s+(?P<x_fhi_aims_controlInOut_band_segment_start3>[-+0-9.]+)"),
                SM (r"\s*\|\s*end\s*(?P<x_fhi_aims_controlInOut_band_segment_end1>[-+0-9.]+)\s+(?P<x_fhi_aims_controlInOut_band_segment_end2>[-+0-9.]+)\s+(?P<x_fhi_aims_controlInOut_band_segment_end3>[-+0-9.]+)"),
                SM (r"\s*\|\s*number of points:\s*[0-9]+"),
                ]),
            # only the first character is important for aims
            SM (r"\s*hse_unit: Unit for the HSE06 hybrid functional screening parameter set to (?P<x_fhi_aims_controlInOut_hse_unit>[a-zA-Z])[a-zA-Z]*\^\(-1\)\.", repeats = True),
            SM (r"\s*hybrid_xc_coeff: Mixing coefficient for hybrid-functional exact exchange modified to\s*(?P<x_fhi_aims_controlInOut_hybrid_xc_coeff>[0-9.]+)\s*\.", repeats = True),
            SM (r"^\s*Found k-point grid:\s+(?P<x_fhi_aims_controlInOut_k1>[0-9]+)\s+(?P<x_fhi_aims_controlInOut_k2>[0-9]+)\s+(?P<x_fhi_aims_controlInOut_k3>[0-9]+)", repeats = True),
            # section fhi_aims_section_MD_detect is just used to detect already in the methods section if a MD run was perfomed
            SM (r"\s*Molecular dynamics time step =\s*(?P<x_fhi_aims_controlInOut_MD_time_step__ps>[0-9.]+) *ps", repeats = True, sections = ['x_fhi_aims_section_MD_detect']),
            SM (r"\s*Scalar relativistic treatment of kinetic energy: (?P<x_fhi_aims_controlInOut_relativistic>[-a-zA-Z\s]+)\.", repeats = True),
            SM (r"\s*(?P<x_fhi_aims_controlInOut_relativistic>Non-relativistic) treatment of kinetic energy\.", repeats = True),
            SM (r"\s*Threshold value for ZORA:\s*(?P<x_fhi_aims_controlInOut_relativistic_threshold>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True),
            # need several regualar expressions to capture all possible output messages of the xc functional
            SM (r"\s*XC: Using (?P<x_fhi_aims_controlInOut_xc>[-_a-zA-Z0-9\s()]+)(?:\.| NOTE)", repeats = True),
            SM (r"\s*XC: Using (?P<x_fhi_aims_controlInOut_xc>HSE-functional) with OMEGA =\s*(?P<x_fhi_aims_controlInOut_hse_omega>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*<units>\.", repeats = True),
            SM (r"\s*XC: Using (?P<x_fhi_aims_controlInOut_xc>Hybrid M11 gradient-corrected functionals) with OMEGA =\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True),
            SM (r"\s*XC:\s*(?P<x_fhi_aims_controlInOut_xc>HSE) with OMEGA_PBE =\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True),
            SM (r"\s*XC: Running (?P<x_fhi_aims_controlInOut_xc>[-_a-zA-Z0-9\s()]+) \.\.\.", repeats = True),
            SM (r"\s*(?P<x_fhi_aims_controlInOut_xc>Hartree-Fock) calculation starts \.\.\.\.\.\.", repeats = True),
            # define some basis set specific SMs
            SM (r"\s*Reading configuration options for species\s*(?P<x_fhi_aims_controlInOut_species_name>[a-zA-Z]+)", repeats=True,
            #SM (r"\s*Reading configuration options for species\s*(?P<atom_type_name>[a-zA-Z]+)", repeats=True,
                sections = ['section_atom_type',"x_fhi_aims_section_controlInOut_atom_species"],
                subFlags = SM.SubFlags.Unordered,
                subMatchers = [
                   SM (r"\s*\|\s*Found\s*request\s*to\s*include\s*pure\s*gaussian\s*fns.\s*:"
                       r"\s+(?P<x_fhi_aims_controlInOut_pure_gaussian>[A-Z]+)"
                       r"\s*", repeats = True),
                       #r"\s+(?P<x_fhi_aims_controlInOut_pure_gaussian>[A-Z]+)"
                       #r"\s*", repeats = True),
                   SM(startReStr = r"\s*\|\s*Found nuclear charge :"
                    #"\s*(?P<x_fhi_aims_controlInOut_species_charge>[.0-9]+\S)\s*",
                    r"\s*(?P<atom_type_charge>[.0-9]+\S)\s*",
                   repeats = True),
                   SM(r"\s*\|\s*Found atomic mass :"
                    #"\s*(?P<x_fhi_aims_controlInOut_species_mass__amu>[.0-9]+)"
                    r"\s*(?P<atom_type_mass__amu>[.0-9]+)"
                    r"\s*",repeats = True),
                   SM(r"\s*\|\s*Found cutoff potl. onset \[A\], width \[A\], scale factor :"
                    r"\s*(?P<x_fhi_aims_controlInOut_species_cut_pot__angstrom>[.0-9]+)"
                    r"\s+(?P<x_fhi_aims_controlInOut_species_cut_pot_width__angstrom>[.0-9]+)"
                    r"\s*(?P<x_fhi_aims_controlInOut_species_cut_pot_scale>[.0-9]+)"
                    r"\s*",repeats = True),
                # Parsing for Gaussian basis starts
                   SM(r"\s*\|\s*Found\s*"
                    #r"(?P<basis_set_atom_centered_unique_name>[-_a-zA-Z0-9\s]+"
                    r"(?P<x_fhi_aims_controlInOut_basis_func_type>[-_a-zA-Z0-9\s]+"
                    #r"(?P<basis_set_atom_centered_unique_name>[-_a-zA-Z0-9\s]+"
                    r"\S)\s*(?:basis function)\s*:\s*L\s*=\s*"
                    r"(?P<x_fhi_aims_controlInOut_basis_func_gauss_l>[0-9]+)"
                    r"\s*,\s*(?P<x_fhi_aims_controlInOut_basis_func_gauss_N>[0-9]+)",
                   repeats = True,
                   #sections = ["x_fhi_aims_section_controlInOut_basis_func"],
                   sections = ["x_fhi_aims_section_controlInOut_basis_func",
                               r"section_basis_set_atom_centered"],
                   subMatchers = [

                           SM(r"\s*\|\s*alpha\s*=\s*"
                              r"(?P<x_fhi_aims_controlInOut_basis_func_gauss_alpha>[-+]?(?:[0-9]+\.?|\.[0-9]+)[-+0-9eEdD]+)"
                            r"\s*weight\s*=\s*"
                            r"(?P<x_fhi_aims_controlInOut_basis_func_gauss_weight>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)",
                           repeats = True,
                           sections = ["x_fhi_aims_section_controlInOut_basis_func"])
                   ]),

                   SM(r"\s*\|\s*Found\s*"
                    #r"(?P<basis_set_atom_centered_unique_name>[-_a-zA-Z0-9\s]+"
                    r"(?P<x_fhi_aims_controlInOut_basis_func_type>[-_a-zA-Z0-9\s]+"
                    #r"(?P<basis_set_atom_centered_unique_name>[-_a-zA-Z0-9\s]+"
                    r"\S)\s*(?:basis function)\s*:\s*"
                    r"(?P<x_fhi_aims_controlInOut_basis_func_gauss_l>[0-9]+)"
                    r"\s*(?P<x_fhi_aims_controlInOut_basis_func_primitive_gauss_alpha>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)",
                   repeats = True,
                   #sections = ["x_fhi_aims_section_controlInOut_basis_func"]),
                   sections = ["x_fhi_aims_section_controlInOut_basis_func",
                               "section_basis_set_atom_centered"]),
                # Parsing for Gaussian basis ends

                # Parsing for NAO basis starts
                   SM(startReStr = r"\s*\|\s*Found free-atom valence",
                      forwardMatch = True,
                      subMatchers = [
                # In FHI-aims for a valence or ion_occ basis function the last digit refers to their occupation
                         SM(r"\s*\|\s*Found\s*"
                           "(?P<x_fhi_aims_controlInOut_basis_func_type>[-_a-zA-Z0-9\s]+"
                           #r"(?P<basis_set_atom_centered_unique_name>[-_a-zA-Z0-9\s]+"
                           r"\S)\s*(?:shell)\s*:\s*"
                           r"(?P<x_fhi_aims_controlInOut_basis_func_n>[0-9]+)"
                           r"\s+(?P<x_fhi_aims_controlInOut_basis_func_l>[a-zA-Z])"
                           r"\s+(?P<x_fhi_aims_controlInOut_basis_func_occ>[.0-9]+)",
                           repeats = True,
                           #sections = ["x_fhi_aims_section_controlInOut_basis_func"]),
                           sections = ["x_fhi_aims_section_controlInOut_basis_func",
                                       "section_basis_set_atom_centered"]),
                # In FHI-aims for a hydrogenic basis function the last digit refers to the effective nuclear charge
                         SM (startReStr = r"\s*\|\s*Found hydrogenic basis",
                            forwardMatch = True,
                            subMatchers = [
                              SM(r"\s*\|\s*Found\s*"
                               "(?P<x_fhi_aims_controlInOut_basis_func_type>[-_a-zA-Z0-9\s]+"
                               #"(?P<basis_set_atom_centered_unique_name>[-_a-zA-Z0-9\s]+"
                               "\S)\s*(?:function)\s*:\s*"
                               "(?P<x_fhi_aims_controlInOut_basis_func_n>[0-9]+)"
                               "\s+(?P<x_fhi_aims_controlInOut_basis_func_l>[a-zA-Z])"
                               "\s+(?P<x_fhi_aims_controlInOut_basis_func_eff_charge>[.0-9]+)",
                               repeats = True,
                               #sections = ["x_fhi_aims_section_controlInOut_basis_func"])
                               sections = ["x_fhi_aims_section_controlInOut_basis_func",
                                           "section_basis_set_atom_centered"])
                             ]),
                # In FHI-aims for a ionic basis function the last digit is equal to the cut-off radius
                         SM (startReStr = r"\s*\|\s*Found ionic basis",
                            forwardMatch = True,
                            subMatchers = [
                              SM(r"\s*\|\s*Found\s*"
                               #"(?P<x_fhi_aims_controlInOut_basis_func_type>[-_a-zA-Z0-9\s]+"
                               "(?P<basis_set_atom_centered_unique_name>[-_a-zA-Z0-9\s]+"
                               "\S)\s*(?:function)\s*:\s*"
                               "(?P<x_fhi_aims_controlInOut_basis_func_n>[0-9]+)"
                               "\s+(?P<x_fhi_aims_controlInOut_basis_func_l>[a-zA-Z])",
                              repeats = True,
                               #sections = ["x_fhi_aims_section_controlInOut_basis_func"])
                               sections = ["x_fhi_aims_section_controlInOut_basis_func",
                                           "section_basis_set_atom_centered"])
                             ])
                    ])
                # Parsing for NAO basis ends
                   ])
                ]), # END ControlInOutLines
        SM (r"\s*-{20}-*", weak = True)
        ])
    ########################################
    # subMatcher for geometry.in
    geometryInSubMatcher = SM (name = 'GeometryIn',
        startReStr = r"\s*Parsing geometry\.in \(first pass over file, find array dimensions only\)\.",
        endReStr = r"\s*Completed first pass over input file geometry\.in \.",
        subMatchers = [
        SM (r"\s*The contents of geometry\.in will be repeated verbatim below"),
        SM (r"\s*unless switched off by setting 'verbatim_writeout \.false\.' \."),
        SM (r"\s*in the first line of geometry\.in \."),
        SM (name = 'GeometryInKeywords',
            startReStr = r"\s*-{20}-*",
            weak = True,
            subFlags = SM.SubFlags.Unordered,
            subMatchers = [
            # Explicitly add ^ to ensure that the keyword is not within a comment.
            # The search is done unordered since the keywords do not appear in a specific order.
            SM (startReStr = r"^\s*(?:atom|atom_frac)\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+\s+[a-zA-Z]+",
                repeats = True,
                subFlags = SM.SubFlags.Unordered,
                subMatchers = [
                SM (r"^\s*constrain_relaxation\s+(?:x|y|z|\.true\.|\.false\.)", repeats = True),
                SM (r"^\s*initial_charge\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True),
                SM (r"^\s*initial_moment\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True),
                SM (r"^\s*velocity\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True)
                ]),
            SM (r"^\s*hessian_block\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True),
            SM (r"^\s*hessian_block_lv\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True),
            SM (r"^\s*hessian_block_lv_atom\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True),
            SM (startReStr = r"^\s*lattice_vector\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+",
                repeats = True,
                subMatchers = [
                SM (r"^\s*constrain_relaxation\s+(?:x|y|z|\.true\.|\.false\.)", repeats = True)
                ]),
            SM (r"^\s*trust_radius\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", repeats = True),
            SM (r"^\s*verbatim_writeout\s+[.a-zA-Z]+")
            ]), # END GeometryInKeywords
        SM (r"\s*-{20}-*", weak = True)
        ])
    ########################################
    # subMatcher for geometry
    # the verbatim writeout of the geometry.in is not considered for getting the structure data
    # using the geometry output of aims has the advantage that it has a clearer structure
    geometrySubMatcher = SM (name = 'Geometry',
        startReStr = r"\s*Reading geometry description geometry\.in\.",
        sections = ['section_system'],
        subMatchers = [
        SM (r"\s*-{20}-*", weak = True),
        SM (r"\s*Input structure read successfully\."),
        SM (r"\s*The structure contains\s*[0-9]+\s*atoms,\s*and a total of\s*[0-9.]+\s*electrons\."),
        SM (r"\s*Input geometry:"),
        SM (r"\s*\|\s*No unit cell requested\."),
        SM (startReStr = r"\s*\|\s*Unit cell:",
            subMatchers = [
            SM (r"\s*\|\s*(?P<x_fhi_aims_geometry_lattice_vector_x__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_lattice_vector_y__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_lattice_vector_z__angstrom>[-+0-9.]+)", repeats = True)
            ]),
        SM (startReStr = r"\s*\|\s*Atomic structure:",
            subMatchers = [
            SM (r"\s*\|\s*Atom\s*x \[A\]\s*y \[A\]\s*z \[A\]"),
            SM (r"\s*\|\s*[0-9]+:\s*Species\s+(?P<x_fhi_aims_geometry_atom_labels>[a-zA-Z]+)\s+(?P<x_fhi_aims_geometry_atom_positions_x__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_positions_y__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_positions_z__angstrom>[-+0-9.]+)", repeats = True)
            ])
        ])
    ########################################
    # subMatcher for geometry after relaxation step
    geometryRelaxationSubMatcher = SM (name = 'GeometryRelaxation',
        startReStr = r"\s*Updated atomic structure:",
        sections = ['section_system'],
        subMatchers = [
        SM (r"\s*x \[A\]\s*y \[A\]\s*z \[A\]"),
        SM (startReStr = r"\s*lattice_vector\s*[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+",
            forwardMatch = True,
            subMatchers = [
            SM (r"\s*lattice_vector\s+(?P<x_fhi_aims_geometry_lattice_vector_x__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_lattice_vector_y__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_lattice_vector_z__angstrom>[-+0-9.]+)", repeats = True)
            ]),
        SM (r"\s*atom\s+(?P<x_fhi_aims_geometry_atom_positions_x__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_positions_y__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_positions_z__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_labels>[a-zA-Z]+)", repeats = True),
        SM (startReStr = r"\s*Fractional coordinates:",
            subMatchers = [
            SM ("\s*L1\s*L2\s*L3"),
            SM (r"\s*atom_frac\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+\s+[a-zA-Z]+", repeats = True),
            SM (r"\s*-{20}-*", weak = True),
            SM (r'\s*Writing the current geometry to file "geometry\.in\.next_step"\.'),
            ])
        ])
    ########################################
    # subMatcher for MD geometry that was used for the finished SCF cycle (see word 'preceding' in the description)
    geometryMDSubMatcher = SM (name = 'GeometryMD',
            startReStr = r"\s*(?:A|Final a)tomic structure \(and velocities\) as used in the preceding time step:",
        sections = ['section_system'],
        subMatchers = [
        SM (r"\s*x \[A\]\s*y \[A\]\s*z \[A\]\s*Atom"),
        SM (startReStr = r"\s*atom\s+(?P<x_fhi_aims_geometry_atom_positions_x__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_positions_y__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_positions_z__angstrom>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_labels>[a-zA-Z]+)",
            repeats = True,
            subMatchers = [
            SM (r"\s*velocity\s+(?P<x_fhi_aims_geometry_atom_velocity_x__angstrom_ps_1>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_velocity_y__angstrom_ps_1>[-+0-9.]+)\s+(?P<x_fhi_aims_geometry_atom_velocity_z__angstrom_ps_1>[-+0-9.]+)")
            ])
        ])
    ########################################
    # submatcher for eigenvalues
    # first define function to build subMatcher for normal case and scalar ZORA
    def build_eigenvaluesGroupSubMatcher(addStr):
        """Builds the SimpleMatcher to parse the normal and the scalar ZORA eigenvalues in aims.

        Args:
            addStr: String that is appended to the metadata names.

        Returns:
            SimpleMatcher that parses eigenvalues with metadata according to addStr.
        """
        # submatcher for eigenvalue list
        EigenvaluesListSubMatcher =  SM (name = 'EigenvaluesLists',
            startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
            sections = ['x_fhi_aims_section_eigenvalues_list%s' % addStr],
            subMatchers = [
            SM (startReStr = r"\s*[0-9]+\s+(?P<x_fhi_aims_eigenvalue_occupation%s>[0-9.eEdD]+)\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+(?P<x_fhi_aims_eigenvalue_eigenvalue%s__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)" % (2 * (addStr,)), repeats = True)
            ])
        return SM (name = 'EigenvaluesGroup',
            startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
            sections = ['x_fhi_aims_section_eigenvalues_group%s' % addStr],
            subMatchers = [
            # spin-polarized
            SM (name = 'EigenvaluesSpin',
                startReStr = r"\s*Spin-(?:up|down) eigenvalues:",
                sections = ['x_fhi_aims_section_eigenvalues_spin%s' % addStr],
                repeats = True,
                subMatchers = [
                # periodic
                SM (startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+(?P<x_fhi_aims_eigenvalue_kpoint1%s>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_eigenvalue_kpoint2%s>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_eigenvalue_kpoint3%s>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+\(in units of recip\. lattice\)" % (3 * (addStr,)),
                    repeats = True,
                    subMatchers = [
                    SM (startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
                        forwardMatch = True,
                        subMatchers = [
                        EigenvaluesListSubMatcher.copy()
                        ])
                    ]),
                # non-periodic
                SM (startReStr = r"\s*State\s+Occupation\s+Eigenvalue *\[Ha\]\s+Eigenvalue *\[eV\]",
                    forwardMatch = True,
                    subMatchers = [
                    EigenvaluesListSubMatcher.copy()
                    ])
                ]), # END EigenvaluesSpin
            # non-spin-polarized, periodic
            SM (name = 'EigenvaluesNoSpinPeriodic',
                startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+.*\s+\(in units of recip\. lattice\)",
                sections = ['x_fhi_aims_section_eigenvalues_spin%s' % addStr],
                forwardMatch = True,
                subMatchers = [
                SM (startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+(?P<x_fhi_aims_eigenvalue_kpoint1%s>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_eigenvalue_kpoint2%s>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_eigenvalue_kpoint3%s>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+\(in units of recip\. lattice\)" % (3 * (addStr,)),
                    repeats = True,
                    subMatchers = [
                    SM (startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
                        forwardMatch = True,
                        subMatchers = [
                        EigenvaluesListSubMatcher.copy()
                        ])
                    ]),
                ]), # END EigenvaluesNoSpinPeriodic
            # non-spin-polarized, non-periodic
            SM (name = 'EigenvaluesNoSpinNonPeriodic',
                startReStr = r"\s*State\s+Occupation\s+Eigenvalue *\[Ha\]\s+Eigenvalue *\[eV\]",
                sections = ['x_fhi_aims_section_eigenvalues_spin%s' % addStr],
                forwardMatch = True,
                subMatchers = [
                EigenvaluesListSubMatcher.copy()
                ]), # END EigenvaluesNoSpinNonPeriodic
            ])
    # now construct the two subMatchers
    EigenvaluesGroupSubMatcher = build_eigenvaluesGroupSubMatcher('')
    EigenvaluesGroupSubMatcherZORA = build_eigenvaluesGroupSubMatcher('_ZORA')
    ########################################
    # submatcher for pertubative GW eigenvalues
    # first define function to build subMatcher
    def build_GWeigenvaluesGroupSubMatcher(addStr):
        """Builds the SimpleMatcher to parse the perturbative GW eigenvalues in aims.

        Args:
            addStr: String that is appended to the metadata names.

        Returns:
            SimpleMatcher that parses eigenvalues with metadata according to addStr.
        """
        # submatcher for eigenvalue list
        GWEigenvaluesListSubMatcher = SM (name = 'perturbativeGW_EigenvaluesLists',
            startReStr = r"\s*state\s+occ_num\s+e_gs\s+e_x\^ex\s+e_xc\^gs\s+e_c\^nloc\s+e_qp",
            sections = ['x_fhi_aims_section_eigenvalues_list%s' % addStr],
            subMatchers = [
            SM (startReStr = r"\s*[0-9]+\s+(?P<x_fhi_aims_eigenvalue_occupation%s>[0-9.eEdD]+)\s+(?P<x_fhi_aims_eigenvalue_ks_GroundState__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+"
                              "(?P<x_fhi_aims_eigenvalue_ExactExchange_perturbativeGW__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_eigenvalue_ks_ExchangeCorrelation__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+"
                              "(?P<x_fhi_aims_eigenvalue_correlation_perturbativeGW__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_eigenvalue_quasiParticle_energy__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)" % (1 * (addStr,)),
            repeats = True)
            ])
        return SM (name = 'perturbativeGW_EigenvaluesGroup',
            startReStr = r"\s*GW quasiparticle calculation starts ...",
            sections = ['section_method', 'section_single_configuration_calculation', 'x_fhi_aims_section_eigenvalues_group%s' % addStr],
            adHoc = lambda parser: parser.superContext.setStartingPointCalculation(parser),
            fixedStartValues = { 'electronic_structure_method': 'G0W0' },
            subMatchers = [
            # non-spin-polarized, non-periodic
            SM (name = 'GW_EigenvaluesNoSpinNonPeriodic',
                startReStr = r"\s*state\s+occ_num\s+e_gs\s+e_x\^ex\s+e_xc\^gs\s+e_c\^nloc\s+e_qp",
                sections = ['x_fhi_aims_section_eigenvalues_spin%s' % addStr],
                forwardMatch = True,
                subMatchers = [
                SM (r"\s*-+"),
               SM (r"\s*-+"),
                GWEigenvaluesListSubMatcher.copy()
                ]), # END EigenvaluesNoSpinNonPeriodic
            ])
    # now construct the two subMatchers
    GWEigenvaluesGroupSubMatcher = build_GWeigenvaluesGroupSubMatcher('_perturbativeGW')

    ########################################
    # submatcher for total energy components during SCF interation
    TotalEnergyScfSubMatcher = SM (name = 'TotalEnergyScf',
        startReStr = r"\s*Total energy components:",
        subMatchers = [
        SM (r"\s*\|\s*Sum of eigenvalues\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_sum_eigenvalues_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*XC energy correction\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_XC_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*XC potential correction\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_XC_potential_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Free-atom electrostatic energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<x_fhi_aims_energy_electrostatic_free_atom_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Hartree energy correction\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_correction_hartree_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Entropy correction\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_correction_entropy_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*-{20}-*", weak = True),
        SM (r"\s*\|\s*Total energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_total_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Total energy, T -> 0\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_total_T0_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Electronic free energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_free_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*Derived energy quantities:"),
        SM (r"\s*\|\s*Kinetic energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<electronic_kinetic_energy_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Electrostatic energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_electrostatic_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Energy correction for multipole"),
        SM (r"\s*\|\s*error in Hartree potential\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_hartree_error_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<energy_sum_eigenvalues_per_atom_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<energy_total_T0_per_atom_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<energy_free_per_atom_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        ])
    ########################################
    # submatcher for total energy components in scalar ZORA post-processing
    TotalEnergyZORASubMatcher = SM (name = 'TotalEnergyZORA',
        startReStr = r"\s*Total energy components:",
        subMatchers = [
        SM (r"\s*\|\s*Sum of eigenvalues\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_sum_eigenvalues__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*XC energy correction\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV"),
        SM (r"\s*\|\s*XC potential correction\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_XC_potential__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Free-atom electrostatic energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV"),
        SM (r"\s*\|\s*Hartree energy correction\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_correction_hartree__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Entropy correction\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_correction_entropy__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*-{20}-*", weak = True),
        SM (r"\s*\|\s*Total energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV"),
        SM (r"\s*\|\s*Total energy, T -> 0\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV"),
        SM (r"\s*\|\s*Electronic free energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV"),
        SM (r"\s*Derived energy quantities:"),
        SM (r"\s*\|\s*Kinetic energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<electronic_kinetic_energy__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Electrostatic energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_electrostatic__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Energy correction for multipole"),
        SM (r"\s*\|\s*error in Hartree potential\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_hartree_error__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<energy_sum_eigenvalues_per_atom__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<energy_total_T0_per_atom__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<energy_free_per_atom__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        ])
    ########################################
    # submatcher for geometry relaxation
    RelaxationSubMatcher = SM (name = 'Relaxation',
        startReStr = r"\s*Geometry optimization: Attempting to predict improved coordinates\.",
        subMatchers = [
        SM (r"\s*Removing unitary transformations \(pure translations, rotations\) from forces on atoms\."),
        SM (r"\s*Maximum force component is\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV/A\."),
        # output of final structure after relaxation not needed since it is just a repetition of the current geometry
        SM (name =  'FinalStructure',
            startReStr = r"\s*Present geometry (?P<x_fhi_aims_geometry_optimization_converged>is converged)\.",
            subMatchers = [
            SM (r"\s*\|\s*-{20}-*", weak = True),
            SM (r"\s*Final atomic structure:"),
            SM (r"\s*x \[A\]\s*y \[A\]\s*z \[A\]"),
            SM (startReStr = r"\s*lattice_vector\s*[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+",
                forwardMatch = True,
                subMatchers = [
                SM (r"\s*lattice_vector\s*[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+", repeats = True)
                ]),
            SM (r"\s*atom\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+\s+[a-zA-Z]+", repeats = True),
            SM (startReStr = r"\s*Fractional coordinates:",
                subMatchers = [
                SM ("\s*L1\s*L2\s*L3"),
                SM (r"\s*atom_frac\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+\s+[a-zA-Z]+", repeats = True),
                SM (r"\s*-{20}-*", weak = True),
                ])
            ]), # END FinalStructure
        SM (name = 'RelaxationStep',
            startReStr = r"\s*Present geometry (?P<x_fhi_aims_geometry_optimization_converged>is not yet converged)\.",
            subMatchers = [
            SM (r"\s*Relaxation step number\s*[0-9]+: Predicting new coordinates\."),
            SM (r"\s*Advancing geometry using (?:trust radius method|BFGS)\.")
            ]) # END RelaxationStep
        ])
    ########################################
    # submatcher for MD
    MDSubMatcher = SM (name = 'MD',
        startReStr = r"\s*Molecular dynamics: Attempting to update all nuclear coordinates\.",
        subMatchers = [
        SM (r"\s*Removing unitary transformations \(pure translations, rotations\) from forces on atoms\."),
        SM (r"\s*Maximum force component is\s*[-+0-9.eEdD]\s*eV/A\."),
        SM (r"\s*Present geometry is converged\."),
        SM (name = 'MDStep',
            startReStr = r"\s*Advancing structure using Born-Oppenheimer Molecular Dynamics:",
            subMatchers = [
            SM (r"\s*Complete information for previous time-step:"),
            SM (r"\s*\|\s*Time step number\s*:\s*[0-9]+"),
            SM (r"\s*-{20}-*", weak = True)
            ]), # END MDStep
        # parse MD geometry that was used for the finished SCF cycle
        geometryMDSubMatcher
        ])
    ########################################
    # submatcher for DOS
    dosSubMatcher = SM (name = 'DOS',
        startReStr = r"\s*Calculating total density of states \.\.\.",
        endReStr = r"\s*\|\s*writing DOS \(raw data\) to file \S+",
        sections = ['section_dos'],
        subMatchers = [
        SM (r"\s*\|\s*writing DOS \(shifted by electron chemical potential\) to file \S+")
        ])
    ########################################
    # submatcher for perturbative DOS
    perturbDosSubMatcher = SM (name = 'perturbDOS',
        startReStr = r"\s*Post-scf processing of Kohn-Sham eigenvalues on a denser k-point grid\.",
        endReStr = r"\s*\|\s*writing perturbative DOS \(raw data\) to file \S+",
        sections = ['section_dos'],
        subMatchers = [
        SM (r"\s*\|\s*writing perturbative DOS \(shifted by electron chemical potential\) to file \S+")
        ])
    ########################################
    # submatcher for species projected DOS
    speciesDosSubMatcher = SM (name = 'speciesDOS',
        startReStr = r"\s*Calculating angular momentum projected density of states \.\.\.",
        sections = ['section_species_projected_dos'],
        subMatchers = [
            SM (r"\s*\|\s*writing(?: spin-(?:up|down))? projected DOS \(shifted by the chemical potential\) for species (?P<x_fhi_aims_species_projected_dos_species_label>[a-zA-Z]+) to file (?P<x_fhi_aims_species_projected_dos_file>\S+[^.])\.", repeats = True)
        ])
    ########################################
    # submatcher for atom projected DOS
    atomDosSubMatcher = SM (name = 'atomDOS',
        startReStr = r"\s*Calculating atom-projected density of states \.\.\.",
        sections = ['section_atom_projected_dos'],
        subMatchers = [
            SM (r"\s*\|\s*writing(?: spin-(?:up|down))? projected DOS \(shifted by the chemical potential\) for species [a-zA-Z]+ to file (?P<x_fhi_aims_atom_projected_dos_file>\S+[^.])\.", repeats = True)
        ])
    ########################################
    # submatcher for band structure
    bandStructureSubMatcher = SM (name = 'BandStructure',
        startReStr = r"\s*Writing the requested band structure output:",
        endReStr = r"\s*Band Structure\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)",
        sections = ['section_k_band'],
        subMatchers = [
        SM (r"\s*\s*-{20}-*", weak = True),
        SM (r"\s*Integrating Hamiltonian matrix: batch-based integration\."),
        SM (r"\s*Time summed over all CPUs for integration: real work\s*[0-9.]+ *s, elapsed\s*[0-9.]+ *s"),
        SM (name = 'BandSegment',
            startReStr = r"\s*Treating all\s*[0-9]+ k-points in band plot segment #\s*(?P<x_fhi_aims_band_segment>[0-9]+):",
            repeats= True,
            subMatchers = [
            SM (r'\s*"Band gap" along reciprocal space direction number:\s*[0-9]+'),
            SM (r"\s*\|\s*Lowest unoccupied state\s*:\s*[-+0-9.]+ *eV"),
            SM (r"\s*\|\s*Highest occupied state\s*:\s*[-+0-9.]+ *eV"),
            SM (r"\s*\|\s*Energy difference\s*:\s*[-+0-9.]+ *eV")
            ]), # END BandSegment
        SM (r'\s*"Band gap" of total set of bands:'),
        SM (r"\s*\|\s*Lowest unoccupied state\s*:\s*[-+0-9.]+ *eV"),
        SM (r"\s*\|\s*Highest occupied state\s*:\s*[-+0-9.]+ *eV"),
        SM (r"\s*\|\s*Energy difference\s*:\s*[-+0-9.]+ *eV")
        ])
    ########################################
    # submatcher for Tkatchenko-Scheffler van der Waals
    TS_VanDerWaalsSubMatcher = SM (name = 'TS_VanDerWaals',
        startReStr = r"\s*Evaluating non-empirical van der Waals correction",
        sections = ['section_method', 'section_single_configuration_calculation', 'x_fhi_aims_section_vdW_TS'],
        repeats = True,
        fixedStartValues = { "van_der_Waals_method": "TS" },
        subMatchers = [
            SM (startReStr = r"\s*\|\s*Atom\s*[0-9]:\s*[a-zA-Z]+",
                repeats = True,
                forwardMatch = True,
                subMatchers = [
                SM (r"\s*\|\s*Atom\s*[0-9]:\s*(?P<x_fhi_aims_atom_type_vdW>[a-zA-Z]+)", repeats = True),
                SM (r"\s*\|\s*Hirshfeld charge\s*:\s*(?P<x_fhi_aims_hirschfeld_charge>[-+0-9.]+)", repeats = True),
                SM (r"\s*\|\s*Free atom volume\s*:\s*(?P<x_fhi_aims_free_atom_volume>[-+0-9.]+)", repeats = True),
                SM (r"\s*\|\s*Hirshfeld volume\s*:\s*(?P<x_fhi_aims_hirschfeld_volume>[-+0-9.]+)", repeats = True)
                ]),
            SM (r"\s*\|\s*van der Waals energy corr.\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<x_fhi_aims_vdW_energy_corr_TS__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV")
        ])
    ########################################
    # submatcher for total energy components during SCF interation
    ScgwEnergyScfSubMatcher = SM(name = "Self consistent  GW",
      startReStr = r"\s*Self-Consistent GW calculation starts\s*\.\.\.\s*",
      sections = ['section_method', 'section_single_configuration_calculation'],
      fixedStartValues = { 'electronic_structure_method': "scGW" },
      subMatchers = [
        SM (name = 'ScgwEnergyScfSubMatcher',
        startReStr = r"\s*\---\s*GW Total Energy Calculation",
                                  sections = ['section_scf_iteration'],
        repeats = True,
        subMatchers = [
        SM (r"\s*\|\s*Galitskii-Migdal Total Energy\s*:\s*(?P<x_fhi_aims_scgw_galitskii_migdal_total_energy__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*GW Kinetic Energy\s*:\s*(?P<x_fhi_aims_scgw_kinetic_energy__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Hartree energy from GW density\s*:\s*(?P<x_fhi_aims_scgw_hartree_energy_sum_eigenvalues_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*GW correlation Energy\s*:\s*(?P<x_fhi_aims_energy_scgw_correlation_energy__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*RPA correlation Energy\s*:\s*(?P<x_fhi_aims_scgw_rpa_correlation_energy__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Sigle Particle Energy\s*:\s*(?P<x_fhi_aims_single_particle_energy__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
        SM (r"\s*\|\s*Fit accuracy for G\(w\)\s*(?P<x_fhi_aims_poles_fit_accuracy>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)")
        ])
      ])
    ########################################
    # return main Parser
    return SM (name = 'Root',
        startReStr = "",
        forwardMatch = True,
        weak = True,
        subMatchers = [
        SM (name = 'NewRun',
            startReStr = r"\s*Invoking FHI-aims \.\.\.",
            endReStr = r"\s*Have a nice day\.",
            repeats = True,
            required = True,
            forwardMatch = True,
            fixedStartValues={'program_name': 'FHI-aims', 'program_basis_set_type': 'numeric AOs' },
            sections = ['section_run'],
            subMatchers = [
            # header specifing version, compilation info, task assignment
            SM (name = 'ProgramHeader',
                startReStr = r"\s*Invoking FHI-aims \.\.\.",
                subMatchers = [
                SM (r"\s*Version\s*(?P<program_version>[0-9a-zA-Z_\.]+).*"),
                SM (r"\s*Compiled on\s*(?P<x_fhi_aims_program_compilation_date>[0-9/]+)\s*at\s*(?P<x_fhi_aims_program_compilation_time>[0-9:]+)\s*on host\s*(?P<program_compilation_host>[-a-zA-Z0-9._]+)\."),
                SM (r"\s*Date\s*:\s*(?P<x_fhi_aims_program_execution_date>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<x_fhi_aims_program_execution_time>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"),
                SM (r"\s*-{20}-*", weak = True),
                SM (r"\s*Time zero on CPU 1\s*:\s*(?P<time_run_cpu1_start>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *s?\."),
                SM (r"\s*Internal wall clock time zero\s*:\s*(?P<time_run_wall_start>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *s\."),
                SM (r"\s*aims_uuid\s*:\s*(?P<raw_id>[a-zA-Z0-9\-]+)"),
                SM (r"\s*FHI-aims\s+version\s*:\s*(?P<program_version>[0-9a-zA-Z_]+)"),
                SM (name = "nParallelTasks",
                    startReStr = r"\s*Using\s*(?P<x_fhi_aims_number_of_tasks>[0-9]+)\s*parallel tasks\.",
                    sections = ["x_fhi_aims_section_parallel_tasks"],
                    subMatchers = [
                    SM (name = 'ParallelTasksAssignement',
                        startReStr = r"\s*Task\s*(?P<x_fhi_aims_parallel_task_nr>[0-9]+)\s*on host\s+(?P<x_fhi_aims_parallel_task_host>[-a-zA-Z0-9._]+)\s+reporting\.",
                        repeats = True,
                        sections = ["x_fhi_aims_section_parallel_task_assignement"])
                    ]) # END nParallelTasks
                ]), # END ProgramHeader
            # parse control and geometry

            SM (name = 'SectionMethodNoVerbatin',
                startReStr = r"\s*Parsing control\.in *(?:\.\.\.|\(first pass over file, find array dimensions only\)\.)",
                # startReStr = r"\s*Parsing control\.in \(first pass over file, find array dimensions only\)\.",
                sections = ['section_method'],
                subMatchers = [
                # parse verbatim writeout of control.in
                controlInSubMatcher,
                # parse verbatim writeout of geometry.in
                geometryInSubMatcher,
                # parse number of spin channels
                SM (name = 'ArraySizeParameters',
                    startReStr = r"\s*Basic array size parameters:",
                    subMatchers = [
                        SM (r"\s*\|\s*Number of species\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Number of atoms\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Number of lattice vectors\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Max\. basis fn\. angular momentum\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Max\. atomic/ionic basis occupied n\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Max\. number of basis fn\. types\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Max\. radial fns per species/type\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Max\. logarithmic grid size\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Max\. radial integration grid size\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Max\. angular integration grid size\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Max\. angular grid division number\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Radial grid for Hartree potential\s*:\s*[0-9]+"),
                        SM (r"\s*\|\s*Number of spin channels\s*:\s*(?P<x_fhi_aims_controlInOut_number_of_spin_channels>[0-9]+)")
                    ]), # END ArraySizeParameters
                # parse settings writeout of aims
                controlInOutSubMatcher,
                # parse geometry writeout of aims
                geometrySubMatcher
                ]), # END SectionMethod

            SM (r"\s*Preparing all fixed parts of the calculation\."),
            # this SimpleMatcher groups a single configuration calculation together with output after SCF convergence from relaxation
            SM (name = 'SingleConfigurationCalculationWithSystemDescription',
                startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                repeats = True,
                forwardMatch = True,
                subMatchers = [
                # the actual section for a single configuration calculation starts here
                SM (name = 'SingleConfigurationCalculation',
                    startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                    repeats = True,
                    forwardMatch = True,
                    sections = ['section_method','section_single_configuration_calculation'],
                    fixedStartValues = { 'electronic_structure_method': 'DFT' },
                    subMatchers = [
                    # initialization of SCF loop, SCF iteration 0
                    SM (name = 'ScfInitialization',
                        startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                        sections = ['section_scf_iteration'],
                        subMatchers = [
                        SM (r"\s*Date\s*:\s*(?P<x_fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<x_fhi_aims_scf_time_start>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"),
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*\| Chemical potential \(Fermi level\) (in eV)?\s*:\s*(?P<x_fhi_aims_energy_reference_fermi__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)(\s*eV)?", name='FermiLevel'),
                        EigenvaluesGroupSubMatcher,
                        TotalEnergyScfSubMatcher,
                        SM (r"\s*End scf initialization - timings\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)"),
                        SM (r"\s*-{20}-*", weak = True)
                        ]), # END ScfInitialization
                    # normal SCF iterations
                    SM (name = 'ScfRestart',
                        startReStr = r"\s*density from restart information",
                        sections = ['section_scf_iteration'],
                        repeats = True,
                        subMatchers = [
                        SM (r"\s*Date\s*:\s*(?P<x_fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<x_fhi_aims_scf_time_start>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"),
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*\| Chemical potential \(Fermi level\) (in eV)?\s*:\s*(?P<x_fhi_aims_energy_reference_fermi__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)(\s*eV)?", name='FermiLevel'),
                        EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        TotalEnergyScfSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        # raw forces
                        SM (name = 'RawForces',
                            startReStr = r"\s*atomic forces \[eV/Ang\]:",
                            subMatchers = [
                            SM (r"\s*Total forces\(\s*[0-9]+\s*\)\s*:\s+(?P<x_fhi_aims_atom_forces_raw_x__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_atom_forces_raw_y__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_atom_forces_raw_z__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True)
                            ]),
                        # SCF convergence info
                        SM (name = 'SCFConvergence',
                            startReStr = r"\s*Self-consistency convergence accuracy:",
                            subMatchers = [
                            SM (r"\s*\|\s*Change of charge(?:/spin)? density\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+0-9.eEdD]*"),
                            SM (r"\s*\|\s*Change of sum of eigenvalues\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV"),
                            SM (r"\s*\|\s*Change of total energy\s*:\s*(?P<energy_change_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                            SM (r"\s*\|\s*Change of forces\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV/A"),
                            SM (r"\s*\|\s*Change of analytical stress\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV/A\*\*3")
                            ]), # END SCFConvergence
                        # after convergence eigenvalues are printed in the end instead of usually in the beginning
                        # EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        # SM (r"\s*(?P<x_fhi_aims_single_configuration_calculation_converged>Self-consistency cycle converged)\."),
                        # SM (r"\s*End self-consistency iteration #\s*[0-9]+\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)")
                        ]), # END ScfIteration
                    SM (name = 'ScfIteration',
                        startReStr = r"\s*Begin self-consistency iteration #\s*[0-9]+",
                        sections = ['section_scf_iteration'],
                        repeats = True,
                        subMatchers = [
                        SM (r"\s*Date\s*:\s*(?P<x_fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<x_fhi_aims_scf_time_start>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"),
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*\| Chemical potential \(Fermi level\) (in eV)?\s*:\s*(?P<x_fhi_aims_energy_reference_fermi__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)(\s*eV)?", name='FermiLevel'),
                        EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        TotalEnergyScfSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        # raw forces
                        SM (name = 'RawForces',
                            startReStr = r"\s*atomic forces \[eV/Ang\]:",
                            subMatchers = [
                            SM (r"\s*Total forces\(\s*[0-9]+\s*\)\s*:\s+(?P<x_fhi_aims_atom_forces_raw_x__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_atom_forces_raw_y__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_atom_forces_raw_z__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True)
                            ]),
                        # SCF convergence info
                        SM (name = 'SCFConvergence',
                            startReStr = r"\s*Self-consistency convergence accuracy:",
                            subMatchers = [
                            SM (r"\s*\|\s*Change of charge(?:/spin)? density\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s+[-+0-9.eEdD]*"),
                            SM (r"\s*\|\s*Change of sum of eigenvalues\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV"),
                            SM (r"\s*\|\s*Change of total energy\s*:\s*(?P<energy_change_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                            SM (r"\s*\|\s*Change of forces\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV/A"),
                            SM (r"\s*\|\s*Change of analytical stress\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *eV/A\*\*3")
                            ]), # END SCFConvergence
                        # after convergence eigenvalues are printed in the end instead of usually in the beginning
                        EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        SM (r"\s*(?P<x_fhi_aims_single_configuration_calculation_converged>Self-consistency cycle converged)\."),
                        SM (r"\s*End self-consistency iteration #\s*[0-9]+\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)")
                        ]), # END ScfIteration

                    # SCF iterations for output_level MD_light
                    SM (name = 'ScfIterationsMDlight',
                        startReStr = r"\s*Convergence:\s*q app.\s*\|\s*density\s*\|\s*eigen \(eV\)\s*\|\s*Etot \(eV\)\s*\|\s*forces \(eV/A\)\s*\|\s*CPU time\s*\|\s*Clock time",
                        subMatchers = [
                            SM (startReStr =r"\s*SCF\s*[0-9]+\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*\|\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*\|\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*\|\s*(?P<energy_change_scf_iteration__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*\|\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?\s*\|\s*[0-9.]+ *s\s*\|\s*[0-9.]+ *s",
                            sections = ['section_scf_iteration'],
                            repeats = True
                            )
                        ]), # END ScfIterationsMDlight
                    # possible scalar ZORA post-processing
                    SM (name = 'ScaledZORAPostProcessing',
                        startReStr = r"\s*Post-processing: scaled ZORA corrections to eigenvalues and total energy.",
                        endReStr = r"\s*End evaluation of scaled ZORA corrections.",
                        subMatchers = [
                        SM (r"\s*Date\s*:\s*[-.0-9/]+\s*,\s*Time\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"),
                        SM (r"\s*-{20}-*", weak = True),
                        # parse eigenvalues
                        SM (name = 'EigenvaluesListsZORA',
                            startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
                            forwardMatch = True,
                            sections = ['x_fhi_aims_section_eigenvalues_ZORA'],
                            subMatchers = [
                            EigenvaluesGroupSubMatcherZORA
                            ]), # END EigenvaluesListsZORA
                        TotalEnergyZORASubMatcher
                        ]), # END ScaledZORAPostProcessing
                    # summary of energy and forces
                    SM (name = 'EnergyForcesSummary',
                        startReStr = r"\s*Energy and forces in a compact form:",
                        subMatchers = [
                        SM (r"\s*\|\s*Total energy uncorrected\s*:\s*(?P<energy_total__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (r"\s*\|\s*Total energy corrected\s*:\s*(?P<energy_total_T0__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (r"\s*\|\s*Electronic free energy\s*:\s*(?P<energy_free__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (name = 'ForcesSummary',
                            startReStr = r"\s*Total atomic forces \(.*\) \[eV/Ang\]:",
                            subMatchers = [
                            SM (r"\s*\|\s*[0-9]+\s+(?P<x_fhi_aims_atom_forces_free_x__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_atom_forces_free_y__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?P<x_fhi_aims_atom_forces_free_z__eV_angstrom_1>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)", repeats = True)
                            ]),
                        SM (r"\s*-{20}-*", weak = True)
                        ]), # END EnergyForcesSummary
                    # DOS
                    dosSubMatcher,
                    # decomposition of the xc energy
                    SM (name = 'DecompositionXCEnergy',
                        startReStr = r"\s*Start decomposition of the XC Energy",
                        endReStr = r"\s*End decomposition of the XC Energy",
                        subMatchers = [
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*Hartree-Fock part\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_hartree_fock_X_scaled__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*X Energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_X__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (r"\s*C Energy GGA\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_C__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (r"\s*Total XC Energy\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<energy_XC_functional__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (r"\s*LDA X and C from self-consistent density"),
                        SM (r"\s*X Energy LDA\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<x_fhi_aims_energy_X_LDA__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (r"\s*C Energy LDA\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)? *Ha\s+(?P<x_fhi_aims_energy_C_LDA__eV>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) *eV"),
                        SM (r"\s*-{20}-*", weak = True),
                        ]), # END DecompositionXCEnergy
                    # calculation was not converged
                    SM (name = 'ScfNotConverged',
                        startReStr = r"\s*\*\s*WARNING! SELF-CONSISTENCY CYCLE DID NOT CONVERGE",
                        subMatchers = [
                        SM (r"\s*\*\s*USING YOUR PRESELECTED ACCURACY CRITERIA\."),
                        SM (r"\s*\*\s*DO NOT RELY ON ANY FINAL DATA WITHOUT FURTHER CHECKS\.")
                        ]),
                    # geometry relaxation
                    RelaxationSubMatcher,
                    # MD
                    MDSubMatcher,
                    # species projected DOS
                    speciesDosSubMatcher,
                    # atom projected DOS
                    atomDosSubMatcher,
                    # perturbative DOS
                    perturbDosSubMatcher,
                    # band structure
                    bandStructureSubMatcher
                    ]), # END SingleConfigurationCalculation
                # TS van der Waals
                TS_VanDerWaalsSubMatcher,
                # self-consistent GW
                ScgwEnergyScfSubMatcher,
                # perturbative GW
                GWEigenvaluesGroupSubMatcher,
                # parse updated geometry for relaxation
                geometryRelaxationSubMatcher,
                ]), # END SingleConfigurationCalculationWithSystemDescription
            SM (r"\s*-{20}-*", weak = True),
            SM (r"\s*Leaving FHI-aims\."),
            SM (r"\s*Date\s*:\s*[-.0-9/]+\s*,\s*Time\s*:\s*[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"),
            # summary of computational steps
            SM (name = 'ComputationalSteps',
                startReStr = r"\s*Computational steps:",
                subMatchers = [
                SM (r"\s*\|\s*Number of self-consistency cycles\s*:\s*[0-9]+"),
                SM (r"\s*\|\s*Number of relaxation steps\s*:\s*[0-9]+"),
                SM (r"\s*\|\s*Number of molecular dynamics steps\s*:\s*[0-9]+"),
                SM (r"\s*\|\s*Number of force evaluations\s*:\s*[0-9]+")
                ]), # END ComputationalSteps
            SM (r"\s*Detailed time accounting\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)")
            ]) # END NewRun
        ]) # END Root

def get_cachingLevelForMetaName(metaInfoEnv):
    """Sets the caching level for the metadata.

    Args:
        metaInfoEnv: metadata which is an object of the class InfoKindEnv in nomadcore.local_meta_info.py.

    Returns:
        Dictionary with metaname as key and caching level as value.
    """
    # manually adjust caching of metadata
    cachingLevelForMetaName = {
                               'x_fhi_aims_atom_projected_dos_file': CachingLevel.Cache,
                               'x_fhi_aims_band_segment': CachingLevel.Cache,
                               'x_fhi_aims_geometry_optimization_converged': CachingLevel.Cache,
                               'x_fhi_aims_section_MD_detect': CachingLevel.Ignore,
                               'x_fhi_aims_single_configuration_calculation_converged': CachingLevel.Cache,
                               'x_fhi_aims_species_projected_dos_file': CachingLevel.Cache,
                               'x_fhi_aims_species_projected_dos_species_label': CachingLevel.Cache,
                               'section_scf_iteration': CachingLevel.ForwardAndCache,
                               'x_fhi_aims_energy_reference_fermi': CachingLevel.ForwardAndCache,
                               'section_dos': CachingLevel.Ignore,
                              }
    # Set all controlIn and controlInOut metadata to Cache to capture multiple occurrences of keywords and
    # their last value is then written by the onClose routine in the FhiAimsParserContext.
    # Set all geometry metadata to Cache as all of them need post-processsing.
    # Set all eigenvalue related metadata to Cache.
    # Set all forces related metadata to Cache.
    for name in metaInfoEnv.infoKinds:
        metaInfo = metaInfoEnv.infoKinds[name]
        if (name.startswith('x_fhi_aims_controlIn_') and
            metaInfo.kindStr == "type_document_content" and
            ("x_fhi_aims_controlIn_method" in metaInfo.superNames or "x_fhi_aims_controlIn_run" in metaInfo.superNames) or
            name.startswith('x_fhi_aims_controlInOut_') and
            metaInfo.kindStr == "type_document_content" and
            "x_fhi_aims_controlInOut_method" in metaInfo.superNames
            or name.startswith('x_fhi_aims_geometry_')
            or name.startswith('x_fhi_aims_eigenvalue_')
            or name.startswith('x_fhi_aims_section_eigenvalues_')
            or name.startswith('x_fhi_aims_atom_forces_')
           ):
            cachingLevelForMetaName[name] = CachingLevel.Cache
    return cachingLevelForMetaName


def getParserInfo():
    return {'name': 'fhi-aims-parser', 'version': '1.0'}


def main():
    """Main function.

    Set up everything for the parsing of the FHI-aims main file and run the parsing.
    """
    # get main file description
    FhiAimsMainFileSimpleMatcher = build_FhiAimsMainFileSimpleMatcher()
    # loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json
    metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
    metaInfoEnv = get_metaInfo(metaInfoPath)
    # set parser info
    parserInfo = getParserInfo()
    # get caching level for metadata
    cachingLevelForMetaName = get_cachingLevelForMetaName(metaInfoEnv)
    # start parsing
    mainFunction(mainFileDescription = FhiAimsMainFileSimpleMatcher,
                 metaInfoEnv = metaInfoEnv,
                 parserInfo = parserInfo,
                 cachingLevelForMetaName = cachingLevelForMetaName,
                 superContext = FhiAimsParserContext())

if __name__ == "__main__":
    main()
