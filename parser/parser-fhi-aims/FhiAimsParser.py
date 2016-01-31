import setup_paths
import numpy as np
import nomadcore.ActivateLogging
from nomadcore.caching_backend import CachingLevel
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.simple_parser import mainFunction
from nomadcore.simple_parser import SimpleMatcher as SM
from nomadcore.unit_conversion.unit_conversion import convert_unit
import json, logging, os, re, sys

############################################################
# REMARKS
# 
# Energies are parsed in eV to get consistent values 
# since not all energies are given in hartree.
############################################################

logger = logging.getLogger("nomad.FhiAimsParser") 

class FhiAimsParserContext(object):
    """Context for parsing FHI-aims main file.

    This class keeps tracks of several aims settings to adjust the parsing to them.
    The onClose_ functions allow processing and writing of cached values after a section is closed.
    They take the following arguments:
        backend: Class that takes care of wrting and caching of metadata.
        gIndex: Index of the section that is closed.
        section: The cached values and sections that were found in the section that is closed.
    """
    def __init__(self):
        self.secMethodIndex = None
        self.secSystemDescriptionIndex = None
        self.scalarZORA = False
        self.periodicCalc = False
        self.MD = False
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        self.scfConvergence = False
        self.geoConvergence = None
        self.eigenvalues_occupation = []
        self.eigenvalues_eigenvalues = []
        self.eigenvalues_kpoints = []
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
        # dictionary for conversion of xc functional name in aims to metadata format
        # TODO hybrid GGA and mGGA functionals and vdW functionals
        self.xcDict = {
                       'Perdew-Wang parametrisation of Ceperley-Alder LDA': 'LDA_C_PW_LDA_X',
                       'Perdew-Zunger parametrisation of Ceperley-Alder LDA': 'LDA_C_PZ_LDA_X',
                       'VWN-LDA parametrisation of VWN5 form': 'LDA_C_VWN_LDA_X',
                       'VWN-LDA parametrisation of VWN-RPA form': 'LDA_C_VWN_RPA_LDA_X',
                       'AM05 gradient-corrected functionals': 'GGA_C_AM05_GGA_X_AM05',
                       'BLYP functional': 'GGA_C_LYP_GGA_X_B88',
                       'PBE gradient-corrected functionals': 'GGA_C_PBE_GGA_X_PBE',
                       'PBEint gradient-corrected functional': 'GGA_C_PBEINT_GGA_X_PBEINT',
                       'PBEsol gradient-corrected functionals': 'GGA_C_PBE_SOL_GGA_X_PBE_SOL',
                       'RPBE gradient-corrected functionals': 'GGA_C_PBE_GGA_X_RPBE',
                       'revPBE gradient-corrected functionals': 'GGA_C_PBE_GGA_X_PBE_R',
                       'PW91 gradient-corrected functionals': 'GGA_C_PW91_GGA_X_PW91',
                       'M06-L gradient-corrected functionals': 'MGGA_C_M06_L_MGGA_X_M06_L',
                       'M11-L gradient-corrected functionals': 'MGGA_C_M11_L_MGGA_X_M11_L',
                       'TPSS gradient-corrected functionals': 'MGGA_C_TPSS_MGGA_X_TPSS',
                       'TPSSloc gradient-corrected functionals': 'MGGA_C_TPSSLOC_MGGA_X_TPSS',
                       'Hartree-Fock': 'HF_X',
                      }
        # dictionary for conversion of relativistic treatment in aims to metadata format
        self.relativisticDict = {
                                 'Non-relativistic': '',
                                 'ZORA': 'scalar_relativistic',
                                 'on-site free-atom approximation to ZORA': 'scalar_relativistic_atomic_ZORA',
                                }

    def startedParsing(self, fInName, parser):
        """Function is called when the parsing starts.

        Get compiled parser.
        Later one can compile a parser for parsing an external file.

        Args:
            fInName: The file name on which the current parser is running.
            parser: The compiled parser. Is an object of the class SimpleParser in nomadcore.simple_parser.py.
        """
        self.parser = parser

    def update_eigenvalues(self, section, addStr):
        """Update eigenvalues and occupations if they were found in the given section.

        addStr allows to use this function for the eigenvalues of scalar ZORA
        by extending the metadata with addStr.

        Args:
            section: Contains the cached sections and values.
            addStr: String that is appended to the metadata names.
        """
        # get cached fhi_aims_section_eigenvalues_group
        sec_ev_group = section['fhi_aims_section_eigenvalues_group%s' % addStr]
        if sec_ev_group is not None:
            # check if only one group was found
            if len(sec_ev_group) != 1:
                logger.warning("Found %d instead of 1 group of eigenvalues. Used entry 0." % len(sec_ev_group))
            # get cached fhi_aims_section_eigenvalues_group
            sec_ev_spins = sec_ev_group[0]['fhi_aims_section_eigenvalues_spin%s' % addStr]
            for sec_ev_spin in sec_ev_spins:
                occs = []
                evs = []
                kpoints = []
                # get cached fhi_aims_section_eigenvalues_list
                sec_ev_lists = sec_ev_spin['fhi_aims_section_eigenvalues_list%s' % addStr]
                # extract occupations and eigenvalues
                for sec_ev_list in sec_ev_lists:
                    occ = sec_ev_list['fhi_aims_eigenvalue_occupation%s' % addStr]
                    if occ is not None:
                        occs.append(occ)
                    ev = sec_ev_list['fhi_aims_eigenvalue_occupation%s' % addStr]
                    if ev is not None:
                        evs.append(ev)
                # extract kpoints
                for i in ['1', '2', '3']:
                    ki = sec_ev_spin['fhi_aims_eigenvalue_kpoint%s%s' % (i, addStr)]
                    if ki is not None:
                        kpoints.append(ki)
                # append values for each spin channel
                self.eigenvalues_occupation.append(occs)
                self.eigenvalues_eigenvalues.append(evs)
                if kpoints:
                    # transpose list
                    kpoints = map(lambda *x: list(x), *kpoints)
                    self.eigenvalues_kpoints.append(kpoints) 

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
        # write for geometry optimization convergence
        if self.geoConvergence is not None:
            backend.addValue('geometry_optimization_converged', self.geoConvergence)
        # write settings
        for k,v in section.simpleValues.items():
            if k.startswith('fhi_aims_controlIn_') or k.startswith('fhi_aims_controlInOut_'):
                # convert control.in keyword values which are strings to lowercase for consistency
                if k.startswith('fhi_aims_controlIn_') and isinstance(v[-1], str):
                    value = v[-1].lower()
                else:
                    value = v[-1]
                backend.superBackend.addValue(k, value)
        # reset all variables
        self.secMethodIndex = None
        self.secSystemDescriptionIndex = None
        self.scalarZORA = False
        self.periodicCalc = False
        self.MD = False
        self.scfIterNr = -1
        self.scfConvergence = False
        self.geoConvergence = None
        self.eigenvalues_occupation = []
        self.eigenvalues_eigenvalues = []
        self.eigenvalues_kpoints = []

    def onClose_section_method(self, backend, gIndex, section):
        """Trigger called when section_method is closed.

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
        def write_k_grid(metaName, valuesDict):
            """Function to write k-grid for controlIn and controlInOut.

            Args:
                metaName: Corresponding metadata name for k.
                valuesDict: Dictionary that contains the cached values of a section.
            """
            k_grid = []
            for i in ['1', '2', '3']:
                ki = valuesDict.get(metaName + i)
                if ki is not None:
                    k_grid.append(ki[-1])
            if k_grid:
                backend.superBackend.addArrayValues(metaName + '_grid', np.asarray(k_grid))

        def write_xc_functional(metaNameStart, valuesDict):
            """Function to write xc settings for controlIn and controlInOut.

            The omega of the HSE-functional is converted to the unit given in the metadata.
            The xc functional from controlInOut is converted to the string required for the metadata XC_functional.

            Args:
                metaNameStart: Base name of metdata for xc. Must be fhi_aims_controlIn or fhi_aims_controlInOut.
                valuesDict: Dictionary that contains the cached values of a section.
            """
            if metaNameStart == 'fhi_aims_controlIn':
                location = 'control.in'
                functional = 'hse06'
                xcWrite = False
            elif metaNameStart == 'fhi_aims_controlInOut':
                location = 'aims output from the parsed control.in'
                functional = 'HSE-functional'
                xcWrite = True
            else:
                logger.error("Unknown metaName %s in function write_xc_functional. Please correct." % metaNameStart)
                return
            # get cached values for xc functional
            xc = valuesDict.get(metaNameStart + '_xc')
            if xc is not None:
                # check if only one xc keyword was found in control.in
                if len(xc) > 1:
                    logger.warning("Found %d settings for the xc functional in %s: %s. This leads to an undefined behavior und no metadata can be written for %s_xc." % (len(xc), location, xc, metaNameStart))
                else:
                    backend.superBackend.addValue(metaNameStart + '_xc', xc[-1])
                    # convert xc functional to metadata XC_functional
                    if xcWrite:
                        xcMetaName = 'XC_functional'
                        xcString = self.xcDict.get(xc[-1])
                        if xcString is not None:
                            backend.addValue(xcMetaName, xcString)
                        else:
                            logger.error("The xc functional '%s' could not be converted to the required string for the metadata '%s'. Please add it to the dictionary xcDict." % (xc[-1], xcMetaName))
                    # hse_omega is only written for HSE06
                    hse_omega = valuesDict.get(metaNameStart + '_hse_omega')
                    if hse_omega is not None and xc[-1] == functional:
                        # get unit from metadata
                        unit = self.parser.parserBuilder.metaInfoEnv.infoKinds[metaNameStart + '_hse_omega'].units
                        # try unit conversion and write hse_omega
                        hse_unit = valuesDict.get(metaNameStart + '_hse_unit')
                        if hse_unit is not None:
                            value = None
                            if hse_unit[-1] in ['a', 'A']:
                                value = convert_unit(hse_omega[-1], 'angstrom**-1', unit)
                            elif hse_unit[-1] in ['b', 'B']:
                                value = convert_unit(hse_omega[-1], 'bohr**-1', unit)
                            if value is not None:
                                backend.superBackend.addValue(metaNameStart + '_hse_omega', value)
                            else:
                                logger.warning("Unknown hse_unit %s in %s. Cannot write %s" % (hse_unit[-1], location, metaNameStart + '_hse_omega'))
                        else:
                            logger.warning("No value found for %s. Cannot write %s" % (metaNameStart + '_hse_unit', metaNameStart + '_hse_omega'))

        # keep track of the latest method section
        self.secMethodIndex = gIndex
        # list of excluded metadata
        # k_grid is written with k1 so that k2 and k2 are not needed.
        # fhi_aims_controlIn_relativistic_threshold is written with fhi_aims_controlIn_relativistic.
        # The xc setting have to be handeled separatly since having more than one gives undefined behavior.
        # hse_omega is only written if HSE was used and converted according to hse_unit which is not written since not needed.
        exclude_list = ['fhi_aims_controlIn_k2',
                        'fhi_aims_controlIn_k3',
                        'fhi_aims_controlInOut_k2',
                        'fhi_aims_controlInOut_k3',
                        'fhi_aims_controlIn_relativistic_threshold',
                        'fhi_aims_controlIn_xc',
                        'fhi_aims_controlInOut_xc',
                        'fhi_aims_controlIn_hse_omega',
                        'fhi_aims_controlInOut_hse_omega',
                        'fhi_aims_controlIn_hse_unit',
                        'fhi_aims_controlInOut_hse_unit',
                       ]
        # write settings
        for k,v in section.simpleValues.items():
            if k.startswith('fhi_aims_controlIn_') or k.startswith('fhi_aims_controlInOut_'):
                if k in exclude_list:
                    continue
                # write k_krid
                elif k == 'fhi_aims_controlIn_k1':
                    write_k_grid('fhi_aims_controlIn_k', section.simpleValues)
                elif k == 'fhi_aims_controlInOut_k1':
                    write_k_grid('fhi_aims_controlInOut_k', section.simpleValues)
                elif k == 'fhi_aims_controlIn_relativistic':
                    # check for scalar ZORA setting and convert to one common name
                    if re.match(r"\s*zora\s+scalar", v[-1], re.IGNORECASE):
                        backend.superBackend.addValue(k, 'zora scalar')
                        # write threshold only for scalar ZORA
                        value = section[k + '_threshold']
                        if value is not None:
                            backend.superBackend.addValue(k + '_threshold', value[-1])
                    else:
                        backend.superBackend.addValue(k, v[-1].lower())
                elif k == 'fhi_aims_controlInOut_relativistic_threshold':
                    # write threshold only for scalar ZORA
                    if section['fhi_aims_controlInOut_relativistic'] is not None:
                        if section['fhi_aims_controlInOut_relativistic'][-1] == 'ZORA':
                            backend.superBackend.addValue(k, v[-1])
                # default writeout
                else:
                    # convert keyword values of control.in which are strings to lowercase for consistency
                    if k.startswith('fhi_aims_controlIn_') and isinstance(v[-1], str):
                        value = v[-1].lower()
                    else:
                        value = v[-1]
                    backend.superBackend.addValue(k, value)
        # detect MD
        if section['fhi_aims_MD_flag'] is not None:
            self.MD = True
        # detect scalar ZORA
        if section['fhi_aims_controlInOut_relativistic'] is not None:
            if section['fhi_aims_controlInOut_relativistic'][-1] == 'ZORA':
                self.scalarZORA = True
        # convert relativistic setting to metadata string
        InOut_relativistic = section['fhi_aims_controlInOut_relativistic']
        if InOut_relativistic is not None:
            relativistic = self.relativisticDict.get(InOut_relativistic[-1])
            if relativistic is not None:
                backend.addValue('relativity_method', relativistic)
            else:
                logger.warning("The relativistic setting '%s' could not be converted to the required string for the metadata 'relativity_method'. Please add it to the dictionary relativisticDict." % InOut_relativistic[-1])
        # handling of xc functional
        write_xc_functional('fhi_aims_controlIn', section.simpleValues)
        write_xc_functional('fhi_aims_controlInOut', section.simpleValues)

    def onClose_section_system_description(self, backend, gIndex, section):
        """Trigger called when section_system_description is closed.

        Writes atomic positions, atom labels and lattice vectors.
        """
        # keep track of the latest system description section
        self.secSystemDescriptionIndex = gIndex
        # write atomic positions
        atom_pos = []
        for i in ['x', 'y', 'z']:
            api = section['fhi_aims_geometry_atom_position_' + i]
            if api is not None:
                atom_pos.append(api)
        if atom_pos is not None:
            # need to transpose array since its shape is given by [number_of_atoms,3] in the metadata
            backend.addArrayValues('atom_position', np.transpose(np.asarray(atom_pos)))
        # write atom labels
        atom_labels = section['fhi_aims_geometry_atom_label']
        if atom_labels is not None:
            backend.addArrayValues('atom_label', np.asarray(atom_labels))
        # write unit cell if present and set flag self.periodicCalc
        unit_cell = []
        for i in ['x', 'y', 'z']:
            uci = section['fhi_aims_geometry_lattice_vector_' + i]
            if uci is not None:
                unit_cell.append(uci)
        if unit_cell:
            # from metadata: "The first index is x,y,z and the second index the lattice vector."
            # => unit_cell has already the right format
            backend.addArrayValues('simulation_cell', np.asarray(unit_cell))
            self.periodicCalc = True

    def onClose_section_single_configuration_calculation(self, backend, gIndex, section):
        """Trigger called when section_single_configuration_calculation is closed.

        Write number of SCF iterations and convergence.
        Check for convergence of geometry optimization.
        Write eigenvalues.
        Write converged energy values (not for scalar ZORA as these values are extracted separatly).
        Write reference to section_method and section_system_description
        """
        # write number of SCF iterations
        backend.addValue('scf_dft_number_of_iterations', self.scfIterNr)
        # write SCF convergence and reset
        backend.addValue('single_configuration_calculation_converged', self.scfConvergence)
        self.scfConvergence = False
        # check for geometry optimization convergence
        if section['fhi_aims_geometry_optimization_converged'] is not None:
            if section['fhi_aims_geometry_optimization_converged'][-1] == 'is converged':
                self.geoConvergence = True
            else:
                self.geoConvergence = False
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        # write eigenvalues if found
        if self.eigenvalues_occupation and self.eigenvalues_eigenvalues:
            occ = np.asarray(self.eigenvalues_occupation)
            ev = np.asarray(self.eigenvalues_eigenvalues)
            # check if there is the same number of spin channels
            if len(occ) == len(ev):
                kpt = None
                if self.eigenvalues_kpoints:
                    kpt = np.asarray(self.eigenvalues_kpoints)
                # check if there is the same number of spin channels for the periodic case
                if kpt is None or len(kpt) == len(ev):
                    gIndexGroup = backend.openSection('section_eigenvalues_group')
                    for i in range(len(occ)):
                        gIndex = backend.openSection('section_eigenvalues')
                        backend.addArrayValues('eigenvalues_occupation', occ[i])
                        backend.addArrayValues('eigenvalues_eigenvalues', ev[i])
                        if kpt is not None:
                            backend.addArrayValues('eigenvalues_kpoints', kpt[i])
                        backend.closeSection('section_eigenvalues', gIndex)
                    backend.closeSection('section_eigenvalues_group', gIndexGroup)
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
                    backend.addValue(k, v)
        # write the references to section_method and section_system_description
        backend.addValue('single_configuration_calculation_method_ref', self.secMethodIndex)
        backend.addValue('single_configuration_calculation_system_description_ref', self.secSystemDescriptionIndex)

    def onClose_fhi_aims_section_eigenvalues_ZORA(self, backend, gIndex, section):
        """Trigger called when fhi_aims_section_eigenvalues_ZORA is closed.

        Eigenvalues are extracted.
        """
        # reset eigenvalues
        self.eigenvalues_occupation = []
        self.eigenvalues_eigenvalues = []
        self.eigenvalues_kpoints = []
        self.update_eigenvalues(section, '_ZORA')

    def onClose_section_scf_iteration(self, backend, gIndex, section):
        """Trigger called when section_scf_iteration is closed.

        Count number of SCF iterations and check for convergence.
        Energy values are tracked (not for scalar ZORA as these values are extracted separatly).
        Eigenvalues are tracked (not for scalar ZORA, see onClose_fhi_aims_section_eigenvalues_ZORA).

        The quantitties that are tracked during the SCF cycle are reset to default.
        I.e. scalar values are allowed to be set None, and lists are set to empty.
        This ensures that after convergence only the values associated with the last SCF iteration are written
        and not some values that appeared in earlier SCF iterations.
        """
        # count number of SCF iterations
        self.scfIterNr += 1
        # check for SCF convergence
        if section['fhi_aims_single_configuration_calculation_converged'] is not None:
            self.scfConvergence = True
        # keep track of energies
        # With scalar ZORA, the correctly scaled energy values and eigenvalues are given in a separate post-processing step,
        # which are read there. Therefore, we do not track the energy values for scalar ZORA.
        if not self.scalarZORA:
            for k in self.totalEnergyList:
                self.totalEnergyList[k] = section[k + '_scf_iteration']
            # reset eigenvalues
            self.eigenvalues_occupation = []
            self.eigenvalues_eigenvalues = []
            self.eigenvalues_kpoints = []
            self.update_eigenvalues(section, '')

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
            subFlags = SM.SubFlags.Unordered,
            subMatchers = [
            # Now follows the list to match the keywords from the control.in.
            # Explicitly add ^ to ensure that the keyword is not within a comment.
            # The search is done unordered since the keywords do not appear in a specific order.
            # Repating occurrences of the same keywords are captured.
            # Closing the section fhi_aims_section_controlIn will write the last captured value
            # since aims uses the last occurrence of a keyword.
            # List the matchers in alphabetical order according to keyword name.
            #
            SM (r"^\s*MD_time_step\s+(?P<fhi_aims_controlIn_MD_time_step__ps>[-+0-9.eEdD]+)", repeats = True),
            # need to distinguish different cases
            SM (r"^\s*occupation_type\s+",
                forwardMatch = True,
                repeats = True,
                subMatchers = [
                SM (r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)\s+(?P<fhi_aims_controlIn_occupation_order>[0-9]+)"),
                SM (r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)")
                ]),
            SM (r"^\s*override_relativity\s+\.?(?P<fhi_aims_controlIn_override_relativity>[-_a-zA-Z]+)\.?", repeats = True),
            SM (r"^\s*charge\s+(?P<fhi_aims_controlIn_charge>[-+0-9.eEdD]+)", repeats = True),
            # only the first character is important for aims
            SM (r"^\s*hse_unit\s+(?P<fhi_aims_controlIn_hse_unit>[a-zA-Z])[-_a-zA-Z0-9]+", repeats = True),
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
            SM (r"^\s*k_grid\s+(?P<fhi_aims_controlIn_k1>[0-9]+)\s+(?P<fhi_aims_controlIn_k2>[0-9]+)\s+(?P<fhi_aims_controlIn_k3>[0-9]+)", repeats = True),
            SM (r"^\s*spin\s+(?P<fhi_aims_controlIn_spin>[-_a-zA-Z]+)", repeats = True),
            # need to distinguish two cases: just the name of the xc functional or name plus number (e.g. for HSE functional)
            SM (r"^\s*xc\s+",
                forwardMatch = True,
                repeats = True,
                subMatchers = [
                SM (r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)\s+(?P<fhi_aims_controlIn_hse_omega>[-+0-9.eEdD]+)"),
                SM (r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)")
                ]),
            SM (r"^\s*verbatim_writeout\s+[.a-zA-Z]+")
            ]), # END ControlInKeywords
        SM (r"\s*-{20}-*", weak = True)
        ])
    ########################################
    # submatcher for aims output from the parsed control.in
    controlInOutSubMatcher = SM (name = 'ControlInOut',
        startReStr = r"\s*Reading file control\.in\.",
        subMatchers = [
        SM (name = 'ControlInOutLines',
            startReStr = r"\s*-{20}-*",
            weak = True,
            subFlags = SM.SubFlags.Unordered,
            subMatchers = [
            # Now follows the list to match the aims output from the parsed control.in.
            # The search is done unordered since the output is not in a specific order.
            # Repating occurrences of the same keywords are captured.
            # List the matchers in alphabetical order according to metadata name.
            #
            # only the first character is important for aims
            SM (r"\s*hse_unit: Unit for the HSE06 hybrid functional screening parameter set to (?P<fhi_aims_controlInOut_hse_unit>[a-zA-Z])[a-zA-Z]*\^\(-1\)\.", repeats = True),
            # metadata fhi_aims_MD_flag is just used to detect already in the methods section if a MD run was perfomed
            SM (r"\s*(?P<fhi_aims_MD_flag>Molecular dynamics time step) =\s*(?P<fhi_aims_controlInOut_MD_time_step__ps>[0-9.]+) *ps", repeats = True),
            SM (r"\s*Scalar relativistic treatment of kinetic energy: (?P<fhi_aims_controlInOut_relativistic>[-a-zA-Z\s]+)\.", repeats = True),
            SM (r"\s*(?P<fhi_aims_controlInOut_relativistic>Non-relativistic) treatment of kinetic energy\.", repeats = True),
            SM (r"\s*Threshold value for ZORA:\s*(?P<fhi_aims_controlInOut_relativistic_threshold>[-+0-9.eEdD]+)", repeats = True),
            # need several regualar expressions to capture all possible output messages of the xc functional
            SM (r"\s*XC: Using (?P<fhi_aims_controlInOut_xc>[-_a-zA-Z0-9\s()]+)(?:\.| NOTE)", repeats = True),
            SM (r"\s*XC: Using (?P<fhi_aims_controlInOut_xc>HSE-functional) with OMEGA =\s*(?P<fhi_aims_controlInOut_hse_omega>[-+0-9.eEdD]+)\s*<units>\.", repeats = True),
            SM (r"\s*XC: Using (?P<fhi_aims_controlInOut_xc>Hybrid M11 gradient-corrected functionals) with OMEGA =\s*[-+0-9.eEdD]+", repeats = True),
            SM (r"\s*XC:\s*(?P<fhi_aims_controlInOut_xc>HSE) with OMEGA_PBE =\s*[-+0-9.eEdD]+", repeats = True),
            SM (r"\s*XC: Running (?P<fhi_aims_controlInOut_xc>[-_a-zA-Z0-9\s()]+) \.\.\.", repeats = True),
            SM (r"\s*(?P<fhi_aims_controlInOut_xc>Hartree-Fock) calculation starts \.\.\.\.\.\.", repeats = True),
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
                SM (r"^\s*initial_charge\s+[-+0.9.eEdD]", repeats = True),
                SM (r"^\s*initial_moment\s+[-+0.9.eEdD]", repeats = True),
                SM (r"^\s*velocity\s+[-+0.9.eEdD]\s+[-+0.9.eEdD]\s+[-+0.9.eEdD]", repeats = True)
                ]),
            SM (r"^\s*hessian_block\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+0-9.eEdD]+", repeats = True),
            SM (r"^\s*hessian_block_lv\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+0-9.eEdD]+", repeats = True),
            SM (r"^\s*hessian_block_lv_atom\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+0-9.eEdD]+", repeats = True),
            SM (startReStr = r"^\s*lattice_vector\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+",
                repeats = True,
                subMatchers = [
                SM (r"^\s*constrain_relaxation\s+(?:x|y|z|\.true\.|\.false\.)", repeats = True)
                ]),
            SM (r"^\s*trust_radius\s+[-+0-9.eEdD]+", repeats = True),
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
        sections = ['section_system_description'],
        subMatchers = [
        SM (r"\s*-{20}-*", weak = True),
        SM (r"\s*The structure contains\s*[0-9]+\s*atoms,\s*and a total of\s*[0-9.]\s*electrons\."),
        SM (r"\s*Input geometry:"),
        SM (r"\s*\|\s*No unit cell requested\."),
        SM (startReStr = r"\s*\|\s*Unit cell:",
            subMatchers = [
            SM (r"\s*\|\s*(?P<fhi_aims_geometry_lattice_vector_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_z__angstrom>[-+0-9.]+)", repeats = True)
            ]),
        SM (startReStr = r"\s*\|\s*Atomic structure:",
            subMatchers = [
            SM (r"\s*\|\s*Atom\s*x \[A\]\s*y \[A\]\s*z \[A\]"),
            SM (r"\s*\|\s*[0-9]+:\s*Species\s+(?P<fhi_aims_geometry_atom_label>[a-zA-Z]+)\s+(?P<fhi_aims_geometry_atom_position_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_z__angstrom>[-+0-9.]+)", repeats = True)
            ])
        ])
    ########################################
    # subMatcher for geometry after relaxation step
    geometryRelaxationSubMatcher = SM (name = 'GeometryRelaxation',
        startReStr = r"\s*Updated atomic structure:",
        sections = ['section_system_description'],
        subMatchers = [
        SM (r"\s*x \[A\]\s*y \[A\]\s*z \[A\]"),
        SM (startReStr = r"\s*lattice_vector\s*[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+",
            forwardMatch = True,
            subMatchers = [
            SM (r"\s*lattice_vector\s+(?P<fhi_aims_geometry_lattice_vector_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_z__angstrom>[-+0-9.]+)", repeats = True)
            ]),
        SM (r"\s*atom\s+(?P<fhi_aims_geometry_atom_position_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_z__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_label>[a-zA-Z]+)", repeats = True),
        SM (startReStr = r"\s*Fractional coordinates:",
            subMatchers = [
            SM ("\s*L1\s*L2\s*L3"),
            SM (r"\s*atom_frac\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+\s+[a-zA-Z]+", repeats = True),
            SM (r'\s*Writing the current geometry to file "geometry\.in\.next_step"\.'),
            SM (r"\s*-{20}-*", weak = True)
            ])
        ])
    ########################################
    # subMatcher for geometry after MD step
    geometryMDSubMatcher = SM (name = 'GeometryMD',
        startReStr = r"\s*Atomic structure \(and velocities\) as used in the preceding time step:",
        sections = ['section_system_description'],
        subMatchers = [
        SM (r"\s*x \[A\]\s*y \[A\]\s*z \[A\]\s*Atom"),
        SM (startReStr = r"\s*atom\s+(?P<fhi_aims_geometry_atom_position_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_z__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_label>[a-zA-Z]+)",
            repeats = True,
            subMatchers = [
            SM (r"^\s*velocity\s+[-+0.9.eEdD]\s+[-+0.9.eEdD]\s+[-+0.9.eEdD]")
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
           SimpleMatcher that parses eigenvalues with metadata ccording to addStr. 
        """
        # submatcher for eigenvalue list
        EigenvaluesListSubMatcher =  SM (name = 'EigenvaluesLists',
            startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
            sections = ['fhi_aims_section_eigenvalues_list%s' % addStr],
            subMatchers = [
            SM (startReStr = r"\s*[0-9]+\s+(?P<fhi_aims_eigenvalue_occupation%s>[0-9.eEdD]+)\s+[-+0-9.eEdD]+\s+(?P<fhi_aims_eigenvalue_eigenvalue%s__eV>[-+0-9.eEdD]+)" % (2 * (addStr,)), repeats = True)
            ])
        return SM (name = 'EigenvaluesGroup',
            startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
            sections = ['fhi_aims_section_eigenvalues_group%s' % addStr],
            subMatchers = [
            # spin-polarized
            SM (name = 'EigenvaluesSpin',
                startReStr = r"\s*Spin-(?:up|down) eigenvalues:",
                sections = ['fhi_aims_section_eigenvalues_spin%s' % addStr],
                repeats = True,
                subMatchers = [
                # periodic
                SM (startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+(?P<fhi_aims_eigenvalue_kpoint1%s>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint2%s>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint3%s>[-+0-9.eEdD]+)\s+\(in units of recip\. lattice\)" % (3 * (addStr,)),
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
                sections = ['fhi_aims_section_eigenvalues_spin%s' % addStr],
                forwardMatch = True,
                subMatchers = [
                SM (startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+(?P<fhi_aims_eigenvalue_kpoint1%s>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint2%s>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint3%s>[-+0-9.eEdD]+)\s+\(in units of recip\. lattice\)" % (3 * (addStr,)),
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
                sections = ['fhi_aims_section_eigenvalues_spin%s' % addStr],
                forwardMatch = True,
                subMatchers = [
                EigenvaluesListSubMatcher.copy()
                ]), # END EigenvaluesNoSpinNonPeriodic
            ])
    # now construct the two subMatchers
    EigenvaluesGroupSubMatcher = build_eigenvaluesGroupSubMatcher('')
    EigenvaluesGroupSubMatcherZORA = build_eigenvaluesGroupSubMatcher('_ZORA')
    ########################################
    # submatcher for total energy components during SCF interation
    TotalEnergyScfSubMatcher = SM (name = 'TotalEnergyScf',
        startReStr = r"\s*Total energy components:",
        subMatchers = [
        SM (r"\s*\|\s*Sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_sum_eigenvalues_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*XC energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*XC potential correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_potential_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Free-atom electrostatic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<fhi_aims_energy_electrostatic_free_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Hartree energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_correction_hartree_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Entropy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_correction_entropy_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*-{20}-*", weak = True),
        SM (r"\s*\|\s*Total energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_total_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Total energy, T -> 0\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_total_T0_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Electronic free energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_free_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*Derived energy quantities:"),
        SM (r"\s*\|\s*Kinetic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<electronic_kinetic_energy_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Electrostatic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_electrostatic_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Energy correction for multipole"),
        SM (r"\s*\|\s*error in Hartree potential\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_hartree_error_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<energy_sum_eigenvalues_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<energy_total_T0_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<energy_free_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        ])
    ########################################
    # submatcher for total energy components in scalar ZORA post-processing
    TotalEnergyZORASubMatcher = SM (name = 'TotalEnergyZORA',
        startReStr = r"\s*Total energy components:",
        subMatchers = [
        SM (r"\s*\|\s*Sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_sum_eigenvalues__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*XC energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_functional__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*XC potential correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_potential__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Free-atom electrostatic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+[-+0-9.eEdD]+ *eV"),
        SM (r"\s*\|\s*Hartree energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_correction_hartree__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Entropy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_correction_entropy__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*-{20}-*", weak = True),
        SM (r"\s*\|\s*Total energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+[-+0-9.eEdD]+ *eV"),
        SM (r"\s*\|\s*Total energy, T -> 0\s*:\s*[-+0-9.eEdD]+ *Ha\s+[-+0-9.eEdD]+ *eV"),
        SM (r"\s*\|\s*Electronic free energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+[-+0-9.eEdD]+ *eV"),
        SM (r"\s*Derived energy quantities:"),
        SM (r"\s*\|\s*Kinetic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<electronic_kinetic_energy__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Electrostatic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_electrostatic__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Energy correction for multipole"),
        SM (r"\s*\|\s*error in Hartree potential\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_hartree_error__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<energy_sum_eigenvalues_per_atom__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<energy_total_T0_per_atom__eV>[-+0-9.eEdD]+) *eV"),
        SM (r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<energy_free_per_atom__eV>[-+0-9.eEdD]+) *eV"),
        ])
    ########################################
    # return main Parser
    return SM (name = 'Root',
        weak = True,
        startReStr = "",
        subMatchers = [
        SM (name = 'NewRun',
            startReStr = r"\s*Invoking FHI-aims \.\.\.",
            endReStr = r"\s*Have a nice day\.",
            repeats = True,
            required = True,
            forwardMatch = True,
            sections   = ['section_run'],
            subMatchers = [
            # header specifing version, compilation info, task assignment
            SM (name = 'ProgramHeader',
                startReStr = r"\s*Invoking FHI-aims \.\.\.",
                subMatchers = [
                SM (r"\s*Version\s*(?P<program_version>[0-9a-zA-Z_.]+)"),
                SM (r"\s*Compiled on\s*(?P<fhi_aims_program_compilation_date>[0-9/]+)\s*at\s*(?P<fhi_aims_program_compilation_time>[0-9:]+)\s*on host\s*(?P<program_compilation_host>[-a-zA-Z0-9._]+)"),
                SM (r"\s*Date\s*:\s*(?P<fhi_aims_program_execution_date>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_program_execution_time>[-+0-9.eEdD]+)"),
                SM (r"\s*-{20}-*", weak = True),
                SM (r"\s*Time zero on CPU 1\s*:\s*(?P<time_run_cpu1_start>[-+0-9.eEdD]+) *s?\."),
                SM (r"\s*Internal wall clock time zero\s*:\s*(?P<time_run_wall_start>[-+0-9.eEdD]+) *s\."),
                SM (name = "nParallelTasks",
                    startReStr = r"\s*Using\s*(?P<fhi_aims_number_of_tasks>[0-9]+)\s*parallel tasks\.",
                    sections = ["fhi_aims_section_parallel_tasks"],
                    subMatchers = [
                    SM (name = 'ParallelTasksAssignement',
                        startReStr = r"\s*Task\s*(?P<fhi_aims_parallel_task_nr>[0-9]+)\s*on host\s+(?P<fhi_aims_parallel_task_host>[-a-zA-Z0-9._]+)\s+reporting\.",
                        repeats = True,
                        sections = ["fhi_aims_section_parallel_task_assignement"])
                    ]), # END nParallelTasks
                SM (r"\s*Performing system and environment tests:"),
                ]), # END ProgramHeader
            SM (r"\s*Obtaining array dimensions for all initial allocations:"),
            # parse control and geometry
            SM (name = 'SectionMethod',
                startReStr = r"\s*Parsing control\.in \(first pass over file, find array dimensions only\)\.",
                sections = ["section_method"],
                subMatchers = [
                # parse verbatim writeout of control.in
                controlInSubMatcher,
                # parse verbatim writeout of geometry.in
                geometryInSubMatcher,
                # parse settings writeout of aims
                controlInOutSubMatcher,
                # parse geometry writeout of aims
                geometrySubMatcher
                ]), # END SectionMethod
            SM (r"\s*Preparing all fixed parts of the calculation\."),
            # this SimpleMatcher groups a single configuration calculation together with output after SCF convergence from Relaxation and MD
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
                    sections = ['section_single_configuration_calculation'],
                    subMatchers = [
                    # initialization of SCF loop, SCF iteration 0
                    SM (name = 'ScfInitialization',
                        startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                        sections = ['section_scf_iteration'],
                        subMatchers = [
                        SM (r"\s*Date\s*:\s*(?P<fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)"),
                        SM (r"\s*-{20}-*", weak = True),
                        EigenvaluesGroupSubMatcher,
                        TotalEnergyScfSubMatcher,
                        SM (r"\s*Full exact exchange energy:\s*[-+0-9.eEdD]+ *eV"),
                        SM (r"\s*End scf initialization - timings\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)"),
                        SM (r"\s*-{20}-*", weak = True)
                        ]), # END ScfInitialization
                    # normal SCF iterations
                    SM (name = 'ScfIteration',
                        startReStr = r"\s*Begin self-consistency iteration #\s*[0-9]+",
                        sections = ['section_scf_iteration'],
                        repeats = True,
                        subMatchers = [
                        SM (r"\s*Date\s*:\s*(?P<fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)"),
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*Full exact exchange energy:\s*[-+0-9.eEdD]+ *eV"),
                        EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        TotalEnergyScfSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        # SCF convergence info
                        SM (name = 'SCFConvergence',
                            startReStr = r"\s*Self-consistency convergence accuracy:",
                            subMatchers = [
                            SM (r"\s*\|\s*Change of charge(?:/spin)? density\s*:\s*[-+0-9.eEdD]+\s+[-+0-9.eEdD]*"),
                            SM (r"\s*\|\s*Change of sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *eV"),
                            SM (r"\s*\|\s*Change of total energy\s*:\s*(?P<energy_change_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
                            SM (r"\s*\|\s*Change of forces\s*:\s*[-+0-9.eEdD]+ *eV/A"),
                            SM (r"\s*\|\s*Change of analytical stress\s*:\s*[-+0-9.eEdD]+ *eV/A\*\*3")
                            ]), # END SCFConvergence
                        # after convergence eigenvalues are printed in the end instead of usually in the beginning
                        EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
                        SM (r"\s*(?P<fhi_aims_single_configuration_calculation_converged>Self-consistency cycle converged)\."),
                        SM (r"\s*End self-consistency iteration #\s*[0-9]+\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)")
                        ]), # END ScfIteration
                    # possible scalar ZORA post-processing
                    SM (name = 'ScaledZORAPostProcessing',
                        startReStr = r"\s*Post-processing: scaled ZORA corrections to eigenvalues and total energy.",
                        endReStr = r"\s*End evaluation of scaled ZORA corrections.",
                        subMatchers = [
                        SM (r"\s*Date\s*:\s*[-.0-9/]+\s*,\s*Time\s*:\s*[-+0-9.eEdD]+"),
                        SM (r"\s*-{20}-*", weak = True),
                        # parse eigenvalues
                        SM (name = 'EigenvaluesListsZORA',
                            startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
                            forwardMatch = True,
                            sections = ['fhi_aims_section_eigenvalues_ZORA'],
                            subMatchers = [
                            EigenvaluesGroupSubMatcherZORA
                            ]), # END EigenvaluesListsZORA
                        TotalEnergyZORASubMatcher
                        ]), # END ScaledZORAPostProcessing
                    # summary of energy and forces
                    SM (name = 'EnergyForcesSummary',
                        startReStr = r"\s*Energy and forces in a compact form:",
                        subMatchers = [
                        SM (r"\s*\|\s*Total energy uncorrected\s*:\s*(?P<energy_total__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*\|\s*Total energy corrected\s*:\s*(?P<energy_total_T0__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*\|\s*Electronic free energy\s*:\s*(?P<energy_free__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*-{20}-*", weak = True)
                        ]), # END EnergyForcesSummary
                    # decomposition of the xc energy
                    SM (name = 'DecompositionXCEnergy',
                        startReStr = r"\s*Start decomposition of the XC Energy",
                        subMatchers = [
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*Hartree-Fock part\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_hartree_fock_X_scaled__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*X Energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_X__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*C Energy GGA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_C__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*Total XC Energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_functional__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*LDA X and C from self-consistent density"),
                        SM (r"\s*X Energy LDA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<fhi_aims_energy_X_LDA__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*C Energy LDA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<fhi_aims_energy_C_LDA__eV>[-+0-9.eEdD]+) *eV"),
                        SM (r"\s*-{20}-*", weak = True),
                        SM (r"\s*End decomposition of the XC Energy"),
                        SM (r"\s*-{20}-*", weak = True)
                        ]), # END DecompositionXCEnergy
                    # caclualtion was not converged
                    SM (name = 'ScfNotConverged',
                        startReStr = r"\s*\*\s*WARNING! SELF-CONSISTENCY CYCLE DID NOT CONVERGE",
                        subMatchers = [
                        SM (r"\s*\*\s*USING YOUR PRESELECTED ACCURACY CRITERIA\."),
                        SM (r"\s*\*\s*DO NOT RELY ON ANY FINAL DATA WITHOUT FURTHER CHECKS\.")
                        ]),
                    # geometry relaxation
                    SM (name = 'Relaxation',
                        startReStr = r"\s*Geometry optimization: Attempting to predict improved coordinates\.",
                        subMatchers = [
                        SM (r"\s*Removing unitary transformations \(pure translations, rotations\) from forces on atoms\."),
                        SM (r"\s*Maximum force component is\s*[-+0-9.eEdD]\s*eV/A\."),
                        SM (r"\s*Present geometry (?P<fhi_aims_geometry_optimization_converged>is converged)\."),
                        SM (name = 'RelaxationStep',
                            startReStr = r"\s*Present geometry (?P<fhi_aims_geometry_optimization_converged>is not yet converged)\.",
                            subMatchers = [
                            SM (r"\s*Relaxation step number\s*[0-9]+: Predicting new coordinates\."),
                            SM (r"\s*Advancing geometry using (?:trust radius method|BFGS)\.")
                            ]) # END RelaxationStep
                        ]), # END Relaxation
                    # MD
                    SM (name = 'MD',
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
                            ]) # END MDStep
                        ]) # END MD
                    ]), # END SingleConfigurationCalculation
                # parse updated geometry for relaxation
                geometryRelaxationSubMatcher,
                # parse updated geometry for MD
                #geometryMDSubMatcher
                ]), # END SingleConfigurationCalculationWithSystemDescription
            SM (r"\s*-{20}-*", weak = True),
            SM (r"\s*Final output of selected total energy values:"),
            SM (r"\s*The following output summarizes some interesting total energy values"),
            SM (r"\s*at the end of a run \(AFTER all relaxation, molecular dynamics, etc\.\)\."),
            SM (r"\s*-{20}-*", weak = True),
            SM (r"\s*-{20}-*", weak = True),
            SM (r"\s*Leaving FHI-aims\."),
            SM (r"\s*Date\s*:\s*[-.0-9/]+\s*,\s*Time\s*:\s*[-+0-9.eEdD]+"),
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
    
def get_metaInfo(filePath):
    """Loads metadata.

    Args:
        filePath: Location of metadata.

    Returns:
       metadata which is an object of the class InfoKindEnv in nomadcore.local_meta_info.py.
    """
    metaInfoEnv, warnings = loadJsonFile(filePath = filePath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)
    return metaInfoEnv

def get_cachingLevelForMetaName(metaInfoEnv):
    """Sets the caching level for the metadata.

    Args:
       metaInfoEnv: metadata which is an object of the class InfoKindEnv in nomadcore.local_meta_info.py.

    Returns:
       Dictionary with metaname as key and caching level as value. 
    """
    # manually adjust caching of metadata
    cachingLevelForMetaName = {
                               'fhi_aims_MD_flag': CachingLevel.Cache,
                               'fhi_aims_geometry_optimization_converged': CachingLevel.Cache,
                               'fhi_aims_single_configuration_calculation_converged': CachingLevel.Cache,
                              }
    # Set all controlIn and controlInOut metadata to Cache to capture multiple occurrences of keywords and
    # their last value is then written by the onClose routine in the FhiAimsParserContext.
    # Set all geometry metadata to Cache as all of them need post-processsing.
    # Set all eigenvalue related metadata to Cache.
    for name in metaInfoEnv.infoKinds:
        if (   name.startswith('fhi_aims_controlIn_')
            or name.startswith('fhi_aims_controlInOut_')
            or name.startswith('fhi_aims_geometry_')
            or name.startswith('fhi_aims_eigenvalue_')
            or name.startswith('fhi_aims_section_eigenvalues_')
           ):
            cachingLevelForMetaName[name] = CachingLevel.Cache
    return cachingLevelForMetaName

def main():
    """Main function.

    Set up everything for the parsing of the FHI-aims main file.
    """
    # get main file description
    FhiAimsMainFileSimpleMatcher = build_FhiAimsMainFileSimpleMatcher()
    # loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json
    metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
    metaInfoEnv = get_metaInfo(metaInfoPath)
    # set parser info
    parserInfo = {'name':'fhi-aims-parser', 'version': '1.0'}
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

