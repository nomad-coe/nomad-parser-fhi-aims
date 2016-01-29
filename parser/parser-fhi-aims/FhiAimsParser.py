import setup_paths
import numpy as np
import nomadcore.ActivateLogging
from nomadcore.simple_parser import SimpleMatcher, mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.caching_backend import CachingLevel
import re, os, sys, json, logging

############################################################
# REMARKS
# 
# Energies are parsed in eV to get consistent values 
# since not all energies are given in hartree.
############################################################

logger = logging.getLogger("nomad.FhiAimsParser") 

class FhiAimsParserContext(object):
    def __init__(self):
        self.secMethodIndex = None
        self.secSystemDescriptionIndex = None
        self.scalarZORA = False
        self.periodicCalc = False
        self.MD = False
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
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
        """
        self.parser = parser

    def update_eigenvalues(self, section, addStr):
        """Update eigenvalues and occupations if they were found in the given section.

        addStr allows to use this function for the eigenvalues of scalar ZORA
        by extending the metadata with addStr.
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

        Write the keywords from control.in and the aims output from the parsed control.in, which belong to settings_run.
        Write the last occurrence of a keyword/setting, i.e. [-1], since aims uses the last occurrence of a keyword.
        Variables are reset to ensure clean start for new run.

        ATTENTION
        backend.superBackend is used here instead of only the backend to write the JSON values,
        since this allows to bybass the caching setting which was used to collect the values for processing.
        However, this also bypasses the checking of validity of the metadata name by the backend.
        The scala part will check the validity nevertheless.
        """
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
            """
            k_grid = []
            for i in ['1', '2', '3']:
                ki = valuesDict.get(metaName + i)
                if ki is not None:
                    k_grid.append(ki[-1])
            if k_grid:
                backend.superBackend.addArrayValues(metaName + '_grid', np.asarray(k_grid))
        # keep track of the latest method section
        self.secMethodIndex = gIndex
        # list of excluded metadata
        # k_grid is written with k1 so that k2 and k2 are not needed.
        # fhi_aims_controlIn_relativistic_threshold is written with fhi_aims_controlIn_relativistic.
        # The xc setting have to be handeled separatly since having more than one gives undefined behavior.
        # hse_omega is only written if HSE was used and converted according to the given unit.
        exclude_list = ['fhi_aims_controlIn_k2',
                        'fhi_aims_controlIn_k3',
                        'fhi_aims_controlInOut_k2',
                        'fhi_aims_controlInOut_k3',
                        'fhi_aims_controlIn_relativistic_threshold',
                        'fhi_aims_controlIn_xc',
                        'fhi_aims_controlInOut_xc',
                        'fhi_aims_controlIn_hse_omega',
                        'fhi_aims_controlInOut_hse_omega',
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
                    # convert keyword values which are strings to lowercase for consistency
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
        # handling xc functional
        In_xc = section['fhi_aims_controlIn_xc']
        if In_xc is not None:
            # check if only one xc keyword was found in control.in
            if len(In_xc) > 1:
                logger.warning("Found %d settings for the xc functional in control.in: %s. This leads to an undefined behavior und no metadata can be written for fhi_aims_controlIn_xc." % (len(In_xc), In_xc))
            else:
                backend.superBackend.addValue('fhi_aims_controlIn_xc', In_xc[-1])
                # hse_omega is only written for HSE06
                In_hse_omega = section['fhi_aims_controlIn_hse_omega']
                if In_hse_omega is not None and re.match('hse06', In_xc[-1], re.IGNORECASE):
                    backend.superBackend.addValue('fhi_aims_controlIn_hse_omega', In_hse_omega[-1])
        InOut_xc = section['fhi_aims_controlInOut_xc']
        if InOut_xc is not None:
            # check if only one xc keyword was found in the aims output from the parsed control.in
            if len(InOut_xc) > 1:
                logger.error("Found %d settings for the xc functional in the aims output from the parsed control.in: %s. This leads to an undefined behavior und no metadata can be written for fhi_aims_controlInOut_xc and XC_functional." % (len(InOut_xc), InOut_xc))
            else:
                backend.superBackend.addValue('fhi_aims_controlInOut_xc', InOut_xc[-1])
                # hse_omega is only written for HSE06
                InOut_hse_omega = section['fhi_aims_controlInOut_hse_omega']
                if InOut_hse_omega is not None and InOut_xc[-1] == 'HSE-functional':
                    backend.superBackend.addValue('fhi_aims_controlInOut_hse_omega', InOut_hse_omega[-1])
                # convert xc functional to metadata string
                xc = self.xcDict.get(InOut_xc[-1])
                if xc is not None:
                    backend.addValue('XC_functional', xc)
                else:
                    logger.error("The xc functional '%s' could not be converted to the required string for the metadata 'XC_functional'. Please add it to the dictionary xcDict." % InOut_xc[-1])

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

        Write number of SCF iterations.
        Write eigenvalues.
        Write converged energy values (not for scalar ZORA as these values are extracted separatly).
        Write reference to section_method and section_system_description
        """
        # write number of SCF iterations
        backend.addValue('scf_dft_number_of_iterations', self.scfIterNr)
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

        Count number of SCF iterations.
        Energy values are tracked (not for scalar ZORA as these values are extracted separatly).
        Eigenvalues are tracked (not for scalar ZORA, see onClose_fhi_aims_section_eigenvalues_ZORA).

        The quantitties that are tracked during the SCF cycle are reset to default.
        I.e. scalar values are allowed to be set None, and lists are set to empty.
        This ensures that after convergence only the values associated with the last SCF iteration are written
        and not some values that appeared in earlier SCF iterations.
        """
        # count number of SCF iterations
        self.scfIterNr += 1
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

########################################
# submatcher for control.in
controlInSubMatcher = SimpleMatcher(name = 'ControlIn',
              startReStr = r"\s*The contents of control\.in will be repeated verbatim below",
              endReStr = r"\s*Completed first pass over input file control\.in \.",
              subMatchers = [
  SimpleMatcher(r"\s*unless switched off by setting 'verbatim_writeout \.false\.' \."),
  SimpleMatcher(r"\s*in the first line of control\.in \."),
  SimpleMatcher(name = 'ControlInKeywords',
                startReStr = r"\s*-{20}-*",
                weak = True,
                subFlags = SimpleMatcher.SubFlags.Unordered,
                subMatchers = [
    # Now follows the list to match the keywords from the control.in.
    # Explicitly add ^ to ensure that the keyword is not within a comment.
    # The search is done unordered since the keywords do not appear in a specific order.
    # Repating occurrences of the same keywords are captured.
    # Closing the section fhi_aims_section_controlIn will write the last captured value
    # since aims uses the last occurrence of a keyword.
    # List the matchers in alphabetical order according to keyword name.
    #
    SimpleMatcher(r"^\s*MD_time_step\s+(?P<fhi_aims_controlIn_MD_time_step__ps>[-+0-9.eEdD]+)", repeats = True),
    # need to distinguish different cases
    SimpleMatcher(r"^\s*occupation_type\s+",
                  forwardMatch = True,
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)\s+(?P<fhi_aims_controlIn_occupation_order>[0-9]+)"),
      SimpleMatcher(r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)")
                              ]),
    SimpleMatcher(r"^\s*override_relativity\s+\.?(?P<fhi_aims_controlIn_override_relativity>[-_a-zA-Z]+)\.?", repeats = True),
    SimpleMatcher(r"^\s*charge\s+(?P<fhi_aims_controlIn_charge>[-+0-9.eEdD]+)", repeats = True),
    # only the first character is important for aims
    SimpleMatcher(r"^\s*hse_unit\s+(?P<fhi_aims_controlIn_hse_unit>[a-zA-Z])[-_a-zA-Z0-9]+", repeats = True),
    # need to distinguish different cases
    SimpleMatcher(r"^\s*relativistic\s+",
                  forwardMatch = True,
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*relativistic\s+(?P<fhi_aims_controlIn_relativistic>[-_a-zA-Z]+\s+[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_relativistic_threshold>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"^\s*relativistic\s+(?P<fhi_aims_controlIn_relativistic>[-_a-zA-Z]+)")
                              ]),
    SimpleMatcher(r"^\s*sc_accuracy_rho\s+(?P<fhi_aims_controlIn_sc_accuracy_rho>[-+0-9.eEdD]+)", repeats = True),
    SimpleMatcher(r"^\s*sc_accuracy_eev\s+(?P<fhi_aims_controlIn_sc_accuracy_eev>[-+0-9.eEdD]+)", repeats = True),
    SimpleMatcher(r"^\s*sc_accuracy_etot\s+(?P<fhi_aims_controlIn_sc_accuracy_etot>[-+0-9.eEdD]+)", repeats = True),
    SimpleMatcher(r"^\s*sc_accuracy_forces\s+(?P<fhi_aims_controlIn_sc_accuracy_forces>[-+0-9.eEdD]+)", repeats = True),
    SimpleMatcher(r"^\s*sc_accuracy_stress\s+(?P<fhi_aims_controlIn_sc_accuracy_stress>[-+0-9.eEdD]+)", repeats = True),
    SimpleMatcher(r"^\s*sc_iter_limit\s+(?P<fhi_aims_controlIn_sc_iter_limit>[0-9]+)", repeats = True),
    SimpleMatcher(r"^\s*k_grid\s+(?P<fhi_aims_controlIn_k1>[0-9]+)\s+(?P<fhi_aims_controlIn_k2>[0-9]+)\s+(?P<fhi_aims_controlIn_k3>[0-9]+)", repeats = True),
    SimpleMatcher(r"^\s*spin\s+(?P<fhi_aims_controlIn_spin>[-_a-zA-Z]+)", repeats = True),
    # need to distinguish two cases: just the name of the xc functional or name plus number (e.g. for HSE functional)
    SimpleMatcher(r"^\s*xc\s+",
                  forwardMatch = True,
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)\s+(?P<fhi_aims_controlIn_hse_omega>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)")
                  ]),
    SimpleMatcher(r"^\s*verbatim_writeout\s+[.a-zA-Z]+")
                ]), # END ControlInKeywords
  SimpleMatcher(r"\s*-{20}-*", weak = True)
              ])
########################################
# submatcher for aims output from the parsed control.in
controlInOutSubMatcher = SimpleMatcher(name = 'ControlInOut',
              startReStr = r"\s*Reading file control\.in\.",
              subMatchers = [
  SimpleMatcher(name = 'ControlInOutLines',
                startReStr = r"\s*-{20}-*",
                weak = True,
                subFlags = SimpleMatcher.SubFlags.Unordered,
                subMatchers = [
    # Now follows the list to match the aims output from the parsed control.in.
    # The search is done unordered since the output is not in a specific order.
    # Repating occurrences of the same keywords are captured.
    # List the matchers in alphabetical order according to metadata name.
    #
    # metadata fhi_aims_MD_flag is just used to detect already in the methods section if a MD run was perfomed
    SimpleMatcher(r"\s*(?P<fhi_aims_MD_flag>Molecular dynamics time step) =\s*(?P<fhi_aims_controlInOut_MD_time_step__ps>[0-9.]+) *ps", repeats = True),
    SimpleMatcher(r"\s*Scalar relativistic treatment of kinetic energy: (?P<fhi_aims_controlInOut_relativistic>[-a-zA-Z\s]+)\.", repeats = True),
    SimpleMatcher(r"\s*(?P<fhi_aims_controlInOut_relativistic>Non-relativistic) treatment of kinetic energy\.", repeats = True),
    SimpleMatcher(r"\s*Threshold value for ZORA:\s*(?P<fhi_aims_controlInOut_relativistic_threshold>[-+0-9.eEdD]+)", repeats = True),
    # need several regualar expressions to capture all possible output messages of the xc functional
    SimpleMatcher(r"\s*XC: Using (?P<fhi_aims_controlInOut_xc>[-_a-zA-Z0-9\s()]+)(?:\.| NOTE)", repeats = True),
    SimpleMatcher(r"\s*XC: Using (?P<fhi_aims_controlInOut_xc>HSE-functional) with OMEGA =\s*(?P<fhi_aims_controlInOut_hse_omega>[-+0-9.eEdD]+)\s*<units>\.", repeats = True),
    SimpleMatcher(r"\s*XC: Using (?P<fhi_aims_controlInOut_xc>Hybrid M11 gradient-corrected functionals) with OMEGA =\s*[-+0-9.eEdD]+", repeats = True),
    SimpleMatcher(r"\s*XC:\s*(?P<fhi_aims_controlInOut_xc>HSE) with OMEGA_PBE =\s*[-+0-9.eEdD]+", repeats = True),
    SimpleMatcher(r"\s*XC: Running (?P<fhi_aims_controlInOut_xc>[-_a-zA-Z0-9\s()]+) \.\.\.", repeats = True),
    SimpleMatcher(r"\s*(?P<fhi_aims_controlInOut_xc>Hartree-Fock) calculation starts \.\.\.\.\.\.", repeats = True),
                ]), # END ControlInOutLines
  SimpleMatcher(r"\s*-{20}-*", weak = True)
              ])
########################################
# subMatcher for geometry.in
geometryInSubMatcher = SimpleMatcher(name = 'GeometryIn',
              startReStr = r"\s*Parsing geometry\.in \(first pass over file, find array dimensions only\)\.",
              endReStr = r"\s*Completed first pass over input file geometry\.in \.",
              subMatchers = [
  SimpleMatcher(r"\s*The contents of geometry\.in will be repeated verbatim below"),
  SimpleMatcher(r"\s*unless switched off by setting 'verbatim_writeout \.false\.' \."),
  SimpleMatcher(r"\s*in the first line of geometry\.in \."),
  SimpleMatcher(name = 'GeometryInKeywords',
                startReStr = r"\s*-{20}-*",
                weak = True,
                subFlags = SimpleMatcher.SubFlags.Unordered,
                subMatchers = [
    # Explicitly add ^ to ensure that the keyword is not within a comment.
    # The search is done unordered since the keywords do not appear in a specific order.
    SimpleMatcher(startReStr = r"^\s*(?:atom|atom_frac)\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+\s+[a-zA-Z]+",
                  repeats = True,
                  subFlags = SimpleMatcher.SubFlags.Unordered,
                  subMatchers = [
      SimpleMatcher(r"^\s*constrain_relaxation\s+(?:x|y|z|\.true\.|\.false\.)", repeats = True),
      SimpleMatcher(r"^\s*initial_charge\s+[-+0.9.eEdD]", repeats = True),
      SimpleMatcher(r"^\s*initial_moment\s+[-+0.9.eEdD]", repeats = True),
      SimpleMatcher(r"^\s*velocity\s+[-+0.9.eEdD]\s+[-+0.9.eEdD]\s+[-+0.9.eEdD]", repeats = True)
                  ]),
    SimpleMatcher(r"^\s*hessian_block\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+0-9.eEdD]+", repeats = True),
    SimpleMatcher(r"^\s*hessian_block_lv\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+0-9.eEdD]+", repeats = True),
    SimpleMatcher(r"^\s*hessian_block_lv_atom\s+[0-9]+\s+[0-9]+" + 9 * r"\s+[-+0-9.eEdD]+", repeats = True),
    SimpleMatcher(startReStr = r"^\s*lattice_vector\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+",
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*constrain_relaxation\s+(?:x|y|z|\.true\.|\.false\.)", repeats = True)
                  ]),
    SimpleMatcher(r"^\s*trust_radius\s+[-+0-9.eEdD]+", repeats = True),
    SimpleMatcher(r"^\s*verbatim_writeout\s+[.a-zA-Z]+")
                ]), # END GeometryInKeywords
  SimpleMatcher(r"\s*-{20}-*", weak = True)
              ])
########################################
# subMatcher for geometry
# the verbatim writeout of the geometry.in is not considered for getting the structure data
# using the geometry output of aims has the advantage that it has a clearer structure
geometrySubMatcher = SimpleMatcher(name = 'Geometry',
              startReStr = r"\s*Reading geometry description geometry\.in\.",
              sections = ['section_system_description'],
              subMatchers = [
  SimpleMatcher(r"\s*-{20}-*", weak = True),
  SimpleMatcher(r"\s*The structure contains\s*[0-9]+\s*atoms,\s*and a total of\s*[0-9.]\s*electrons\."),
  SimpleMatcher(r"\s*Input geometry:"),
  SimpleMatcher(r"\s*\|\s*No unit cell requested\."),
  SimpleMatcher(startReStr = r"\s*\|\s*Unit cell:",
                subMatchers = [
    SimpleMatcher(r"\s*\|\s*(?P<fhi_aims_geometry_lattice_vector_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_z__angstrom>[-+0-9.]+)", repeats = True)
                ]),
  SimpleMatcher(startReStr = r"\s*\|\s*Atomic structure:",
                subMatchers = [
    SimpleMatcher(r"\s*\|\s*Atom\s*x \[A\]\s*y \[A\]\s*z \[A\]"),
    SimpleMatcher(r"\s*\|\s*[0-9]+:\s*Species\s+(?P<fhi_aims_geometry_atom_label>[a-zA-Z]+)\s+(?P<fhi_aims_geometry_atom_position_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_z__angstrom>[-+0-9.]+)", repeats = True)
                ])
              ])
########################################
# subMatcher for geometry after relaxation step
geometryRelaxationSubMatcher = SimpleMatcher(name = 'GeometryRelaxation',
              startReStr = r"\s*Updated atomic structure:",
              sections = ['section_system_description'],
              subMatchers = [
  SimpleMatcher(r"\s*x \[A\]\s*y \[A\]\s*z \[A\]"),
  SimpleMatcher(startReStr = r"\s*lattice_vector\s*[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+",
                forwardMatch = True,
                subMatchers = [
    SimpleMatcher(r"\s*lattice_vector\s+(?P<fhi_aims_geometry_lattice_vector_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_z__angstrom>[-+0-9.]+)", repeats = True)
                ]),
  SimpleMatcher(r"\s*atom\s+(?P<fhi_aims_geometry_atom_position_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_z__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_label>[a-zA-Z]+)", repeats = True),
  SimpleMatcher(startReStr = r"\s*Fractional coordinates:",
                subMatchers = [
    SimpleMatcher("\s*L1\s*L2\s*L3"),
    SimpleMatcher(r"\s*atom_frac\s+[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+\s+[a-zA-Z]+", repeats = True),
  SimpleMatcher(r'\s*Writing the current geometry to file "geometry\.in\.next_step"\.'),
  SimpleMatcher(r"\s*-{20}-*", weak = True)
                ])
              ])
########################################
# subMatcher for geometry after MD step
geometryMDSubMatcher = SimpleMatcher(name = 'GeometryMD',
              startReStr = r"\s*Atomic structure \(and velocities\) as used in the preceding time step:",
              sections = ['section_system_description'],
              subMatchers = [
  SimpleMatcher(r"\s*x \[A\]\s*y \[A\]\s*z \[A\]\s*Atom"),
  SimpleMatcher(startReStr = r"\s*atom\s+(?P<fhi_aims_geometry_atom_position_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_z__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_label>[a-zA-Z]+)",
                repeats = True,
                subMatchers = [
    SimpleMatcher(r"^\s*velocity\s+[-+0.9.eEdD]\s+[-+0.9.eEdD]\s+[-+0.9.eEdD]")
                ]),
              ])
########################################
# submatcher for eigenvalues
# first define function to build submatcher for normal case and scalar ZORA
def generateEigenvaluesGroupSubMatcher(addStr):
    # submatcher for eigenvalue list
    EigenvaluesListSubMatcher =  SimpleMatcher(name = 'EigenvaluesLists',
              startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
              sections = ['fhi_aims_section_eigenvalues_list%s' % addStr],
              subMatchers = [
      SimpleMatcher(startReStr = r"\s*[0-9]+\s+(?P<fhi_aims_eigenvalue_occupation%s>[0-9.eEdD]+)\s+[-+0-9.eEdD]+\s+(?P<fhi_aims_eigenvalue_eigenvalue%s__eV>[-+0-9.eEdD]+)" % (2 * (addStr,)), repeats = True)
                  ])
    return SimpleMatcher(name = 'EigenvaluesGroup',
              startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
              sections = ['fhi_aims_section_eigenvalues_group%s' % addStr],
              subMatchers = [
  # spin-polarized
  SimpleMatcher(name = 'EigenvaluesSpin',
                startReStr = r"\s*Spin-(?:up|down) eigenvalues:",
                sections = ['fhi_aims_section_eigenvalues_spin%s' % addStr],
                repeats = True,
                subMatchers = [
    # periodic
    SimpleMatcher(startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+(?P<fhi_aims_eigenvalue_kpoint1%s>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint2%s>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint3%s>[-+0-9.eEdD]+)\s+\(in units of recip\. lattice\)" % (3 * (addStr,)),
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
                    forwardMatch = True,
                    subMatchers = [
        EigenvaluesListSubMatcher.copy()
                    ])
                  ]),
    # non-periodic
    SimpleMatcher(startReStr = r"\s*State\s+Occupation\s+Eigenvalue *\[Ha\]\s+Eigenvalue *\[eV\]",
                  forwardMatch = True,
                  subMatchers = [
      EigenvaluesListSubMatcher.copy()
                  ])
                ]), # END EigenvaluesSpin
  # non-spin-polarized, periodic
  SimpleMatcher(name = 'EigenvaluesNoSpinPeriodic',
                startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+.*\s+\(in units of recip\. lattice\)",
                sections = ['fhi_aims_section_eigenvalues_spin%s' % addStr],
                forwardMatch = True,
                subMatchers = [
    SimpleMatcher(startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+(?P<fhi_aims_eigenvalue_kpoint1%s>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint2%s>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint3%s>[-+0-9.eEdD]+)\s+\(in units of recip\. lattice\)" % (3 * (addStr,)),
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
                    forwardMatch = True,
                    subMatchers = [
        EigenvaluesListSubMatcher.copy()
                    ])
                  ]),
                ]), # END EigenvaluesNoSpinPeriodic
  # non-spin-polarized, non-periodic
  SimpleMatcher(name = 'EigenvaluesNoSpinNonPeriodic',
                startReStr = r"\s*State\s+Occupation\s+Eigenvalue *\[Ha\]\s+Eigenvalue *\[eV\]",
                sections = ['fhi_aims_section_eigenvalues_spin%s' % addStr],
                forwardMatch = True,
                subMatchers = [
    EigenvaluesListSubMatcher.copy()
                ]), # END EigenvaluesNoSpinNonPeriodic
              ])
EigenvaluesGroupSubMatcher = generateEigenvaluesGroupSubMatcher('')
EigenvaluesGroupSubMatcherZORA = generateEigenvaluesGroupSubMatcher('_ZORA')
########################################
# submatcher for total energy components during SCF interation
TotalEnergyScfSubMatcher = SimpleMatcher(name = 'TotalEnergyScf',
              startReStr = r"\s*Total energy components:",
              subMatchers = [
  SimpleMatcher(r"\s*\|\s*Sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_sum_eigenvalues_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*XC energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*XC potential correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_potential_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Free-atom electrostatic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<fhi_aims_energy_electrostatic_free_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Hartree energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_correction_hartree_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Entropy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_correction_entropy_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*-{20}-*", weak = True),
  SimpleMatcher(r"\s*\|\s*Total energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_total_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Total energy, T -> 0\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_total_T0_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Electronic free energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_free_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*Derived energy quantities:"),
  SimpleMatcher(r"\s*\|\s*Kinetic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<electronic_kinetic_energy_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Electrostatic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_electrostatic_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Energy correction for multipole"),
  SimpleMatcher(r"\s*\|\s*error in Hartree potential\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_hartree_error_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<energy_sum_eigenvalues_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<energy_total_T0_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<energy_free_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
              ])
########################################
# submatcher for total energy components in scalar ZORA post-processing
TotalEnergyZORASubMatcher = SimpleMatcher(name = 'TotalEnergyZORA',
              startReStr = r"\s*Total energy components:",
              subMatchers = [
  SimpleMatcher(r"\s*\|\s*Sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_sum_eigenvalues__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*XC energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_functional__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*XC potential correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_potential__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Free-atom electrostatic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Hartree energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_correction_hartree__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Entropy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_correction_entropy__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*-{20}-*", weak = True),
  SimpleMatcher(r"\s*\|\s*Total energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Total energy, T -> 0\s*:\s*[-+0-9.eEdD]+ *Ha\s+[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Electronic free energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*Derived energy quantities:"),
  SimpleMatcher(r"\s*\|\s*Kinetic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<electronic_kinetic_energy__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Electrostatic energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_electrostatic__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Energy correction for multipole"),
  SimpleMatcher(r"\s*\|\s*error in Hartree potential\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_hartree_error__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<energy_sum_eigenvalues_per_atom__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<energy_total_T0_per_atom__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<energy_free_per_atom__eV>[-+0-9.eEdD]+) *eV"),
              ])

########################################
# main Parser
mainFileDescription = SimpleMatcher(name = 'Root',
              weak = True,
              startReStr = "",
              subMatchers = [
  SimpleMatcher(name = 'NewRun',
                startReStr = r"\s*Invoking FHI-aims \.\.\.",
                endReStr = r"\s*Have a nice day\.",
                repeats = True,
                required = True,
                forwardMatch = True,
                sections   = ['section_run'],
                subMatchers = [
    # header specifing version, compilation info, task assignment
    SimpleMatcher(name = 'ProgramHeader',
                  startReStr = r"\s*Invoking FHI-aims \.\.\.",
                  subMatchers = [
      SimpleMatcher(r"\s*Version\s*(?P<program_version>[0-9a-zA-Z_.]+)"),
      SimpleMatcher(r"\s*Compiled on\s*(?P<fhi_aims_program_compilation_date>[0-9/]+)\s*at\s*(?P<fhi_aims_program_compilation_time>[0-9:]+)\s*on host\s*(?P<program_compilation_host>[-a-zA-Z0-9._]+)"),
      SimpleMatcher(r"\s*Date\s*:\s*(?P<fhi_aims_program_execution_date>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_program_execution_time>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"\s*-{20}-*", weak = True),
      SimpleMatcher(r"\s*Time zero on CPU 1\s*:\s*(?P<time_run_cpu1_start>[-+0-9.eEdD]+) *s?\."),
      SimpleMatcher(r"\s*Internal wall clock time zero\s*:\s*(?P<time_run_wall_start>[-+0-9.eEdD]+) *s\."),
      SimpleMatcher(name = "nParallelTasks",
                    startReStr = r"\s*Using\s*(?P<fhi_aims_number_of_tasks>[0-9]+)\s*parallel tasks\.",
                    sections = ["fhi_aims_section_parallel_tasks"],
                    subMatchers = [
        SimpleMatcher(name = 'ParallelTasksAssignement',
                      startReStr = r"\s*Task\s*(?P<fhi_aims_parallel_task_nr>[0-9]+)\s*on host\s+(?P<fhi_aims_parallel_task_host>[-a-zA-Z0-9._]+)\s+reporting\.",
                      repeats = True,
                      sections = ["fhi_aims_section_parallel_task_assignement"])
                    ]), # END nParallelTasks
      SimpleMatcher(r"\s*Performing system and environment tests:"),
                  ]), # END ProgramHeader
    SimpleMatcher(r"\s*Obtaining array dimensions for all initial allocations:"),
    # parse control and geometry
    SimpleMatcher(name = 'SectionMethod',
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
    SimpleMatcher(r"\s*Preparing all fixed parts of the calculation\."),
    # this SimpleMatcher groups a single configuration calculation together with output after SCF convergence from Relaxation and MD
    SimpleMatcher(name = 'SingleConfigurationCalculationWithSystemDescription',
                  startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                  repeats = True,
                  forwardMatch = True,
                  subMatchers = [
      # the actual section for a single configuration calculation starts here
      SimpleMatcher(name = 'SingleConfigurationCalculation',
                    startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                    repeats = True,
                    forwardMatch = True,
                    sections = ['section_single_configuration_calculation'],
                    subMatchers = [
        # initialization of SCF loop, SCF iteration 0
        SimpleMatcher(name = 'ScfInitialization',
                      startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                      sections = ['section_scf_iteration'],
                      subMatchers = [
          SimpleMatcher(r"\s*Date\s*:\s*(?P<fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)"),
          SimpleMatcher(r"\s*-{20}-*", weak = True),
          EigenvaluesGroupSubMatcher,
          TotalEnergyScfSubMatcher,
          SimpleMatcher(r"\s*Full exact exchange energy:\s*[-+0-9.eEdD]+ *eV"),
          SimpleMatcher(r"\s*End scf initialization - timings\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)"),
          SimpleMatcher(r"\s*-{20}-*", weak = True)
                       ]), # END ScfInitialization
        # normal SCF iterations
        SimpleMatcher(name = 'ScfIteration',
                      startReStr = r"\s*Begin self-consistency iteration #\s*[0-9]+",
                      sections = ['section_scf_iteration'],
                      repeats = True,
                      subMatchers = [
          SimpleMatcher(r"\s*Date\s*:\s*(?P<fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)"),
          SimpleMatcher(r"\s*-{20}-*", weak = True),
          SimpleMatcher(r"\s*Full exact exchange energy:\s*[-+0-9.eEdD]+ *eV"),
          EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
          TotalEnergyScfSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
          # SCF convergence info
          SimpleMatcher(name = 'SCFConvergence',
                        startReStr = r"\s*Self-consistency convergence accuracy:",
                        subMatchers = [
            SimpleMatcher(r"\s*\|\s*Change of charge(?:/spin)? density\s*:\s*[-+0-9.eEdD]+\s+[-+0-9.eEdD]*"),
            SimpleMatcher(r"\s*\|\s*Change of sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *eV"),
            SimpleMatcher(r"\s*\|\s*Change of total energy\s*:\s*(?P<energy_change_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
            SimpleMatcher(r"\s*\|\s*Change of forces\s*:\s*[-+0-9.eEdD]+ *eV/A"),
            SimpleMatcher(r"\s*\|\s*Change of analytical stress\s*:\s*[-+0-9.eEdD]+ *eV/A\*\*3")
                        ]), # END SCFConvergence
          # after convergence eigenvalues are printed in the end instead of usually in the beginning
          EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
          SimpleMatcher(r"\s*End self-consistency iteration #\s*[0-9]+\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)"),
          SimpleMatcher(r"\s*Self-consistency cycle converged\.")
                      ]), # END ScfIteration
        # possible scalar ZORA post-processing
        SimpleMatcher(name = 'ScaledZORAPostProcessing',
                      startReStr = r"\s*Post-processing: scaled ZORA corrections to eigenvalues and total energy.",
                      endReStr = r"\s*End evaluation of scaled ZORA corrections.",
                      subMatchers = [
          SimpleMatcher(r"\s*Date\s*:\s*[-.0-9/]+\s*,\s*Time\s*:\s*[-+0-9.eEdD]+"),
          SimpleMatcher(r"\s*-{20}-*", weak = True),
          # parse eigenvalues
          SimpleMatcher(name = 'EigenvaluesListsZORA',
                        startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
                        forwardMatch = True,
                        sections = ['fhi_aims_section_eigenvalues_ZORA'],
                        subMatchers = [
            EigenvaluesGroupSubMatcherZORA
                        ]), # END EigenvaluesListsZORA
          TotalEnergyZORASubMatcher
                      ]), # END ScaledZORAPostProcessing
        # summary of energy and forces
        SimpleMatcher(name = 'EnergyForcesSummary',
                      startReStr = r"\s*Energy and forces in a compact form:",
                      subMatchers = [
          SimpleMatcher(r"\s*\|\s*Total energy uncorrected\s*:\s*(?P<energy_total__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*\|\s*Total energy corrected\s*:\s*(?P<energy_total_T0__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*\|\s*Electronic free energy\s*:\s*(?P<energy_free__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*-{20}-*", weak = True)
                      ]), # END EnergyForcesSummary
        # decomposition of the xc energy
        SimpleMatcher(name = 'DecompositionXCEnergy',
                      startReStr = r"\s*Start decomposition of the XC Energy",
                      subMatchers = [
          SimpleMatcher(r"\s*-{20}-*", weak = True),
          SimpleMatcher(r"\s*Hartree-Fock part\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_hartree_fock_X_scaled__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*-{20}-*", weak = True),
          SimpleMatcher(r"\s*X Energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_X__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*C Energy GGA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_C__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*Total XC Energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_functional__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*LDA X and C from self-consistent density"),
          SimpleMatcher(r"\s*X Energy LDA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<fhi_aims_energy_X_LDA__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*C Energy LDA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<fhi_aims_energy_C_LDA__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*-{20}-*", weak = True),
          SimpleMatcher(r"\s*End decomposition of the XC Energy"),
          SimpleMatcher(r"\s*-{20}-*", weak = True)
                      ]), # END DecompositionXCEnergy
        # geometry relaxation
        SimpleMatcher(name = 'Relaxation',
                      startReStr = r"\s*Geometry optimization: Attempting to predict improved coordinates\.",
                      subMatchers = [
          SimpleMatcher(r"\s*Removing unitary transformations \(pure translations, rotations\) from forces on atoms\."),
          SimpleMatcher(r"\s*Maximum force component is\s*[-+0-9.eEdD]\s*eV/A\."),
          SimpleMatcher(r"\s*Present geometry is converged\."),
          SimpleMatcher(name = 'RelaxationStep',
                        startReStr = r"\s*Present geometry is not yet converged\.",
                        subMatchers = [
            SimpleMatcher(r"\s*Relaxation step number\s*[0-9]+: Predicting new coordinates\."),
            SimpleMatcher(r"\s*Advancing geometry using (?:trust radius method|BFGS)\.")
                        ]) # END RelaxationStep
                      ]), # END Relaxation
        # MD
        SimpleMatcher(name = 'MD',
                startReStr = r"\s*Molecular dynamics: Attempting to update all nuclear coordinates\.",
                      subMatchers = [
          SimpleMatcher(r"\s*Removing unitary transformations \(pure translations, rotations\) from forces on atoms\."),
          SimpleMatcher(r"\s*Maximum force component is\s*[-+0-9.eEdD]\s*eV/A\."),
          SimpleMatcher(r"\s*Present geometry is converged\."),
          SimpleMatcher(name = 'MDStep',
                        startReStr = r"\s*Advancing structure using Born-Oppenheimer Molecular Dynamics:",
                        subMatchers = [
            SimpleMatcher(r"\s*Complete information for previous time-step:"),
            SimpleMatcher(r"\s*\|\s*Time step number\s*:\s*[0-9]+"),
            SimpleMatcher(r"\s*-{20}-*", weak = True)
                        ]) # END MDStep
                      ]) # END MD
                    ]), # END SingleConfigurationCalculation
      # parse updated geometry for relaxation
      geometryRelaxationSubMatcher,
      # parse updated geometry for MD
      #geometryMDSubMatcher
                  ]), # END SingleConfigurationCalculationWithSystemDescription
    SimpleMatcher(r"\s*-{20}-*", weak = True),
    SimpleMatcher(r"\s*Final output of selected total energy values:"),
    SimpleMatcher(r"\s*The following output summarizes some interesting total energy values"),
    SimpleMatcher(r"\s*at the end of a run \(AFTER all relaxation, molecular dynamics, etc\.\)\."),
    SimpleMatcher(r"\s*-{20}-*", weak = True),
    SimpleMatcher(r"\s*-{20}-*", weak = True),
    SimpleMatcher(r"\s*Leaving FHI-aims\."),
    SimpleMatcher(r"\s*Date\s*:\s*[-.0-9/]+\s*,\s*Time\s*:\s*[-+0-9.eEdD]+"),
    # summary of computational steps
    SimpleMatcher(name = 'ComputationalSteps',
                  startReStr = r"\s*Computational steps:",
                  subMatchers = [
      SimpleMatcher(r"\s*\|\s*Number of self-consistency cycles\s*:\s*[0-9]+"),
      SimpleMatcher(r"\s*\|\s*Number of relaxation steps\s*:\s*[0-9]+"),
      SimpleMatcher(r"\s*\|\s*Number of molecular dynamics steps\s*:\s*[0-9]+"),
      SimpleMatcher(r"\s*\|\s*Number of force evaluations\s*:\s*[0-9]+")
                  ]), # END ComputationalSteps
    SimpleMatcher(r"\s*Detailed time accounting\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)")
                ]) # END NewRun
              ]) # END Root

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json
metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)
# dictionary for functions to call on close of a section metaname
onClose = {}
# parser info
parserInfo = {'name':'fhi-aims-parser', 'version': '1.0'}
# adjust caching of metadata
cachingLevelForMetaName = {
                           'fhi_aims_MD_flag': CachingLevel.Cache,
                          }

if __name__ == "__main__":
    # Set all controlIn and controlInOut metadata to Cache to capture multiple occurrences of keywords and
    # their last value is then written by the onClose routine in the FhiAimsParserContext.
    # Set all geometry metadata to Cache as all of them need post-processsing.
    # Set all eigenvalue related metadata to Cache.
    for name in metaInfoEnv.infoKinds:
        if (  name.startswith('fhi_aims_controlIn_')
            or name.startswith('fhi_aims_controlInOut_')
            or name.startswith('fhi_aims_geometry_')
            or name.startswith('fhi_aims_eigenvalue_')
            or name.startswith('fhi_aims_section_eigenvalues_')
           ):
            cachingLevelForMetaName[name] = CachingLevel.Cache
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo, superContext = FhiAimsParserContext(), cachingLevelForMetaName = cachingLevelForMetaName, onClose = onClose)

