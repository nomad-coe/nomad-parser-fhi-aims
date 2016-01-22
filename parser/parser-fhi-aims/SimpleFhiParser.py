import setup_paths
import numpy as np
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

logger = logging.getLogger("FhiAimsParser") 
logger.addHandler(logging.StreamHandler())

class FhiAimsParserContext(object):
    def __init__(self):
        self.secMethodIndex = None
        self.secSystemDescriptionIndex = None
        self.scalarZORA = False
        self.periodicCalc = False
        self.xc = None
        self.energy_hartree_fock_X = None
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        self.eigenvalues_occupation = []
        self.eigenvalues_occupation_spin = []
        self.eigenvalues_occupation_spin_tmp = []
        self.eigenvalues_eigenvalues = []
        self.eigenvalues_eigenvalues_spin = []
        self.eigenvalues_eigenvalues_spin_tmp = []
        self.eigenvalues_kpoints_spin = []
        self.eigenvalues_kpoints_spin_tmp = []
        # dictonary of energy values, which are tracked between scf iterations and printed after convergence
        self.totalEnergyList = {
                                'energy_sum_eigenvalues': None,
                                'energy_XC': None,
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
        # dictonary for conversion of xc functional name in aims to metadata format
        # TODO hybrid GGA and mGGA functionals and vdW functionals
        self.xcDict = {
                       'pw-lda': 'LDA_C_PW_LDA_X',
                       'pz-lda': 'LDA_C_PZ_LDA_X',
                       'vwn': 'LDA_C_VWN_LDA_X',
                       'vwn-gauss': 'LDA_C_VWN_RPA_LDA_X',
                       'am05': 'GGA_C_AM05_GGA_X_AM05',
                       'blyp': 'GGA_C_LYP_GGA_X_B88',
                       'pbe': 'GGA_C_PBE_GGA_X_PBE',
                       'pbeint': 'GGA_C_PBEINT_GGA_X_PBEINT',
                       'pbesol': 'GGA_C_PBE_SOL_GGA_X_PBE_SOL',
                       'rpbe': 'GGA_C_PBE_GGA_X_RPBE',
                       'revpbe': 'GGA_C_PBE_GGA_X_PBE_R',
                       'pw91_gga': 'GGA_C_PW91_GGA_X_PW91',
                       'm06-l': 'MGGA_C_M06_L_MGGA_X_M06_L',
                       'm11-l': 'MGGA_C_M11_L_MGGA_X_M11_L',
                       'tpss': 'MGGA_C_TPSS_MGGA_X_TPSS',
                       'tpssloc': 'MGGA_C_TPSSLOC_MGGA_X_TPSS',
                       'hf': 'HF_X',
                      }

    def write_system_description(self, backend, gIndex, section):
        """writes atomic positions, atom labels and lattice vectors"""
        # write atomic positions
        atom_pos = []
        for i in ['x', 'y', 'z']:
            api = section.simpleValues.get('fhi_aims_geometry_atom_position_' + i)
            if api is not None:
                atom_pos.append(api)
        if len(atom_pos) == 3:
            if all(len(x) == len(atom_pos[0]) for x in atom_pos[1:-1]):
                # Need to transpose array since its shape is given by [number_of_atoms,3] in the metadata
                backend.addArrayValues('atom_position', np.transpose(np.asarray(atom_pos)))
            else:
                raise Exception("The number of atomic positions is not consistent for all components in geometry specification %d:" + len(atom_pos) * " %d" % tuple(map(len, atom_pos)) + (gIndex,))
        else:
            raise Exception("Found atomic positions but only %d instead of 3 components in geometry specification %d." % (len(atom_pos), gIndex))
        # write atom labels
        atom_labels = section.simpleValues.get('fhi_aims_geometry_atom_label')
        if atom_labels is not None:
            if len(atom_labels) == len(atom_pos[0]):
                backend.addArrayValues('atom_label', np.asarray(atom_labels))
            else:
                raise Exception("The number of atom labels % d is not equal the number of atomic positions %d in geometry specification %d." % (len(atom_labels), len(atom_pos[0]), gIndex))
        else:
            raise Exception("Found no atom labels in geometry specification %d." % gIndex)
        # write unit cell if present and set flag self.periodicCalc
        unit_cell = []
        for i in ['x', 'y', 'z']:
            uci = section.simpleValues.get('fhi_aims_geometry_lattice_vector_' + i)
            if uci is not None:
                unit_cell.append(uci)
                self.periodicCalc = True
        if self.periodicCalc:
            if len(uci) == 3:
                if all(len(x) == 3 for x in unit_cell):
                    # From metadata: The first index is x,y,z and the second index the lattice vector.
                    # => unit_cell has already the right format
                    backend.addArrayValues('simulation_cell', np.asarray(unit_cell))
                    self.periodicCalc = True
                else:
                    raise Exception("The number of lattice vectors is not consistent for all components in geometry specification %d:" + len(unit_cell) * " %d" % tuple(map(len, unit_cell)) + (gIndex,))
            else:
                raise Exception("Found lattice vectors but only %d instead of 3 components in geometry specification %d." % (len(unit_cell), gIndex))

    def onClose_fhi_aims_section_controlIn(self, backend, gIndex, section):
        """trigger called when section fhi_aims_section_controlIn is closed"""
        # write the last occurrence of a keyword, i.e. [-1], since aims uses the last occurrence of a keyword
        # convert keyword values which are strings to lowercase for consistency
        for k,v in section.simpleValues.items():
            # write k_krid if all 3 values are present
            if k == 'fhi_aims_controlIn_k1':
                k_grid = []
                for i in ['1', '2', '3']:
                    ki = section.simpleValues.get('fhi_aims_controlIn_k' + i)
                    if ki is not None:
                        k_grid.append(ki[-1])
                if len(k_grid) == 3:
                    backend.superBackend.addArrayValues('fhi_aims_controlIn_k_grid', np.asarray(k_grid))
                else:
                    raise Exception("Found keyword k_grid but only %d instead of 3 components." % len(k_grid))
            # k_grid was already written, therefore, do nothing
            elif k in ['fhi_aims_controlIn_k2', 'fhi_aims_controlIn_k3']:
                pass
            elif k == 'fhi_aims_controlIn_relativistic_label':
                # check for scalar ZORA setting and convert to one common name
                if re.match(r"\s*zora\s+scalar", v[-1], re.IGNORECASE):
                    self.scalarZORA = True
                    backend.superBackend.addValue('fhi_aims_controlIn_relativistic', 'zora scalar')
                    # write threshold only for scalar ZORA
                    threshold = section.simpleValues.get('fhi_aims_controlIn_relativistic_threshold')[-1]
                    backend.superBackend.addValue('fhi_aims_controlIn_relativistic_threshold', threshold)
                else:
                    backend.superBackend.addValue('fhi_aims_controlIn_relativistic', v[-1])
            # write threshold only for scalar ZORA
            elif k == 'fhi_aims_controlIn_relativistic_threshold':
                pass
            # get value for xc
            elif k == 'fhi_aims_controlIn_xc':
                value = v[-1].lower()
                self.xc = value
                backend.superBackend.addValue(k, value)
            # default writing
            else:
                if isinstance(v[-1], str):
                    value = v[-1].lower()
                else:
                    value = v[-1]
                backend.superBackend.addValue(k, value)

    def onClose_section_method(self, backend, gIndex, section):
        """trigger called when section_method is closed"""
        # keep track of the latest method section
        self.secMethodIndex = gIndex
        # write xc functional
        xc = self.xcDict.get(self.xc)
        if xc is not None:
            backend.addValue('XC_functional', xc)
        else:
            logger.warning("The xc functional %s could not be converted to the required string for the metadata. Please add it to the dictionary xcDict." % self.xc)

    def onClose_section_system_description(self, backend, gIndex, section):
        """trigger called when section_system_description is closed"""
        # keep track of the latest system description section
        self.secSystemDescriptionIndex = gIndex
        self.write_system_description(backend, gIndex, section)

    def onClose_section_single_configuration_calculation(self, backend, gIndex, section):
        """trigger called when section_single_configuration_calculation is closed"""
        # write number of SCF iterations
        backend.addValue('scf_dft_number_of_iterations', self.scfIterNr)
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        # write eigenvalues if found
        if self.eigenvalues_occupation_spin and self.eigenvalues_eigenvalues_spin:
            gIndexGroup = backend.openSection('section_eigenvalues_group')
            occ = np.asarray(self.eigenvalues_occupation_spin)
            ev = np.asarray(self.eigenvalues_eigenvalues_spin)
            if self.periodicCalc:
                kpt = np.asarray(self.eigenvalues_kpoints_spin)
            # in onClose_fhi_aims_section_eigenvalues_list it was already checked 
            # that the lowest lying lists in occupation and eigenvalues have the same length
            # therefore, eigenvalues_occupation_spin and eigenvalues_eigenvalues_spin should be of the same form
            if occ.shape != ev.shape:
                raise Exception("Array of collected eingenvalue occupations and eigenvalues do not have same shape, %s vs. %s" % (occ.shape, ev.shape))
            # check if the first two dimensions (number of spin chhannels and number of kpoints) are equal
            if self.periodicCalc:
                if occ.shape[0:2] != kpt.shape[0:2]:
                    raise Exception("Array of collected eingenvalue occupations and kpoints do not have same shape for the first two dimensions, %s vs. %s" % (occ.shape[0:2], kpt.shape[0:2]))
            for i in range (occ.shape[0]):
                gIndex = backend.openSection('section_eigenvalues')
                backend.addArrayValues('eigenvalues_occupation', occ[i])
                backend.addArrayValues('eigenvalues_eigenvalues', ev[i])
                self.eigenvalues_occupation_spin = []
                self.eigenvalues_eigenvalues_spin = []
                if self.periodicCalc:
                    backend.addArrayValues('eigenvalues_kpoints', kpt[i])
                    self.eigenvalues_kpoints = []
                backend.closeSection('section_eigenvalues', gIndex)
            backend.closeSection('section_eigenvalues_group', gIndexGroup)
        # write converged energy values
        # with scalar ZORA, the correctly scaled energy values are given in a separate post-processing step, which are read there
        # therefore, we don't write the energy values for scalar ZORA
        if not self.scalarZORA:
            for k,v in self.totalEnergyList.items():
                if v is not None:
                    # need to change name of energy_XC to energy_XC_functional according to metadata
                    if k == 'energy_XC':
                        k = k + '_functional'
                    backend.addValue(k, v)
        if self.energy_hartree_fock_X is not None:
            backend.addValue('energy_hartree_fock_X', self.energy_hartree_fock_X)
        # write the references to section_method and section_system_description
        backend.addValue('single_configuration_calculation_method_ref', self.secMethodIndex)
        backend.addValue('single_configuration_calculation_system_description_ref', self.secSystemDescriptionIndex)

    def onClose_section_scf_iteration(self, backend, gIndex, section):
        """trigger called when section_scf_iteration is closed"""
        self.scfIterNr += 1
        # extract current value for the energy values
        # with scalar ZORA, the correctly scaled energy values are given in a separate post-processing step, which are read there
        # therefore, we do not track the energy values for scalar ZORA
        if not self.scalarZORA:
            for k in self.totalEnergyList:
                self.totalEnergyList[k] = section.simpleValues.get(k + '_scf_iteration')
        # keep track of full exact exchange energy
        self.energy_hartree_fock_X = section.simpleValues.get('fhi_aims_energy_hartree_fock_X_scf_iteration')

    def onClose_fhi_aims_section_eigenvalues_list(self, backend, gIndex, section):
        """trigger called when fhi_aims_section_eigenvalues_list is closed"""
        # get cached values for occupation and eigenvalues
        occ = section.simpleValues.get('fhi_aims_eigenvalue_occupation')
        ev = section.simpleValues.get('fhi_aims_eigenvalue_eigenvalue')
        # check if values were found
        if not occ or not ev:
            raise Exception("Regular expression for eigenvalues were found, but no values could be extracted.")
        # check for consistent list length of obtained values
        if len(occ) != len(ev):
            raise Exception("Found %d eigenvalue occupations but %d eigenvalues." % (len(occ), len(ev)))
        if self.eigenvalues_occupation:
            if len(self.eigenvalues_occupation[-1]) != len(occ):
                raise Exception("Inconsistent number of eigenvalue occupations bewtween kpoints found in SCF cycle %d." % self.scfIterNr + 1)
        if self.eigenvalues_eigenvalues:
            if len(self.eigenvalues_eigenvalues[-1]) != len(ev):
                raise Exception("Inconsistent number of eigenvalues between kpoints found in SCF cycle %d." % self.scfIterNr + 1)
        # append values for current k-point
        self.eigenvalues_occupation.append(occ)
        self.eigenvalues_eigenvalues.append(ev)

    def onClose_fhi_aims_section_eigenvalues_spin(self, backend, gIndex, section):
        """trigger called when fhi_aims_section_eigenvalues_spin is closed"""
        # append values for current spin channel
        self.eigenvalues_occupation_spin_tmp.append(self.eigenvalues_occupation)
        self.eigenvalues_eigenvalues_spin_tmp.append(self.eigenvalues_eigenvalues)
        self.eigenvalues_occupation = []
        self.eigenvalues_eigenvalues = []
        # collect kpoints
        if self.periodicCalc:
            kpoints = []
            for i in ['1', '2', '3']:
                ki = section.simpleValues.get('fhi_aims_eigenvalue_kpoint' + i)
                if ki is not None:
                    kpoints.append(ki)
            if len(kpoints) == 3:
                if all(len(x) == len(kpoints[0]) for x in kpoints[1:-1]):
                    # transpose list
                    kpoints = map(lambda *x: list(x), *kpoints)
                    # append the result
                    self.eigenvalues_kpoints_spin_tmp.append(kpoints)
                else:
                    raise Exception("The number of kpoints of the eigenvalues is not consistent for all components:" + len(kpoints) * " %d" % tuple(map(len, kpoints)))
            else:
                raise Exception("Found kpoints for eigenvalues but only %d instead of 3 components." % len(kpoints))

    def onClose_fhi_aims_section_eigenvalues_group(self, backend, gIndex, section):
        """trigger called when fhi_aims_section_eigenvalues_group is closed"""
        # override values since we need only the latest eigenvalues which are then written in onClose_section_single_configuration_calculation
        self.eigenvalues_occupation_spin = self.eigenvalues_occupation_spin_tmp
        self.eigenvalues_eigenvalues_spin = self.eigenvalues_eigenvalues_spin_tmp
        self.eigenvalues_occupation_spin_tmp = []
        self.eigenvalues_eigenvalues_spin_tmp = []
        if self.periodicCalc:
            self.eigenvalues_kpoints_spin = self.eigenvalues_kpoints_spin_tmp
            self.eigenvalues_kpoints_spin_tmp = []

########################################
# submatcher for controlIn
controlInSubMatcher = SimpleMatcher(name = 'ControlIn',
              startReStr = r"\s*Parsing control\.in *(?:\.\.\.|\(first pass over file, find array dimensions only\)\.)",
              endReStr = r"\s*Completed first pass over input file control\.in \.",
              sections = ['fhi_aims_section_controlIn'],
              subMatchers = [
  SimpleMatcher(r"\s*The contents of control\.in will be repeated verbatim below"),
  SimpleMatcher(r"\s*unless switched off by setting 'verbatim_writeout \.false\.' \."),
  SimpleMatcher(r"\s*in the first line of control\.in \."),
  SimpleMatcher(r"\s*-{20}-*",
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
    # need to distinguish different cases
    SimpleMatcher(r"^\s*occupation_type\s+",
                  forwardMatch = True,
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)\s+(?P<fhi_aims_controlIn_occupation_order>[0-9]+)"),
      SimpleMatcher(r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)")
                              ]),
    SimpleMatcher(r"^\s*override_relativity\s+\.?(?P<fhi_aims_controlIn_override_relativity>[-_a-zA-Z]+)\.?", repeats = True),
    # need to distinguish different cases
    SimpleMatcher(r"^\s*charge\s+(?P<fhi_aims_controlIn_charge>[-+0-9.eEdD]+)", repeats = True),
    # only the first character is important for aims
    SimpleMatcher(r"^\s*hse_unit\s+(?P<fhi_aims_controlIn_hse_unit>[a-zA-Z])[-_a-zA-Z0-9]*", repeats = True),
    SimpleMatcher(r"^\s*relativistic\s+",
                  forwardMatch = True,
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*relativistic\s+(?P<fhi_aims_controlIn_relativistic_label>[-_a-zA-Z]+\s*[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_relativistic_threshold>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"^\s*relativistic\s+(?P<fhi_aims_controlIn_relativistic_label>[-_a-zA-Z]+)")
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
      SimpleMatcher(r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)\s+(?P<fhi_aims_controlIn_xc_omega>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)")
                  ]),
                ])
              ])
########################################
# subMatcher for geometry
# the verbatim writeout of the geometry.in is not considered
# using the geometry output of aims has the advantage that it has a clearer structure
geometrySubMatcher = SimpleMatcher(name = 'Geometry',
              startReStr = r"\s*Reading geometry description geometry\.in\.",
              sections = ['section_system_description'],
              subMatchers = [
  SimpleMatcher(r"\s*The structure contains\s*[0-9]+\s*atoms,\s*and a total of\s*[0-9.]\s*electrons\."),
  SimpleMatcher(r"\s*Input geometry:"),
  SimpleMatcher(r"\s*\|\s*No unit cell requested\."),
  SimpleMatcher(startReStr = r"\s*\|\s*Unit cell:",
                subMatchers = [
    SimpleMatcher(startReStr = r"\s*\|\s*(?P<fhi_aims_geometry_lattice_vector_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_lattice_vector_z__angstrom>[-+0-9.]+)", repeats = True)
                ]),
  SimpleMatcher(startReStr = r"\s*\|\s*Atomic structure:",
                subMatchers = [
    SimpleMatcher(r"\s*\|\s*Atom\s*x \[A\]\s*y \[A\]\s*z \[A\]"),
    SimpleMatcher(startReStr = r"\s*\|\s*[0-9]+:\s*Species\s*(?P<fhi_aims_geometry_atom_label>[a-zA-z]+)\s+(?P<fhi_aims_geometry_atom_position_x__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_y__angstrom>[-+0-9.]+)\s+(?P<fhi_aims_geometry_atom_position_z__angstrom>[-+0-9.]+)", repeats = True)
                ])
              ])
########################################
# submatcher for eigenvalue list
EigenvaluesListSubMatcher =  SimpleMatcher(name = 'EigenvaluesLists',
              startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
              sections = ['fhi_aims_section_eigenvalues_list'],
              subMatchers = [
  SimpleMatcher(startReStr = r"\s*[0-9]+\s+(?P<fhi_aims_eigenvalue_occupation>[0-9.eEdD]+)\s+[-+0-9.eEdD]+\s+(?P<fhi_aims_eigenvalue_eigenvalue__eV>[-+0-9.eEdD]+)", repeats = True)
              ])
########################################
# submatcher for eigenvalues
EigenvaluesGroupSubMatcher = SimpleMatcher(name = 'EigenvaluesGroup',
              startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
              sections = ['fhi_aims_section_eigenvalues_group'],
              subMatchers = [
  # spin-polarized
  SimpleMatcher(name = 'EigenvaluesSpin',
                startReStr = r"\s*Spin-(?:up|down) eigenvalues:",
                sections = ['fhi_aims_section_eigenvalues_spin'],
                repeats = True,
                subMatchers = [
    # Periodic
    SimpleMatcher(startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+(?P<fhi_aims_eigenvalue_kpoint1>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint2>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint3>[-+0-9.eEdD]+)\s+\(in units of recip\. lattice\)",
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
                    forwardMatch = True,
                    subMatchers = [
        EigenvaluesListSubMatcher
                    ])
                  ]),
    # Non-periodic
    SimpleMatcher(startReStr = r"\s*State\s+Occupation\s+Eigenvalue *\[Ha\]\s+Eigenvalue *\[eV\]",
                  forwardMatch = True,
                  subMatchers = [
      EigenvaluesListSubMatcher.copy() # need copy since SubMatcher already used above
                  ])
                ]),
  # non-spin-polarized, periodic
  SimpleMatcher(name = 'EigenvaluesNoSpinPeriodic',
                startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+.*\s+\(in units of recip\. lattice\)",
                sections = ['fhi_aims_section_eigenvalues_spin'],
                forwardMatch = True,
                subMatchers = [
    SimpleMatcher(startReStr = r"\s*K-point:\s*[0-9]+\s+at\s+(?P<fhi_aims_eigenvalue_kpoint1>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint2>[-+0-9.eEdD]+)\s+(?P<fhi_aims_eigenvalue_kpoint3>[-+0-9.eEdD]+)\s+\(in units of recip\. lattice\)",
                  repeats = True,
                  subMatchers = [
      SimpleMatcher(startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
                    forwardMatch = True,
                    subMatchers = [
        EigenvaluesListSubMatcher.copy() # need copy since SubMatcher already used above
                    ])
                  ]),
                ]),
  # non-spin-polarized, non-periodic
  SimpleMatcher(name = 'EigenvaluesNoSpinNonPeriodic',
                startReStr = r"\s*State\s+Occupation\s+Eigenvalue *\[Ha\]\s+Eigenvalue *\[eV\]",
                sections = ['fhi_aims_section_eigenvalues_spin'],
                forwardMatch = True,
                subMatchers = [
    EigenvaluesListSubMatcher.copy() # need copy since SubMatcher already used above
                ]),
              ])
########################################
# submatcher for total energy components during SCF interation
TotalEnergyScfSubMatcher = SimpleMatcher(name = 'TotalEnergyScf',
              startReStr = r"\s*Total energy components:",
              subMatchers = [
  SimpleMatcher(r"\s*\|\s*Sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_sum_eigenvalues_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*XC energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_scf_iteration__eV>[-+0-9.eEdD]+) eV"),
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
  SimpleMatcher(r"\s*\|\s*XC energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC_functional__eV>[-+0-9.eEdD]+) eV"),
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
                repeats = True,
                required = True,
                forwardMatch = True,
                sections   = ['section_run'],
                subMatchers = [
    SimpleMatcher(name = 'ProgramHeader',
                  startReStr = r"\s*Invoking FHI-aims \.\.\.",
                  subMatchers = [
      SimpleMatcher(r"\s*Version\s*(?P<program_version>[0-9a-zA-Z_.]+)"),
      SimpleMatcher(r"\s*Compiled on\s*(?P<fhi_aims_program_compilation_date>[0-9/]+)\s*at\s*(?P<fhi_aims_program_compilation_time>[0-9:]+)\s*on host\s*(?P<program_compilation_host>[-a-zA-Z0-9._]+)"),
      SimpleMatcher(r"\s*Date\s*:\s*(?P<fhi_aims_program_execution_date>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_program_execution_time>[-+0-9.eEdD]+)"),
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
                    ])
                  ]),
    # parse verbatim writeout of control.in
    controlInSubMatcher,
    # parse geometry
    geometrySubMatcher,
    # trigger the output of the metadata connected to section_method
    SimpleMatcher(name = 'ParallelTasksAssignement',
                  startReStr = r"\s*Preparing all fixed parts of the calculation\.",
                  sections = ["section_method"]),
    # this regex should detect the first start of a single configuration calculation and the subsequent starts of relaxation and MD steps with updated configurations
    SimpleMatcher(name = 'SingleConfigurationCalculationWithSystemDescription',
                  startReStr = r"\s*(?:Begin self-consistency loop: Initialization\.|Geometry optimization: Attempting to predict improved coordinates\.|Molecular dynamics: Attempting to update all nuclear coordinates\.)",
                  repeats = True,
                  forwardMatch = True,
                  subMatchers = [
      # parse updated geometry for relaxation
      SimpleMatcher(name = 'GeometryRelaxation',
                    startReStr = r"\s*Geometry optimization: Attempting to predict improved coordinates\.",
                    subMatchers = [
                    ]),
      # parse updated geometry for MD
      SimpleMatcher(name = 'GeometryRelaxation',
                    startReStr = r"\s*Molecular dynamics: Attempting to update all nuclear coordinates\.",
                    subMatchers = [
                    ]),
      # the actual section for a single configuration calculation starts here
      SimpleMatcher(name = 'SingleConfigurationCalculation',
                    startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                    forwardMatch = True,
                    sections = ['section_single_configuration_calculation'],
                    subMatchers = [
        SimpleMatcher(name = 'ScfInitialization',
                      startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                      sections = ['section_scf_iteration'],
                      subMatchers = [
          SimpleMatcher(r"\s*Date\s*:\s*(?P<fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)"),
          EigenvaluesGroupSubMatcher,
          TotalEnergyScfSubMatcher,
          SimpleMatcher(r"\s*Full exact exchange energy:\s*(?P<fhi_aims_energy_hartree_fock_X_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*End scf initialization - timings\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)")
                       ]),
        SimpleMatcher(name = 'ScfIteration',
                      startReStr = r"\s*Begin self-consistency iteration #\s*[0-9]+",
                      sections = ['section_scf_iteration'],
                      repeats = True,
                      subMatchers = [
          SimpleMatcher(r"\s*Date\s*:\s*(?P<fhi_aims_scf_date_start>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)"),
          SimpleMatcher(r"\s*Full exact exchange energy:\s*(?P<fhi_aims_energy_hartree_fock_X_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
          EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
          TotalEnergyScfSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
          SimpleMatcher(r"\s*Self-consistency convergence accuracy:"),
          SimpleMatcher(r"\s*\|\s*Change of charge(?:/spin)? density\s*:\s*[-+0-9.eEdD]+\s+[-+0-9.eEdD]*"),
          SimpleMatcher(r"\s*\|\s*Change of sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *eV"),
          SimpleMatcher(r"\s*\|\s*Change of total energy\s*:\s*(?P<energy_change_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
          SimpleMatcher(r"\s*\|\s*Change of forces\s*:\s*[-+0-9.eEdD]+ *eV/A"),
          # After convergence eigenvalues are printed in the end instead of usually in the beginning
          EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
          SimpleMatcher(r"\s*End self-consistency iteration #\s*[0-9]+\s*:\s*max\(cpu_time\)\s+wall_clock\(cpu1\)"),
          SimpleMatcher(r"\s*Self-consistency cycle converged\.")
                      ]),
        # possible scalar ZORA post-processing
        SimpleMatcher(name = 'ScaledZORAPostProcessing',
                      startReStr = r"\s*Post-processing: scaled ZORA corrections to eigenvalues and total energy.",
                      endReStr = r"\s*End evaluation of scaled ZORA corrections.",
                      subMatchers = [
          EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
          TotalEnergyZORASubMatcher
                      ]),
        SimpleMatcher(r"\s*Energy and forces in a compact form:"),
        SimpleMatcher(r"\s*\|\s*Total energy uncorrected\s*:\s*(?P<energy_total__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*\|\s*Total energy corrected\s*:\s*(?P<energy_total_T0__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*\|\s*Electronic free energy\s*:\s*(?P<energy_free__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*Start decomposition of the XC Energy"),
        SimpleMatcher(r"\s*Hartree-Fock part\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_hartree_fock_X_scaled__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*X Energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_X__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*C Energy GGA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_C__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*Total XC Energy\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*LDA X and C from self-consistent density"),
        SimpleMatcher(r"\s*X Energy LDA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<fhi_aims_energy_X_LDA__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*C Energy LDA\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<fhi_aims_energy_C_LDA__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*End decomposition of the XC Energy"),
                    ])
                  ])
                ])
              ])

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json
metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)
# dictionary for functions to call on close of a section metaname
onClose = {}
# parser info
parserInfo = {'name':'fhi-aims-parser', 'version': '1.0'}
# adjust caching of metadata
cachingLevelForMetaName = {
                           'fhi_aims_energy_hartree_fock_X_scf_iteration': CachingLevel.Cache,
                           'fhi_aims_section_eigenvalues_list': CachingLevel.Ignore,
                           'fhi_aims_section_eigenvalues_spin': CachingLevel.Ignore,
                           'fhi_aims_section_eigenvalues_group': CachingLevel.Ignore,
                           'fhi_aims_eigenvalue_occupation': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_eigenvalue': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_kpoint1': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_kpoint2': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_kpoint3': CachingLevel.Cache,
                          }

if __name__ == "__main__":
    # set all controlIn metadata to Cache to capture multiple occurrences of keywords
    # their last value is then written by the onClose routine in the FhiAimsParserContext
    # set all geometry metadata to Cache as all of them need post-processsing
    for name in metaInfoEnv.infoKinds:
        if name.startswith('fhi_aims_controlIn_') or name.startswith('fhi_aims_geometry_'):
            cachingLevelForMetaName[name] = CachingLevel.Cache
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo, superContext = FhiAimsParserContext(), cachingLevelForMetaName = cachingLevelForMetaName, onClose = onClose)

