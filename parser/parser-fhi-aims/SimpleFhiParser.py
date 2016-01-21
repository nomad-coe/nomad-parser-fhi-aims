import setup_paths
import numpy as np
from nomadcore.simple_parser import SimpleMatcher, mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.caching_backend import CachingLevel
import re, os, sys, json, logging

class FhiAimsParserContext(object):
    def __init__(self):
        self.scalarZORA = False
        self.energy_hartree_fock_X = None
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        self.eigenvalues_occupation = []
        self.eigenvalues_eigenvalues = []
        self.eigenvalues_occupation_spin = []
        self.eigenvalues_occupation_spin_tmp = []
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


    def onClose_fhi_aims_section_controlIn(self, backend, gIndex, section):
        """trigger called when section fhi_aims_section_controlIn is closed"""
        # check for scalar ZORA setting and convert to one common name
        rel = section.simpleValues.get('fhi_aims_controlIn_relativistic_zora')
        if rel:
            # we get back a list of values and aims uses the last occurence of a keyword
            rel = rel[-1]
            if re.match(r"\s*zora\s+scalar", rel, re.IGNORECASE):
                self.scalarZORA = True
                backend.addValue('fhi_aims_controlIn_relativistic', 'zora scalar')
        # write k_grid if it was found
        k1 = section.simpleValues.get('fhi_aims_controlIn_k1')
        k2 = section.simpleValues.get('fhi_aims_controlIn_k2')
        k3 = section.simpleValues.get('fhi_aims_controlIn_k3')
        if all(x is not None for x in [k1, k2, k3]):
            # we get back a list of values and aims uses the last occurence of a keyword
            backend.addArrayValues('fhi_aims_controlIn_k_grid', np.asarray([k1[-1],k2[-1],k3[-1]]))

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
            if self.eigenvalues_kpoints_spin:
                kpt = np.asarray(self.eigenvalues_kpoints_spin)
            else:
                kpt = None
            # in onClose_fhi_aims_section_eigenvalues_list it was already checked 
            # that the lowest lying lists in occupation and eigenvalues have the same length
            # therefore, eigenvalues_occupation_spin and eigenvalues_eigenvalues_spin should be of the same form
            if occ.shape != ev.shape:
                raise Exception("Array of collected eingenvalue occupations and eigenvalues do not have same shape, %s vs. %s" % (occ.shape, ev.shape))
            # check if the first two dimensions (number of spin chhannels and number of kpoints) are equal
            if kpt is not None:
                if occ.shape[0:2] != kpt.shape[0:2]:
                    raise Exception("Array of collected eingenvalue occupations and kpoints do not have same shape for the first two dimensions, %s vs. %s" % (occ.shape[0:2], kpt.shape[0:2]))
            for i in range (occ.shape[0]):
                gIndex = backend.openSection('section_eigenvalues')
                backend.addArrayValues('eigenvalues_occupation', occ[i])
                backend.addArrayValues('eigenvalues_eigenvalues', ev[i])
                self.eigenvalues_occupation_spin = []
                self.eigenvalues_eigenvalues_spin = []
                if kpt is not None:
                    backend.addArrayValues('eigenvalues_kpoints', kpt[i])
                    self.eigenvalues_kpoints = []
                backend.closeSection('section_eigenvalues', gIndex)
            backend.closeSection('section_eigenvalues_group', gIndexGroup)
        # print converged energy values
        # with scalar ZORA, the correctly scaled energy values are given in a separate post-processing step, which are read there
        # therefore, we do print the energy values for scalar ZORA
        if not self.scalarZORA:
            for k,v in self.totalEnergyList.items():
                #if v is not None:
                backend.addValue(k, v)
        if self.energy_hartree_fock_X is not None:
            backend.addValue('energy_hartree_fock_X', self.energy_hartree_fock_X)

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
            raise Exception("Regular expression for eigenvalues were found, but now values could be extracted.")
        # check for consistent list length of obtained values
        if len(occ) != len(ev):
            raise Exception("Found %d eigenvalue occupations but %d eigenvalues." % (len(occ), len(ev)))
        if self.eigenvalues_occupation:
            if len(self.eigenvalues_occupation[-1]) != len(occ):
                raise Exception("Inconsistent number of eigenvalue occupations found in SCF cycle %d." % self.scfIterNr + 1)
        if self.eigenvalues_eigenvalues:
            if len(self.eigenvalues_eigenvalues[-1]) != len(ev):
                raise Exception("Inconsistent number of eigenvalues found in SCF cycle %d." % self.scfIterNr + 1)
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
        k1 = section.simpleValues.get('fhi_aims_eigenvalue_kpoint1')
        k2 = section.simpleValues.get('fhi_aims_eigenvalue_kpoint2')
        k3 = section.simpleValues.get('fhi_aims_eigenvalue_kpoint3')
        tmp_list = []
        if all(x is not None for x in [k1, k2, k3]):
            if len(k1) == len(k2) and len(k2) == len(k3):
                # first create a list wich contains all kpoints as lists of length 3
                for i in range (len(k1)):
                    tmp_list.append([k1[i], k2[i], k3[i]])
                # now append this list
                self.eigenvalues_kpoints_spin_tmp.append(tmp_list)
            else:
                raise Exception("Inconsistent number of kpoint components found for eingevalue list: #k1 = %d, #k2 = %d, #k3 = %d" % (len(k1), len(k2), len(k3)))

    def onClose_fhi_aims_section_eigenvalues_group(self, backend, gIndex, section):
        """trigger called when fhi_aims_section_eigenvalues_group is closed"""
        # override values since we need only the latest eigenvalues which are then printed in onClose_section_single_configuration_calculation
        self.eigenvalues_occupation_spin = self.eigenvalues_occupation_spin_tmp
        self.eigenvalues_eigenvalues_spin = self.eigenvalues_eigenvalues_spin_tmp
        self.eigenvalues_kpoints_spin = self.eigenvalues_kpoints_spin_tmp
        self.eigenvalues_occupation_spin_tmp = []
        self.eigenvalues_eigenvalues_spin_tmp = []
        self.eigenvalues_kpoints_spin_tmp = []

geometryInSubMatchers = []

########################################
# submatcher for controlIn
controlInSubMatcher = SimpleMatcher(name = 'controlIn',
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
    # now follows the list to match the keywords from the control.in
    # explicitly add ^ to ensure that the keyword is not within a comment
    # the search is done unordered since the keywords do not appear in a specific order
    # list the matchers in alphabetical order according to keyword name
    # need to distinguish different cases
    SimpleMatcher(r"^\s*occupation_type\s+",
                  forwardMatch = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)\s+(?P<fhi_aims_controlIn_occupation_order>[0-9]+)"),
      SimpleMatcher(r"^\s*occupation_type\s+(?P<fhi_aims_controlIn_occupation_type>[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_occupation_width>[-+0-9.eEdD]+)")
                              ]),
    SimpleMatcher(r"^\s*override_relativity\s+\.?(?P<fhi_aims_controlIn_override_relativity>[-_a-zA-Z]+)\.?"),
    # need to distinguish different cases
    SimpleMatcher(r"^\s*charge\s+(?P<fhi_aims_controlIn_charge>[-+0-9.eEdD]+)"),
    # only the first character is important for aims
    SimpleMatcher(r"^\s*hse_unit\s+(?P<fhi_aims_controlIn_hse_unit>[a-zA-Z])[-_a-zA-Z0-9]*"),
    SimpleMatcher(r"^\s*relativistic\s+",
                  forwardMatch = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*relativistic\s+(?P<fhi_aims_controlIn_relativistic_zora>[-_a-zA-Z]+\s*[-_a-zA-Z]+)\s+(?P<fhi_aims_controlIn_relativistic_threshold>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"^\s*relativistic\s+(?P<fhi_aims_controlIn_relativistic>[-_a-zA-Z]+)")
                              ]),
    SimpleMatcher(r"^\s*sc_accuracy_rho\s+(?P<fhi_aims_controlIn_sc_accuracy_rho>[-+0-9.eEdD]+)"),
    SimpleMatcher(r"^\s*sc_accuracy_eev\s+(?P<fhi_aims_controlIn_sc_accuracy_eev>[-+0-9.eEdD]+)"),
    SimpleMatcher(r"^\s*sc_accuracy_etot\s+(?P<fhi_aims_controlIn_sc_accuracy_etot>[-+0-9.eEdD]+)"),
    SimpleMatcher(r"^\s*sc_accuracy_forces\s+(?P<fhi_aims_controlIn_sc_accuracy_forces>[-+0-9.eEdD]+)"),
    SimpleMatcher(r"^\s*sc_accuracy_stress\s+(?P<fhi_aims_controlIn_sc_accuracy_stress>[-+0-9.eEdD]+)"),
    SimpleMatcher(r"^\s*sc_iter_limit\s+(?P<fhi_aims_controlIn_sc_iter_limit>[0-9]+)"),
    SimpleMatcher(r"^\s*k_grid\s+(?P<fhi_aims_controlIn_k1>[0-9]+)\s+(?P<fhi_aims_controlIn_k2>[0-9]+)\s+(?P<fhi_aims_controlIn_k3>[0-9]+)"),
    SimpleMatcher(r"^\s*spin\s+(?P<fhi_aims_controlIn_spin>[-_a-zA-Z]+)"),
    # need to distinguish two cases: just the name of the xc functional or name plus number (e.g. for HSE functional)
    SimpleMatcher(r"^\s*xc\s+",
                  forwardMatch = True,
                  subMatchers = [
      SimpleMatcher(r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)\s+(?P<fhi_aims_controlIn_xc_omega>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"^\s*xc\s+(?P<fhi_aims_controlIn_xc>[-_a-zA-Z0-9]+)")
                  ]),
                ])
              ])
########################################
# subMatcher for geometryIn
# the verbatim writeout of the geometry.in is not considered
# using the geometry output of aims has the advantage that it has a clearer structure
geometryInSubMatcher = SimpleMatcher(name = 'geometryIn',
              startReStr = r"\s*Reading geometry description geometry\.in\.",
              endReStr = r"\s*Preparing all fixed parts of the calculation\.",
              sections = ['fhi_aims_section_geometryIn'],
              subMatchers = [
  SimpleMatcher(r"\s*The structure contains\s*[0-9]+\s*atoms,\s*and a total of\s*[0-9.]\s*electrons\."),
  SimpleMatcher(r"\s*Input geometry:"),
  SimpleMatcher(startReStr = r"\s*\|\sUnit cell:",
                subMatchers = [
                ]),
  SimpleMatcher(startReStr = r"\s*\|\sAtomic structure:",
                subMatchers = [
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
              #forwardMatch = True,
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
  SimpleMatcher(r"\s*\|\s*XC energy correction\s*:\s*[-+0-9.eEdD]+ *Ha\s+(?P<energy_XC__eV>[-+0-9.eEdD]+) eV"),
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
mainFileDescription = SimpleMatcher(name = 'root',
              weak = True,
              startReStr = "",
              subMatchers = [
  SimpleMatcher(name = 'newRun',
                startReStr = r"\s*Invoking FHI-aims \.\.\.\s*",
                repeats = True,
                required = True,
                forwardMatch = True,
                sections   = ['section_run'],
                subMatchers = [
    SimpleMatcher(name = 'ProgramHeader',
                  startReStr = r"\s*Invoking FHI-aims \.\.\.\s*",
                  subMatchers = [
      SimpleMatcher(r"\s*Version\s*(?P<program_version>[0-9a-zA-Z_.]*)"),
      SimpleMatcher(r"\s*Compiled on\s*(?P<fhi_aims_program_compilation_date>[0-9/]+)\s*at\s*(?P<fhi_aims_program_compilation_time>[0-9:]+)\s*on host\s*(?P<program_compilation_host>[-a-zA-Z0-9._]+)"),
      SimpleMatcher(r"\s*Date\s*:\s*(?P<fhi_aims_program_execution_date>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_program_execution_time>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"\s*Time zero on CPU 1\s*:\s*(?P<time_run_cpu1_start>[-+0-9.eEdD]+) *s?\."),
      SimpleMatcher(r"\s*Internal wall clock time zero\s*:\s*(?P<time_run_wall_start>[-+0-9.eEdD]+) *s\."),
      SimpleMatcher(name = "nParallelTasks",
                    startReStr = r"\s*Using\s*(?P<fhi_aims_number_of_tasks>[0-9]+)\s*parallel tasks\.",
                    sections = ["fhi_aims_section_parallel_tasks"],
                    subMatchers = [
        SimpleMatcher(name = 'parallelTasksAssignement',
                      startReStr = r"\s*Task\s*(?P<fhi_aims_parallel_task_nr>[0-9]+)\s*on host\s+(?P<fhi_aims_parallel_task_host>[-a-zA-Z0-9._]+)\s+reporting\.",
                      repeats = True,
                      sections = ["fhi_aims_section_parallel_task_assignement"])
                    ])
                  ]),
    # parse verbatim writeout of control.in
    controlInSubMatcher,
    # parse geometry
    geometryInSubMatcher,
    # this regex should detect the first start of a single configuration calculation and the subsequent starts of relaxation and MD steps with updated configurations
    SimpleMatcher(name = 'SingleConfigurationCalculation',
                  startReStr = r"\s*(?:Begin self-consistency loop: Initialization\.|Geometry optimization: Attempting to predict improved coordinates\.|Molecular dynamics: Attempting to update all nuclear coordinates\.)",
                  repeats = True,
                  forwardMatch = True,
                  sections = ['section_single_configuration_calculation'],
                  subMatchers = [
      # need to eat start line for new relaxation and MD step due to forwardMatch
      # the first expression is taken care of below
      SimpleMatcher(r"\s*(?:Geometry optimization: Attempting to predict improved coordinates\.|Molecular dynamics: Attempting to update all nuclear coordinates\.)"),
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

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json
metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)
# dictionary for functions to call on close of a section metaname
onClose = {}
# parser info
parserInfo = {'name':'fhi-aims-parser', 'version': '1.0'}
# adjust caching of metadata
cachingLevelForMetaName = {
                           'fhi_aims_controlIn_relativistic': CachingLevel.Forward,
                           'fhi_aims_controlIn_k_grid': CachingLevel.Forward,
                           'fhi_aims_controlIn_k1': CachingLevel.Cache,
                           'fhi_aims_controlIn_k2': CachingLevel.Cache,
                           'fhi_aims_controlIn_k3': CachingLevel.Cache,
                           'fhi_aims_controlIn_relativistic_zora': CachingLevel.Cache,
                           'fhi_aims_energy_hartree_fock_X_scf_iteration': CachingLevel.Cache,
                           'fhi_aims_section_eigenvalues_list': CachingLevel.Ignore,
                           'fhi_aims_section_eigenvalues_spin': CachingLevel.Ignore,
                           'fhi_aims_section_eigenvalues_group': CachingLevel.Ignore,
                           'fhi_aims_eigenvalue_occupation': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_eigenvalue': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_kpoint1': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_kpoint2': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_kpoint3': CachingLevel.Cache,
                           'scf_dft_number_of_iterations': CachingLevel.Forward,
                           'energy_sum_eigenvalues': CachingLevel.Forward,
                           'energy_XC': CachingLevel.Forward,
                           'energy_XC_potential': CachingLevel.Forward,
                           'energy_correction_hartree': CachingLevel.Forward,
                           'energy_correction_entropy': CachingLevel.Forward,
                           'energy_van_der_Waals_value': CachingLevel.Forward,
                           'electronic_kinetic_energy': CachingLevel.Forward,
                           'energy_electrostatic': CachingLevel.Forward,
                           'energy_hartree_error': CachingLevel.Forward,
                           'energy_sum_eigenvalues_per_atom': CachingLevel.Forward,
                           'energy_total_T0_per_atom': CachingLevel.Forward,
                           'energy_free_per_atom': CachingLevel.Forward,
                           'energy_hartree_fock_X': CachingLevel.Forward,
                          }

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo, superContext = FhiAimsParserContext(), cachingLevelForMetaName = cachingLevelForMetaName, onClose = onClose)
