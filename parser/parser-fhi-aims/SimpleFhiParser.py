import setup_paths
import numpy as np
from nomadcore.simple_parser import SimpleMatcher, mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.caching_backend import CachingLevel
import re, os, sys, json, logging

class FhiAimsParserContext(object):
    def __init__(self):
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1
        self.eigenvalues_occupation = []
        self.eigenvalues_eigenvalues = []

    def onClose_section_single_configuration_calculation(self, backend, gIndex, section):
        """trigger called when section_single_configuration_calculation is closed"""
        backend.addValue('scf_dft_number_of_iterations', self.scfIterNr)
        # start with -1 since zeroth iteration is the initialization
        self.scfIterNr = -1

    def onClose_section_scf_iteration(self, backend, gIndex, section):
        """trigger called when section_scf_iteration is closed"""
        self.scfIterNr += 1

    def onClose_fhi_aims_section_eigenvalues_list(self, backend, gIndex, section):
        """trigger called when _aims_section_eigenvalues_list is closed"""
        # get cached values for occupation and eigenvalues
        occ = section.simpleValues['fhi_aims_eigenvalue_occupation']
        ev = section.simpleValues['fhi_aims_eigenvalue_eigenvalue']
        # check for consistent list length of obtained values
        if len(occ) != len(ev):
            raise Exception("Found %d eigenvalue occupations but %d eigenvalues." % (len(occ), len(ev)))
        if self.eigenvalues_occupation:
            logging.getLogger("nomadcore.parsing").info("YYYY %s", self.eigenvalues_occupation)
            if len(self.eigenvalues_occupation[-1]) != len(occ):
                raise Exception("Inconsistent number of eigenvalue occupations found in SCF cycle %d." % self.scfIterNr + 1)
        if self.eigenvalues_eigenvalues:
            if len(self.eigenvalues_eigenvalues[-1]) != len(ev):
                raise Exception("Inconsistent number of eigenvalues found in SCF cycle %d." % self.scfIterNr + 1)
        # append values for current k-point
        self.eigenvalues_occupation.append(occ)
        self.eigenvalues_eigenvalues.append(ev)

    def onClose_section_eigenvalues(self, backend, gIndex, section):
        """trigger called when section_eigenvalues is closed"""
        backend.addArrayValues('eigenvalues_occupation', np.asarray(self.eigenvalues_occupation))
        backend.addArrayValues('eigenvalues_eigenvalues', np.asarray(self.eigenvalues_eigenvalues))
        self.eigenvalues_occupation = []
        self.eigenvalues_eigenvalues = []

# description of the input
controlInSubMatchers = []
geometryInSubMatchers = []

# TODO k-point extraction
# submatcher for eigenvalue list
EigenvaluesListSubMatcher =  SimpleMatcher(name = 'EigenValuesLists',
              startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
              sections = ['fhi_aims_section_eigenvalues_list'],
              subMatchers = [
  SimpleMatcher(startReStr = r"\s*[0-9]+\s*(?P<fhi_aims_eigenvalue_occupation>[0-9.eEdD]+)\s*(?P<fhi_aims_eigenvalue_eigenvalue__hartree>[-+0-9.eEdD]+)\s*[-+0-9.eEdD]+", repeats = True)
                  ])
# submatcher for eigenvalue group
EigenvaluesGroupSubMatcher = SimpleMatcher(name = 'EigenvaluesGroup',
              startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
              sections = ['section_eigenvalues_group'],
              forwardMatch = True,
              subMatchers = [
  SimpleMatcher(startReStr = r"\s*(?:Writing Kohn-Sham eigenvalues\.|Spin-down eigenvalues:)",
                repeats = True,
                subMatchers = [
    SimpleMatcher(r"\s*Spin-up eigenvalues:"),
    # for periodic calculations
    SimpleMatcher(name = 'EigenValuesKPoints',
                  sections = ['section_eigenvalues'],
                  startReStr = r"\s*K-point:\s*[0-9]+\s*at\s*.*\s*\(in units of recip\. lattice\)",
                  repeats = True,
                  subMatchers = [EigenvaluesListSubMatcher]
                  ),
    # for non-periodic calculations
    SimpleMatcher(name = 'EigenValuesKPoints',
                  sections = ['section_eigenvalues'],
                  startReStr = r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]",
                  forwardMatch = True,
                  subMatchers = [EigenvaluesListSubMatcher.copy()] # need copy since SubMatcher already used above
                  ),
                ])
              ])
# submatcher for total energy components
TotalEnergyScfSubMatcher = SimpleMatcher(name = 'TotalEnergyScf',
              startReStr = r"\s*Total energy components:",
              subMatchers = [
  SimpleMatcher(r"\s*\|\s*Sum of eigenvalues\s*:\s*(?P<energy_sum_eigenvalues_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*XC energy correction\s*:\s*(?P<energy_XC_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ eV"),
  SimpleMatcher(r"\s*\|\s*XC potential correction\s*:\s*(?P<energy_XC_potential_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Free-atom electrostatic energy\s*:\s*(?P<fhi_aims_energy_electrostatic_free_atom_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Hartree energy correction\s*:\s*(?P<energy_correction_hartree_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Entropy correction\s*:\s*(?P<energy_correction_entropy_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*van der Waals energy corr\.\s*:\s*(?P<energy_van_der_Waals_value__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*-{20}-*", weak = True),
  SimpleMatcher(r"\s*\|\s*Total energy\s*:\s*(?P<energy_total_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Total energy, T -> 0\s*:\s*(?P<energy_total_T0_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Electronic free energy\s*:\s*(?P<energy_free_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*Derived energy quantities:"),
  SimpleMatcher(r"\s*\|\s*Kinetic energy\s*:\s*(?P<electronic_kinetic_energy_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Electrostatic energy\s*:\s*(?P<energy_electrostatic_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Energy correction for multipole"),
  SimpleMatcher(r"\s*\|\s*error in Hartree potential\s*:\s*(?P<energy_hartree_error_scf_iteration__hartree>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
  SimpleMatcher(r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<energy_sum_eigenvalues_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<energy_total_T0_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
  SimpleMatcher(r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<energy_free_per_atom_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
              ])

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
      SimpleMatcher(r" *Version *(?P<program_version>[0-9a-zA-Z_.]*)"),
      SimpleMatcher(r" *Compiled on *(?P<fhi_aims_program_compilation_date>[0-9/]+) at (?P<fhi_aims_program_compilation_time>[0-9:]+) *on host *(?P<program_compilation_host>[-a-zA-Z0-9._]+)"),
      SimpleMatcher(r"\s*Date\s*:\s*(?P<fhi_aims_program_execution_date>[-.0-9/]+)\s*,\s*Time\s*:\s*(?P<fhi_aims_program_execution_time>[-+0-9.eEdD]+)"),
      SimpleMatcher(r"\s*Time zero on CPU 1\s*:\s*(?P<time_run_cpu1_start>[-+0-9.eEdD]+) *s?\."),
      SimpleMatcher(r"\s*Internal wall clock time zero\s*:\s*(?P<time_run_wall_start>[-+0-9.eEdD]+) *s\."),
      SimpleMatcher(name = "nParallelTasks",
                    startReStr = r"\s*Using\s*(?P<fhi_aims_number_of_tasks>[0-9]+)\s*parallel tasks\.",
                    sections = ["fhi_aims_section_parallel_tasks"],
                    subMatchers = [
        SimpleMatcher(name = 'parallelTasksAssignement',
                      startReStr = r"\s*Task\s*(?P<fhi_aims_parallel_task_nr>[0-9]+)\s*on host\s*(?P<fhi_aims_parallel_task_host>[-a-zA-Z0-9._]+)\s*reporting\.",
                      sections = ["fhi_aims_section_parallel_task_assignement"])
                    ])
                  ]),
    SimpleMatcher(name = 'controlInParser',
                  startReStr = r"\s*Parsing control\.in *(?:\.\.\.|\(first pass over file, find array dimensions only\)\.)",
                  subMatchers = [
      SimpleMatcher(r"\s*The contents of control\.in will be repeated verbatim below"),
      SimpleMatcher(r"\s*unless switched off by setting 'verbatim_writeout \.false\.' \."),
      SimpleMatcher(r"\s*in the first line of control\.in \."),
      SimpleMatcher(r"\s*-{10}-*", endReStr = r" *-{10}-*",
                  subMatchers = controlInSubMatchers)
                  ]),
    #controlInOutputs,
    SimpleMatcher(name = 'GeometryInParser',
                  startReStr = r"\s*Parsing geometry\.in *(?:\.\.\.|\(first pass over file, find array dimensions only\)\.)",
                  subMatchers = [
      SimpleMatcher(r"\s*The contents of geometry\.in will be repeated verbatim below"),
      SimpleMatcher(r"\s*unless switched off by setting 'verbatim_writeout \.false\.' \."),
      SimpleMatcher(r"\s*in the first line of geometry\.in \."),
      SimpleMatcher(r" *-{10}-*", endReStr = r" *-{10}-*",
                  subMatchers = geometryInSubMatchers)
                  ]),
    # geometryInOutputs,
    SimpleMatcher(name = 'SinglePointEvaluation',
                  startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                  repeats = True,
                  forwardMatch = True,
                  sections = ['section_single_configuration_calculation'],
                  subMatchers = [
      SimpleMatcher(name = 'ScfInitialization',
                    startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                    sections = ['section_scf_iteration'],
                    subMatchers = [
        SimpleMatcher(r"\s*Date *: *(?P<fhi_aims_scf_date_start>[-.0-9/]+) *, *Time *: *(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)"),
        EigenvaluesGroupSubMatcher,
        TotalEnergyScfSubMatcher
                     ]),
      SimpleMatcher(name = 'ScfIteration',
                    startReStr = r"\s*Begin self-consistency iteration #\s*[0-9]+",
                    sections = ['section_scf_iteration'],
                    repeats = True,
                    subMatchers = [
        SimpleMatcher(r"\s*Date *: *(?P<fhi_aims_scf_date_start>[-.0-9/]+) *, *Time *: *(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)"),
        EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
        TotalEnergyScfSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
        SimpleMatcher(r"\s*Self-consistency convergence accuracy:"),
        SimpleMatcher(r"\s*\|\s*Change of charge(?:/spin)? density\s*:\s*[-+0-9.eEdD]+\s*[-+0-9.eEdD]*"),
        SimpleMatcher(r"\s*\|\s*Change of sum of eigenvalues\s*:\s*[-+0-9.eEdD]+ *eV"),
        SimpleMatcher(r"\s*\|\s*Change of total energy\s*:\s*(?P<energy_change_scf_iteration__eV>[-+0-9.eEdD]+) *eV"),
        SimpleMatcher(r"\s*\|\s*Change of forces\s*:\s*[-+0-9.eEdD]+ *eV/A"),
        # After convergence eigenvalues are printed in the end instead of usually in the beginning
        EigenvaluesGroupSubMatcher.copy(), # need copy since SubMatcher already used for ScfInitialization
        SimpleMatcher(r"\s*End self-consistency iteration #\s*[0-9]+\s*:\s*max\(cpu_time\)\s*wall_clock\(cpu1\)"),
        SimpleMatcher(r"\s*Self-consistency cycle converged\.")
                    ]),
      SimpleMatcher(r"\s*Energy and forces in a compact form:"),
      SimpleMatcher(r"\s*\|\s*Total energy uncorrected\s*:\s*(?P<energy_total_scf_converged>[-+0-9.eEdD]+) *eV"),
      SimpleMatcher(r"\s*\|\s*Total energy corrected\s*:\s*(?P<energy_total_T0_scf_converged>[-+0-9.eEdD]+) *eV"),
      SimpleMatcher(r"\s*\|\s*Electronic free energy\s*:\s*(?P<energy_free_scf_converged>[-+0-9.eEdD]+) *eV")
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
cachingLevelForMetaName = {'fhi_aims_section_eigenvalues_list': CachingLevel.Ignore,
                           'fhi_aims_eigenvalue_occupation': CachingLevel.Cache,
                           'fhi_aims_eigenvalue_eigenvalue': CachingLevel.Cache,
                           'eigenvalues_occupation': CachingLevel.Forward,
                           'eigenvalues_eigenvalues': CachingLevel.Forward,
                           'scf_dft_number_of_iterations': CachingLevel.Forward}

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv, parserInfo, superContext = FhiAimsParserContext(), cachingLevelForMetaName = cachingLevelForMetaName, onClose = onClose)
