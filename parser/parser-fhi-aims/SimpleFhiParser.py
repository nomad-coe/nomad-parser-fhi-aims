import setup_paths
from nomadcore.simple_parser import SimpleMatcher, mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
import os, sys, json

# description of the input

controlInSubMatchers = []
geometryInSubMatchers = []

# Eigenvalues need to be done
# submatcher for a single SCF Interation
ScFIterationSubMatcher = SimpleMatcher(r"\s*Date *: *(?P<fhi_aims_scf_date_start>[-.0-9/]+) *, *Time *: *(?P<fhi_aims_scf_time_start>[-+0-9.eEdD]+)",
                subMatchers = [
    SimpleMatcher(name = 'Eigenvalues',
                  startReStr = r"\s*Writing Kohn-Sham eigenvalues\.",
                  sections = ['section_eigenvalues'],
                  subMatchers = [
      SimpleMatcher(r"\s*K-point:\s*(?P<eigenvalues_kpoints_number>[0-9]+) *at\s*.*\s*\(in units of recip\. lattice\)"),
      SimpleMatcher(r"\s*State\s*Occupation\s*Eigenvalue *\[Ha\]\s*Eigenvalue *\[eV\]"),
      #SimpleMatcher(r"\s*(?P<eigenvalues_eigenvalues_number>[0-9]+)", repeats = True),
      #SimpleMatcher(r"\s*(?P<eigenvalues_eigenvalues_number>[0-9]+)\s*(?P<eigenvalues_occupation>[0-9.eEdD]+)\s*(?P<eigenvalues_eigenvalues>[-+0-9.eEdD]+)\s*[-+0-9.eEdD]+", repeats = True),
                  ]),
    SimpleMatcher(name = 'TotalEnergyScf0',
                  startReStr = r"\s*Total energy components:",
                  subMatchers = [
      SimpleMatcher(r"\s*\|\s*Sum of eigenvalues\s*:\s*(?P<energy_sum_eigenvalues_sc_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*XC energy correction\s*:\s*(?P<energy_XC_sc_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ eV"),
      SimpleMatcher(r"\s*\|\s*XC potential correction\s*:\s*(?P<energy_XC_potential_sc_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*Free-atom electrostatic energy\s*:\s*(?P<fhi_aims_energy_electrostatic_free_atom_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*Hartree energy correction\s*:\s*(?P<energy_correction_hartree_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*Entropy correction\s*:\s*(?P<energy_correction_entropy_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*van der Waals energy corr\.\s*:\s*(?P<energy_van_der_Waals_value>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*-{20}-*", weak = True),
      SimpleMatcher(r"\s*\|\s*Total energy\s*:\s*(?P<energy_total_sc_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*Total energy, T -> 0\s*:\s*(?P<energy_total_T0_sc_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*Electronic free energy\s*:\s*(?P<energy_free_sc_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*Derived energy quantities:"),
      SimpleMatcher(r"\s*\|\s*Kinetic energy\s*:\s*(?P<electronic_kinetic_energy_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*Electrostatic energy\s*:\s*(?P<energy_electrostatic_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*Energy correction for multipole"),
      SimpleMatcher(r"\s*\|\s*error in Hartree potential\s*:\s*(?P<energy_hartree_error_scf>[-+0-9.eEdD]+) *Ha\s*[-+0-9.eEdD]+ *eV"),
      SimpleMatcher(r"\s*\|\s*Sum of eigenvalues per atom\s*:\s*(?P<energy_sum_eigenvalues_per_atom_scf>[-+0-9.eEdD]+) *eV"),
      SimpleMatcher(r"\s*\|\s*Total energy \(T->0\) per atom\s*:\s*(?P<energy_total_T0_sc_per_atom_scf>[-+0-9.eEdD]+) *eV"),
      SimpleMatcher(r"\s*\|\s*Electronic free energy per atom\s*:\s*(?P<energy_free_sc_per_atom_scf>[-+0-9.eEdD]+) *eV"),
                  ])
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
                  sections = ['section_single_point_evaluation'],
                  subMatchers = [
      SimpleMatcher(name = 'ScfInitialization',
                    startReStr = r"\s*Begin self-consistency loop: (?:I|Re-i)nitialization\.",
                    sections = ['section_scf_iteration'],
                    subMatchers = [ScFIterationSubMatcher]
                    ),
      SimpleMatcher(name = 'ScfIteration',
                    startReStr = r"\s*Begin self-consistency iteration #\s*(?P<scf_dft_niter>[0-9]+)",
                    sections = ['section_scf_iteration'],
                    repeats = True,
                    subMatchers = [ScFIterationSubMatcher.copy()]
                    )
                  ])
                ])
              ])

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json

metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv)
