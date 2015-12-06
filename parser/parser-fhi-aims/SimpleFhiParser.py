import setup_paths
from nomadcore.simple_parser import SimpleMatcher, mainFunction
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
import os, sys, json

# description of the input

mainFileDescription = SimpleMatcher(name = "root",
              weak = True,
              startReStr = "",
              subMatchers = [
  SimpleMatcher(name = 'newRun',
                startReStr = r"\W*Invoking FHI-aims \.\.\.\W*",
                repeats  = True,
                required   = True,
                forwardMatch = True,
                sections   = ["section_run"],
                subMatchers = [
    SimpleMatcher(name = "ProgramHeader",
                  startReStr = "\W*Invoking FHI-aims \.\.\.\W*",
                  subMatchers=[
      SimpleMatcher(r" *Version +(?P<program_version>[0-9a-zA-Z_.]*)"),
      SimpleMatcher(r" *Compiled on +(?P<fhi_aims_program_compilation_date>[0-9/]+) at (?P<fhi_aims_program_compilation_time>[0-9:]+) *on host +(?P<program_compilation_host>[-a-zA-Z0-9._]+)"),
      SimpleMatcher(r" *Date *: *(?P<fhi_aims_program_execution_date>[-.0-9/]+) *, *Time *: *(?P<fhi_aims_program_execution_time>[-+0-9.EDed]+)"),
      SimpleMatcher(r" *Time zero on CPU 1 *: *(?P<time_run_cpu1_0>[-+0-9.EeDd]+) *s\."),
      SimpleMatcher(r" *Internal wall clock time zero *: *(?P<time_run_wall_0>[-+0-9.EeDd]+) *s\."),
      SimpleMatcher(name = "nParallelTasks",
                    startReStr = r" *Using *(?P<fhi_aims_number_of_tasks>[0-9]+) *parallel tasks\.",
                    sections = ["fhi_aims_section_parallel_tasks"],
                    subMatchers = [
        SimpleMatcher(name = "parallelTasksAssignement",
                      startReStr = r" *Task *(?P<fhi_aims_parallel_task_nr>[0-9]+) *on host *(?P<fhi_aims_parallel_task_host>[-a-zA-Z0-9._]+) *reporting\.",
                      sections = ["fhi_aims_section_parallel_task_assignement"])
                    ])
                  ])
                ])
              ])

# loading metadata from nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json

metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)

if __name__ == "__main__":
    mainFunction(mainFileDescription, metaInfoEnv)
