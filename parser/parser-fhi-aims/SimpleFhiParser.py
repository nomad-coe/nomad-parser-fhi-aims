import setup_paths
from nomadcore.simple_parser import SimpleParserBuilder, SimpleParser, SimpleMatcher, PushbackLineFile
from nomadcore.parser_backend import JsonParseEventsWriterBackend
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.parse_streamed_dicts import ParseStreamedDicts
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

def parseFile(parserBuilder, uri, path, backend):
    with open(path, "r") as fIn:
        backend.startedParsingSession(uri,
                                      {"name":"fhi-aims-parser", "version": "1.0"})
        parser = parserBuilder.buildParser(PushbackLineFile(fIn), backend)
        parser.parse()
        backend.finishedParsingSession(uri,
                                       {"name":"fhi-aims-parser", "version": "1.0"})

if __name__ == "__main__":
    usage = """{exeName} [--meta-info] [--help] [--specialize] [--stream] [--uri uri] [path/toFile]

    --meta-info outputs the meta info supported by this parser
    --help prints this message
    --specialize expects inclusion and exclusion of meta info to parse via a json disctionary on stdin
    --stream expects the files to parse via dictionary on stdin

    If a path to a file is given this is parsed
""".format(exeName = os.path.basename(sys.argv[0] if len(sys.argv) > 0 else "fhi_aims.py"))
    outF = sys.stdout
    metaInfo = False
    specialize = False
    stream = False
    fileToParse = None
    uri = None
    ii = 1
    while ii < len(sys.argv):
        arg = sys.argv[ii]
        ii += 1
        if arg == "--meta-info":
            metaInfo = True
        elif arg == "--help":
            sys.stderr.write(usage)
            sys.exit(0)
        elif arg == "--specialize":
            specialize = True
        elif arg == "--stream":
            stream = True
        elif arg == "--uri":
            if ii >= len(sys.argv):
                raise Exception("missing uri after --uri")
            uri = arg
            ii += 1
        elif not fileToParse:
            fileToParse = arg
        else:
            sys.stderr.write("unexpected argument " + arg + "\n")
            sys.stderr.write(usage)
            sys.exit(1)
    if uri is None and fileToParse:
        uri = "file://" + fileToParse
    outF.write("[")
    writeComma = False
    if metaInfo:
        if writeComma:
            outF.write(", ")
        else:
            writeComma = True
        metaInfoEnv.embedDeps()
        metaInfoEnv.serialize(outF.write, subGids = True, selfGid = True)
        outF.flush()
    dictReader = ParseStreamedDicts(sys.stdin)
    toOuput = list(metaInfoEnv.infoKinds.keys())
    backend = JsonParseEventsWriterBackend(metaInfoEnv, sys.stdout)
    if specialize:
        specializationInfo = dictReader.readNextDict()
        if specializationInfo is None or specializationInfo.get("type", "") != "nomad_parser_specialization_1_0":
            raise Exception("expected a nomad_parser_specialization_1_0 as first dictionary, got " + json.dumps(specializationInfo))
        specializationInfo

    # initialize the parser builder
    parserBuilder = SimpleParserBuilder(mainFileDescription, metaInfoEnv)
    #sys.stderr.write("matchers:\n  ")
    #parserBuilder.writeMatchers(sys.stderr, 2)
    #sys.stderr.write("\n")
    parserBuilder.compile()
    #sys.stderr.write("compiledMatchers:\n  ")
    #parserBuilder.writeCompiledMatchers(sys.stderr, 2)
    #sys.stderr.write("\n\n")

    if fileToParse:
        if writeComma:
            outF.write(", ")
        else:
            writeComma = True
        parseFile(parserBuilder, uri, fileToParse, backend)
    if stream:
        while True:
            if writeComma:
                outF.write(", ")
            else:
                writeComma = True
            toRead = dictReader.readNextDict()
            if toRead is None:
                break
            parseFile(parseFile, toRead['mainFileUri'], toRead['mainFile'])
    outF.write("]\n")
