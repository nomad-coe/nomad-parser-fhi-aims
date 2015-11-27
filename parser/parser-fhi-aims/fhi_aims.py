#!/bin/env python
import setup_paths
from nomadcore.parser_backend import JsonParseEventsWriterBackend
from nomadcore.local_meta_info import loadJsonFile, InfoKindEl
from nomadcore.parse_streamed_dicts import ParseStreamedDicts
import os, os.path, sys
import FhiAimsParser

def parseFile(path, backend):
    backend.startedParsingSession(os.path.abspath(path),
                                  {"name":"fhi-aims-parser", "version": "1.0"})
    parser = FhiAimsParser.FhiAimsParser(backend, None, fileToParse)
    parser.parse()
    backend.finishedParsingSession(os.path.abspath(path),
                                   {"name":"fhi-aims-parser", "version": "1.0"})


metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../../nomad-meta-info/meta_info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
metaInfoEnv, warnings = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)

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
    if fileToParse:
        if writeComma:
            outF.write(", ")
        else:
            writeComma = True
        parseFile(fileToParse, backend)
    if stream:
        while True:
            if writeComma:
                outF.write(", ")
            else:
                writeComma = True
            toRead = dictReader.readNextDict()
            if toRead is None:
                break
            parseFile(toRead['mainFile'])
    outF.write("]\n")
