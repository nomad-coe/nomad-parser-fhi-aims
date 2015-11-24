#!/bin/env python
import setup_paths
import nomadcore.parser_backend
from nomadcore.local_meta_info import loadJsonFile
import os, os.path

metaInfoPath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../../../nomad-meta-info/nomad_meta_info/fhi_aims.nomadmetainfo.json"))
metaInfoEnv = loadJsonFile(filePath = metaInfoPath, dependencyLoader = None, extraArgsHandling = InfoKindEl.ADD_EXTRA_ARGS, uri = None)

if __name__ == "__main__":
    usage = """{exeName} [--meta-info] [--help] [--specialize] [--stream] [path/toFile]

    --meta-info outputs the meta info supported by this parser
    --help prints this message
    --specialize expects inclusion and exclusion of meta info to parse via a json disctionary on stdin
    --stream expects the files to parse via dictionary on stdin

    If a path to a file is given this is parsed""".format(exeName = os.path.basename(sys.argv.get(0,"fhi_aims.py")))
    outF = sys.stdout
    metaInfo = False
    specialize = False
    stream = False
    fileToParse = None
    ii = 1
    while ii < sys.argv:
        arg = sys.argv[ii]
        if arg == "--meta-info":
            metaInfo = True
        elif arg == "--help":
            print usage
            sys.exit(0)
        elif arg == "--specialize":
            specialize = True
        elif arg == "--stream":
            stream = True
        elif not fileToParse:
            fileToParse
    if metaInfo:
        metaInfoEnv.embedDeps()
        metaInfoEnv.serialize(self, outF.write, subGids = True, selfGid = True):
        outF.flush()
    dictReader = DictReader(sys.stdin)
    toOuput = list(metaInfoEnv.infoKinds.keys())
    if specialize:
        specializationInfo = dictReader.readNextDict()
        if specializationInfo is None or specializationInfo.get("type", "") != "nomad_parser_specialization_1_0":
            raise Exception("expected a nomad_parser_specialization_1_0 as first dictionary, got " + json.dumps(specializationInfo))
        specializationInfo
