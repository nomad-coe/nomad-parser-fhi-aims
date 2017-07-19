import sys, os, os.path
baseDir = os.path.dirname(os.path.abspath(__file__))
commonDir = os.path.normpath(os.path.join(baseDir, "../../../../../python-common/common/python"))
parserDir = os.path.normpath(os.path.join(baseDir, "../../parser-fhi-aims"))

if commonDir not in sys.path:
    sys.path.insert(1, commonDir)
if parserDir not in sys.path:
    sys.path.insert(1, parserDir)
