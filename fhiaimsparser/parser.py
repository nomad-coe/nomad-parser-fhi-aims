# Copyright 2015-2018 Franz Knuth, Fawzi Mohamed, Wael Chibani, Ankit Kariryaa, Lauri Himanen, Danio Brambila
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import logging
import sys
from nomadcore.baseclasses import ParserInterface, MainHierarchicalParser

from fhiaimsparser.FhiAimsParser import FhiAimsParserContext, build_FhiAimsMainFileSimpleMatcher, get_cachingLevelForMetaName, getParserInfo


class FHIaimsParser(ParserInterface):
    """This class provides an object-oriented access to the parser.
    """
    def __init__(
            self, metainfo_to_keep=None, backend=None,
            default_units=None, metainfo_units=None, debug=False,
            log_level=logging.ERROR, store=True):
        super(FHIaimsParser, self).__init__(
            metainfo_to_keep, backend, default_units,
            metainfo_units, debug, log_level, store)

    def setup_version(self):
        """Setups the version by possbily investigating the output file and the
        version specified in it.
        """
        # Setup the root folder to the fileservice that is used to access files
        dirpath, filename = os.path.split(self.parser_context.main_file)
        dirpath = os.path.abspath(dirpath)
        self.parser_context.file_service.setup_root_folder(dirpath)
        self.parser_context.file_service.set_file_id(filename, "output")

        # Setup the correct main parser possibly based on the version
        self.main_parser = FHIaimsMainParser(self.parser_context)

    @staticmethod
    def get_mainfile_regex():
        regex_str = (
            "\s*Invoking FHI-aims \.\.\.\n"
            "\s*Version "
        )
        return regex_str

    def get_metainfo_filename(self):
        return "fhi_aims.nomadmetainfo.json"

    def get_parser_info(self):
        return getParserInfo()


fhiAimsMainFileSimpleMatcher = build_FhiAimsMainFileSimpleMatcher()


class FHIaimsMainParser(MainHierarchicalParser):
    """The main parser class that is called for all run types.
    """
    def __init__(self, parser_context):
        """
        """
        super(FHIaimsMainParser, self).__init__(parser_context)
        self.root_matcher = fhiAimsMainFileSimpleMatcher
        self.caching_levels = get_cachingLevelForMetaName(
            parser_context.metainfo_env)
        self.super_context = FhiAimsParserContext()


def main():
    parser = FHIaimsParser()
    results = parser.parse(sys.argv[1])
    return results


if __name__ == "__main__":
    main()
