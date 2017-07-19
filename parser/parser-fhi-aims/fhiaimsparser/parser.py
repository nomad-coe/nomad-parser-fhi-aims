import os
import logging
from nomadcore.baseclasses import ParserInterface, MainHierarchicalParser
from fhiaimsparser.FhiAimsParser import FhiAimsParserContext, build_FhiAimsMainFileSimpleMatcher, get_cachingLevelForMetaName, getParserInfo
logger = logging.getLogger("nomad")


class FHIaimsParser(ParserInterface):
    """This class provides an object-oriented access to the parser.
    """
    def __init__(
            self, main_file, metainfo_to_keep=None, backend=None,
            default_units=None, metainfo_units=None, debug=False,
            log_level=logging.ERROR, store=True):
        super(FHIaimsParser, self).__init__(
            main_file, metainfo_to_keep, backend, default_units,
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
        self.main_parser = FHIaimsMainParser(
            self.parser_context.main_file,
            self.parser_context)

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


class FHIaimsMainParser(MainHierarchicalParser):
    """The main parser class that is called for all run types.
    """
    def __init__(self, filepath, parser_context):
        """
        """
        super(FHIaimsMainParser, self).__init__(filepath, parser_context)
        self.root_matcher = build_FhiAimsMainFileSimpleMatcher()
        self.caching_levels = get_cachingLevelForMetaName(
            parser_context.metainfo_env)
        self.super_context = FhiAimsParserContext()
