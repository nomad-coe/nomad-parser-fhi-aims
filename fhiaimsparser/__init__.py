from .metainfo import m_env

from nomad.parsing.parser import FairdiParser
from fhiaimsparser.fhiaims_parser import FHIAimsParserInterface


class FHIAimsParser(FairdiParser):
    def __init__(self):
        super().__init__(
            name='parsers/fhi-aims', code_name='FHI-aims',
            code_homepage='https://aimsclub.fhi-berlin.mpg.de/',
            mainfile_contents_re=(
                r'^(.*\n)*'
                r'?\s*Invoking FHI-aims \.\.\.'))

    def parse(self, filepath, archive, logger=None):
        self._metainfo_env = m_env

        parser = FHIAimsParserInterface(filepath, archive, logger)

        parser.parse()
