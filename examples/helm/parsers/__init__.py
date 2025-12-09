from .math import MATHParser
from .wmt_14 import WMT14Parser

parsers = {
    'math'   : MATHParser,
    'wmt_14' : WMT14Parser,
}