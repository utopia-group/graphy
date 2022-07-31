from typing import List

from lib.eval.benchmark import Benchmark
from lib.neural_parser.parser import QueryNeuralParser
from lib.program import Program
from lib.synthesizer.program_synthesizer import ProgramSynthesizer


class TimeoutException(Exception):
    pass


def alarm_handler(signum, frame):
    print("Alarm signal received.")
    raise TimeoutException


class TopLevelSynthesizer:
    """
    Top level synthesizer
    """

    def __init__(self, query_parser, synthesizer, timeout=300):
        self.query_parser: QueryNeuralParser = query_parser
        self.synthesizer: ProgramSynthesizer = synthesizer

        self.goal_type_enumerated_count = 0
        self.timeout = timeout

    def get_spec(self, b: Benchmark):
        return self.query_parser.parse(b)

    def synthesize(self, b: Benchmark, k=5, limit=-1) -> List[Program]:
        raise NotImplementedError
