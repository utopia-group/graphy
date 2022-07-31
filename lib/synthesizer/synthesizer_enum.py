import time
from collections import OrderedDict
from typing import List

from lib.eval.benchmark import Benchmark
from lib.eval.output import NeuralParserOutput
from lib.program import Program
from lib.synthesizer.synthesizer import TopLevelSynthesizer
from lib.type.type_system import reset_formula_id


class TopLevelSynthesizerEnum(TopLevelSynthesizer):
    """
    A top-level synthesizer that implements a pure enumeration of all possible programs
    which does not terminate.
    This class should be deprecated.
    """

    def __init__(self, query_parser, synthesizer, timeout=300):
        super().__init__(query_parser, synthesizer, timeout)

    def synthesize(self, b: Benchmark, k=5, limit=-1) -> List[Program]:

        start = time.time()

        synthesized_programs: OrderedDict[Program] = OrderedDict()
        parsed_output: NeuralParserOutput = self.query_parser.parse(b)
        b.data.update_input_type(
            parsed_output.prob_output['field'][:-1], all_columns=True)
        reset_formula_id()

        goal_type_enumerator = parsed_output.get_next_goal_type(b.data)
        goal_type_enumerated = []

        self.synthesizer.partial_prog_visited = 0
        self.synthesizer.solution_explored = 0

        self.synthesizer.synthesize_init(b.data)

        try:
            while len(synthesized_programs) < k:

                # check timeout
                if time.time() - start > self.timeout:
                    raise TimeoutError

                if limit == -1 or len(goal_type_enumerated) < limit:
                    goal_type = next(goal_type_enumerator)
                    goal_type_enumerated.append(goal_type)
                else:
                    raise StopIteration

                print("{} new goal type: {}".format(len(goal_type_enumerated), goal_type))
                progs: List[Program] = self.synthesizer.synthesize(goal_type, b.data)

                for prog in progs:
                    synthesized_programs[prog] = ""

        except StopIteration:
            # printd(traceback.format_exc())
            print("No more goal type available")
        except TimeoutError:
            print("Timeout")

        print('synthesized_programs: ', synthesized_programs.keys())

        end = time.time()
        print('time: ', end - start)
        print('size of synthesized programs: ',
              len(synthesized_programs.keys()))
        print('partial prog visited: ', self.synthesizer.partial_prog_visited)
        print('solution explored: ', self.synthesizer.solution_explored)

        self.goal_type_enumerated_count = len(goal_type_enumerated)

        return list(synthesized_programs.keys())