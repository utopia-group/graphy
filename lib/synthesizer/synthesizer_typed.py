import time
from collections import defaultdict, OrderedDict
from typing import List

from lib.eval.benchmark import Benchmark
from lib.eval.output import NeuralParserOutput
from lib.program import Program
from lib.synthesizer.program_synthesizer_typed import ProgramSynthesizerTyped
from lib.synthesizer.synthesizer import TopLevelSynthesizer
from lib.type.ref_type import BaseRefType
from lib.type.type_system import reset_formula_id
from lib.utils.misc_utils import printd
from lib.utils.synth_utils import check_goal_type, check_disallow_prov


class TopLevelSynthesizerTyped(TopLevelSynthesizer):
    """
    A top-level synthesizer that is our approach.
    """

    def __init__(self, query_parser, synthesizer, timeout=300):
        """
        :param ablation: if True, we are running the experiment that is comparing to a enumeration baseline, which means the evaluation strategy is different
        """
        super(TopLevelSynthesizerTyped, self).__init__(query_parser, synthesizer, timeout)

    def synthesize(self, b: Benchmark, k=5, limit=-1) -> List[Program]:

        # TODO: avoid re-synthesize the same goal type (this issue should no longer exist once we modify the synthesizer to automatically enumerate x and y)

        assert isinstance(self.synthesizer, ProgramSynthesizerTyped)

        start = time.time()

        self.synthesizer.learned_conflicts = {}
        self.synthesizer.learned_conflicts_based = defaultdict(list)
        self.synthesizer.goal_type_pruned = 0

        synthesized_programs: OrderedDict[Program] = OrderedDict()
        parsed_output: NeuralParserOutput = self.query_parser.parse(b)
        b.data.update_input_type(parsed_output.prob_output['field'][:-1])
        reset_formula_id()
        self.synthesizer.explored_progs_types = []
        self.synthesizer.partial_prog_visited = 0
        self.synthesizer.solution_explored = 0

        goal_type_enumerator = parsed_output.get_next_goal_type(b.data)
        goal_type_enumerated = []

        try:
            i = 0
            while len(synthesized_programs) < k:  # and i < 5:
                i += 1

                if limit == -1 or len(goal_type_enumerated) < limit:
                    goal_type: BaseRefType = next(goal_type_enumerator)
                    goal_type_str = goal_type.__repr__()
                else:
                    raise StopIteration

                if goal_type_str not in goal_type_enumerated:
                    goal_type_enumerated.append(goal_type_str)
                    print("{} new goal type: {}".format(len(goal_type_enumerated), goal_type))
                    if check_goal_type(goal_type, self.synthesizer.learned_conflicts_based) and check_disallow_prov(goal_type, self.synthesizer.learned_conflicts_prov):
                        progs: List[Program] = self.synthesizer.synthesize(goal_type, b.data)
                        if len(progs) == 0:
                            print("no instantiation found")

                        for prog in progs:
                            prog_str = prog.__repr__()
                            if 'scatter' in prog_str and 'bin' in prog_str:
                                continue
                            else:
                                synthesized_programs[prog] = ""
                    else:
                        print("Goal type {} is not valid".format(goal_type_str))
                        # goal_type_pruned += 1
                        # print(self.synthesizer.learned_conflicts)
                        # assert False

        except StopIteration:
            print("No more goal type available")

        print('synthesized programs:', synthesized_programs)

        end = time.time()

        print(self.synthesizer.learned_conflicts)
        print(self.synthesizer.learned_conflicts_prov)
        print("{} goal types pruned".format(self.synthesizer.goal_type_pruned))
        printd('len(explored type):', len(
            self.synthesizer.explored_progs_types))
        print('size of synthesized_programs: ',
              len(synthesized_programs.keys()))
        printd('len(enumerated goal type):', len(goal_type_enumerated))
        print('time: ', end - start)
        print('partial prog visited: ', self.synthesizer.partial_prog_visited)
        print('solution explored: ', self.synthesizer.solution_explored)

        self.goal_type_enumerated_count = len(goal_type_enumerated)

        # print('synthesized programs:', synthesized_programs)

        return list(synthesized_programs.keys())