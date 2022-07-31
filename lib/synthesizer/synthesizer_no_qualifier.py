import time
from collections import OrderedDict
from typing import List

from lib.eval.benchmark import Benchmark
from lib.eval.output import NeuralParserOutput
from lib.program import Program
from lib.synthesizer.synthesizer import TopLevelSynthesizer
from lib.type.ref_type import BaseRefType
from lib.type.type_system import reset_formula_id
from lib.utils.misc_utils import printd
from lib.utils.synth_utils import check_goal_type, check_disallow_prov, compile_and_check_table, check_prov_constraint, create_pseudo_goal_type


class TopLevelSynthesizerNoQualifier(TopLevelSynthesizer):
    """
    A top-level synthesizer that either no datatype or no refinement type.
    """

    def __init__(self, query_parser, synthesizer, timeout=300, no_provenance=False, no_table=False, base_type=True):
        super(TopLevelSynthesizerNoQualifier, self).__init__(query_parser, synthesizer, timeout)
        self.no_provenance = no_provenance
        self.no_table = no_table
        self.base_type = base_type

    def synthesize(self, b: Benchmark, k=5, limit=-1) -> List[Program]:

        start = time.time()

        synthesized_programs: OrderedDict[Program] = OrderedDict()
        parsed_output: NeuralParserOutput = self.query_parser.parse(b)

        if self.no_provenance and self.no_table:
            b.data.update_input_type_without_card(parsed_output.prob_output['field'][:-1])
        elif self.no_provenance:
            b.data.update_input_type(parsed_output.prob_output['field'][:-1])
        elif self.no_table:
            b.data.update_input_type_without_card(parsed_output.prob_output['field'][:-1])

        valid_synthesized_programs: OrderedDict[Program] = OrderedDict()
        valid_synthesized_vegalite = {}

        reset_formula_id()
        self.synthesizer.explored_progs_types = []
        self.synthesizer.partial_prog_visited = 0
        self.synthesizer.solution_explored = 0

        goal_type_enumerator = parsed_output.get_next_goal_type(b.data)
        goal_type_enumerated = []

        try:
            i = 0
            while len(valid_synthesized_programs) < k:

                # check timeout
                if time.time() - start > self.timeout:
                    raise TimeoutError

                i += 1

                if limit == -1 or len(goal_type_enumerated) < limit:
                    goal_type: BaseRefType = next(goal_type_enumerator)
                    goal_type_str = goal_type.__repr__()
                else:
                    raise StopIteration

                # if i < 2 or i >= 3:
                #     continue

                # print(i)

                if goal_type_str not in goal_type_enumerated:
                    goal_type_enumerated.append(goal_type_str)
                    print("{} new goal type: {}".format(len(goal_type_enumerated), goal_type))

                    pass_check = True
                    if self.no_provenance:
                        pass
                    elif check_goal_type(goal_type, self.synthesizer.learned_conflicts_based) and check_disallow_prov(goal_type, self.synthesizer.learned_conflicts_prov):
                        assert not self.no_provenance
                        pass
                    else:
                        pass_check = False

                    if pass_check:

                        if self.no_provenance:
                            # here we create a pseudo-goal type with no provenance constraint
                            goal_type_to_synthesizer = create_pseudo_goal_type(goal_type)
                            print("goal type to synthesizer: {}".format(goal_type_to_synthesizer))
                        else:
                            goal_type_to_synthesizer = goal_type

                        progs: List[Program] = self.synthesizer.synthesize(goal_type_to_synthesizer, b.data)
                        if len(progs) == 0:
                            print("no instantiation found")

                        for prog in progs:
                            # print("synthesized program:", prog)

                            synthesized_programs[prog] = ""

                            if self.no_provenance:
                                if not check_prov_constraint(b, prog, goal_type):
                                    continue

                            if not compile_and_check_table(b, prog, no_table=self.no_table):
                                continue

                            prog_str = str(prog)
                            if 'scatter' in prog_str and 'bin' in prog_str:
                                continue

                            vl_spec = prog.to_vega_lite()
                            if str(vl_spec) not in valid_synthesized_vegalite:
                                valid_synthesized_programs[prog] = ""
                    else:
                        if not self.no_provenance:
                            print("Goal type {} is not valid".format(goal_type_str))
                        else:
                            assert False
                # print("valid_synthesized_programs:", valid_synthesized_programs)
                # assert False

            # printd(traceback.format_exc())
        except StopIteration:
            print("No more goal type available")
        except TimeoutError:
            print("Timeout")

        printd('synthesized programs:', synthesized_programs)

        end = time.time()

        printd('len(explored type):', len(self.synthesizer.explored_progs_types))
        print('size of synthesized_programs: ', len(synthesized_programs.keys()))
        printd('len(enumerated goal type):', len(goal_type_enumerated))
        print('time: ', end - start)
        print('partial prog visited: ', self.synthesizer.partial_prog_visited)
        print('solution explored: ', self.synthesizer.solution_explored)
        print("num valid programs: ", len(valid_synthesized_programs))
        print("valid programs: ", valid_synthesized_programs.keys())

        self.num_total_synthesized_prog = len(synthesized_programs)
        self.goal_type_enumerated_count = len(goal_type_enumerated)

        return list(valid_synthesized_programs.keys())