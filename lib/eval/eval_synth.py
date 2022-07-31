import itertools
import json
import time

import torch
from typing import Tuple, List, Dict

from lib.eval.benchmark import Benchmark
from lib.eval.output import EvalOutput
from lib.eval.eval import EvalEngine
from lib.grammar.typed_cfg import TypedCFG
from lib.neural_parser.parser import QueryNeuralParser
from lib.program import Program
from lib.synthesizer.program_synthesizer_typed import ProgramSynthesizerTyped
from lib.synthesizer.synthesizer_typed import TopLevelSynthesizerTyped
from parse_args import args

import lib.utils.eval_utils as eval_utils
import lib.utils.trinity_utils.eval_utils as trinity_eval_utils

"""
new eval engine for new-proposed stuff (goal-type directed synthesis)
"""


class EvalSynth(EvalEngine):

    def __init__(self, timeout=300, disable_lemma=False, k=10, include_parsing_time=False):
        super().__init__()

        print(args)

        if torch.cuda.is_available() and args.gpu is not None:
            args.device = torch.device(('cuda:' + args.gpu))
        else:
            args.device = 'cpu'

        self.k = k
        self.include_parsing_time = include_parsing_time

        grammar = TypedCFG()
        self.synthesizer = TopLevelSynthesizerTyped(QueryNeuralParser(args), ProgramSynthesizerTyped(grammar, disable_lemma=disable_lemma), timeout=timeout)

        # init query parser
        self.parser_init: bool = False

        # factory method for generating bid for interactive mode
        self.interactive_bid = itertools.count(start=1)

    def load_cache(self):
        self.synthesizer.query_parser.load_cache()

    def save_cache(self):
        self.synthesizer.query_parser.save_cache()

    def eval(self, benchmark: Benchmark) -> EvalOutput:

        benchmark.data, data_constraint, falx_data = self.get_data(benchmark, mode='synth', generate_sythesis_constraint=True)

        # benchmark.data.init_ref_type()
        vlspec_gt = self.read_gt(benchmark)

        if self.include_parsing_time:
            start = time.time()

        spec = self.synthesizer.get_spec(benchmark)

        if not self.include_parsing_time:
            start = time.time()

        synthesized_programs: List[Program] = self.synthesizer.synthesize(benchmark, k=self.k)
        end = time.time()

        output: EvalOutput = EvalOutput(benchmark.get_id(), benchmark.nl, synthesized_programs)
        output.spec = spec
        output.time = (end - start)
        output.goal_type_enumerated = self.synthesizer.goal_type_enumerated_count
        output.new_partial_prog_explored = self.synthesizer.synthesizer.partial_prog_visited
        output.new_solution_explored = self.synthesizer.synthesizer.solution_explored

        if len(synthesized_programs) == 0:
            output.correct = False

        synthesized_prog_vlspec: List[Dict] = [prog.to_vega_lite() for prog in synthesized_programs]
        if not self.ablation:
            output.correct = eval_utils.check_spec_equiv(falx_data, vlspec_gt, synthesized_prog_vlspec)
        else:
            # For efficiency reason, we are using a much faster equivalent checker
            output.correct = trinity_eval_utils.check_spec_equiv(vlspec_gt, synthesized_prog_vlspec)

        output.gt = json.dumps(vlspec_gt)

        return output

    def prepare_input_interactive(self, nl: str, dataset: str) -> Benchmark:
        """
        Prepare input for the interactive mode
        """
        b = Benchmark(dataname=dataset, bname='user-{}'.format(next(self.interactive_bid)), nl=nl, gtname='null', benchmark_set='chi21')
        return b

    def write_to_overall_output(self, res: EvalOutput) -> Dict:

        output_to_write = {}
        output_to_write['query'] = res.query
        output_to_write['spec'] = res.spec
        output_to_write['gt'] = str(res.gt)

        if self.ablation:
            output_to_write['size'] = str(len(res.output)) if res.output is not None else '0',
            output_to_write['time({})'.format(self.get_name())] = res.time
            output_to_write['top-n correct'] = any(res.correct if len(res.correct) > 0 else "False")

            if res.old_time is not None:
                output_to_write['time_old({})'.format(self.get_name())] = res.old_time
                output_to_write['old_partial_prog_enumerated'] = res.old_partial_prog_explored
                output_to_write['old_solution_enumerated'] = res.old_solution_explored
        else:

            output_to_write['output({})'.format(self.get_name())] = str('\n'.join([str(o) for o in res.output])) \
                if res.output is not None else "None"
            output_to_write['output_vl({})'.format(self.get_name())] = str([json.dumps(o.to_vega_lite()) for o in res.output]) \
                if res.output is not None else "None"
            output_to_write['top-1 correct'] = res.correct[0] if len(res.correct) > 0 else "False"
            output_to_write['top-5 correct'] = any(res.correct[:5]) if len(res.correct) > 0 else "False"
            output_to_write['top-10 correct'] = any(res.correct[:10]) if len(res.correct) > 0 else "False"
            output_to_write['field_match({})'.format(self.get_name())] = res.field_match
            output_to_write['goal_type_enumerated'] = res.goal_type_enumerated
            output_to_write['partial_prog_enumerated'] = res.new_partial_prog_explored
            output_to_write['solution_enumerated'] = res.new_solution_explored
            output_to_write['time({})'.format(self.get_name())] = res.time

            if res.old_time is not None:
                output_to_write['time_old({})'.format(self.get_name())] = res.old_time
                output_to_write['old_partial_prog_enumerated'] = res.old_partial_prog_explored
                output_to_write['old_solution_enumerated'] = res.old_solution_explored

        return output_to_write