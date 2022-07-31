import json
import time
from typing import Dict, List

import torch

from lib.eval.benchmark import Benchmark
from lib.eval.eval import EvalEngine
from lib.eval.output import EvalOutput
from lib.grammar.typed_cfg import TypedCFG
from lib.neural_parser.parser import QueryNeuralParser
from lib.program import Program
from lib.synthesizer.program_synthesizer_enum import ProgramSynthesizerEnum
from lib.synthesizer.synthesizer import TopLevelSynthesizer
from lib.synthesizer.synthesizer_enum import TopLevelSynthesizerEnum
from lib.utils.eval_utils import check_spec_equiv
from parse_args import args


class EvalEnumCheck(EvalEngine):

    """
    This is the class for the baseline that first enumerates all possible programs and then checks if the
    program is equivalent to the specification.
    Note that this baseline almost never terminates because it enumerates all possible programs.
    """
    def __init__(self, ablation=False, timeout=300):
        super().__init__()
        print(args)

        if torch.cuda.is_available() and args.gpu is not None:
            args.device = torch.device(('cuda:' + args.gpu))
        else:
            args.device = 'cpu'

        self.ablation = ablation

        grammar = TypedCFG()
        self.synthesizer = TopLevelSynthesizerEnum(QueryNeuralParser(args), ProgramSynthesizerEnum(grammar, based_typed=True, colname_hack=True), timeout=timeout, ablation=ablation)

    def load_cache(self):
        self.synthesizer.query_parser.load_cache()

    def save_cache(self):
        self.synthesizer.query_parser.save_cache()

    def eval(self, benchmark: Benchmark, new_old_ablation=False) -> EvalOutput:

        benchmark.data, data_constraint, falx_data = self.get_data(benchmark, mode='synth', generate_sythesis_constraint=True)
        # benchmark.data.init_ref_type()
        vlspec_gt = self.read_gt(benchmark)
        spec = self.synthesizer.get_spec(benchmark)
        start = time.time()
        synthesized_programs: List[Program] = self.synthesizer.synthesize(benchmark, k=10)
        end = time.time()
        output: EvalOutput = EvalOutput(benchmark.get_id(), benchmark.nl, synthesized_programs)
        output.spec = spec
        output.time = (end - start)
        output.goal_type_enumerated = self.synthesizer.goal_type_enumerated_count

        # TODO: not sure what to do in timeout cases
        if output.time > 300:
            output.time = 300

        if len(synthesized_programs) == 0:
            output.correct = False

        # Don't check equivalence if we are running ablation
        if not self.ablation:
            synthesized_prog_vlspec: List[Dict] = [prog.to_vega_lite() for prog in synthesized_programs]
            output.correct = check_spec_equiv(falx_data, vlspec_gt, synthesized_prog_vlspec)
        output.gt = json.dumps(vlspec_gt)

        return output

    def write_to_overall_output(self, res: EvalOutput) -> Dict:

        if self.ablation:
            output_to_write = {'query': res.query,
                               'spec': res.spec,
                               'gt': str(res.gt),
                               'size': str(len(res.output)) if res.output is not None else '0',
                               'goal_type_enumerated': res.goal_type_enumerated,
                               'time({})'.format(self.get_name()): res.time
                               }
        else:
            output_to_write = {'query': res.query,
                               'spec': res.spec,
                               'gt': str(res.gt),
                               'output({})'.format(self.get_name()): str('\n'.join([str(o) for o in res.output])) \
                                   if res.output is not None else "None",
                               'output_vl({})'.format(self.get_name()): str([json.dumps(o.to_vega_lite()) for o in res.output]) \
                                   if res.output is not None else "None",
                               'top-1 correct': res.correct[0] if len(res.correct) > 0 else "False",
                               'top-5 correct': any(res.correct[:5]) if len(res.correct) > 0 else "False",
                               'top-10 correct': any(res.correct[:10]) if len(res.correct) > 0 else "False",
                               'field_match({})'.format(self.get_name()): res.field_match,
                               'time({})'.format(self.get_name()): res.time,
                               'cost({})'.format(self.get_name()): res.cost
                               }
        return output_to_write
