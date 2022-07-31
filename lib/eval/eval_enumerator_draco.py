import json
import time
from typing import List, Dict

import torch.cuda

from lib.eval.benchmark import Benchmark
from lib.eval.eval import EvalEngine
from lib.eval.output import EvalOutput
from lib.neural_parser.parser import QueryNeuralParser
from lib.prev_synthesizer.enumerator import Enumerator
from lib.prev_synthesizer.interpreter import Interpreter
from lib.prev_synthesizer.translator import Translator
from lib.prev_synthesizer.vis_completor import complete_vis
from lib.utils.enum_utils import process_raw_prediction_for_bid
from lib.utils.eval_utils import check_spec_equiv
from parse_args import args


class EvalSynthEnum(EvalEngine):
    """
    Eval object for the previous translator + interpretor approach
    I put the main synthesis loop in the eval method other than create a synthesis object which is a hack
    translator here is a keyword approach
    """

    def __init__(self, timeout=300):
        super().__init__()

        if torch.cuda.is_available() and args.gpu is not None:
            args.device = torch.device(('cuda:' + args.gpu))
        else:
            args.device = 'cpu'

        self.timeout = timeout
        self.qparser = QueryNeuralParser(args)
        self.enumerator = Enumerator()
        self.translator = Translator(None)
        self.interpreter = Interpreter()
        self.k = 10
        self.spec_limit = -1

        self.incorrect_field = []
        self.none_output = []

        # self.cache_fname = '.cache/evalsynth.pkl'
        # self.cache = {"parsing_field":{}, "parsing": {}}
        # self.load_cache()

    def load_cache(self):
        self.qparser.load_cache()

    def save_cache(self):
        self.qparser.save_cache()

    def eval(self, benchmark: Benchmark, new_old_ablation: bool = False) -> EvalOutput:
        self.enumerator.reinit()

        benchmark.data, data_constraint, falx_data = self.get_data(benchmark, mode='synth', generate_sythesis_constraint=True, analyze_data_syn=True)
        # benchmark.data.init_ref_type()
        vlspec_gt = self.read_gt(benchmark)
        spec = self.qparser.parse(benchmark)
        fields = spec.prob_output['field'][:-1]
        processed_spec = process_raw_prediction_for_bid({benchmark.bname: spec.prob_output}, benchmark.bname)

        start = time.time()

        # implement the enumerator synthesis procedure
        enumerator = self.enumerator.enumerate_yield(processed_spec, limit=-1)
        synthesized_prog_vlspec: List[Dict] = []
        synthesized_prog_vlspec_enumerated = set()
        enumerated_spec = []
        try:
            while len(synthesized_prog_vlspec) < self.k:

                if time.time() - start > self.timeout:
                    raise TimeoutError

                curr_spec = next(enumerator)
                # vis_progs, not_allowed_aggregations, not_allowed_enc = complete_vis(curr_spec, fields, benchmark.data)
                vis_progs, not_allowed_aggregations, not_allowed_enc = curr_spec.get_draco_spec(fields, benchmark.data)
                for prog in vis_progs:
                    print("num enumerated spec")
                    print(len(enumerated_spec))
                    if self.spec_limit == -1 or len(enumerated_spec) < self.spec_limit:
                        enumerated_spec.append(prog)
                    else:
                        raise StopIteration

                    if len(synthesized_prog_vlspec) > self.k:
                        raise StopIteration

                    constraints = self.translator.translate(prog, benchmark.data.colnames, data_url=benchmark.data.get_data_path_constraint())
                    draco_encoding = self.interpreter.to_asp_prev(benchmark.data.get_constraints(), constraints, self.translator,
                                                                  not_allowed_agg=not_allowed_aggregations, not_allowed_enc=not_allowed_enc)

                    outputs, cost = self.interpreter.get_vis(draco_encoding, multiple_solution=True, remove_column=('column' in not_allowed_enc), remove_color=('color' in not_allowed_enc))
                    print("outputs:", outputs)
                    # assert False
                    if outputs is not None:
                        assert len(outputs) == 1
                        if not str(outputs[0]) in synthesized_prog_vlspec_enumerated:
                            synthesized_prog_vlspec.append(outputs[0])
                            synthesized_prog_vlspec_enumerated.add(str(outputs[0]))
                    else:
                        pass
                        # print("not satisfying assignment")

        except StopIteration:
            print("No more goal type available")
        except TimeoutError:
            print("Timeout")

        end = time.time()
        output: EvalOutput = EvalOutput(benchmark.get_id(), benchmark.nl, synthesized_prog_vlspec)
        output.spec = spec
        output.time = (end - start)
        output.cost = -1

        if len(synthesized_prog_vlspec) == 0:
            output.correct = False

        output.correct = check_spec_equiv(falx_data, vlspec_gt, synthesized_prog_vlspec)
        output.gt = json.dumps(vlspec_gt)
        return output

    def write_to_overall_output(self, res: EvalOutput) -> Dict:

        output_to_write = {}
        output_to_write['query'] = res.query
        output_to_write['spec'] = res.spec
        output_to_write['gt'] = str(res.gt)
        output_to_write['output({})'.format(self.get_name())] = str('\n'.join([str(o) for o in res.output])) \
            if res.output is not None else "None"
        output_to_write['top-1 correct'] = res.correct[0] if res.correct else False
        output_to_write['top-5 correct'] = any(res.correct[:5])
        output_to_write['top-10 correct'] = any(res.correct[:10])
        output_to_write['field_match({})'.format(self.get_name())] = res.field_match
        output_to_write['time({})'.format(self.get_name())] = res.time
        output_to_write['cost({})'.format(self.get_name())] = res.cost

        return output_to_write
