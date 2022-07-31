import json
import os
import pickle
from typing import Tuple, Union, Dict

from lib.eval.benchmark import Dataset, Benchmark
from lib.eval.output import EvalOutput
from lib.utils.data_utils import get_data
from lib.utils.eval_utils import clean_vlspec

"""
Class for eval a particular method
its role:
- read benchmark
- run the ? engine
- compare the result with the gt and produce the metric 
"""


class EvalEngine:
    def __init__(self):
        self.eval_dir = 'eval'
        self.gt_dir = os.path.join(self.eval_dir, "gt")
        self.data_dir = os.path.join(self.eval_dir, "data")

        self.ablation = False

        # self.cache = None
        # self.cache_fname = ''

    # TODO: improve caching mechanism
    def load_cache(self):
        raise NotImplementedError

    def save_cache(self):
        raise NotImplementedError

    def read_gt(self, benchmark) -> Dict:
        if benchmark.benchmark_set == "nl4dv":
            gt_path = "{}/{}-{}.vl.json".format(self.gt_dir, benchmark.dataname, benchmark.bname)
        else:
            gt_path = "{}/{}.vl.json".format(self.gt_dir, benchmark.gtname)
        with open(gt_path, "r") as f:
            vlspec = json.load(f)
        return clean_vlspec(vlspec)

    def get_data(self, benchmark: Benchmark, mode=None, read_data_constraint=True, generate_sythesis_constraint=False, analyze_data_syn=False) -> \
            Union[Tuple[Dataset, list, object], Tuple]:
        return get_data(self.data_dir, benchmark, mode, read_data_constraint, generate_sythesis_constraint, analyze_data_syn)

    def get_benchmark_path(self, benchmark_set) -> str:
        return "{}/benchmarks-{}.csv".format(self.eval_dir, benchmark_set)

    def eval(self, benchmark: Benchmark) -> EvalOutput:
        raise NotImplementedError

    def print_additional_output(self):
        pass

    def write_to_overall_output(self, res: EvalOutput) -> Dict:
        output_to_write = {}
        output_to_write['query'] = res.query
        output_to_write['spec'] = res.spec
        output_to_write['gt'] = str(res.gt)
        # print("OUTPUT")
        # print(res.output)
        output_to_write['output({})'.format(self.get_name())] = str('\n'.join([str(o) for o in res.output])) \
            if res.output is not None else "None"
        # only get distinct length
        # res_output_str_set = set([str(output) for output in res.output]) if isinstance(res.output, list) else set()
        # output_to_write['len({})'.format(self.get_name())] = len(res_output_str_set)
        output_to_write['field_match({})'.format(self.get_name())] = res.field_match
        output_to_write['time({})'.format(self.get_name())] = res.time
        output_to_write['cost({})'.format(self.get_name())] = res.cost
        output_to_write['res({})'.format(self.get_name())] = res.correct
        return output_to_write

    def get_name(self):
        return self.__class__.__name__

    def run_parsing(self, benchmark: Benchmark):
        raise NotImplementedError
