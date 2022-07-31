import os
import torch

from lib.eval.eval_synth import EvalSynth
from lib.synthesizer.synthesizer_typed import TopLevelSynthesizerTyped
from lib.utils.csv_utils import read_csv_to_dict, List, Dict
from lib.eval.benchmark import Benchmark
from lib.utils.benchmark_utils import create_benchmark
from lib.utils.data_utils import get_data
from lib.program import Program
from lib.neural_parser.parser import QueryNeuralParser
from lib.synthesizer.program_synthesizer_typed import ProgramSynthesizerTyped
from parse_args import args
from lib.grammar.typed_cfg import TypedCFG

"""
This is the user query handler script for the user study, should connect to the flask in some way
"""

class UserQueryHandler:

    def __init__(self):

        os.chdir("datavis")

        if torch.cuda.is_available() and args.gpu is not None:
            args.device = torch.device(('cuda:' + args.gpu))
        else:
            args.device = 'cpu'

        grammar = TypedCFG()

        self.synth = TopLevelSynthesizerTyped(QueryNeuralParser(args), ProgramSynthesizerTyped(grammar), timeout=300)
        data_dir = "eval/data"
        self.benchmark = Benchmark(dataname="cars", bname="user", nl="", benchmark_set="chi21")
        data, data_constraint, falx_data = get_data(data_dir, self.benchmark, mode='synth', generate_synthesis_constraint=True)
        self.benchmark.data = data
        self.counter = 0

        os.chdir("..")

    def run_user_query_benchmark(self, query, dataset_name):

        os.chdir("datavis")
        benchmark_set = "chi21"
        eval_engine = EvalSynth()
        benchmarks = read_csv_to_dict(eval_engine.get_benchmark_path(benchmark_set))

        for b in benchmarks:
            if b["query"] == query:
                benchmark: Benchmark = create_benchmark(b, benchmark_set)

        print(query)
        res = eval_engine.eval(benchmark)
        synthesized_prog_vlspec: List[Dict] = [prog.to_vega_lite() for prog in res.output]
        os.chdir("..")
        return synthesized_prog_vlspec

    def run_user_query(self, query, k):

        os.chdir("datavis")

        self.benchmark.nl = query
        self.benchmark.bname = "user-" + str(self.counter)
        self.counter += 1
        res: List[Program] = self.synth.synthesize(self.benchmark, k=k)
        synthesized_prog_vlspec: List[Dict] = [prog.to_vega_lite() for prog in res]

        os.chdir("..")

        return synthesized_prog_vlspec


if __name__ == '__main__':
    handler = UserQueryHandler()
    res = handler.run_user_query('how does MPG compare to displacement, broken out by region?', 10)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~1st~~~~~~~~~~~~~~~~~~~~~~~")
    print(res)
    res = handler.run_user_query('show the relationship between acceleration and cylinders', 10)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~2nd~~~~~~~~~~~~~~~~~~~~~~~")
    print(res)
    # res = run_user_query_benchmark('how does MPG compare to displacement, broken out by region?', "Cars")
