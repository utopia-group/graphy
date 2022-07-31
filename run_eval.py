import argparse
import os
import traceback
import random
from collections import defaultdict
import cProfile, pstats, io

from lib.eval.eval_enumerator_draco import EvalSynthEnum
from lib.eval.eval_synth_no_prov import EvalSynthNoProv
from lib.eval.eval_synth_no_qualifier import EvalSynthNoQualifier
from lib.eval.eval_synth_no_table import EvalSynthNoTable
from parse_args import set_args
from lib.config.config import PROFILING
from lib.eval.eval import EvalEngine
from lib.eval.eval_enum import EvalEnumCheck
from lib.eval.eval_nl4dv import EvalNL4DV
from lib.eval.eval_synth import EvalSynth
from lib.utils.csv_utils import *
from lib.eval.benchmark import Benchmark
from lib.utils.benchmark_utils import create_benchmark
from lib.utils.preprocess_utils import check_benchmark_support


"""
Main evaluation file that is used to run all benchmarks on Graphy and its baselines in the paper (except the neural ones)
"""

os.chdir(os.getcwd())
print(os.getcwd())

overall_results = defaultdict(dict)
print_str = ""


def run_eval(benchmark_set, eval_engine: EvalEngine, eval_dataset: str, parsing_only=False):
    global print_str
    benchmarks = read_csv_to_dict(eval_engine.get_benchmark_path(benchmark_set))

    if eval_dataset == 'cars':
        held_out_test_benchmarks = convert_list_data_to_dict_data(read_csv_to_dict('eval/datavis/test_held_out.txt'), 'id')
    elif eval_dataset == 'superstore':
        held_out_test_benchmarks = convert_list_data_to_dict_data(read_csv_to_dict('eval/datavis/test_superstore.txt'), 'id')
    elif eval_dataset == 'movies':
        held_out_test_benchmarks = convert_list_data_to_dict_data(read_csv_to_dict('eval/datavis/test_movies.txt'), 'id')
    else:
        raise ValueError('Unknown dataset: {}'.format(eval_dataset))

    correct = 0
    correct_top_5 = 0
    correct_top_10 = 0
    correct_top_k = 0
    total = 0
    wrong_bnames = []
    omitted_bnames = []

    pr = cProfile.Profile()
    pr.enable()

    try:
        for b in benchmarks:
            # print(b)
            b_fullname = "{}-{}".format(b["data"].lower(), b["id"])

            not_support = "" if "not supported" not in b else b["not supported"]
            if not check_benchmark_support(not_support):
                overall_results[b_fullname]["query"] = b['query']
                overall_results[b_fullname]['output({})'.format(eval_engine.get_name())] = "NA"
                omitted_bnames.append(b_fullname)
                continue

            if eval_dataset == 'cars':
                # used to be 122-431
                if not (122 <= int(b["id"]) <= 431):
                    continue
            elif eval_dataset == 'movies':
                if not (432 <= int(b["id"]) <= 702):
                    continue
            elif eval_dataset == 'superstore':
                if not (703 <= int(b["id"]) <= 935):
                    continue

            # recall that some of the benchmarks are used as the dev_set
            if not b["id"] in held_out_test_benchmarks:
                continue

            if 'labeled' not in b or b["labeled"].lower() == 'y':

                benchmark: Benchmark = create_benchmark(b, benchmark_set, held_out_test_benchmarks[b["id"]])

                print("> benchmark: ", b_fullname)
                print("> nl: ", benchmark.nl)

                if not parsing_only:
                    res = eval_engine.eval(benchmark)

                    print(res.correct)

                    correct_val = res.check_correct(_print=True)
                    if correct_val == 'OMITTED':
                        omitted_bnames.append(b_fullname)
                        continue
                    elif correct_val == 'CORRECT':
                        correct += 1
                        correct_top_5 += 1
                        correct_top_10 += 1
                        correct_top_k += 1
                    elif correct_val == 'CORRECT-5':
                        correct_top_5 += 1
                        correct_top_10 += 1
                        correct_top_k += 1
                    elif correct_val == 'CORRECT-10':
                        correct_top_10 += 1
                        correct_top_k += 1
                    elif correct_val == 'CORRECT-K':
                        correct_top_k += 1
                        # Still add this benchmark to the wrong_bnames list if it is not in ablation mode
                        if not eval_engine.ablation:
                            wrong_bnames.append(b_fullname)
                    else:
                        wrong_bnames.append(b_fullname)

                    overall_results[benchmark.get_id()] = eval_engine.write_to_overall_output(res)
                else:
                    spec = eval_engine.run_parsing(benchmark)
                    overall_results[b_fullname]['spec'] = spec
                    print("res spec:", spec)

                total += 1

            else:
                overall_results[b_fullname]['output({})'.format(eval_engine.get_name())] = "NA"
    except Exception:
        traceback.print_exc()

    pr.disable()
    if total > 0:
        print("overall correct (top 1) rate: {}/{}={}".format(correct, total, correct / total))
        print("overall correct (top 5) rate: {}/{}={}".format(correct_top_5, total, correct_top_5 / total))
        print("overall correct (top 10) rate: {}/{}={}".format(correct_top_10, total, correct_top_10 / total))
        print("overall correct (top k) rate: {}/{}={}".format(correct_top_k, total, correct_top_k / total))
        print("incorrect bnames: ", wrong_bnames)
        print("omitted bnames: ", omitted_bnames)
        eval_engine.print_additional_output()

        print_str += "{} overall correct rate: {}/{}={}\n".format(eval_engine.get_name(), correct, total, correct / total)
    print("omitted count: {}".format(len(omitted_bnames)))

    if PROFILING:
        s = io.StringIO()
        if sys.version_info >= (3, 7):
            from pstats import SortKey
            sortby = SortKey.CUMULATIVE
        else:
            sortby = 2
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        with open('eval_output/synthesis_profiling.txt', 'w+') as f:
            f.write(s.getvalue())


if __name__ == '__main__':

    random.seed(233)

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_set', type=str, default='chi21', help='benchmark set to use')
    parser.add_argument('--eval_dataset', type=str, default='movies', help='dataset to evaluate on')
    parser.add_argument('--save_res', action='store_true', help='save results to file')
    parser.add_argument('--disable_lemma', action='store_true', help='disable pruning using lemmas')
    parser.add_argument('--parsing_only', action='store_true', help='only run parsing')
    parser.add_argument('--include_parsing_time', action='store_true', help='record parsing time')
    parser.add_argument('--timeout', type=int, default=60, help='timeout for each benchmark')
    parser.add_argument('--draco', action='store_true', help='use draco')
    parser.add_argument('--enum', action='store_true', help='use enum')
    parser.add_argument('--no_qualifier', action='store_true', help='disable all qualifier')
    parser.add_argument('--no_prov', action='store_true', help='disable provenance only')
    parser.add_argument('--no_table', action='store_true', help='disable table properties only')
    parser.add_argument('--top_k', type=int, default=10, help='top k results')
    parser.add_argument('--nl4dv', action='store_true', help='use nl4dv')
    args, unknown = parser.parse_known_args()

    print("args:", args)
    set_args(args.eval_dataset) # set the model for the parser depending on the dataset

    if args.nl4dv:
        eval_engine = EvalNL4DV()
    elif args.enum:
        eval_engine = EvalEnumCheck(timeout=args.timeout)
    elif args.draco:
        eval_engine = EvalSynthEnum(timeout=args.timeout)
    elif args.no_qualifier:
        eval_engine = EvalSynthNoQualifier(timeout=args.timeout, k=args.top_k)
    elif args.no_prov:
        eval_engine = EvalSynthNoProv(timeout=args.timeout, k=args.top_k)
    elif args.no_table:
        eval_engine = EvalSynthNoTable(timeout=args.timeout, k=args.top_k)
    else:
        eval_engine = EvalSynth(timeout=args.timeout, disable_lemma=args.disable_lemma, k=args.top_k, include_parsing_time=args.include_parsing_time)

    run_eval(args.benchmark_set, eval_engine, args.eval_dataset, parsing_only=args.parsing_only)

    # save output to some csv
    if args.save_res:
        new_overall_results = [dict({"bid": k, **v}) for k, v in overall_results.items()]
        if args.parsing_only:
            new_overall_results_file = "{}_{}_{}_run_parsing_only_res".format(eval_engine.get_name(), args.benchmark_set, args.eval_dataset)
        elif args.include_parsing_time:
            new_overall_results_file = "{}_{}_{}_run_parsing_time_res".format(eval_engine.get_name(), args.benchmark_set, args.eval_dataset)
        elif args.disable_lemma:
            new_overall_results_file = "{}_{}_{}_run_parsing_no_lemma_res".format(eval_engine.get_name(), args.benchmark_set, args.eval_dataset)
        else:
            new_overall_results_file = "{}_{}_{}_run_parsing_res".format(eval_engine.get_name(), args.benchmark_set, args.eval_dataset)
        save_dict_to_csv("eval_output/{}.csv".format(new_overall_results_file), new_overall_results)
        # save_dict_to_csv("{}.csv".format(new_overall_results_file), new_overall_results)

        print(print_str)
        # os.system('mv {0}.csv {0}_1.csv'.format(new_overall_results_file))
