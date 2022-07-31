from lib.eval.benchmark import Benchmark


def create_benchmark(b: dict, benchmark_set, test_set_info=None) -> Benchmark:
    gtname = b["gtname"] if "gtname" in b else ""

    benchmark = Benchmark(dataname=b["data"].lower(), bname=b["id"], nl=b["query"], gtname=gtname,
                          benchmark_set=benchmark_set)

    if b.get("query-fixed") is not None and not b["query-fixed"] == "":
        benchmark.nl = b["query-fixed"]

    if test_set_info is not None:
        benchmark.fields = [col_str[1:-1] for col_str in test_set_info['fields'][1:-1].split(', ')[:-1]]

    return benchmark
