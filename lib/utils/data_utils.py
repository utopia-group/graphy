"""
utility function to help obtain the correct data (csv file) and cache the read data
"""
from collections import defaultdict
from typing import Union, Tuple

from lib.dracopy.draco.helper import read_data_to_asp
from lib.eval.benchmark import Benchmark, Dataset, InputDataset
from lib.falx.utils.table_utils import load_and_clean_table
from lib.utils.csv_utils import read_csv_to_dict

data_map = {}
data_constraint_map = {}
data_falx_map = {}
data_space_map = {}
data_underscore_map = {}


def get_data(data_dir: str, benchmark: Benchmark, mode=None, read_data_constraint=True, generate_synthesis_constraint=False, analyze_data_syn=False) -> \
        Union[Tuple[Dataset, list, object], Tuple]:
    if mode == 'synth':
        data_identifier = "{}_{}".format(benchmark.dataname,
                                         benchmark.benchmark_set) if benchmark.dataname.lower() == \
                                                                     'movies' else benchmark.dataname
        if data_identifier not in data_map:
            if generate_synthesis_constraint:
                data = InputDataset(benchmark.dataname, benchmark.benchmark_set, data_dir)
                data.init_ref_type()
            else:
                data = Dataset(benchmark.dataname, benchmark.benchmark_set, data_dir, analyze_data_nl2sql=True, analyze_data_syn=analyze_data_syn)
            data_map[data_identifier] = data
            data_falx_map[data_identifier] = load_and_clean_table(data.data)
            if read_data_constraint:
                data_constraints = read_data_to_asp(data.data_path)
                colname_to_contraint = defaultdict(list)
                for constraint in data_constraints:
                    constraint_parse = constraint.split('"')
                    if len(constraint_parse) == 3:
                        field_name = constraint_parse[1]
                        colname_to_contraint[field_name].append(constraint)
                data_constraint_map[data_identifier] = colname_to_contraint
            else:
                data_constraint_map[data_identifier] = None

        return data_map[data_identifier], data_constraint_map[data_identifier], data_falx_map[data_identifier]
    else:
        dataset_path = "{0}/{1}_{2}/{1}.csv".format(data_dir, benchmark.dataname, benchmark.benchmark_set) if \
            benchmark.dataname.lower() == "movies" else "{0}/{1}/{1}.csv".format(data_dir, benchmark.dataname)

        if not dataset_path in data_underscore_map:
            data_underscore_map[dataset_path] = load_and_clean_table(read_csv_to_dict(dataset_path))
            fieldnames = []
            for k in data_underscore_map[dataset_path][0].keys():
                fieldnames.append(k.replace("_", " "))
            data_space_map[dataset_path] = load_and_clean_table(read_csv_to_dict(dataset_path, fieldnames))

        data_underscore = data_underscore_map[dataset_path]
        data_space = data_space_map[dataset_path]

        return data_underscore, data_space
