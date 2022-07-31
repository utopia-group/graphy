import argparse
import collections
import copy
import itertools
import json
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import subprocess
import time

from pprint import pprint

from lib.dracopy.draco.utils import cql_to_asp, data_to_asp
from lib.dracopy.draco.run import run, run_clingo, DEBUG
import lib.dracopy.draco.refine_strategies as refine_strategies
from lib.dracopy.post_process.extract_plot_semantics import extract_plot_semantics
from lib.dracopy.post_process.extract_position_encodings import adjust_position_encodings, extract_position_encodings
from lib.dracopy.post_process.count_with_size import adjust_count_with_size_encoding


parser = argparse.ArgumentParser(description="Draco visualization synthesis.")
parser.add_argument("--input_dir", default="../../wikisql-datasets", help="Input directory containing tables and fine types.")
parser.add_argument("--output_dir", default="../../vis-viewer/wikisql-outputs", help="The output directory.")
parser.add_argument("--table_name", default=None, help="The table that will be used in enumeration.")
parser.add_argument("--inline_data", default=False, help="Whether we inline data in the vega-lite script (or alternatively store them separately in a csv file)")
parser.add_argument("--dataset_url", default=None, help="the url for accessing dataset, this is useful when we don't inline data")

# enumeration optinos for each tale
parser.add_argument("--sample_for_each_col_comb", type=int, default=None, help="How many visualization samples do we want to choose for each column combination in a dataset.")
parser.add_argument("--enhance_with_sorting", default=False, help="Whether we want to add sorting options in visualization enumeration")
parser.add_argument("--enum_table_filter_num", default=0, help="Whether we want to add random filters to visualization datasets (and how many filters we want to consider for each table)")
parser.add_argument("--max_field_comb_size", default=2, help="The maximum number of fields we'd consider in each column combination.")
parser.add_argument("--max_num_field_combs", default=256, help="The number of field combinations we'd consider for one table.")

# dataset level options
parser.add_argument("--ignore_used_dataset", default=False, help="If visualizations for one dataset already exist, prevent enumerating for that.")
parser.add_argument("--dataset_limit", default=None, help="Limit the number of dataset being considered to dataset_limit.")

# --- Viz enumeration hyperparameters --- #


SEM_TYPE_TO_VEGA_ENC_TYPE = {
    'COUNT': 'quantitative',
    'QUANT_MONEY': 'quantitative',
    'QUANT_WITH_UNIT': 'quantitative',
    'RAW_INTEGER': 'quantitative',
    'RAW_NUMBER': 'quantitative',
    'RATIO': 'quantitative',
    'RATIO_PERCENT': 'quantitative',
    'ORDINAL': 'ordinal',
    'ORDINAL_INDEX': 'ordinal',
    "TEMPORAL": 'temporal',
    "TEMPORAL_YEAR": 'temporal',
    "TEMPORAL_MONTH": 'temporal',
    'GEO_CITY': 'nominal',
    'GEO_STATE': 'nominal',
    'GEO_COUNTRY': 'nominal',
    'CATEGORICAL': 'nominal',
    'NOMINAL': 'nominal',
    'STRING': 'nominal'
}


VEGA_ENC_TYPES_TO_SEM_TYPES = {
    'quantitative': ['COUNT', 'QUANT_MONEY', 'QUANT_WITH_UNIT', 'RAW_INTEGER', 'RAW_NUMBER', 'RATIO', 'RATIO_PERCENT'],
    'ordinal': ['ORDINAL', 'ORDINAL_INDEX'],
    'temporal': ['TEMPORAL', 'TEMPORAL_YEAR', 'TEMPORAL_MONTH'],
    'nominal': ['GEO_CITY', 'GEO_STATE', 'GEO_COUNTRY', 'CATEGORICAL', 'NOMINAL', 'STRING']
}


def process_column(columns):
    """process column name so that they are compatible with ASP syntax for draco enumeration
    Returns:
        new_cols: new column names
        inv_map: a map that maps new column names to their original names (for recovery purpose)
    """
    new_cols = [re.sub('[^a-zA-Z0-9 \n\.]', '', s.lower()).replace(" ", "") for s in columns]
    new_cols = [f"{c}{i}" for i, c in enumerate(new_cols)]
    inv_mp = { new_cols[i] :  s for i, s in enumerate(columns)}
    return new_cols, inv_mp


def run_draco(in_csv, df, target_fields, fine_types, top_k=5):
    """given the dataset and a compass query, 
       obtain a vega-lite spec recommened from draco """

    type_map = { df.columns[i] : fine_types[i] for i in range(len(df.columns)) }

    data = json.loads(df.to_json(orient="records"))

    if DEBUG:
        print(target_fields)

    cql_query = {
        "data": {"url": in_csv},
        "encodings": [{"field": f, "type": SEM_TYPE_TO_VEGA_ENC_TYPE[type_map[f]]} for f in target_fields]
    }

    if DEBUG:
        print([(f, type_map[f]) for f in target_fields])
        pprint(cql_query["encodings"])

    asp_query = cql_to_asp(cql_query)

    # prevent generating aggregations
    for i, f in enumerate(target_fields):
        eid = "e{}".format(i)
        if type_map[f] not in {"QUANT_MONEY", "RATIO", "RATIO_PERCENT", "RAW_NUMBER", "QUANT_WITH_UNIT"}:
            asp_query.append(":- aggregate({},_).".format(eid))

    projected_data = [{ f : r[f] for f in r if f in target_fields } for r in data]
    projected_fine_type = [type_map[f] for f in target_fields]

    data_asp = data_to_asp(projected_data, projected_fine_type)
    if DEBUG:
        print(data_asp)

    program = asp_query + data_asp

    results = run(program, multiple_solution=True, debug=DEBUG)

    if results is not None:
        return [result.as_vl() for result in results]
    else:
        return None


def score_column_comb(column_comb):
    SINGLE_FIELD_SCORE = 1.0
    DOUBLE_FIELD_SCORE = 10.0
    TRIPLE_FIELD_SCORE = 5.0
    if len(column_comb) == 1:
        score = SINGLE_FIELD_SCORE
        column, fine_type = column_comb[0]
    elif len(column_comb) == 2:
        score = DOUBLE_FIELD_SCORE
        fine_types = []
        for column, fine_type in column_comb:
            fine_types.append(fine_type)
        fine_type_signature = frozenset(fine_types)
        if fine_type_signature in refine_strategies.BANNED_TUPLE_FIELD_TYPES:
            return 0
        if fine_type_signature in refine_strategies.BONUS_TUPLE_FIELD_TYPES:
            score += refine_strategies.BONUS_TUPLE_FIELD_TYPES[fine_type_signature]
    elif len(column_comb) == 3:
        score = TRIPLE_FIELD_SCORE
        fine_types = []
        for column, fine_type in column_comb:
            fine_types.append(fine_type)
        if frozenset(fine_types) in refine_strategies.BANNED_TRIPLE_FIELD_TYPES:
            return 0
    else:
        raise NotImplementedError
    return score


def enum_column_combinations(columns, fine_types, markers, max_num_column_combs, max_field_comb_size):
    """markers mark whether a column is valid or not """
    columns_with_type = [(columns[i], fine_types[i]) for i in range(len(columns)) if markers[i]] #list(zip(columns, fine_types))
    col_combs = []
    for k in range(1, max_field_comb_size + 1):
        if len(columns) < k: break
        col_combs += [(x, score_column_comb(x)) for x in list(itertools.combinations(columns_with_type, k))]

    # only select at most $max_num_column_combs column combinations to visualize
    chosen_cols = [col_comb for col_comb, score in sorted(col_combs, key=lambda x:x[1], reverse=True)[:max_num_column_combs]
                   if score > 0]
    return chosen_cols


def enum_visualizations(df, target_columns, fine_types, file_url, config):
    """Enumerate visdualizations for the given columns in a dataset
    Args:
        df: the target dataset that will be used in enumeration
        target_columns: column combinations that will be used
        fine_types: fine data types of the given visualization
        file_url: the url link for the dataset (only going to be used as a placeholder)
    Returns:
        a list of valid visualizations
    """
    # run draco for each column combinations

    enhance_with_sorting = config["enhance_with_sorting"]

    data = json.loads(df.to_json(orient="records"))

    specs = run_draco(file_url, df, target_columns, fine_types)
    num_specs = len(specs) if specs is not None else 0
    print('{}: {} pre-filtering'.format(target_columns, num_specs))
    if specs is None:
        return []

    final_specs = []
    for spec in specs:
        # inspect = spec['mark'] == 'bar'
        # if inspect:
        #     print(json.dumps(spec, indent=4))
        #     import pdb
        #     pdb.set_trace()
        # post process specs and remove broken line or area charts
        spec = refine_strategies.post_process_vl_spec(spec, data)

        if spec is None:
            continue
        # if inspect:
        #     print(2)
        #     print(json.dumps(spec, indent=4))

        spec = refine_strategies.filter_broken_chart_and_process(spec, data)

        if spec is None: 
            continue
        # if inspect:
        #     print(3)
        #     print(json.dumps(spec, indent=4))

        if enhance_with_sorting:
            spec_candidates = refine_strategies.add_sorting(spec)
        else:
            spec_candidates = [spec]
        final_specs += spec_candidates

    return final_specs


def enum_visualizations_per_dataset(csv_data_file, fine_type_file, field_markers, config, out_dir):
    """Given the csv data file and fine_type_file, enumerate visualizations on this dataset
    Args:
        csv_data_file: csv data used for visualization
        fine_type_file: fine types of columns in the dataset
    Returns:
        a list of vl_specs with embedded data 
    """
    num_total_specs = 0

    start = time.time()

    with open(csv_data_file, "r", encoding='utf-8') as f:
        df = pd.read_csv(f, nrows=500)

    with open(fine_type_file, "r") as g:
        fine_types = []
        for s in g.readlines():
            if s.strip():
                column_fine_types = s.strip().split('\t')
                if column_fine_types[0] == 'NOMINAL' and 'CATEGORICAL' in column_fine_types:
                    fine_types.append('CATEGORICAL')
                else:
                    fine_types.append(column_fine_types[0])

    assert(len(df.columns) == len(fine_types))

    raw_data = json.loads(df.to_json(orient="records"))
    
    new_cols, inv_map = process_column(df.columns)
    df.columns = new_cols

    # select columns that will be used in visualization
    refined_cols, refined_types = refine_strategies.select_fields_of_interests(df.columns, fine_types)

    # enumerate column combinations
    col_combs = enum_column_combinations(
                    refined_cols, refined_types, field_markers, 
                    config["max_num_field_combs"], config["max_field_comb_size"])

    filter_candidates = [(df, None)]

    if config["enum_table_filter_num"] > 0:
        # enumerate filtering predicates
        type_map = { df.columns[i] : fine_types[i] for i in range(len(df.columns)) }
        enc_types = [SEM_TYPE_TO_VEGA_ENC_TYPE.get(type_map[f], None) for f in df.columns]
        filter_candidates += refine_strategies.enum_filters(df, enc_types, limit=config["enum_table_filter_num"])
        # TODO: iterate over predicate candidates to enumerate different visualizations

    vl_specs = collections.defaultdict(list)
    for _df, filter_expr in filter_candidates:
       
        if filter_expr != None:
            c = filter_expr['col']
            op = filter_expr['op']
            const = filter_expr['val'] if not isinstance(filter_expr['val'],str) else "'{}'".format(filter_expr['val'])
            filter_obj = {"filter": f"datum['{inv_map[c]}'] {op} {const}"}

        specs = collections.defaultdict(list)
        for cols in col_combs:
            if DEBUG:
                print(cols)
            target_columns = [c for c, _ in cols]
            target_cols_sig = ':::'.join(sorted(target_columns))
            specs_cols = enum_visualizations(_df, target_columns, fine_types, csv_data_file, config)
            num_total_specs += len(specs_cols)
            # sample visualizations per column combination if there is a number limit
            sample_size = config["sample_for_each_col_comb"]
            if sample_size is not None and len(specs_cols) > sample_size:
                specs_cols = random.sample(specs_cols, sample_size)
            # add filter experssion to data
            if filter_expr != None:
                for spec in specs_cols:
                    spec["transform"] = [filter_obj]
            # fix names generated from draco (since draco uses internal names)
            for spec in specs_cols:
                if "title" in spec and spec["title"] in inv_map:
                    spec["title"] = inv_map[spec["title"]]

                for channel, enc in spec["encoding"].items():
                    if "field" in spec["encoding"][channel]:
                        enc["field"] = inv_map[enc["field"]]

                        # handle name escape, this deals with cases where name contains ".", "[", "]" that requires escaping
                        if "." in enc["field"] or "[" in enc["field"] or "]" in enc["field"]:
                            enc["title"] = enc["field"]
                            enc["field"] = enc["field"].replace(".", "\\.").replace("[", "\\[").replace("]", "\\]")

                if config["inline_data"]:
                    # inline data in the vl spec for the purpose of easy viewing
                    spec["data"] = {"values": raw_data}
                else:
                    if config["dataset_url"] is not None:
                        spec["data"] = {
                            "url": os.path.join(config["dataset_url"], os.path.basename(out_dir), os.path.basename(csv_data_file)),
                            "format": {"type": "csv"}
                        }

            specs[target_cols_sig] = specs_cols

        for target_cols_sig in specs:
            vl_specs[target_cols_sig].extend(specs[target_cols_sig])

    time_lapse = time.time() - start
    if DEBUG:
        print('{} columns in table, {} column combinations considered, {} passed'.format(
            len(df.columns), len(col_combs), time_lapse))

    return raw_data, vl_specs, col_combs, num_total_specs


def enum_table_visualizations(table_name, in_dir, out_dir, config, finished_folders):
    """Given a table name and the input dir, the method will look for 
        <table_name>.norm.csv and <table_name>.sem.types and then enumerate outputs 
        and store them in out_dir/<table_name>
        inline_data: whether inline raw data directly in visualization"""
    for file_name in os.listdir(in_dir):
        if file_name.endswith('.norm.csv'):
            table_name = os.path.basename(file_name)[:-9]
            break

    if table_name not in finished_folders:
        print('enumerate visualizations for table {}'.format(table_name))
        in_csv = os.path.join(in_dir, '{}.norm.csv'.format(table_name))
        in_types = os.path.join(in_dir, '{}.sem.types'.format(table_name))
        valid_field_marker_file = os.path.join(in_dir, "{}.valid.field.markers".format(table_name))
        viz_field_marker_file = os.path.join(in_dir, "{}.viz.field.markers".format(table_name))
        with open(valid_field_marker_file, "r", encoding="utf-8") as f:
            valid_field_markers = [int(m) for m in f.readlines()[0].strip().split(",")]
        with open(viz_field_marker_file, "r", encoding="utf-8") as f:
            viz_field_markers = [int(m) for m in f.readlines()[0].strip().split(",")]
        field_markers = [x * y for x, y in zip(valid_field_markers, viz_field_markers)]
        out_plot_dir = os.path.join(out_dir, table_name)
        if not os.path.exists(out_plot_dir):
            os.mkdir(out_plot_dir)
        else:
            if config["ignore_used_dataset"]:
                print("[ignore dataset] {}".format(out_dir))
                return

        print('saving to {}'.format(out_plot_dir))
        data, specs, col_combs, num_specs = enum_visualizations_per_dataset(in_csv, in_types, field_markers, config, out_dir)

        data_json = os.path.join(out_plot_dir, '{}.data.json'.format(table_name))
        with open(data_json, "w") as f:
            json.dump(data, f, indent=2)

        with open(os.path.join(out_plot_dir, 'field_combinations.txt'), 'w') as o_f:
            for col_comb in sorted(col_combs):
                o_f.write('{}\n'.format('\t'.join([c for c, _ in col_comb])))
        with open(os.path.join(out_plot_dir, 'type_combinations.txt'), 'w') as o_f:
            for col_comb in sorted(col_combs):
                o_f.write('{}\n'.format('\t'.join([t for _, t in col_comb])))

        out_id = 0
        for cols_sig in specs:
            # # post process specs
            # spec = refine_strategies.post_process_vl_spec(spec, data)
            # if refine_strategies.is_broken_line_area_charts(spec, data):
            #     continue
            
            # subdir = os.path.join(out_plot_dir, cols_sig)
            # if len(specs[cols_sig]) > 0 and not os.path.exists(subdir):
            #     os.mkdir(subdir)
            subdir = os.path.join(out_plot_dir, cols_sig)
            if len(specs[cols_sig]) > 0 and not os.path.exists(subdir):
                os.mkdir(subdir)
            position_encodings_dict = dict()
            for spec in specs[cols_sig]:
                adjust_count_with_size_encoding(spec)
                semantic_id = extract_plot_semantics(spec)
                if not semantic_id in position_encodings_dict:
                    position_encodings_dict[semantic_id] = collections.defaultdict(list)
                position_encoding_id = extract_position_encodings(spec)
                position_encodings_dict[semantic_id][position_encoding_id].append(spec)
            for semantic_id in position_encodings_dict:
                if '/' in semantic_id:
                    plot_semantic_id=semantic_id.replace('/', '-')
                else:
                    plot_semantic_id=semantic_id
                plot_dir = os.path.join(subdir, plot_semantic_id)
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
                for position_encoding_id in position_encodings_dict[semantic_id]:
                    specs_ = position_encodings_dict[semantic_id][position_encoding_id]
                    if len(specs_) > 1:
                        adjust_position_encodings(specs_[0])
                    out_json = os.path.join(plot_dir, 'v{}.vl.json'.format(out_id))

                    out_id += 1
                    with open(out_json, 'w') as o_f:
                        print('saving to {}'.format(out_json))
                        json.dump(specs_[0], o_f, indent=2)

        return len(col_combs), num_specs


def enum_table_visualizations_in_batch(in_dir, out_dir, config):
    """iterate over all files in the in_dir and save to out_dir """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # datasets = [in_csv.split('.', 1)[0] for in_csv in os.listdir(in_dir) if in_csv.endswith(".norm.csv")]
    datasets = [os.path.join(in_dir, subdir) for subdir in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, subdir))]

    if config["dataset_limit"] != None:
        datasets = datasets[:config["dataset_limit"]]

    finished_folders = [folder.replace(out_dir, '') for folder in os.listdir(out_dir)]

    for dataset_subdir in datasets:
        enum_table_visualizations(os.path.basename(dataset_subdir), dataset_subdir, out_dir, config, finished_folders)


if __name__ == '__main__':
    args = parser.parse_args()
    config = {
        "sample_for_each_col_comb": args.sample_for_each_col_comb, 
        "inline_data": args.inline_data,
        "dataset_url": args.dataset_url,
        "enhance_with_sorting": args.enhance_with_sorting,
        "enum_table_filter_num": args.enum_table_filter_num,
        "max_field_comb_size": args.max_field_comb_size,
        "max_num_field_combs": args.max_num_field_combs,
        "ignore_used_dataset": args.ignore_used_dataset,
        "dataset_limit": args.dataset_limit
    }


    if args.table_name is None:
        enum_table_visualizations_in_batch(
            args.input_dir, args.output_dir, config)
    else:
        enum_table_visualizations(
            args.table_name, args.input_dir, args.output_dir, config)
