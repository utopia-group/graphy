import argparse
import os
import pandas as pd
import subprocess

from pprint import pprint

import lib.dracopy.draco.vis_enum
import random

# parameters used to generate the wikisql dataset
from lib.dracopy.draco import vis_enum

config = {
    "sample_for_each_col_comb": 2,
    "inline_data": False,
    "dataset_url": "https://aqua-data.github.io/data/vega-datasets/",
    "enhance_with_sorting": True,
    "enum_table_filter_num": 0,
    "max_field_comb_size": 2,
    "max_num_field_combs": 256,
    "ignore_used_dataset": True,
    "dataset_limit": 100000
}

table_name = "earthquakes"

random.seed(2020)

parser = argparse.ArgumentParser(description="Draco visualization synthesis.")
parser.add_argument("--input_dir", default="../../vega-datasets", help="Input directory containing tables and fine types.")
parser.add_argument("--output_dir", default="../../vis-viewer/sample-outputs/vega-datasets/", help="The output directory.")
# parser.add_argument("--table_name", default="wikisql_1-11391448-2", help="The table that will be used in enumeration.")
# parser.add_argument("--inline_data", default=True, help="Whether we inline data in the vega-lite script (or alternatively store them separately in a csv file)")

# # enumeration optinos for each tale
# parser.add_argument("--sample_for_each_col_comb", type=int, default=3, help="How many visualization samples do we want to choose for each column combination in a dataset.")
# parser.add_argument("--enhance_with_sorting", default=False, help="Whether we want to add sorting options in visualization enumeration")
# parser.add_argument("--enum_table_filter_num", default=0, help="Whether we want to add random filters to visualization datasets (and how many filters we want to consider for each table)")
# parser.add_argument("--max_field_comb_size", default=2, help="The maximum number of fields we'd consider in each column combination.")
# parser.add_argument("--max_num_field_combs", default=256, help="The number of field combinations we'd consider for one table.")

# # dataset level options
# parser.add_argument("--ignore_used_dataset", default=True, help="If visualizations for one dataset already exist, prevent enumerating for that.")
# parser.add_argument("--dataset_limit", default=None, help="Limit the number of dataset being considered to dataset_limit.")

if __name__ == '__main__':
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    if table_name is None:
        vis_enum.enum_table_visualizations_in_batch(input_dir, output_dir, config)
    else:
        vis_enum.enum_table_visualizations(table_name, input_dir, output_dir, config)
