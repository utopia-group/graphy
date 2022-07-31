"""
csv utils:
- dict <==>(save, read) csv
- given a list of dict data, output a dict of list of data
"""

import csv
import sys
from typing import Dict, List

csv.field_size_limit(sys.maxsize)


def save_dict_to_csv(path, records):

    print('save_dict_to_csv:', path)
    print(len(records))

    if len(records) == 0:
        return

    keys = records[0].keys()
    for r in records:
        if len(r.keys()) > len(keys):
            keys = r.keys()
    # keys = records[0].keys()
    # print("keys:", keys)
    # print("records:", records)
    with open(path, "w", encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(records)


def read_csv_to_dict(path, fieldname=None) -> List[Dict]:
    records = []
    with open(path, 'r', encoding='utf-8') as input_file:
        if fieldname is not None:
            rd = csv.DictReader(input_file, delimiter=',', skipinitialspace=True, fieldnames=fieldname)
            next(rd)
        else:
            rd = csv.DictReader(input_file, delimiter=',', skipinitialspace=True)
        for row in rd:
            records.append(row)
    return records


def convert_list_data_to_dict_data(data: List[Dict], key: str) -> Dict:

    new_data = {}
    for d in data:
        new_data[d[key]] = d

    return new_data