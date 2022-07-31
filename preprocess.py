from copy import deepcopy

from transformers import BertTokenizer

from lib.eval.eval import EvalEngine
from lib.keyword_parser.derivations import NL4DVDerivations
from lib.keyword_parser.lexicons import NL4DVLexicons
from lib.keyword_parser.parser import QueryKeywordParser
from parse_args import args

import math
import os
import json
import pickle
import random

from collections import namedtuple

import pandas as pd
from pandas import DataFrame

from lib.neural_parser.tokenizer import Tokenizer
from lib.neural_parser.labels import CLASSES
from lib.utils.csv_utils import *
from lib.utils.misc_utils import get_processed_data_path, get_train_val_test_file_name
from lib.utils.benchmark_utils import create_benchmark
from lib.utils.nlp_utils import standardize_neural
from lib.utils.preprocess_utils import read_spec_gt, filter_data, format_single_data, df_data
from lib.utils.vegalite_utils import parse_vl_spec

"""
This is the preprocessing file to generate training data for nvBench and also for our parser
"""


def generate_nv_bench_files():
    random.seed = 233
    benchmark_file = 'eval/benchmarks-chi21.csv'
    model = 'ncnet'  # ncnet, nvbench, bart
    gt_path = 'eval/gt'
    data_path = 'eval/data'
    save_dir = '../nvBench/data/chi21'
    sub_dataset = 'superstore'

    # train_data = 'eval/datavis/train_held_out.txt'
    # test_data = 'eval/datavis/test_held_out.txt'
    # dev_data = 'eval/datavis/val_held_out.txt'
    train_data = 'eval/datavis/train_{}.txt'.format(sub_dataset)
    test_data = 'eval/datavis/test_{}.txt'.format(sub_dataset)
    dev_data = 'eval/datavis/val_{}.txt'.format(sub_dataset)

    nvbench_train = 'eval/nvbench/nvbench_train.csv'
    nvbench_db_table_columns = json.load(open('eval/nvbench/nvbench_db_tables_columns.json', 'r'))

    mix_nvbench_data = False
    nvbench_only = True
    assert (mix_nvbench_data & (model != 'nvbench')) or not mix_nvbench_data

    data_info_cache = {}
    nvbench_info_cache = {}
    gt_cache = {}

    BartDataEntry = namedtuple('BartDataEntry', ['id', 'db_id', 'chart', 'hardness', 'query', 'question', 'question_bart', 'vega_zero'])
    DataEntry = namedtuple('DataEntry', ['id', 'db_id', 'chart', 'hardness', 'query', 'question', 'vega_zero'])

    def read_nvbench_data_info(dataset_name):

        if dataset_name not in nvbench_info_cache:
            datainfo_str = dataset_name + ' '
            datainfo_str += ' '.join([key + ' ' + ' '.join(columns) for key, columns in nvbench_db_table_columns[dataset_name].items()])
            nvbench_info_cache[dataset_name] = datainfo_str

        return nvbench_info_cache[dataset_name]

    def read_data_info(dataset_name):

        if dataset_name not in data_info_cache:
            dataset_folder = dataset_name
            if dataset_folder.lower() == 'movies':
                dataset_folder = 'movies_chi21'
            csv_file = '{}/{}/{}.csv'.format(data_path, dataset_folder, dataset_name.lower())
            csv_dict = read_csv_to_dict(csv_file)
            assert len(csv_dict) > 0
            colnames = [c.lower() for c in csv_dict[0].keys()]
            data_info_cache[dataset_name] = '{} {}'.format(dataset_name.lower(), ' '.join(colnames))

        return data_info_cache[dataset_name]

    def read_gt(gt_filename, file_type):

        key = '{}.{}'.format(gt_filename, file_type)

        if gt_cache.get(key) is None:
            with open('{}/{}.{}'.format(gt_path, gt_filename, file_type)) as f:
                gt = f.read().strip()
            gt_cache[key] = gt

        return gt_cache[key]

    def write_benchmark(benchmark_data, data, mode='train'):

        output_data = []

        for entry in benchmark_data:

            # if entry['dataset'] != sub_dataset:
            #     continue

            if entry['id'] not in data:
                continue

            if entry['query-fixed'] == "":
                query = entry['query']
            else:
                query = entry['query-fixed']

            if 'Horespower' in query:
                query = query.replace('Horespower', 'Horsepower')

            if model == 'bart':
                # here we need to prepare the dataset information, with the following format:
                # dbname column_names <SEP> NL query
                query_bart = '{} <SEP> {}'.format(read_data_info(entry['data']), query)
            else:
                query_bart = ""

            vql = read_gt(entry['gtname'], 'vql')
            vega_zero = read_gt(entry['gtname'], 'vegazero')

            if 'scatter' in entry['visId'].lower() or 'histogram' in entry['visId'].lower():
                chart = 'Scatter'
            elif 'bar' in entry['visId'].lower():
                chart = 'Bar'
            elif 'line' in entry['visId'].lower():
                chart = 'Line'
            else:
                raise Exception

            if entry['data'].lower() == 'movies':
                db_id = 'movies_chi21'
            else:
                db_id = entry['data'].lower()

            if model == 'bart':
                output_data.append(BartDataEntry(entry['id'], db_id, chart, 'Easy', vql, query, query_bart, vega_zero)._asdict())
            else:
                output_data.append(DataEntry(entry['id'], db_id, chart, 'Easy', vql, query, vega_zero)._asdict())

        if (mix_nvbench_data or nvbench_only) and mode == 'train':
            # first duplicate the chi training data 10 times
            duplicated_train_data = []

            if mix_nvbench_data:
                for i in range(10):
                    for d in output_data:
                        new_d = deepcopy(d)
                        new_d['id'] = '{}_{}'.format(d['id'], i)

                        duplicated_train_data.append(new_d)

            nvbench_data = read_csv_to_dict(nvbench_train)
            # clean nvbench data
            new_nvbench_data = []
            for i, d in enumerate(nvbench_data):
                new_d = {'id': 'nvbench_{}'.format(i)}
                filter_old_d = dict([(key, value) for key, value in d.items() if key != 'tvBench_id'])
                new_d.update(filter_old_d)
                # d['id'] = 'nvbench_{}'.format(i)
                # del d['tvBench_id']
                if model == 'bart':
                    new_d['question_bart'] = '{} <SEP> {}'.format(read_nvbench_data_info(d['db_id']), d['question'])
                new_nvbench_data.append(new_d)
            output_data = new_nvbench_data + duplicated_train_data
            # shuffle the data
            random.shuffle(output_data)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if model == 'bart':
            if mix_nvbench_data:
                save_dict_to_csv('{}/chi21_nvbench_{}_{}_{}_flat.txt'.format(save_dir, sub_dataset, mode, model), output_data)
            else:
                save_dict_to_csv('{}/chi21_{}_{}_{}_flat.txt'.format(save_dir, sub_dataset, mode, model), output_data)
        else:
            if mix_nvbench_data:
                save_dict_to_csv('{}/chi21_nvbench_{}_{}_flat.txt'.format(save_dir, sub_dataset, mode), output_data)
            elif nvbench_only:
                save_dict_to_csv('{}/chi21_nvbench_only_{}_{}_flat.txt'.format(save_dir, sub_dataset, mode), output_data)
            else:
                save_dict_to_csv('{}/chi21_{}_{}_flat.txt'.format(save_dir, sub_dataset, mode), output_data)

    benchmark_data = read_csv_to_dict(benchmark_file)

    train_data = convert_list_data_to_dict_data(read_csv_to_dict(train_data), 'id')
    test_data = convert_list_data_to_dict_data(read_csv_to_dict(test_data), 'id')
    dev_data = convert_list_data_to_dict_data(read_csv_to_dict(dev_data), 'id')

    write_benchmark(benchmark_data, train_data, 'train')
    write_benchmark(benchmark_data, test_data, 'test')
    write_benchmark(benchmark_data, dev_data, 'dev')


def generate_spec_ground_truth():
    gt_dir = 'eval/gt'

    for root, _, files in os.walk(gt_dir):
        for fn in files:
            if not fn.endswith('vl.json'):
                continue
            fn_new_name = "{}.spec".format(fn.split('.')[0])
            fpath = os.path.join(root, fn)

            print("translating {}...".format(fn))

            with open(fpath, 'r') as f:
                vl_spec = json.load(f)

            print("vl_spec:", vl_spec)

            parsed_spec = parse_vl_spec(vl_spec)

            with open("{}/{}".format(gt_dir, fn_new_name), 'w+') as f:
                f.write(','.join(parsed_spec))


def generate_training_data(args):
    eval_dir = 'eval'
    gt_dir = '{}/{}'.format(eval_dir, 'gt')
    output_dir = '{}/{}'.format(eval_dir, 'datavis')

    field_correct = 0
    field_containment = 0

    qparser = QueryKeywordParser(NL4DVLexicons(), NL4DVDerivations())

    def generate_formatted_data(data: List[Dict], oracle_field_test=True) -> List[Dict]:

        nonlocal field_correct, field_containment

        formatted_data = []
        for d in data:
            gtname = '{}.spec'.format(d['gtname']) if 'gtname' in d else '{}-{}.spec'.format(d['data'], d['id'])
            all_fields, gt = read_spec_gt(gt_dir, gtname, args.bert_model)

            if not oracle_field_test:

                benchmark = create_benchmark(d, 'chi21')
                benchmark.data, _, _ = EvalEngine().get_data(benchmark, mode='synth')
                parsed_fields = qparser.parse(benchmark, field_only=True)

                parsed_fields_tmp = [item[6:-1] for item in parsed_fields]
                all_fields_tmp = all_fields[:-1]
                print("parsed_fields:", parsed_fields_tmp)
                print("all_fields:", all_fields_tmp)
                # assert False

                if set(parsed_fields_tmp) == set(all_fields_tmp):
                    field_correct += 1
                if set(all_fields_tmp).issubset(set(parsed_fields_tmp)):
                    field_containment += 1

                formatted_data.append(format_single_data(d, parsed_fields, (all_fields, gt), model_name=args.bert_model))

            else:
                formatted_data.append(format_single_data(d, None, (all_fields, gt), model_name=args.bert_model))

        return formatted_data

    # read two dataset
    chi_21_data = filter_data(read_csv_to_dict('{}/benchmarks-chi21.csv'.format(eval_dir)))  # 75-25 split
    nl4dv_data = filter_data(read_csv_to_dict('{}/benchmarks-nl4dv.csv'.format(eval_dir)))  # all of them use for training

    print('filtered chi21 size: ', len(chi_21_data))
    print('filtered nl4dv size: ', len(nl4dv_data))

    # split chi_21_data
    if args.held_out:
        # all_data = chi_21_data + nl4dv_data

        # filter out car dataset as the validation + train data
        val_test = []
        train = []
        for d in chi_21_data:
            if d['data'].lower() == args.test_set:
                val_test.append(d)
            else:
                train.append(d)

        for d in nl4dv_data:
            if not d['data'].lower() == args.test_set:
                train.append(d)

        # shuffle the data
        random.shuffle(val_test)
        random.shuffle(train)

        # pick out 10% of car dataset for validation
        val_len = math.floor(len(val_test) * 0.1)
        val = val_test[:val_len]
        test = val_test[(val_len + 1):]
    else:
        random.shuffle(chi_21_data)
        train_len = math.floor(len(chi_21_data) * 0.75)
        chi_21_train = chi_21_data[:train_len]

        train_val = chi_21_train + nl4dv_data

        random.shuffle(train_val)
        train_len = math.floor(len(train_val) * 0.95)

        train = train_val[:train_len]
        val = train_val[(train_len + 1):]
        test = chi_21_data[(train_len + 1):]

    print("train size: ", len(train))
    print("val size: ", len(val))
    print('test size: ', len(test))

    # read gt for each instance and create a new data to store at the output dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    train_file_name, val_file_name, test_file_name = get_train_val_test_file_name(args)
    train_formatted = generate_formatted_data(train, oracle_field_test=True)
    save_dict_to_csv('{}/{}'.format(output_dir, train_file_name), train_formatted)
    val_formatted = generate_formatted_data(val, oracle_field_test=True)
    save_dict_to_csv('{}/{}'.format(output_dir, val_file_name), val_formatted)
    test_formatted = generate_formatted_data(test, oracle_field_test=args.oracle_field_test)
    save_dict_to_csv('{}/{}'.format(output_dir, test_file_name), test_formatted)

    if not args.oracle_field_test:
        print("field_prediction acc:", field_correct / len(test_formatted))
        print("field_prediction containment acc:", field_containment / len(test_formatted))

    # prepare pickle file of dataframe to be read directly
    tokenizer = Tokenizer(args.bert_model)
    dataset = {'train': df_data(args, train_formatted, tokenizer),
               'val': df_data(args, val_formatted, tokenizer),
               'test': df_data(args, test_formatted, tokenizer)}

    out_pkl = get_processed_data_path(args)
    with open(out_pkl, 'wb') as f:
        pickle.dump(dataset, f)
        print('Processed data dumped to {}'.format(out_pkl))

    # synth_eval_engine.save_cache()


def generate_training_data_prev(args):
    eval_dir = 'eval'
    gt_dir = '{}/{}'.format(eval_dir, 'gt')
    output_dir = '{}/{}'.format(eval_dir, 'datavis')

    gt_cache = {}

    def read_spec_gt(fname: str) -> dict:

        def get_spec_type(gt_split, spec_type):
            if spec_type == 'plot':
                spec = [s for s in gt_split if s.startswith('plot(')][0]
                spec = spec[(spec.find("(") + 1):spec.find(")")].lower()
            else:
                spec = [s for s in gt_split if s.startswith('{}('.format(spec_type))]
                if len(spec) == 0:
                    spec = 'null'
                else:
                    spec = spec_type

            spec = CLASSES[spec_type].index(spec)

            return spec

        # print('fname:', fname)
        if fname not in gt_cache:
            with open('{}/{}'.format(gt_dir, fname), 'r') as f:
                gt_str = f.read().strip()
            # print('gt_str:', gt_str)
            gt_split = gt_str.split(',')
            all_spec = dict([(key, get_spec_type(gt_split, key)) for key in CLASSES.keys()])

            gt_cache[fname] = all_spec

        return gt_cache[fname]

    def filter_data(data: List[Dict]) -> List[Dict]:

        if 'not supported' not in data[0]:
            return data

        new_data = [d for d in data if not (
                'rect' in d['not supported'] or 'pie' in d['not supported'] or 'strip' in d['not supported'] or 'box' in d['not supported'] or 'table' in d['not supported'] or '?' in d[
            'labeled'])]
        return new_data

    def generate_formatted_data(data: List[Dict]) -> List[Dict]:

        formatted_data = []
        for d in data:
            # print(d)
            id = d['id']
            data_name = d['data']
            query = standardize_neural(d['query']) if d['query-fixed'] == '' else standardize_neural(d['query-fixed'])
            gtname = '{}.spec'.format(d['gtname']) if 'gtname' in d else '{}-{}.spec'.format(d['data'], d['id'])
            gt = read_spec_gt(gtname)
            formatted_data.append(dict(**{'id': id, 'data': data_name, 'query': query}, **gt))

        return formatted_data

    def df_data(data: List[Dict], tokenizer) -> DataFrame:
        for d in data:
            d['encoding'] = tokenizer.encode_plus(
                d['query'],
                add_special_tokens=True,
                max_length=args.input_dim,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )

        return pd.DataFrame(data)

    # read two dataset
    chi_21_data = filter_data(read_csv_to_dict('{}/benchmarks-chi21.csv'.format(eval_dir)))  # 75-25 split
    nl4dv_data = filter_data(read_csv_to_dict('{}/benchmarks-nl4dv.csv'.format(eval_dir)))  # all of them use for training

    print('filtered chi21 size: ', len(chi_21_data))
    print('filtered nl4dv size: ', len(nl4dv_data))

    # split chi_21_data
    if args.held_out:
        # all_data = chi_21_data + nl4dv_data

        # filter out car dataset as the validation + train data
        val_test = []
        train = []
        for d in chi_21_data:
            if d['data'].lower() == 'cars':
                val_test.append(d)
            else:
                train.append(d)

        for d in nl4dv_data:
            if not d['data'].lower() == 'cars':
                train.append(d)

        # shuffle the data
        random.shuffle(val_test)
        random.shuffle(train)

        # pick out 10% of car dataset for validation
        val_len = math.floor(len(val_test) * 0.1)
        val = val_test[:val_len]
        test = val_test[(val_len + 1):]
    else:
        random.shuffle(chi_21_data)
        train_len = math.floor(len(chi_21_data) * 0.75)
        chi_21_train = chi_21_data[:train_len]

        train_val = chi_21_train + nl4dv_data

        random.shuffle(train_val)
        train_len = math.floor(len(train_val) * 0.95)

        train = train_val[:train_len]
        val = train_val[(train_len + 1):]
        test = chi_21_data[(train_len + 1):]

    print("train size: ", len(train))
    print("val size: ", len(val))
    print('test size: ', len(test))

    # read gt for each instance and create a new data to store at the output dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    train_file_name, val_file_name, test_file_name = get_train_val_test_file_name(args)
    train_formatted = generate_formatted_data(train)
    save_dict_to_csv('{}/{}'.format(output_dir, train_file_name), train_formatted)
    val_formatted = generate_formatted_data(val)
    save_dict_to_csv('{}/{}'.format(output_dir, val_file_name), val_formatted)
    test_formatted = generate_formatted_data(test)
    save_dict_to_csv('{}/{}'.format(output_dir, test_file_name), test_formatted)

    # prepare pickle file of dataframe to be read directly
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    dataset = {'train': df_data(train_formatted, tokenizer),
               'val': df_data(val_formatted, tokenizer),
               'test': df_data(test_formatted, tokenizer)}

    out_pkl = get_processed_data_path(args)
    with open(out_pkl, 'wb') as f:
        pickle.dump(dataset, f)
        print('Processed data dumped to {}'.format(out_pkl))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='dummy parser')
    # args = parser.parse_args()
    # args.seed = 233
    # args.bert_model = 'bert-large-uncased'
    # args.input_dim = 100
    # args.dataset = 'datavis'
    # args.data_path = 'eval/datavis'
    # args.held_out = True
    # args.max_field_num = 10
    # args.max_token_num_per_field = 5
    # args.oracle_field_test = False

    print(args)

    # random.seed(args.seed)
    generate_nv_bench_files()
    # generate_spec_ground_truth()
    # generate_training_data(args)
