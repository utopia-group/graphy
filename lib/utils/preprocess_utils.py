"""
helper functions for the preprocess.py
"""
from typing import Tuple, List, Dict, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from lib.neural_parser.labels import CLASSES
from lib.neural_parser.tokenizer import Tokenizer
from lib.utils.nlp_utils import standardize_neural


def preprocess_field(field_str):
    return field_str.replace('_', ' ')


def format_single_data(d: Dict, parsed_fields, gt_info, model_name):

    id = d['id']
    data_name = d['data']
    query = standardize_neural(
        d['query']) if d['query-fixed'] == '' else standardize_neural(d['query-fixed'])

    if parsed_fields is not None:
        # print(d['id'])
        # obtain field name through nl2sql and other method
        # NOTE: change field to lower since the roberta model is cased (bert model we use is uncased so it doesn't affect the results)
        if model_name == 'bert-large-uncased':
            parsed_fields = list(set([item[6:-1]
                                 for item in parsed_fields])) + ['null']
        else:
            parsed_fields = list(set([item[6:-1].lower()
                                 for item in parsed_fields])) + ['null']

        if gt_info is not None:

            all_fields, gt = gt_info

            if set(parsed_fields) == set(all_fields):
                return dict(**{'id': id, 'data': data_name, 'query': query, 'fields': all_fields}, **gt)
            else:
                new_gt = {}
                for intent, value in gt.items():
                    if 'field' in intent:
                        field_str = all_fields[value]
                        if field_str in parsed_fields:
                            new_gt[intent] = parsed_fields.index(field_str)
                        else:
                            # if the gt is no longer in the field, assign -1 for now
                            new_gt[intent] = -1
                    else:
                        new_gt[intent] = value

                return dict(**{'id': id, 'data': data_name, 'query': query, 'fields': parsed_fields}, **new_gt)
        else:
            # assign all gt to -1 if there is no gt (just try to be consistent here)
            new_gt = {}
            for intent in CLASSES.keys():
                new_gt[intent] = -1
                new_gt['{}-{}'.format(intent, 'field')] = -1
            return dict(**{'id': id, 'data': data_name, 'query': query, 'fields': parsed_fields}, **new_gt)
    else:
        all_fields, gt = gt_info

        return dict(**{'id': id, 'data': data_name, 'query': query, 'fields': all_fields}, **gt)


def df_data(args, data: List[Dict], tokenizer: Tokenizer) -> DataFrame:

    for d in data:

        # print('d: ', d)

        # tokenize_utterance
        query_tokenized, sep_index = tokenizer.tokenize_with_sep_pos(
            d['query'])
        field_start_index = sep_index + 1

        # first pad the field index
        # dim: max_num_field
        all_fields_masks = [1] * len(d['fields']) + \
            [0] * (args.max_field_num - len(d['fields']))
        d['fields_masks'] = np.array(all_fields_masks)

        # tokenize all_fields
        all_fields_indices = []  # dim: max_num_field * max_token_num_per_field
        all_fields_token_masks = []  # dim: max_num_field * max_token_num_per_field
        all_fields_tokenized = []   # dim: len(tokenized_field) * num_field
        curr_field_token_index = field_start_index

        for i in range(args.max_field_num):
            if i < len(d['fields']):
                field = preprocess_field(d['fields'][i])
                field_tokenized = tokenizer.tokenize(
                    field, new_word=(not i == 0))
                assert len(field_tokenized) <= args.max_token_num_per_field
                curr_field_token_indices = []
                curr_field_token_masks = []
                for ft_i in range(args.max_token_num_per_field):
                    if ft_i < len(field_tokenized):
                        curr_field_token_indices.append(curr_field_token_index)
                        curr_field_token_index += 1
                        curr_field_token_masks.append([1])
                    else:
                        curr_field_token_indices.append(sep_index)
                        curr_field_token_masks.append([0])
                all_fields_tokenized.extend(field_tokenized)
                all_fields_indices.append(curr_field_token_indices)
                all_fields_token_masks.append(curr_field_token_masks)

                assert len(curr_field_token_indices) == len(
                    curr_field_token_masks)

            else:
                all_fields_indices.append(
                    [sep_index] * args.max_token_num_per_field)
                all_fields_token_masks.append(
                    [[0]] * args.max_token_num_per_field)

        d['fields_indices'] = np.array(all_fields_indices)
        d['fields_token_masks'] = np.array(all_fields_token_masks)

        # generate encoding for '[CLS] utterance [SEP] field_tok1 field_tok2 ... [SEP]'
        d['encoding'] = tokenizer.encode_plus(
            d['query'],
            all_fields_tokenized,
            add_special_tokens=True,
            max_length=args.input_dim,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        # verify the generation process is correct
        recovered_tokens = tokenizer.convert_ids_to_tokens(
            d['encoding']['input_ids'][0])
        field_tokens = [field for idx, field in enumerate([tokenizer.recover_origin_field_from_tokens([recovered_tokens[field_i_indices[tok_i]]
                                                                                                       for tok_i, field_i_token_i_mask in enumerate(all_fields_token_masks[field_i])
                                                                                                       if field_i_token_i_mask[0] == 1])
                                                           for field_i, field_i_indices in enumerate(all_fields_indices)]) if all_fields_masks[idx] == 1]
        # print('recovered_tokens:', recovered_tokens)
        # print("all_fields_indices:", all_fields_indices)
        # print("all_fields_token_masks:", all_fields_token_masks)
        # print('field_tokens:', field_tokens)
        # print('gt_token:', [f.lower() for f in d['fields']])

        assert set(field_tokens) == set([f.lower() for f in d['fields']])

    return pd.DataFrame(data)


# def df_data(data: List[Dict], tokenizer) -> DataFrame:

    #     for d in data:

    #         # print('d: ', d)

    #         # tokenize_utterance
    #         query_tokenized = tokenizer.tokenize(d['query'])
    #         field_start_index = len(query_tokenized) + 2
    #         sep_index = len(query_tokenized) + 1

    #         # first pad the field index
    #         # all_fields_masks = [1] * len(d['fields']) + [0] * (args.max_field_num - len(d['fields']))   # dim: max_num_field
    #         all_fields_masks = [1] * len(d['fields'])
    #         d['fields_masks'] = np.array(all_fields_masks)

    #         # tokenize all_fields
    #         all_fields_indices = []  # dim: max_num_field * max_token_num_per_field
    #         all_fields_token_masks = []  # dim: max_num_field * max_token_num_per_field
    #         all_fields_tokenized = []   # dim: len(tokenized_field) * num_field
    #         curr_field_token_index = field_start_index

    #         for i in range(len(d['fields'])):
    #             if i < len(d['fields']):
    #                 field = preprocess_field(d['fields'][i])
    #                 field_tokenized = tokenizer.tokenize(field)
    #                 assert len(field_tokenized) <= args.max_token_num_per_field
    #                 curr_field_token_indices = [curr_field_token_index, (curr_field_token_index + len(field_tokenized))]
    #                 curr_field_token_index += len(field_tokenized)
    #                 curr_field_token_masks = []
    #                 # for ft_i in range(len(field_tokenized)):
    #                 #     if ft_i < len(field_tokenized):
    #                 #         curr_field_token_indices.append(curr_field_token_index)
    #                 #         curr_field_token_index += 1
    #                 #         curr_field_token_masks.append([1])
    #                 #     else:
    #                 #         curr_field_token_indices.append(sep_index)
    #                 #         curr_field_token_masks.append([0])
    #                 all_fields_tokenized.extend(field_tokenized)
    #                 all_fields_indices.append(curr_field_token_indices)
    #                 all_fields_token_masks.append(curr_field_token_masks)

    #                 # assert len(curr_field_token_indices) == len(curr_field_token_masks)

    #             else:
    #                 all_fields_indices.append([sep_index] * args.max_token_num_per_field)
    #                 all_fields_token_masks.append([[0]] * args.max_token_num_per_field)

    #         d['fields_indices'] = np.array(all_fields_indices)
    #         d['fields_token_masks'] = np.array(all_fields_token_masks)

    #         # generate encoding for '[CLS] utterance [SEP] field_tok1 field_tok2 ... [SEP]'
    #         d['encoding'] = tokenizer.encode_plus(
    #             d['query'],
    #             all_fields_tokenized,
    #             add_special_tokens=True,
    #             max_length=args.input_dim,
    #             return_token_type_ids=False,
    #             padding='max_length',
    #             return_attention_mask=True,
    #             return_tensors='pt'
    #         )

    #         # verify the generation process is correct
    #         recovered_tokens = tokenizer.convert_ids_to_tokens(d['encoding']['input_ids'][0])
    #         field_tokens = [field for idx, field in enumerate([recover_origin_field_from_tokens(recovered_tokens[field_i_indices[0]:field_i_indices[1]]) for field_i, field_i_indices in enumerate(all_fields_indices)]) if all_fields_masks[idx] == 1]
    #         # print('recovered_tokens:', recovered_tokens)
    #         # print("all_fields_indices:", all_fields_indices)
    #         # print("all_fields_token_masks:", all_fields_token_masks)
    #         print('field_tokens:', field_tokens)
    #         print('gt_token:', [f.lower() for f in d['fields']])

    #         assert set(field_tokens) == set([f.lower() for f in d['fields']])

    #     return pd.DataFrame(data)


gt_cache = {}


def read_spec_gt(gt_dir, fname: str, model_name:str) -> Tuple[List, dict]:
    def get_spec_type(gt_split, spec_type) -> int:
        if spec_type == 'plot':
            spec = [s for s in gt_split if s.startswith('plot(')][0]
            spec = spec[(spec.find("(") + 1):spec.find(")")].lower()
        else:
            spec = [s for s in gt_split if s.startswith(
                '{}('.format(spec_type))]

            if len(spec) == 0:
                spec = 'null'
            else:
                # spec = spec[0]
                spec = spec_type

        spec = CLASSES[spec_type].index(spec)

        return spec

    def get_spec_type_field(gt_split) -> Tuple[List, List]:

        all_fields = {}
        type_field = {}

        for spec_type in list(CLASSES.keys()) + ['x', 'y']:
            if spec_type == 'plot':
                type_field[spec_type] = 'null'
            else:
                spec = [s for s in gt_split if s.startswith(
                    '{}('.format(spec_type))]

                if len(spec) == 0:
                    type_field[spec_type] = 'null'
                else:
                    spec = spec[0]
                    # NOTE: change field font to lower to handle model issue
                    if model_name == 'bert-large-uncased':
                        spec_field = spec[(spec.find('(') + 1):spec.find(')')]
                    else:
                        spec_field = spec[(spec.find('(') + 1):spec.find(')')].lower()
                    all_fields[spec_field] = ''
                    type_field[spec_type] = spec_field

        all_fields = list(all_fields.keys())
        all_fields.append('null')

        # index everything
        type_field_indexed = [('{}-{}'.format(spec_type, 'field'), all_fields.index(field))
                              for spec_type, field in type_field.items()]

        return all_fields, type_field_indexed

    if fname not in gt_cache:
        with open('{}/{}'.format(gt_dir, fname), 'r') as f:
            gt_str = f.read().strip()
        # print('gt_str:', gt_str)
        gt_split = gt_str.split(',')
        all_intent_spec = [(key, get_spec_type(gt_split, key))
                           for key in CLASSES.keys()]
        all_fields, all_field_spec = get_spec_type_field(gt_split)
        gt_cache[fname] = (all_fields, dict(all_intent_spec + all_field_spec))

    return gt_cache[fname]


def check_benchmark_support(not_support_field):
    return not ('rect' in not_support_field or 'pie' in not_support_field or 'strip' in not_support_field or 'box' in not_support_field or 'table' in not_support_field)


def filter_data(data: List[Dict]) -> List[Dict]:

    if 'not supported' not in data[0]:
        return data

    new_data = [d for d in data if check_benchmark_support(
        d['not supported']) and not('?' in d['labeled'])]
    return new_data
