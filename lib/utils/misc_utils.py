"""
utility functions for those don't belong anywhere else, including:
- print probabilities
- get file signatures for all the neural part
- create a benchmark object (which probably doesn't belong here)
"""

import os
from typing import List

from lib.config.config import DEBUG_PRINT
from itertools import chain, combinations


def list_hash(l: List[str]) -> int:
    """
    Generate hash code for list of strings to guarantee that
    lists with the same elements have the same hash code
    """
    return sum([hash(e) for e in l])

def powerset(iterable):
    s = list(iterable)
    x = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
    return [list(item) for item in list(x) if len(item) <= 3]


def printd(*args):
    if DEBUG_PRINT:
        print(*args)


def printc(c, *args):
    if c:
        print(*args)


# def print_probs(probs, class_type='plot', field_vocab=None):
#     # probs = list(np.around(np.array(probs[0].cpu()), 2))
#     probs = np.array(probs[0].cpu())
#     log_probs = np.log(probs)
#     if field_vocab is None:
#         output_dict = {label: prob for prob, label in zip(list(probs), CLASSES[class_type])}
#         output_str = ','.join(['{}({:.5f})'.format(label, prob) for prob, label in zip(list(probs), CLASSES[class_type])])
#     else:
#         output_dict = {label: prob for prob, label in zip(list(probs), field_vocab)}
#         output_str = ','.join(['{}({:.5f})'.format(label, prob) for prob, label in zip(list(probs), field_vocab)])
#     return output_str, output_dict


def get_data_signature(args) -> str:

    base_signature = '{}.{}.{}.{}.{}.{}.{}.'.format(args.dataset, args.test_set, 'query', args.bert_model, str(args.input_dim), str(args.max_field_num), str(args.max_token_num_per_field))

    if args.held_out:
        base_signature = '{}{}.'.format(base_signature, 'held-out')

    if args.oracle_field_test:
        base_signature = '{}{}.'.format(base_signature, 'oracle-field-test')

    return base_signature


def get_model_signature(args) -> str:
    if args.held_out:
        return '{}.{}.{}.{}.{}.{}.{}.{}.p{}.{}.{}.cr{}.{}.{}.cn{}.{}.{}.ct{}.{}.{}.m{}.{}.{}.s{}.{}.{}.{}'.format(args.dataset, args.test_set, args.bert_model, str(args.batch_size), str(args.input_dim), str(args.hidden_size), str(args.max_field_num), str(args.max_token_num_per_field), str(args.model_params["plot_epochs"]), str(args.model_params["plot_lr"]), str(args.model_params["plot_wr"]), str(args.model_params["color_epochs"]), str(args.model_params["color_lr"]), str(args.model_params["color_wr"]), str(args.model_params["column_epochs"]), str(args.model_params["column_lr"]), str(args.model_params["column_wr"]), str(args.model_params["count_epochs"]), str(args.model_params["count_lr"]), str(args.model_params["count_wr"]), str(args.model_params["mean_epochs"]), str(args.model_params["mean_lr"]), str(args.model_params["mean_wr"]), str(args.model_params["sum_epochs"]), str(args.model_params["sum_lr"]), str(args.model_params["sum_wr"]), 'held_out')
        
    else:
        return '{}.{}.{}.{}.{}.{}.{}.{}.p{}.{}.{}.cr{}.{}.{}.cn{}.{}.{}.ct{}.{}.{}.m{}.{}.{}.s{}.{}.{}'.format(args.dataset, args.test_set, args.bert_model, str(args.batch_size), str(args.input_dim), str(args.hidden_size), str(args.max_field_num), str(args.max_token_num_per_field), str(args.model_params["plot_epochs"]), str(args.model_params["plot_lr"]), str(args.model_params["plot_wr"]), str(args.model_params["color_epochs"]), str(args.model_params["color_lr"]), str(args.model_params["color_wr"]), str(args.model_params["column_epochs"]), str(args.model_params["column_lr"]), str(args.model_params["column_wr"]), str(args.model_params["count_epochs"]), str(args.model_params["count_lr"]), str(args.model_params["count_wr"]), str(args.model_params["mean_epochs"]), str(args.model_params["mean_lr"]), str(args.model_params["mean_wr"]), str(args.model_params["sum_epochs"]), str(args.model_params["sum_lr"]), str(args.model_params["sum_wr"]))


def get_processed_data_path(args):
    data_sig = get_data_signature(args)
    return os.path.join(args.data_path, '{}pkl'.format(data_sig))


def get_model_dir(args):
    return 'model/{}'.format(get_model_signature(args))


def get_model_path(gt_type, args, best=True):
    if best:
        return os.path.join('{}/{}-best-model-state.bin'.format(get_model_dir(args), gt_type))
    else:
        if args.model_id is None:
            raise ValueError
        return os.path.join('{}/{}-{}.bin'.format(get_model_dir(args), gt_type, args.model_id))
    

def get_saved_prediction_pkl_path(args):
    if args.ts is None:
        prediction_dict_save_fname = 'inference_temp'
    else:
        prediction_dict_save_fname = 'inference_{}'.format(args.ts)

    full_path = '{}/{}.pkl'.format(get_model_dir(args), prediction_dict_save_fname)

    return full_path


def get_train_val_test_file_name(args):
    base_train, base_val, base_test = 'train', 'val', 'test'

    if args.held_out:
        base_train, base_val, base_test = '{}_{}'.format(base_train, args.test_set),'{}_{}'.format(base_val, args.test_set), '{}_{}'.format(base_test, args.test_set),

    if args.oracle_field_test:
        base_test = '{}_oracle_field'.format(base_test)

    return '{}.txt'.format(base_train), '{}.txt'.format(base_val), '{}.txt'.format(base_test)
