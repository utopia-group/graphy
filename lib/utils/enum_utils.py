"""
utils function for naive prev_synthesizer (synth + draco, Enumerator object) version
it generates p(intent) * p(field) and sort the list
"""

import time
from collections import defaultdict
from typing import Tuple, Dict

from lib.utils.eval_utils import parse_single_spec


def test_enumerators(enum1, enum2, processed_prediction):

    start = time.time()
    enumerated_list = enum1.enumerate(processed_prediction, limit=-1)
    end1 = time.time()
    enumerated_list_2 = enum2.enumerate_slow(processed_prediction, limit=-1)
    end2 = time.time()

    for prog_1, prog_2 in zip(enumerated_list, enumerated_list_2):
        print(">> enumerated list")
        print(repr(prog_1))
        print("<< enumerated list 2")
        print(repr(prog_2))

        print((repr(prog_1).split(',')[0] == repr(prog_2).split(',')[0]))

    print("total length 1:", len(enumerated_list))
    print("total length 2:", len(enumerated_list_2))

    print("time 1: ", end1 - start)
    print("time 2: ", end2 - end1)


# sort each intent
# field * intent
def process_raw_prediction_for_bid(test_predictions: dict, bid: str):

    processed_prediction = {}

    for class_type, preds in test_predictions[bid].items():
        if class_type == 'field':
            continue

        if class_type == 'plot':
            preds = list(preds['intent'].items())
            preds.sort(key=lambda v: v[1], reverse=True)
            processed_prediction[class_type] = preds
        else:
            new_preds = []
            # print("preds:", preds)
            for intent, prob_i in preds['intent'].items():
                for field, prob_f in preds['field'].items():
                    if intent == 'null' and not field == 'null':
                        continue
                    new_preds.append(('{}({})'.format(intent, field), prob_i * prob_f))
            new_preds.sort(key=lambda v: v[1], reverse=True)
            processed_prediction[class_type] = new_preds

    return processed_prediction


def get_pred_and_diff(process_pred, class_type):
    preds = process_pred[class_type]
    diff = [preds[i - 1][1] - preds[i][1] for i in range(1, len(preds))]

    return preds, diff


def get_score(spec_prog):
    """
    return the score of the spec_prog
    """
    return -spec_prog.score


def get_fields_with_encoding(preds) -> Tuple[Dict, Dict]:

    field_to_encoding_map = defaultdict(list)
    encoding_to_null_map = {}
    preds = list([v[0] for v in preds.values()])[1:]

    # print("preds:", preds)

    for pred in preds:
        if pred.startswith('null'):
            continue

        intent, field = parse_single_spec(pred)

        if not intent == 'null' and not field == 'null':
            field_to_encoding_map[field].append(intent)
        elif not intent == 'null' and field == 'null':
            if intent == 'color' or intent == 'column':
                encoding_to_null_map[intent] = None

    return field_to_encoding_map, encoding_to_null_map