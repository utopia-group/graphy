"""
utils for nl2sql parser, include:
- filter invalid predicate
"""
import numbers

from lib.eval.benchmark import Benchmark


def filter_invalid_pred(benchmark: Benchmark, pred: dict) -> dict:
    print('pred:', pred)

    if pred is None:
        return pred

    if not isinstance(pred, dict):
        return None

    assert len(pred) == 1
    top_op, values = list(pred.items())[0]
    if top_op == 'and':
        return pred

    if isinstance(values, list):
        if not isinstance(values[0], str) or not (isinstance(values[1], str) or isinstance(values[1],
                                                                                        numbers.Number)):
            return None

        field_name = values[0].split('.')[1]
    else:
        assert isinstance(values, str)
        field_name = values.split('.')[1]

    if benchmark.data.field_to_type_mapping[field_name] == 'number' and isinstance(values[1], str):
        return None
    if benchmark.data.field_to_type_mapping[field_name] == 'text' and \
            len(benchmark.data.field_to_val_mapping[field_name]) < 10:
        if isinstance(values[1], str):
            if not values[1].lower() in benchmark.data.field_to_val_mapping[field_name]:
                return None
        else:
            return None

    if isinstance(values[1], str):
        values[1] = values[1].replace("VALUE", '').strip()
        if values[1] == '':
            return None

    return {top_op: values}