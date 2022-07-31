import itertools
from typing import List, Tuple

from lib.eval.benchmark import Dataset
from lib.prev_synthesizer.spec import SpecProg
from lib.utils.enum_utils import get_fields_with_encoding

aggregate_intent = ['mean', 'count', 'sum']
visual_intent = ['color', 'column']


# TODO: we actually need a worklist here
# I am implement a dumb one here
def complete_vis(spec_prog: SpecProg, fields: List, data: Dataset) -> Tuple[List, List, List]:

    all_progs = []

    # print("spec_prog: ", repr(spec_prog))
    # print("fields: ", fields)
    # print("data properties: ", data.properties)

    current_field_with_encoding, current_null_field_encoding = get_fields_with_encoding(spec_prog.pred)
    # print(current_field_with_encoding)

    complete_spec = dict([(intent, field) for field, intents in current_field_with_encoding.items() for intent in intents])
    complete_spec['plot'] = spec_prog.pred['plot'][0]

    all_possible_spec = []

    # let's hallucinate
    # NOTE: find each category variable and add it as the field to encoding
    categorical_vars = data.get_categorical()
    current_null_field_encoding = list(current_null_field_encoding)
    if len(current_null_field_encoding) == 0:
        all_possible_spec.append(complete_spec)
    elif len(current_null_field_encoding) == 1:
        intent = current_null_field_encoding[0]
        for f in categorical_vars:
            new_spec = complete_spec.copy()
            new_spec[intent] = f
            all_possible_spec.append(new_spec)
    elif len(current_null_field_encoding) == 2:
        intent1 = current_null_field_encoding[0]
        intent2 = current_null_field_encoding[1]
        if categorical_vars == 1:
            pass
        else:
            for v1, v2 in itertools.combinations(categorical_vars, 2):
                new_spec1 = complete_spec.copy()
                new_spec1[intent1] = v1
                new_spec1[intent2] = v2
                all_possible_spec.append(new_spec1)
                new_spec2 = complete_spec.copy()
                new_spec2[intent1] = v2
                new_spec2[intent2] = v1
                all_possible_spec.append(new_spec2)

    # assign x y field
    for spec in all_possible_spec:
        # we need to deal with 'count' more cautiously
        if 'count' in complete_spec:
            count_field = complete_spec['count']

            spec['x'] = count_field
            # spec['y'] = count_field

            all_progs.append(['{}({})'.format(intent, field) for intent, field in spec.items()])

        else:
            for field1, field2 in itertools.combinations(fields, 2):
                if current_field_with_encoding.get(field1) in aggregate_intent:
                    spec['y'] = field1
                    spec['x'] = field2
                elif current_field_with_encoding.get(field2) in aggregate_intent:
                    spec['y'] = field2
                    spec['x'] = field1
                elif data.datatype_properties[field1]['type-f'] == 'date':
                    spec['y'] = field1
                    spec['x'] = field2
                elif data.datatype_properties[field2]['type-f'] == 'date':
                    spec['y'] = field2
                    spec['x'] = field1
                else:
                    spec['x'] = field1
                    spec['y'] = field2

                all_progs.append(['{}({})'.format(intent, field) for intent, field in spec.items()])

    # print("all_progs:", all_progs)

    not_allowed_aggregation = [e for e in aggregate_intent if e not in complete_spec]
    not_allowed_enc = [e for e in visual_intent if not any([e in s for s in spec])]

    return all_progs, not_allowed_aggregation, not_allowed_enc
