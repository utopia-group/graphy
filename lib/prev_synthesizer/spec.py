import hashlib
import itertools
from typing import List

from lib.eval.benchmark import InputDataset
from lib.type.predicate import Binding, Prov
from lib.type.ref_type import BaseRefType
from lib.type.type_system import get_base_type, create_cnf_formula_instance
from lib.utils.enum_utils import get_fields_with_encoding

aggregate_intents = ['mean', 'count', 'sum']
visual_intents = ['color', 'column', 'x', 'y']
visual_intents_no_x_y = ['color', 'column']


def prog_hash(pred: dict):
    s = str([p[0] for p in pred.values()])
    hs = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    return hs


class SpecProg:
    def __init__(self, pred=None, score=None, spec=None):
        self.pred: dict = pred
        self.score: float = score
        self.spec: dict = spec

    def instantiate_partial_spec(self, fields, data):
        current_field_with_encoding, current_null_field_encoding = get_fields_with_encoding(self.pred)

        # print("curr field with enc:", current_field_with_encoding)
        # print("curr null field with enc:", current_null_field_encoding)

        complete_spec = dict([(intent, field) for field, intents in current_field_with_encoding.items() for intent in intents])

        # print("complete_spec:", complete_spec)

        # i think we should create all the possible spec first lol
        enumerated_spec_temp = []

        # visual encoding hallucinate, by considering all the categorical variables in the dataset
        # TODO: we should modify in the future so that the hallucinate part happen at the synthesizer stage other than the spec parsing stage. I am just put this here to save time.
        #         In the current implementation we are ignoring all the aggregation part that needs to be hallucinate because it is difficult
        #        we just leave this as an option and let the synthesizer to figure out

        categorical_vars = data.get_categorical()
        current_null_field_encoding = list(current_null_field_encoding)

        # print("categorical_vars:", categorical_vars)

        if len(current_null_field_encoding) == 0:
            """
            nothing to hallucinate
            """
            enumerated_spec_temp.append(complete_spec)
        elif len(current_null_field_encoding) == 1:
            """
            one field need to hallucinate
            if there is one field to hallucinate, prioritize it with the colname in the prediction is there exists 
            otherwise enumerate the category field in the dataset 
            """

            intent = current_null_field_encoding[0]
            # first figure out what are the category fields in the predicated colnames that has not been used
            if intent == 'column' or intent == 'color':
                filter_field = complete_spec.get('color') if intent == 'column' else complete_spec.get('column')
                if filter_field is not None:
                    categorical_vars_in_predictions = [f for f in complete_spec.values() if f in categorical_vars and f != filter_field]

                    hallucinate_fields = categorical_vars_in_predictions if len(categorical_vars_in_predictions) > 0 else [f for f in fields if f in categorical_vars and f != filter_field]
                else:
                    categorical_vars_in_predictions = [f for f in complete_spec.values() if f in categorical_vars]
                    hallucinate_fields = categorical_vars_in_predictions if len(categorical_vars_in_predictions) > 0 else [f for f in fields if f in categorical_vars]
            else:
                categorical_vars_in_prediction = [f for f in complete_spec.values() if f in categorical_vars]
                hallucinate_fields = categorical_vars_in_prediction if len(categorical_vars_in_prediction) > 0 else [f for f in fields if f in categorical_vars]

            # categorical_vars_in_prediction = [f for f in fields if f in categorical_vars]
            # categorical_vars_in_prediction = []

            # print('hallucinate_fields:', hallucinate_fields)

            for f in hallucinate_fields:
                new_spec = complete_spec.copy()
                new_spec[intent] = f
                enumerated_spec_temp.append(new_spec)

        elif len(current_null_field_encoding) == 2:
            """
            at max we have 2 field to hallucinate
            here we are making sure that there is no duplicate
            """
            intent1 = current_null_field_encoding[0]
            intent2 = current_null_field_encoding[1]
            categorical_vars_in_prediction = [f for f in complete_spec.values() if f in categorical_vars]
            hallucinate_fields = categorical_vars_in_prediction if len(categorical_vars_in_prediction) > 0 else [f for f in fields if f in categorical_vars]
            if categorical_vars == 1:
                pass
            else:
                for v1, v2 in itertools.combinations(hallucinate_fields, 2):
                    new_spec1 = complete_spec.copy()
                    new_spec1[intent1] = v1
                    new_spec1[intent2] = v2
                    enumerated_spec_temp.append(new_spec1)
                    new_spec2 = complete_spec.copy()
                    new_spec2[intent1] = v2
                    new_spec2[intent2] = v1
                    enumerated_spec_temp.append(new_spec2)

        enumerated_spec = []

        # print(" self.pred['plot'][0]:",  self.pred['plot'][0])

        # assign x y field
        # TODO: we need a better strategy here, too much possibility get enumerated
        """
        if there exists a aggregation enc => make that field y
        if there exists a date var => make that field y 
        if there exists a color/column enc => don't make that column x (unless there is no option) 
        """
        for spec in enumerated_spec_temp:

            used_fields = [f for i, f in spec.items() if i == 'color' or i == 'column']
            available_fields = [f for f in fields if f not in used_fields]

            # print("used_fields:", used_fields)
            # print("available_fields:", available_fields)

            if 'count' in complete_spec:

                """
                if we know the count field, we know the x and y values
                since we finished hallucination at this point, if there is more available fields left, that's probably meaningless
                suppose we have another 2 more field on unuse, it is very unlikely that it is a histogram 
                TODO: i think we really have to find some way to encode this during synthesis other than me enumerating all the heuristics here 
                """

                if len(available_fields) <= 2:
                    new_spec = spec.copy()
                    count_field = complete_spec['count']

                    new_spec['x'] = count_field
                    # new_spec['y'] = count_field

                    enumerated_spec.append(new_spec)

            else:
                # need to first compute the free variable (anything other than the color and column enc) to be assigned

                def assign_x_y_field(_spec, _field1, _field2):
                    _new_spec = _spec.copy()
                    # print('field1:', field1)
                    # print('field2:', field2)
                    if current_field_with_encoding.get(_field1) is not None and any(ai in current_field_with_encoding.get(_field1) for ai in aggregate_intents):
                        _new_spec['y'] = _field1
                        _new_spec['x'] = _field2
                    elif current_field_with_encoding.get(_field2) is not None and any(ai in current_field_with_encoding.get(_field2) for ai in aggregate_intents):
                        _new_spec['y'] = _field2
                        _new_spec['x'] = _field1
                    elif data.datatype_properties[_field1]['type-f'] == 'date':
                        _new_spec['y'] = _field1
                        _new_spec['x'] = _field2
                    elif data.datatype_properties[_field2]['type-f'] == 'date':
                        _new_spec['x'] = _field2
                        _new_spec['y'] = _field1
                    else:
                        if 'color' in spec and (spec['color'] == _field1 or spec['color'] == _field2):
                            _new_spec['x'] = _field1 if _field1 == spec['color'] else _field2
                            _new_spec['y'] = _field1 if _field1 != spec['color'] else _field2
                        else:
                            _new_spec['x'] = _field1
                            _new_spec['y'] = _field2
                    return _new_spec

                if len(available_fields) == 0:
                    # print("spec:", spec)
                    # raise NotImplementedError
                    pass
                elif len(available_fields) == 1:
                    curr_field = available_fields[0]

                    if 'color' in spec:
                        # if we have to choose one existing enc for x and color is present, this is always the best choice
                        new_spec = spec.copy()
                        new_spec['y'] = curr_field
                        new_spec['x'] = spec['color']

                        enumerated_spec.append(new_spec)
                    else:
                        for x in used_fields:
                            new_spec = spec.copy()
                            new_spec['y'] = curr_field
                            new_spec['x'] = x
                            enumerated_spec.append(new_spec)

                elif len(available_fields) == 2:
                    enumerated_spec.append(assign_x_y_field(spec, available_fields[0], available_fields[1]))

                    if 'color' in spec and (data.datatype_properties[available_fields[0]]['type-f'] == 'string' or data.datatype_properties[available_fields[1]]['type-f'] == 'string'):
                        # sometimes we might prefer use color other than the actual x if it is a nominal var
                        enumerated_spec.append(assign_x_y_field(spec, available_fields[0], spec['color']))
                        enumerated_spec.append(assign_x_y_field(spec, available_fields[1], spec['color']))
                else:
                    for field1, field2 in itertools.combinations(available_fields, 2):
                        enumerated_spec.append(assign_x_y_field(spec, field1, field2))

                # print(current_field_with_encoding)
        # print(" self.pred['plot'][0]:",  self.pred['plot'][0])
        # print("enumerated_spec:", enumerated_spec)
        return enumerated_spec

    def get_draco_spec(self, fields: List, data: InputDataset):

        all_progs = []

        current_field_with_encoding, current_null_field_encoding = get_fields_with_encoding(self.pred)
        complete_spec = dict([(intent, field) for field, intents in current_field_with_encoding.items() for intent in intents])

        enumerated_spec = self.instantiate_partial_spec(fields, data)

        for spec in enumerated_spec:
            all_progs.append(['{}({})'.format(intent, field) for intent, field in spec.items()])

        if len(all_progs) == 0:
            return [], None, None

        # print(all_progs)

        not_allowed_aggregation = [e for e in aggregate_intents if e not in complete_spec]
        not_allowed_enc = [e for e in visual_intents_no_x_y if not any([e in s for s in spec])]

        return all_progs, not_allowed_aggregation, not_allowed_enc

    def get_pred_type(self, fields: List, data: InputDataset) -> List[BaseRefType]:
        """
        return the **complete** refinement types tha represent the spec. Might be multiple ones exist
        TODO: the spec should actually include both the aggregation and visual encoding that should not include
                we need to do that for the aggregation, but visual encoding can be omitted because of how the enumeration works
                but just keep in mind that technically we need that too (so something to update in the future )
        """

        return_types: List[BaseRefType] = []

        enumerated_spec = self.instantiate_partial_spec(fields, data)

        for spec in enumerated_spec:

            # print("spec:", spec)

            formula = []
            field_used = list(set(spec.values()))

            contain_count = False

            for vis_intent in visual_intents:
                if spec.get(vis_intent) is None:
                    if vis_intent == 'x':
                        raise ValueError('{} axis has to be non-empty'.format(vis_intent))
                else:
                    formula.append([(Binding(spec[vis_intent], vis_intent), False)])

            for agg_intent in aggregate_intents:
                if spec.get(agg_intent) is None:
                    for field in field_used:
                        formula.append([(Prov(field, agg_intent), True)])
                else:
                    if agg_intent == 'count':
                        contain_count = True
                    field = spec[agg_intent]
                    formula.append([(Prov(field, agg_intent), False)])
                    for other_field in field_used:
                        if other_field is not field:
                            formula.append([(Prov(other_field, agg_intent), True)])

            constraint = create_cnf_formula_instance(formula)

            # print("constraint:", constraint)
            # assert False
            plot_str = self.pred['plot'][0] if not contain_count else 'Histogram'
            output_type = BaseRefType(get_base_type(plot_str), constraint=constraint, fields=field_used, prob=self.score)
            # output_type = BaseRefType(get_base_type(plot_str), constraint=constraint, fields=fields, prob=self.score)

            return_types.append(output_type)

        return return_types

    def __hash__(self):
        return prog_hash(self.pred)

    def __repr__(self):
        return str((list([v[0] for v in self.pred.values()]), self.score))
