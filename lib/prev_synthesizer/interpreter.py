import json
from collections import defaultdict
from typing import Union, List, Tuple, Optional

from lib.dracopy.draco.helper import get_violation
from lib.prev_synthesizer.translator import Translator
from lib.utils.prev_synth_utils import parse_spec
import lib.dracopy.draco as draco


# Parse the intermediate language from Translator to asp encoding that can be interpret as draco
class Interpreter:
    def __init__(self):
        pass

    def to_asp(self, data_constraint: dict, constraints: list, translator: Translator) -> list:

        asp_program = []

        encoding_related = []
        plot_related = []
        data_related = []

        if not translator.data_url == '':
            data_related.append(translator.data_url + '.')

        top_level_encoding_constraint = []

        for constraint in constraints:
            if constraint.startswith('e'):
                encoding_related.append(':~ not encoding({0}). [1@3, {0}]'.format(constraint))
                top_level_encoding_constraint.append('encoding({})'.format(constraint))
                encoding_related.append(":~ not field({0},\"{1}\"). [1@3, {0}]".format(constraint,
                                                                                       translator.encode_var_id_to_name[constraint]))
                data_related.extend(data_constraint[translator.encode_var_id_to_name[constraint]])
                if 'string' in data_constraint[translator.encode_var_id_to_name[constraint]][0]:
                    encoding_related.append(":~ not type({0}, nominal). [1@3, {0}]".format(constraint))

            if constraint.startswith('f'):
                args_str = parse_spec(constraint)[1]
                args = args_str.split(',')
                encoding_id = args[0]
                func_id = args[1]
                func_str = translator.var_constraint_map[func_id.strip()]

                if func_str == 'count':
                    encoding_var_id = translator.get_or_create_var_id('field(count_encode)', 'e')
                    encoding_related.append(":~ not encoding({0}). [1@2, {0}]".format(encoding_var_id))
                    top_level_encoding_constraint.append('encoding({})'.format(encoding_var_id))
                    # if encoding_id == '*':
                    #     encoding_related.append(':~ ' + \
                    #                             ','.join(["not field({}, \"{}\")".format(encoding_var_id, name)
                    #                                       for eid, name in translator.encode_var_id_to_name.items() if
                    #                                       not name == 'count_encode']) + \
                    #                             '. [1@1, {}]'.format(encoding_var_id))
                    # else:
                    #     encoding_related.append(':~ not field({0},\"{1}\"). [1@1, {0}]'.format(encoding_var_id,
                    #                                                   translator.encode_var_id_to_name[encoding_id]))

                    encoding_related.append(":~ not type({0},quantitative). [1@2, {0}]".format(encoding_var_id))
                    encoding_related.append(":~ not aggregate({0},count). [1@2, {0}]".format(encoding_var_id))
                elif func_str == 'mean' or func_str == 'sum':
                    if encoding_id == '*':
                        encoding_related.append(':~' + \
                                                ','.join(["not aggregate({0}, {1})".format(eid, func_str)
                                                          for eid, name in translator.encode_var_id_to_name.items()
                                                          if not name == 'count_encode']) + \
                                                '. [1@2, {}]'.format(','.join(
                                                    eid for eid, name in translator.encode_var_id_to_name.items()
                                                    if not name == 'count_encode')))
                    else:
                        encoding_related.append(':~ not aggregate({0},{1}). [1@2, {0}]'.format(encoding_id, func_str))
                elif func_str == 'color' or func_str == 'column':
                    if encoding_id == '*':
                        encoding_related.append(':~ ' + \
                                                ','.join(["not channel({}, {})".format(eid, func_str)
                                                          for eid, name in translator.encode_var_id_to_name.items()
                                                          if not name == 'count_encode']) + \
                                                '. [1@2, {}]'.format(','.join(
                                                    eid for eid, name in translator.encode_var_id_to_name.items()
                                                    if not name == 'count_encode')))
                    else:
                        encoding_related.append(':~ not channel({0},{1}). [1@2, {0}]'.format(encoding_id, func_str))
                elif func_str == 'channel':
                    encoding_related.append('channel(E, color);channel(E,size);channel(E,column) :- some_channel(E).')
                    if encoding_id == '*':
                        encoding_related.append(':~ ' + \
                                                ','.join(['not some_channel({})'.format(eid)
                                                          for eid, name in translator.encode_var_id_to_name.items()
                                                          if not name == 'count_encode']) + \
                                                '. [1@2, {}]'.format(','.join(
                                                    eid for eid, name in translator.encode_var_id_to_name.items()
                                                    if not name == 'count_encode')))
                        top_level_encoding_constraint.extend(['some_channel({0})'.format(eid) for eid, name in
                                                              translator.encode_var_id_to_name.items() if not name == 'count_encode'])
                    else:
                        encoding_related.append(':~ not some_channel({0}). [1@2, {0}]'.format(encoding_id))
                        top_level_encoding_constraint.append('some_channel({0})'.format(encoding_id))
                else:
                    raise NotImplementedError(func_str)
            if constraint.startswith('p'):
                spec = translator.var_constraint_map[constraint]
                plot_type = parse_spec(spec)[1]
                if plot_type == 'scatter':
                    plot_related.append(':~ not mark(point). [1@3]')
                elif plot_type == 'bar':
                    plot_related.append(':~ not mark(bar). [1@3]')
                elif plot_type == 'line':
                    plot_related.append(':~ not mark(line). [1@3]')
                elif plot_type == 'area':
                    plot_related.append(':~ not mark(area). [1@3]')
                else:
                    print("NotImplementedError({})".format(plot_type))
                    pass

        return data_related + plot_related + encoding_related + \
               ['{{{}}}.'.format(';'.join(top_level_encoding_constraint))]

    def to_asp_prev(self, data_constraint: dict, constraints: list, translator: Translator, not_allowed_agg=None, not_allowed_enc=None) -> list:

        asp_program = []

        encoding_related = []
        plot_related = []
        data_related = []

        if not translator.data_url == '':
            data_related.append(translator.data_url + '.')

        encoding_to_intent = defaultdict(list)
        plot_type = ''

        for constraint in constraints:
            if constraint.startswith('e'):
                encoding_related.append('encoding({0}).'.format(constraint))
                encoding_related.append("field({0},\"{1}\").".format(constraint,
                                                                     translator.encode_var_id_to_name[
                                                                         constraint]))
                data_related.extend(data_constraint[translator.encode_var_id_to_name[constraint]])
                if 'string' in data_constraint[translator.encode_var_id_to_name[constraint]][0]:
                    encoding_related.append("type({0}, nominal).".format(constraint))

            if constraint.startswith('f'):
                args_str = parse_spec(constraint)[1]
                args = args_str.split(',')
                encoding_id = args[0]
                func_id = args[1]
                func_str = translator.var_constraint_map[func_id.strip()]

                if func_str == 'count':
                    encoding_var_id = translator.get_or_create_var_id('field(count_encode)', 'e')
                    encoding_related.append("encoding({0}).".format(encoding_var_id))
                    # if encoding_id == '*':
                    #     encoding_related.append(';'.join(["field({}, \"{}\")".format(encoding_var_id, name)
                    #                                       for eid, name in translator.encode_var_id_to_name.items() if
                    #                                       not name == 'count_encode']) + '.')
                    # else:
                    #     encoding_related.append('field({0},\"{1}\").'.format(encoding_var_id,
                    #                                                          translator.encode_var_id_to_name[encoding_id]))

                    encoding_related.append("type({0},quantitative).".format(encoding_var_id))
                    encoding_related.append("aggregate({0},count). ".format(encoding_var_id))
                    encoding_to_intent[encoding_var_id].append('count')
                elif func_str == 'mean' or func_str == 'sum':
                    if encoding_id == '*':
                        encoding_related.append(';'.join(["aggregate({0}, {1})".format(eid, func_str)
                                                          for eid, name in translator.encode_var_id_to_name.items()
                                                          if not name == 'count_encode']) + '.')
                    else:
                        encoding_related.append('aggregate({0},{1}).'.format(encoding_id, func_str))
                        encoding_to_intent[encoding_id].append(func_str)
                elif func_str == 'color' or func_str == 'column' or func_str == 'x' or func_str == 'y':
                    if encoding_id == '*':
                        encoding_related.append(';'.join(["channel({}, {})".format(eid, func_str)
                                                          for eid, name in translator.encode_var_id_to_name.items()
                                                          if not name == 'count_encode']) + '.')
                    else:
                        encoding_related.append('channel({0},{1}).'.format(encoding_id, func_str))
                        encoding_to_intent[encoding_id].append(func_str)
                elif func_str == 'channel':
                    if encoding_id == '*':
                        encoding_related.append(';'.join(['some_channel({})'.format(eid)
                                                          for eid, name in translator.encode_var_id_to_name.items()
                                                          if not name == 'count_encode']) + '.')
                    else:
                        encoding_related.append('some_channel({0}).'.format(encoding_id))
                    encoding_related.append('channel(E, color);channel(E,size);channel(E,column) :- some_channel(E).')
                else:
                    raise NotImplementedError(func_str)

            if constraint.startswith('p'):
                spec = translator.var_constraint_map[constraint]
                plot_type = parse_spec(spec)[1]
                if plot_type == 'scatter':
                    plot_related.append('mark(point).')
                elif plot_type == 'bar':
                    plot_related.append('mark(bar).')
                elif plot_type == 'line':
                    plot_related.append('mark(line).')
                elif plot_type == 'area':
                    plot_related.append('mark(area).')
                else:
                    raise NotImplementedError(plot_type)

        # add additional constraint for encoding to specify what operation is not allowed
        for eid, intents in encoding_to_intent.items():
            for not_intent in not_allowed_agg:
                if not_intent not in intents:
                    encoding_related.append('not aggregate({},{}).'.format(eid, not_intent))
            for not_enc in not_allowed_enc:
                if not_enc not in intents:
                    encoding_related.append('not channel({},{}).'.format(eid, not_enc))

            if not plot_type == 'bar':
                encoding_related.append('not bin({},10).'.format(eid))

        return data_related + plot_related + encoding_related

    def get_vis(self, asp_spec: list, allow_get_violation=False, multiple_solution=False, soft=False, remove_column=False, remove_color=False) -> Tuple[Optional[List],
                                                                                                                           int]:
        # print("asp_spec:", asp_spec)
        # res = draco.run2(asp_spec, debug=True, multiple_solution=True)
        res = draco.run(asp_spec, debug=True, multiple_solution=multiple_solution, soft=soft, remove_column=remove_column, remove_color=remove_color)
        if res is None and allow_get_violation:
            print(get_violation(asp_spec))
        # print(res)

        if res is None:
            return None, -1

        if isinstance(res, list):
            all_vis = []
            all_vis_str_set = set()
            cost = -1
            for r in res:
                # print(r)
                # print("get_vis res:", json.dumps(r.as_vl()), r.cost)
                # all_vis.append((json.dumps(r.as_vl()), r.cost))
                # all_vis.append(r.as_vl())
                vl_str = str(r.as_vl())
                if vl_str not in all_vis_str_set:
                    all_vis.append(r.as_vl())
                    all_vis_str_set.add(vl_str)
                cost = r.cost
            return all_vis, cost
        # return [(json.dumps(res.as_vl()), res.cost)]
        return [res.as_vl()], res.cost

    def generate_vis(self, data_constraint: dict, constraints: list, translator: Translator):
        asp_spec = self.to_asp(data_constraint, constraints, translator)
        return self.get_vis(asp_spec)
