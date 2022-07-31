import itertools
import re
from typing import Tuple, List

from lib.utils.prev_synth_utils import parse_spec


def translate_func(spec, src_func, targ_func):
    raise NotImplementedError


def constraint_str_gen(func, arg):
    return "{}({})".format(func, arg)


# Translate the parsed spec to a intermediate language so that can be taken as input to the interpreter class
class Translator:
    def __init__(self, grammar):
        self.var_id_counter = itertools.count(start=1)
        self.constraint_var_map = {}
        self.var_constraint_map = {}
        self.encode_var_id_to_name = {}

        self.field_var_to_encode_type = {}

        self.data_url = ""

    def reinit(self):
        self.var_id_counter = itertools.count(start=1)
        self.constraint_var_map = {}
        self.var_constraint_map = {}
        self.encode_var_id_to_name = {}
        self.data_url = ""

    def get_or_create_var_id(self, constraint, _type, func=None) -> str:

        if _type == 'e' and (func == 'color' or func == 'column'):
            constraint = constraint + '_' + func

        var_id = self.constraint_var_map.get(constraint)
        if var_id is None:
            var_id = "{}{}".format(_type, str(next(self.var_id_counter)))
            self.constraint_var_map[constraint] = var_id
            self.var_constraint_map[var_id] = constraint
            if _type == 'e':
                self.encode_var_id_to_name[var_id] = parse_spec(constraint)[1]
        return var_id

    def find_field_var(self, colnames: list, field: str, func: str):

        # if field.islower():
        #     field = field[0].upper() + field[1:]
        # if field.isupper():
        #     field = field[0] + field[1:].lower()

        if colnames is None:
            constraint = constraint_str_gen('field', field)
            return self.get_or_create_var_id(constraint, _type='e', func=func)

        # TODO: this statement can clearly be optimized
        colname_lower_to_colname_map = {c.lower(): c for c in colnames}

        # print("colname_lower_to_colname_map: ", colname_lower_to_colname_map)

        for col, c in colname_lower_to_colname_map.items():
            if field.lower() == col:
                return self.get_or_create_var_id(constraint_str_gen('field', c), _type='e', func=func)

        # TODO: i just list some common case here, need a more principled way in the future
        if ' ' in field:
            # print("field:", field)
            field_new = field.lower().replace(' ', '_')
            # print("field_new:", field_new)
            for col, c in colname_lower_to_colname_map.items():
                # print("col:", col)
                if field_new == col:
                    return self.get_or_create_var_id(constraint_str_gen('field', c), _type='e', func=func)

            # one of the str is in colname
            field_split = field.lower().split(' ')
            # TODO: only take the first one that matches otherwise i don't know how to resolve
            for f_split in field_split:
                for col, c in colname_lower_to_colname_map.items():
                    if f_split == col:
                        return self.get_or_create_var_id(constraint_str_gen('field', c), _type='e', func=func)

        # str is part of the colname
        for cn in colnames:
            if '_' in cn:
                cn_split = [c.lower() for c in cn.split('_')]
                if field.lower() in cn_split:
                    return self.get_or_create_var_id(constraint_str_gen('field', cn), _type='e', func=func)

        return None

    def translate(self, specs: List[str], colnames: List[str] = None, data_url: str = None):

        # print("colnames:", colnames)

        # i think i should directly reinit here
        self.reinit()

        if data_url is not None:
            self.data_url = data_url

        to_be_solved_constraints_clause = []
        transformation_constraints = []

        if 'trend(*)' in specs:
            specs.append('plot(line)')
            specs.append('field(Year)')  # this is a heuristic, ideally we should find all field with datetime type

        for spec in specs:
            if spec.startswith('data'):
                self.data_url = spec
            elif spec.startswith('field'):
                field_names = spec.split('^')
                for field_name in field_names:
                    var_id = self.find_field_var(colnames, parse_spec(field_name)[1], func='field')
                    if var_id is not None:
                        to_be_solved_constraints_clause.append(var_id)
            elif spec.startswith('transform'):
                # TODO: implement this
                pass
            elif spec.startswith('trend'):
                pass
            elif spec.startswith('plot'):
                var_id = self.get_or_create_var_id(spec, _type='p')
                to_be_solved_constraints_clause.append(var_id)
            else:
                func, args = parse_spec(spec)
                # print(spec)
                if args == '*':
                    func_var_id = self.get_or_create_var_id(func, _type='f')
                    to_be_solved_constraints_clause.append("{}(*, {})".format('f', func_var_id))
                    # if func == 'count':
                    #     func_var_id = self.get_or_create_var_id(func, _type='f')
                    #     to_be_solved_constraints_clause.append("{}(*, {})".format('f', func_var_id))
                    # else:
                    #     raise NotImplementedError
                else:
                    args_var_id = self.find_field_var(colnames, args, func)
                    if args_var_id is not None:
                        to_be_solved_constraints_clause.append(args_var_id)
                        func_var_id = self.get_or_create_var_id(func, _type='f')
                        to_be_solved_constraints_clause.append("{}({},{})".format('f', args_var_id, func_var_id))

        return list(set(to_be_solved_constraints_clause))
