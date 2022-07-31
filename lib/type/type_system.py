import itertools
from typing import Dict, Optional, Tuple, List

from lib.type.base_type import ListType, ConstColType, CellType, NullType, NominalType, OrdinalType, AggregateType, QualitativeType, QuantitativeType, Histogram, TemporalType, NullOpType
from lib.type.predicate import Predicate
from lib.type.formula import CNFFormula, XORFormula
from lib.type.ref_type import BaseRefType, FunctionType
from lib.type.base_type import BaseType, PlotType, BarPlot, ScatterPlot, LinePlot, AreaPlot, TableType, DiscreteType, ContinuousType, ConstType, AlphaType
from lib.utils.misc_utils import list_hash

"""
This is a static class
TODO: ideally we should generate this class from some existing type system implementation, leave this for future
"""

# create a dict of basetype so we don't have to always create a basetype
base_types: Dict[str, BaseType] = {'Plot': PlotType(),
                                   'BarPlot': BarPlot(),
                                   'Histogram': Histogram(),
                                   'bar': BarPlot(),
                                   'ScatterPlot': ScatterPlot(),
                                   'scatter': ScatterPlot(),
                                   'LinePlot': LinePlot(),
                                   'line': LinePlot(),
                                   'AreaPlot': AreaPlot(),
                                   'area': AreaPlot(),
                                   'Table': TableType(),
                                   'Discrete': DiscreteType(),
                                   'Nominal': NominalType(),
                                   'Ordinal': OrdinalType(),
                                   'Temporal': TemporalType(),
                                   'Qualitative': QualitativeType(),
                                   'Quantitative': QuantitativeType(),
                                   'Cell': CellType(),
                                   'Continuous': ContinuousType(),
                                   'Aggregate': AggregateType(),
                                   'Const': ConstType(),
                                   'ConstCol': ConstColType(),
                                   'Alpha': AlphaType(),
                                   'Null': NullType(),
                                   'NullOp': NullOpType()}

# the subtyping relation is actually reflect from the class hierarchy, here we encode a string version for efficiency when looking up valid productions
# NOTE: assumption here is that each key has only one superclass, which is true so far but maybe need to revise in the future
subtyping_relation: Dict[str, str] = {
    'BarPlot': 'Plot',
    'Histogram': 'Plot',
    'ScatterPlot': 'Plot',
    'LinePlot': 'Plot',
    'AreaPlot': 'Plot',
    'Quantitative': 'Cell',
    'Qualitative': 'Cell',
    'Discrete': 'Quantitative',
    'Continuous': 'Quantitative',
    'Aggregate': 'Quantitative',
    'Nominal': 'Qualitative',
    'Ordinal': 'Qualitative',
    'Temporal': 'Qualitative'
}

# for each plot, encode the allowed datatype for each enc here
# TODO: we might want to automate this according to the subtyping relation in the future
allowed_data_type_for_plot: Dict[str, Dict[str, List[str]]] = {
    'BarPlot': {'x': ['Qualitative', 'Discrete'], 'y': ['Quantitative'], 'color': ['Qualitative', 'Discrete'], 'column': ['Qualitative', 'Discrete']},
    'Histogram': {'x': ['Qualitative', 'Quantitative'], 'y': ['Quantitative'], 'color': ['Qualitative', 'Discrete'], 'column': ['Qualitative', 'Discrete']},
    'ScatterPlot': {'x': ['Quantitative', 'Ordinal'], 'y': ['Quantitative', 'Ordinal'], 'color': ['Discrete', 'Qualitative'], 'column': ['Qualitative', 'Discrete']},
    'LinePlot': {'x': ['Ordinal', 'Temporal'], 'y': ['Quantitative'], 'color': ['Discrete', 'Qualitative'], 'column': ['Discrete', 'Qualitative']},
    'AreaPlot': {'x': ['Ordinal', 'Temporal'], 'y': ['Quantitative'], 'color': ['Discrete', 'Qualitative'], 'column': ['Discrete', 'Qualitative']},
}

# For genreq: For each input base type, list the functions can take this as an input and its corresponding output type
# This actually looks very similar to generate_base_type_lemma, but we create this data structure to make things easier to query
# TODO: what should we do in the bot case?
# input_type -> output_type -> (func_names, must_be_used/not_used)
legit_input_to_output_type: Dict[str, Dict[str, Tuple[List[str], bool]]] = {
    'Discrete': {'Continuous': (['mean', 'sum'], False), 'Discrete': (['mean', 'sum'], True)},
    'Continuous': {'Continuous': (['count'], True), 'Discrete': (['count'], False)},
    'Qualitative': {'Discrete': (['count'], False), 'Qualitative': (['mean', 'sum', 'count'], True)}
}


def get_base_type(name: str) -> BaseType:
    """
    get the base type object indexed by name
    """
    return base_types[name]


"""
type creation methods
"""


def create_list_type(name: str) -> ListType:
    """
    get list type of a specific basetype
    """
    assert isinstance(get_base_type(name), CellType)
    return ListType(get_base_type(name))


def create_ref_type(base_name: str, constraints: Optional[CNFFormula] = None) -> BaseRefType:
    """
    create a new base refinement type object
    """
    return BaseRefType(get_base_type(base_name), constraints)


def create_func_type(base_name_input: str, base_name_output: str, constraints_input: Optional[CNFFormula] = None, constraints_output: Optional[CNFFormula] = None) -> FunctionType:
    """
    create a new function refinement type object
    """
    return FunctionType(create_ref_type(base_name_input, constraints_input), create_ref_type(base_name_output, constraints_output))


formula_id_factory = itertools.count(0)


def get_next_formula_id() -> int:
    """
    get the next formula id
    """
    return next(formula_id_factory)


def reset_formula_id():
    global formula_id_factory
    formula_id_factory = itertools.count(0)


def create_cnf_formula_instance(args: List[List[Tuple[Predicate, bool]]]) -> CNFFormula:
    """
    shortcut to create the entire formula at once
    also include some analysis information
    args -> [c1, c2, c3]    # conjunction
    c1 -> (l1, l2, l3)  # disjunction
    l1 -> (a, true/false)    # atom or negation of atom
    a # atom
    """
    return CNFFormula(get_next_formula_id(), init_clauses=args)


def create_xor_formula_instance(args: List[List[Tuple[Predicate, bool]]]) -> XORFormula:
    return XORFormula(get_next_formula_id(), init_clauses=args)


def generate_context_auto_helper(src_type: BaseType, dst_type: BaseType) -> BaseType:
    """
    find the most general type that can differentiate src_type and dst_type
    """
    src_type_superclass = set([c.__name__ for c in src_type.__class__.__mro__])
    dst_type_superclass = set([c.__name__ for c in dst_type.__class__.__mro__])

    intersected_superclass = src_type_superclass.intersection(dst_type_superclass)

    # print("src_type_superclass: ", src_type_superclass)
    # print("dst_type_superclass: ", dst_type_superclass)
    # print("intersected_superclass:", intersected_superclass)
    # print("dst_type_superclass / intersected_superclass: ", dst_type_superclass.difference(intersected_superclass))

    dst_type_superclass_left = dst_type_superclass.difference(intersected_superclass)

    if len(dst_type_superclass_left) == 0:
        return dst_type
    elif len(dst_type_superclass_left) == 1:
        type_name = dst_type_superclass_left.pop().split('Type')[0]
        return get_base_type(type_name)
    elif len(dst_type_superclass_left) == 2 and dst_type.__class__.__name__ in dst_type_superclass_left:
        dst_type_superclass_left.discard(dst_type.__class__.__name__)
        assert len(dst_type_superclass_left) == 1
        type_name = dst_type_superclass_left.pop().split('Type')[0]
        return get_base_type(type_name)
    else:
        raise NotImplementedError


def generate_base_type_lemma_auto_helper(src_type: BaseType, dst_type: BaseType, k: int) -> Optional[Tuple[List[List], List]]:
    """
    For genReq, essentially the idea here is to find a function that can goes from input type to output type in k step.
    arg1: src type
    arg2: dst type
    should return two forms: [1] operations that allowed, or [2] must not allowed
    [1]: List of list :: outer list represents the OR structure, inner list represents the AND structure
    [2]: List :: this should represent AND only
    Note that in legit_input_to_output_type, we have three options: Continuous, Discrete and Qualitative
    """

    def get_type_idx_string(some_type: BaseType) -> str:
        if isinstance(some_type, QualitativeType):
            type_idx_string = 'Qualitative'
        elif isinstance(some_type, ContinuousType):
            type_idx_string = 'Continuous'
        elif isinstance(some_type, DiscreteType):
            type_idx_string = 'Discrete'
        else:
            raise ValueError("Cannot handle finding key for type {}".format(src_type.name))

        return type_idx_string

    type_space = list(legit_input_to_output_type.keys())
    src_type_idx_string = get_type_idx_string(src_type)
    dst_type_idx_string = get_type_idx_string(dst_type)

    if k == 1:
        """
        base case: is arg2 reachable from arg1 in one step 
        """

        if dst_type_idx_string in legit_input_to_output_type[src_type_idx_string]:
            func_names, must_not_use_flag = legit_input_to_output_type[src_type_idx_string].get(dst_type_idx_string)
            if must_not_use_flag:
                return [], [func_names]
            else:
                return [[fn] for fn in func_names], []
        else:
            return None

    else:
        """
        recursive case: is arg2 reachable from arg1 in k step
        the lemma should be the union of in 1 step and in k-1 step
        k-1 step:  
        FIXME: I think I am doing something dumb here again
        Assumption: src_type and dst_type is all related to one column name in particular 
        """
        # (1) compute the lemma that goes from src_type to dst_type in 1 step
        one_step_funcs = generate_base_type_lemma_auto_helper(src_type, dst_type, 1)

        # (2) compute the lemma that goes from src_type to dst_type in k step
        allowed_funcs = []
        allowed_funcs_hashcode = []
        disallowed_funcs = []
        for tmp_dst_type in type_space:
            # for all f such that src_type ->(f) tmp_dst_type, check if tmp_dst_type is reachable to dst_type in k-1 step
            tmp_one_step_funcs = generate_base_type_lemma_auto_helper(src_type, get_base_type(tmp_dst_type), 1)
            if tmp_one_step_funcs is None:
                continue

            one_use_func_names, one_not_use_func_name = tmp_one_step_funcs
            if len(one_use_func_names) == 0:
                continue

            tmp_k_step_funcs = generate_base_type_lemma_auto_helper(get_base_type(tmp_dst_type), dst_type, (k - 1))
            if tmp_k_step_funcs is None:
                continue

            k_use_func_names, k_not_use_func_name = tmp_k_step_funcs
            if len(k_use_func_names) == 0:
                continue

            # need to do a cartesian product between tmp_k_step_funcs and tmp_one_step_funcs
            for one_use_op in one_use_func_names:
                for k_use_op in k_use_func_names:

                    # check if any of the one_use_op shows up in k_not_use_op or vise versa
                    # in this case we cannot form a and operator (because of there is already a conflict)
                    if len(set(one_use_op).intersection(set(k_not_use_func_name))) > 0 or \
                            len(set(k_use_op).intersection(set(one_not_use_func_name))) > 0:
                        continue

                    combined_use_op = list(set(one_use_op + k_use_op))
                    combined_use_op_hash = list_hash(combined_use_op)
                    if combined_use_op_hash not in allowed_funcs_hashcode:
                        allowed_funcs.append(combined_use_op)
                        allowed_funcs_hashcode.append(combined_use_op_hash)

        combined_allowed_funcs = allowed_funcs + one_step_funcs[0] if one_step_funcs is not None else allowed_funcs
        combined_disallowed_funcs = disallowed_funcs + one_step_funcs[1] if one_step_funcs is not None else disallowed_funcs

        # make sure no duplicate
        combined_allowed_funcs.sort()
        combined_allowed_funcs = list(val for val, _ in itertools.groupby(combined_allowed_funcs))

        combined_disallowed_funcs.sort()
        combined_disallowed_funcs = list(val for val, _ in itertools.groupby(combined_disallowed_funcs))

        if len(combined_allowed_funcs) == 0 and len(combined_disallowed_funcs) == 0:
            return None
        elif len(combined_allowed_funcs) == 0 and len(combined_disallowed_funcs) > 0:
            return [], combined_disallowed_funcs
        else:
            return combined_allowed_funcs, []
