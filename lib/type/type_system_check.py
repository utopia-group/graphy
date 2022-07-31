from typing import Iterator, Tuple, Dict, List, Optional

from lib.solver.solver import SolverOutput, QuickSolver, Z3Solver
from lib.synthesizer.lemma_learning import generate_base_type_lemma, Lemma
from lib.type.base_type import BaseType, TableType, ListType
from lib.type.formula import CNFFormula, Implication, XORFormula, Atom, get_atom_pred
from lib.type.predicate import Prov
from lib.type.ref_type import RefType, FunctionType, BaseRefType
from lib.type.type_system import subtyping_relation, allowed_data_type_for_plot
from lib.utils.misc_utils import printd

"""
All the subtyping and compatibility checks 
"""

# all the solver instance
quick_solver = QuickSolver()
z3_solver = Z3Solver()


def check_base_subtyping(arg1: BaseType, arg2: BaseType, compute_lemma: bool = False, comatibility: bool = False) -> SolverOutput:
    """
    subtyping check of the base type, which we don't need a formal decision procedure i think
    """
    if isinstance(arg1, TableType):
        if not isinstance(arg2, TableType):
            output = SolverOutput(False)
            return output
        else:
            """
            arg1 <: arg2
            which means arg1 can contains more field than arg2
            and for each field arg2 has, args1[field] <: args2[field]
            """
            for col, _type in arg2.get_record().items():
                if col not in arg1.get_record():
                    # print("here3")
                    output = SolverOutput(False)
                    return output
                if not check_base_subtyping(arg1.get_record()[col], _type, comatibility=comatibility).res:
                    output = SolverOutput(False)
                    # generate the base type lemma here
                    if compute_lemma:
                        output.lemma = generate_base_type_lemma((col, arg1.get_record()[col]), (col, _type))
                    return output
            output = SolverOutput(True)
            return output
    elif isinstance(arg1, ListType):
        if not isinstance(arg2, ListType):
            output = SolverOutput(False)
            return output
        else:
            output = check_base_subtyping(arg1.T, arg2.T, comatibility=comatibility)
            return output
    else:
        if arg1 == arg2:
            return SolverOutput(True)
        else:
            # print("arg1:", arg1, "arg2:", arg2)
            if isinstance(arg1, arg2.__class__):
                # print("here1")
                return SolverOutput(True)
            else:
                if comatibility and issubclass(arg2.__class__, arg1.__class__):
                    # print("here2")
                    return SolverOutput(True)
                else:
                    # print("here3")
                    return SolverOutput(False)


def check_base_compatibility(arg1: BaseType, arg2: BaseType, compute_lemma: bool = False) -> SolverOutput:
    return check_base_subtyping(arg1, arg2, compute_lemma, comatibility=True)


def check_implication(constraint1: CNFFormula, constraint2: CNFFormula, force_z3=False, assume_precedent=True) -> SolverOutput:
    """
    decide to invoke which solver
    check constraint1 ^ constraint1 -> constraint2 (this is equivalent to constraint1 ^ constraint2) if assume_precedent is True
    """

    imp = Implication(constraint1, constraint2)
    if assume_precedent and not constraint1.contain_disjunction() and not constraint2.contain_disjunction() and not force_z3:
        printd("quick solver")
        return quick_solver.solve(imp)

    printd("z3 solver")
    return z3_solver.solve(imp, use_encode=(not assume_precedent))


def get_implication_model(constraint1: CNFFormula, constraint2: XORFormula, force_z3=False) -> Iterator[Tuple[Atom, Prov]]:
    imp = Implication(constraint1, constraint2)

    if not constraint1.contain_disjunction() and not force_z3:
        # print("quick solver")
        model_generator = quick_solver.get_model(imp)

        while True:
            next_model = next(model_generator)

            yield next_model
    else:
        # print("z3 solver")
        model_generator = z3_solver.get_model(imp)

        while True:
            next_model = next(model_generator)

            # TODO: probably some possible post-processing here
            #   find out the literals that are true in the model
            for i in range(len(next_model)):
                var_id = next_model[i]
                if next_model[var_id]:
                    var_id = str(var_id)
                    # print("var_id {} is true".format(var_id))
                    yield get_atom_pred(str(var_id))


def check_bindings(constraint1: CNFFormula, constraint2: CNFFormula, table_record: Dict[str, ListType], viz_base_type: str) -> bool:
    # All bindings in constraint1 must be in constraint2
    # print("constraint1: {}".format(constraint1))
    # print("constraint2: {}".format(constraint2))
    for enc, literal_id in constraint2.binding_enc_to_literal_id.items():
        constraint2_atom = constraint2.literal_id_to_atom[literal_id]
        if enc not in constraint1.binding_enc_to_literal_id or constraint2_atom != constraint1.literal_id_to_atom[constraint1.binding_enc_to_literal_id[enc]]:
            # print("binding false 1")
            return False
    # We need to check that all columns selected in the table program are actually used in the viz program
    table_record_check = {col: False for col in table_record}
    for enc, literal_id in constraint1.binding_enc_to_literal_id.items():
        constraint1_atom = constraint1.literal_id_to_atom[literal_id]
        col = constraint1_atom.predicate.col
        # If there are any bindings in constraint1 that aren't in constraint2, the column must be None
        if enc not in constraint2.binding_enc_to_literal_id and col != 'None':
            # print("binding false 2")
            return False
        if col == 'None':
            continue
        if col not in table_record:
            # print("binding false 3")
            return False
        table_record_check[col] = True
        col_cell_type = table_record[col].T.name
        col_cell_supertype = subtyping_relation[col_cell_type]
        allowed_types = allowed_data_type_for_plot[viz_base_type][enc]
        if col_cell_type not in allowed_types and col_cell_supertype not in allowed_types:
            # print("binding false 4")
            return False
    # Returns true if all selected in table program are used

    return all(table_record_check.values())


def check_compatibility(arg1: RefType, arg2: RefType, compute_lemma=False) -> SolverOutput:
    """
    main compatibility check function
    check if arg1 is compatible with arg2
    return a SolverOutput object with some error code
    1  -> type check
    2  -> ref type failed
    3  -> base type failed
    """

    if arg1 is None:
        return SolverOutput(False, error_code=3)

    # print("arg1:", arg1)
    # print("arg2:", arg2)
    if isinstance(arg1, FunctionType) and isinstance(arg2, FunctionType):
        raise NotImplementedError("FunctionType not implemented")
        # return check_compatibility(arg1.input, arg2.input) and check_compatibility(arg1.output, arg2.output)

    elif isinstance(arg1, BaseRefType) and isinstance(arg2, BaseRefType):

        if arg1.is_bot or arg2.is_bot:
            return SolverOutput(False, error_code=2)

        # first subtype base type then subtype check constraints
        base_check_output = check_base_compatibility(arg1.base, arg2.base, compute_lemma)
        if not base_check_output.res:
            base_check_output.error_code = 3
            return base_check_output

        if arg1.constraint is not None and arg2.constraint is not None:
            ref_check_output = check_implication(arg1.constraint, arg2.constraint, assume_precedent=True)
            if not ref_check_output.res:
                # print(2)
                ref_check_output.error_code = 2
                return ref_check_output

        # print(1)
        return SolverOutput(True, error_code=1)

    else:
        printd("TypeError: not able to perform compatibility checking between {} and {}".format(arg1, arg2))
        raise TypeError


def check_subtyping(arg1: RefType, arg2: RefType) -> SolverOutput:
    """
    main subtyping check function
    check if arg1 is a subtype of arg2
    return some error code:
    1 -> type check
    2 -> ref type failed
    3 -> base type failed
    """
    if arg1 is None:
        return SolverOutput(False, error_code=3)

    if isinstance(arg1, FunctionType) and isinstance(arg2, FunctionType):
        raise NotImplementedError("FunctionType not implemented")
    elif isinstance(arg1, BaseRefType) and isinstance(arg2, BaseRefType):
        if arg1.is_bot or arg2.is_bot:
            return SolverOutput(False, error_code=2)

        base_check_output = check_base_subtyping(arg1.base, arg2.base)
        if not base_check_output.res:
            base_check_output.error_code = 3
            return base_check_output

        if arg1.constraint is not None and arg2.constraint is not None:
            ref_check_output = check_implication(arg1.constraint, arg2.constraint, assume_precedent=False)
            if not ref_check_output.res:
                ref_check_output.error_code = 2
                return ref_check_output
    else:
        printd("TypeError: not able to perform compatibility checking between {} and {}".format(arg1, arg2))
        raise TypeError


"""
Following are the lemma checking:
"""


class LemmaCheckingOutput:
    def __init__(self, res: bool, violated_lemma: Optional[Lemma] = None, is_infeasible: bool = False):
        self.res: bool = res
        self.violated_lemma: Optional[Lemma] = violated_lemma
        self.is_infeasible: bool = is_infeasible

    def __repr__(self):
        return "LemmaCheckingOutput(res={}, violated_lemma={}, is_infeasible={})".format(self.res, self.violated_lemma, self.is_infeasible)


def check_compatibility_lemma_prov_helper(prov_req: List[List[Tuple[Prov, bool]]], goal_type: BaseRefType) -> bool:
    """
    Helper function for checking the lemma req compatibility that involves provenance constraint.
    """

    def check_boolean_consistency(is_negated_1: bool, is_negated_2: bool) -> bool:
        if is_negated_1 == is_negated_2:
            return True
        else:
            return False

    # first obtain all the provenance thing from the goal type
    provenance_predicates_str_to_id = goal_type.constraint.prov_str_to_literal_id
    for and_prov_req in prov_req:
        res = True
        for prov_pair in and_prov_req:
            prov = prov_pair[0]
            is_negated_1 = prov_pair[1]
            if str(prov) not in provenance_predicates_str_to_id:
                continue
            else:
                literal_id = provenance_predicates_str_to_id[str(prov)]
                is_negated_2 = goal_type.constraint.literal_id_to_negation[literal_id]
                if not check_boolean_consistency(is_negated_1, is_negated_2):
                    res = False
                    break
        if res:
            return True

    return False


def check_lemma(goal_type: BaseRefType, lemma: Lemma) -> LemmaCheckingOutput:
    """
    check if a goal type satisfies a lemma
    Here is the procedure:
    1. check if goal type is a subtype of the context
    2. if it subtypes the context, then check if req is satisfied
    If it is satisfied, return True, otherwise return False
    """
    assert isinstance(lemma.context, BaseRefType)
    if lemma.context.constraint is None:
        if check_base_subtyping(goal_type.base, lemma.context.base).res:
            if not check_compatibility(lemma.req, goal_type).res:
                if lemma.base_only:
                    return LemmaCheckingOutput(False, violated_lemma=lemma, is_infeasible=lemma.req.is_bot)
                else:
                    assert lemma.prov_req_only
                    # need to further check if the prov requirement is satisfied
                    if not check_compatibility_lemma_prov_helper(lemma.prov_req, goal_type):
                        # print("goal type:", goal_type)
                        # print("lemma:", lemma)
                        # assert False
                        return LemmaCheckingOutput(False, violated_lemma=lemma)

        return LemmaCheckingOutput(True)
    else:
        raise NotImplementedError


def check_lemmas(goal_type: BaseRefType, lemmas: Dict[str, Lemma]) -> LemmaCheckingOutput:
    """
    check if a goal type satisfies a set of lemmas
    return True if all lemmas are satisfied, otherwise return False
    """
    # print("lemmas:", lemmas)
    for lemma in lemmas.values():
        output = check_lemma(goal_type, lemma)
        if not output.res:
            return output
    return LemmaCheckingOutput(True)
