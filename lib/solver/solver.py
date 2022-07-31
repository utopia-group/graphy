from typing import Dict, Tuple, Iterator

from lib.synthesizer.lemma_learning import Lemma
from lib.type.formula import Implication, CNFFormula, Atom, XORFormula
from z3 import *

from lib.type.predicate import RelationPred, Binding, Prov, Constant, Predicate
from lib.utils.misc_utils import printd


class SolverOutput:
    """
    output of the solver
    we create this object to include the interpolant
    """
    def __init__(self, res: bool, error_code: int = 0, lemma: Lemma = None):
        self.res: bool = res
        self.error_code: int = error_code
        self.lemma: Lemma = lemma
        self.model = None

    def __repr__(self):
        return "SolverOutput(res={}, error_code={}, lemma={})".format(self.res, self.error_code, self.lemma)


class Solver:
    """
    A general solver
    """
    def __init__(self):
        pass

    def solve(self, imp: Implication) -> SolverOutput:
        pass

    def get_model(self, imp: Implication):
        pass


class QuickSolver(Solver):
    """
    Hacked solver for formula of particular form
    """
    def __init__(self):
        super().__init__()

    def solve(self, imp: Implication) -> SolverOutput:
        """
        Need to be explicit about the assumption here
        1. there is no disjunction (i.e. each clause only have 1 literal)
        2. there is no negation for relation predicate
        we are checking satisfiability of a ^ a -> b
        """

        def check_valid_interval(interval: Tuple[float, bool, float, bool]) -> bool:

            lower_bound, lower_bound_eq, upper_bound, upper_bound_eq = interval

            if lower_bound_eq and upper_bound_eq and lower_bound > upper_bound:
                return False

            if not lower_bound_eq and not upper_bound_eq and lower_bound >= upper_bound:
                return False

            if (lower_bound_eq != upper_bound_eq) and lower_bound >= upper_bound:
                return False

            return True

        def generate_interval(_op, value, inverse=False) -> Tuple[float, float, bool]:
            if not inverse:
                if _op == 'eq':
                    var_range = (value, value, True)
                elif 'ge' in _op:
                    var_range = (value, math.inf, 'eq' in _op)
                elif 'le' in _op:
                    var_range = (-math.inf, value, 'eq' in _op)
                else:
                    raise ValueError
            else:
                if _op == 'eq':
                    var_range = (value, value, True)
                elif 'ge' in _op:
                    var_range = (-math.inf, value, 'eq' in _op)
                elif 'le' in _op:
                    var_range = (value, math.inf, 'eq' in _op)
                else:
                    raise ValueError

            return var_range

        def intersect_interval(interval1, interval2) -> Tuple[float, bool, float, bool]:
            """
            I wrote a couple very dumb code lol
            """

            if interval1[0] < interval2[0]:
                lower_bound = interval2[0]
                lower_bound_eq = interval2[2]
            elif interval1[0] > interval2[0]:
                lower_bound = interval1[0]
                lower_bound_eq = interval1[2]
            else:
                assert interval1[0] == interval2[0]
                lower_bound = interval1[0]
                # eq should be false if one of them is not eq
                lower_bound_eq = interval1[2] and interval2[2]

            if interval1[1] < interval2[1]:
                upper_bound = interval1[1]
                upper_bound_eq = interval1[2]
            elif interval1[1] > interval2[1]:
                upper_bound = interval2[1]
                upper_bound_eq = interval1[2]
            else:
                assert interval1[1] == interval2[1]
                upper_bound = interval1[1]
                # eq should be false if one of them is not eq
                upper_bound_eq = interval1[2] and interval2[2]

            return lower_bound, lower_bound_eq, upper_bound, upper_bound_eq

        assert isinstance(imp.antecedent, CNFFormula)
        assert isinstance(imp.consequent, CNFFormula)

        f1: CNFFormula = imp.antecedent
        f2: CNFFormula = imp.consequent

        f1_atom_assignment: Dict[int, bool] = {}      # TODO: we can definitely cache the f1 assignment when constructing the formula
        f1_term_assignment: Dict[int, Tuple[float, float, bool]] = {}     # for each term, we are tracking its min and max value, the last element is whether we consider equal or not

        # build dict for f1 (context), loop through each clause in f2 (which you know the length is 1) and see if there is conflict assignment
        for cid in f1.clause_id_to_literal_id.keys():
            assert len(f1.clause_id_to_literal_id[cid]) == 1
            lit_id: str = f1.clause_id_to_literal_id[cid][0]
            atom: Atom = f1.literal_id_to_atom[lit_id]
            neg = f1.literal_id_to_negation[lit_id]

            assert not (isinstance(atom.predicate, RelationPred) and neg), "RelationPred should not contain negation"

            if isinstance(atom.predicate, RelationPred):
                """
                the context should have relationpred of the following form : a binop b
                where either a or b can be a constant, but not both
                and a non variable/constant term should show up in either a or b
                we enumerate all the scenario in the following
                """
                op = atom.predicate.op_name
                arg1 = atom.predicate.arg1
                arg2 = atom.predicate.arg2
                assert not (isinstance(arg1, Constant) and isinstance(arg2, Constant)), "RelationPred does not allow LHS and RHS to be both constants"
                assert not (not isinstance(arg1, Constant) and not isinstance(arg2, Constant)), "RelationPred in f2 does not allow LHS and RHS both not contain constants"

                var_id = None
                var_value = None

                if not isinstance(arg1, Constant) and isinstance(arg2, Constant):
                    var_id = arg1.id
                    var_value = arg2.arg

                    f1_term_assignment[var_id] = generate_interval(op, var_value)

                elif isinstance(arg1, Constant) and not isinstance(arg2, Constant):
                    var_id = arg2.id
                    var_value = arg1.arg

                    f1_term_assignment[var_id] = generate_interval(op, var_value, inverse=True)

            elif isinstance(atom.predicate, Binding) or isinstance(atom.predicate, Prov):
                # print('id for {}:{}'.format(lit.atom.predicate, lit.atom.id))
                f1_atom_assignment[atom.id] = not neg
            else:
                raise TypeError

        printd("f1 atom assignment:", f1_atom_assignment)
        printd("f1 term assignment:", f1_term_assignment)

        for cid in f2.clause_id.keys():

            assert len(f2.clause_id_to_literal_id[cid]) == 1

            lid: str = f2.clause_id_to_literal_id[cid][0]
            atom: Atom = f2.literal_id_to_atom[lid]
            neg = f2.literal_id_to_negation[lid]

            # printd("f2 atom:", atom)

            assert not (isinstance(atom.predicate, RelationPred) and neg), "RelationPred should not contain negation"

            if isinstance(atom.predicate, RelationPred):

                op = atom.predicate.op_name
                arg1 = atom.predicate.arg1
                arg2 = atom.predicate.arg2

                if not isinstance(arg1, Constant) and isinstance(arg2, Constant):
                    """
                    all we are doing here is to check whether 
                    """
                    if arg1.id in f1_term_assignment:

                        new_interval = generate_interval(op, arg2.arg)
                        if not check_valid_interval(intersect_interval(f1_term_assignment[arg1.id], new_interval)):
                            output = SolverOutput(False)
                            return output

                elif not isinstance(arg1, Constant) and not isinstance(arg2, Constant):
                    # print("arg1:", arg1)
                    # print("arg2:", arg2)
                    # print("arg1:", arg1.id)
                    # print("arg2:", arg2.id)
                    if arg1.id in f1_term_assignment and arg2.id in f1_term_assignment:
                        """
                        both arg need to be in the context, otherwise if we have a free var, there is nothing we can infer 
                        """

                        # print("f1_term_assignment:", f1_term_assignment[arg1.id])
                        # print("f1_term_assignment:", f1_term_assignment[arg2.id])
                        # print("op:", op)

                        # special case: all equal
                        if f1_term_assignment[arg1.id][0] == f1_term_assignment[arg1.id][1] and f1_term_assignment[arg2.id][0] == f1_term_assignment[arg2.id][1]:
                            if op == 'eq':
                                res = f1_term_assignment[arg1.id][0] == f1_term_assignment[arg2.id][0]
                                if not res:
                                    output = SolverOutput(res)
                                    return output
                            elif op == 'geq':
                                res = f1_term_assignment[arg1.id][0] >= f1_term_assignment[arg2.id][0]
                                if not res:
                                    output = SolverOutput(res)
                                    return output
                            elif op == 'leq':
                                res = f1_term_assignment[arg1.id][0] <= f1_term_assignment[arg2.id][0]
                                if not res:
                                    output = SolverOutput(res)
                                    return output
                            elif op == 'ge':
                                res = f1_term_assignment[arg1.id][0] > f1_term_assignment[arg2.id][0]
                                if not res:
                                    output = SolverOutput(res)
                                    return output
                            elif op == 'le':
                                res = f1_term_assignment[arg1.id][0] < f1_term_assignment[arg2.id][0]
                                if not res:
                                    output = SolverOutput(res)
                                    return output
                            else:
                                raise Exception("Unsupported operator: {}".format(op))
                        else:
                            if not check_valid_interval(intersect_interval(f1_term_assignment[arg1.id], f1_term_assignment[arg2.id])):
                                output = SolverOutput(False)
                                return output

                elif isinstance(arg1, Constant) and not isinstance(arg2, Constant):
                    if arg2.id in f1_term_assignment:
                        new_interval = generate_interval(op, arg1.arg, inverse=True)
                        if not check_valid_interval(intersect_interval(f1_term_assignment[arg2.id], new_interval)):
                            output = SolverOutput(False)
                            return output
                else:
                    raise ValueError

            elif isinstance(atom.predicate, Binding) or isinstance(atom.predicate, Prov):
                # printd('id for {}:{}'.format(atom.predicate, atom.id))
                if atom.id in f1_atom_assignment:
                    # printd(f1_atom_assignment[atom.id])
                    if (not f1_atom_assignment[atom.id]) ^ neg:
                        # printd("False")
                        output = SolverOutput(False)
                        return output
            else:
                raise TypeError

        return SolverOutput(True)

    def get_model(self, imp: Implication) -> Iterator[Tuple[Atom, Predicate]]:

        """
        A highly specialized get_model function that is used to get model for the formula structure
        prov_predicates -> prod_predicates
        NOTE THAT prod_predicates is a XORFormula
        """

        assert isinstance(imp.antecedent, CNFFormula)
        assert isinstance(imp.consequent, XORFormula)

        prov_preds: CNFFormula = imp.antecedent
        prod_preds: XORFormula = imp.consequent

        f1_atom_assignment: Dict[int, bool] = {}  # TODO: we can definitely cache the f1 assignment when constructing the formula

        # build dict for f1 (context), loop through each clause in f2 (which you know the length is 1) and see if there is conflict assignment
        for cid in prov_preds.clause_id.keys():
            assert len(prov_preds.clause_id_to_literal_id[cid]) == 1
            lit_id: str = prov_preds.clause_id_to_literal_id[cid][0]
            atom: Atom = prov_preds.literal_id_to_atom[lit_id]
            neg = prov_preds.literal_id_to_negation[lit_id]

            assert isinstance(atom.predicate, Prov)
            f1_atom_assignment[atom.id] = not neg

        # print(f1_atom_assignment)

        cid: str
        lid: str
        for cid in prod_preds.clause_id.keys():
            for lid in prod_preds.clause_id_to_literal_id[cid]:

                atom: Atom = prod_preds.literal_id_to_atom[lid]
                neg = prod_preds.literal_id_to_negation[lid]

                assert not neg
                assert isinstance(atom.predicate, Prov)

                if atom.id in f1_atom_assignment:
                    # printd(f1_atom_assignment[atom.id])
                    if f1_atom_assignment[atom.id]:
                        # this means that prov is set to true
                        yield atom, atom.predicate
                else:
                    # this means that prov never specify anything, which means it can be true
                    yield atom, atom.predicate

        raise StopIteration


class Z3Solver(Solver):
    """
    Solver that secretly invoke Z3
    """
    def __init__(self):
        super().__init__()
        self.solver: z3.Solver = z3.Solver()

    def solve(self, imp: Implication, use_encode=False) -> SolverOutput:
        """
        encode the formula into z3 format and call z3
        if we use encode, then we do not assume the precedent is true
        if we not use encode, then this is a validity check
        """
        if use_encode:
            claim = imp.encode()
            # print("claim:", claim)
            self.solver.add(claim)
        else:
            premise = imp.antecedent.get_z3_formula()
            claim = And(premise, imp.get_z3_formula())
            self.solver.add(claim)

        res = self.solver.check()
        self.solver.reset()
        if res == unsat:
            return SolverOutput(False)
        elif res == sat:
            return SolverOutput(True)
        elif res == unknown:
            raise Z3Exception('Fail to solve')

    def get_model(self, imp: Implication) -> Iterator:

        premise = imp.antecedent.get_z3_formula()
        claim = And(premise, imp.get_z3_formula())
        self.solver.add(claim)
        # print(self.solver)

        res = self.solver.check()

        while res == sat:
            model = self.solver.model()
            yield model

            block = []
            for var in model:
                block.append(var() != model[var])
            self.solver.add(Or(block))
            res = self.solver.check()

        self.solver.reset()
        # print("reset!")
        raise StopIteration
