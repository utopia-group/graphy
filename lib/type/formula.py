import itertools
from collections import OrderedDict, defaultdict

from typing import List, Dict, Optional, Tuple, FrozenSet, Callable

from z3 import Implies, And, Or, Not, Bool, Xor

from lib.type.predicate import Predicate, Binding, Prov, RelationPred, Constant, Term, Variable, Card


class Atom:
    def __init__(self, _id: int, predicate: Predicate):
        self.id = _id
        self.predicate = predicate

        # if predicate is relationpred, then we need to save the id for each of its argument
        self.id_arg1 = None
        self.id_arg2 = None

    def get_z3_formula(self):
        """
        The z3 formula for atom depends on what types of predicate
        """
        if isinstance(self.predicate, Binding):
            return Bool('{}{}'.format('b', self.id))
        elif isinstance(self.predicate, Prov):
            return Bool('{}{}'.format('p', self.id))
        elif isinstance(self.predicate, RelationPred):
            return self.predicate.get_z3_formula()
        else:
            raise TypeError

    def encode(self, context: Dict):
        """
        The encode output depends on what types of predicate
        """
        if isinstance(self.predicate, Binding):
            return Bool('{}{}'.format('b', self.id))
        elif isinstance(self.predicate, Prov):
            return Bool('{}{}'.format('p', self.id))
        elif isinstance(self.predicate, RelationPred):
            return self.predicate.encode(context)
        else:
            raise TypeError

    def __repr__(self):
        return repr(self.predicate)


class Formula:
    """
    A formula is a set of certain form of clauses
    Just put a general abstraction here
    """

    def __init__(self, _id: int):
        self.id = _id

        self.clause_id: OrderedDict[str, str] = OrderedDict()
        self.clause_id_to_literal_id: Dict[str, List[str]] = defaultdict(list)
        self.clause_id_to_contain_disjunction: Dict[str, bool] = {}
        self.literal_id_to_negation: Dict[str, bool] = {}
        self.literal_id_to_atom: Dict[str, Atom] = {}

        self.clause_factory = itertools.count(0)
        self.literal_factory = itertools.count(0)

    def duplicate(self, _id: int) -> 'Formula':
        pass

    def add_clauses(self, clauses: List[List[Tuple[Predicate, bool]]]):
        for clause in clauses:
            new_clause_id = self.get_next_clause_id()
            self.clause_id[new_clause_id] = ''
            self.add_clause_to_id(clause, new_clause_id)

    def get_next_clause_id(self) -> str:
        return "{}_{}".format(self.id, next(self.clause_factory))

    def get_next_literal_id(self, clause_id) -> str:
        return "{}_{}".format(clause_id, next(self.literal_factory))

    def add_clause_to_id(self, clause: List[Tuple[Predicate, bool]], clause_id: str):
        raise NotImplementedError

    def conjunct_formula(self, _id, f) -> 'Formula':
        pass

    def disjunct_formula(self, _id, f) -> 'Formula':
        pass

    def get_z3_formula(self):
        pass

    def encode(self, context: Dict):
        pass

    def __repr__(self):
        pass


class CNFFormula(Formula):
    """
    A formula is a set of Clauses
    Assumption: we have a CNF formula already, otherwise this doesn't work
    TODO: pretty sure we need some map here
    """

    def __init__(self, _id: int, init_clauses: Optional[List[List[Tuple[Predicate, bool]]]] = None):

        super().__init__(_id)
        self.read_only: bool = False    # read only means this formula should not be used to do conjunction/disjunction

        self.binding_clause_ids: List[str] = []
        self.prov_clause_ids: List[str] = []

        self.binding_enc_to_literal_id: Dict[str, str] = {}
        self.prov_op_to_literal_id: Dict[str, str] = {}

        self.binding_str_to_literal_id: Dict[str, str] = {}  # don't remember why I need this
        self.prov_str_to_literal_id: Dict[str, str] = {}  # don't remember why I need this

        self.card_col_to_literal_id: Dict[str, str] = {}

        self.value_assign: Optional[str] = None

        if init_clauses is not None:
            self.add_clauses(init_clauses)

    def duplicate(self, _id) -> 'CNFFormula':

        new_formula = CNFFormula(_id)

        new_formula.clause_id = self.clause_id.copy()
        new_formula.clause_id_to_literal_id = self.clause_id_to_literal_id.copy()
        new_formula.clause_id_to_contain_disjunction = self.clause_id_to_contain_disjunction.copy()
        new_formula.literal_id_to_negation = self.literal_id_to_negation.copy()
        new_formula.literal_id_to_atom = self.literal_id_to_atom.copy()

        new_formula.binding_clause_ids = self.binding_clause_ids.copy()
        new_formula.prov_clause_ids = self.prov_clause_ids.copy()

        new_formula.binding_enc_to_literal_id = self.binding_enc_to_literal_id.copy()
        new_formula.prov_op_to_literal_id = self.prov_op_to_literal_id.copy()

        new_formula.binding_str_to_literal_id = self.binding_str_to_literal_id.copy()
        new_formula.prov_str_to_literal_id = self.prov_str_to_literal_id.copy()

        new_formula.card_col_to_literal_id = self.card_col_to_literal_id.copy()

        new_formula.clause_factory = itertools.tee(self.clause_factory)[1]
        new_formula.literal_factory = itertools.tee(self.literal_factory)[1]

        new_formula.value_assign = self.value_assign
        new_formula.read_only = self.read_only

        return new_formula

    def conjunct_formula(self, _id, f: 'CNFFormula') -> 'CNFFormula':

        new_f = self.duplicate(_id)

        # print('self:', self)
        # print('new_f:', new_f)
        # print('f:', f)

        # print("self.prov_claise_id:", self.prov_clause_ids)
        # print("f.prov_claise_id:", f.prov_clause_ids)

        new_f.clause_id.update(f.clause_id)
        new_f.clause_id_to_literal_id.update(f.clause_id_to_literal_id)
        new_f.clause_id_to_contain_disjunction.update(f.clause_id_to_contain_disjunction)
        new_f.literal_id_to_negation.update(f.literal_id_to_negation)
        new_f.literal_id_to_atom.update(f.literal_id_to_atom)

        new_f.binding_clause_ids.extend(f.binding_clause_ids)
        new_f.prov_clause_ids.extend(f.prov_clause_ids)

        new_f.binding_enc_to_literal_id.update(f.binding_enc_to_literal_id)
        new_f.prov_op_to_literal_id.update(f.prov_op_to_literal_id)

        new_f.binding_str_to_literal_id.update(f.binding_str_to_literal_id)
        new_f.prov_str_to_literal_id.update(f.prov_str_to_literal_id)

        new_f.card_col_to_literal_id.update(f.card_col_to_literal_id)

        new_f.clause_factory = itertools.count(max(new_f.clause_factory.__next__(), max(f.clause_factory.__next__(), 0)))
        new_f.literal_factory = itertools.count(max(new_f.literal_factory.__next__(), max(f.literal_factory.__next__(), 0)))

        # print("new_f:", new_f)

        return new_f

    def disjunct_formula(self, _id, f: 'CNFFormula') -> 'CNFFormula':
        raise NotImplementedError

    def add_literal_to_clause(self, literal: Tuple[Predicate, bool], clause_id: str) -> str:

        pred, neg = literal
        atom = create_atom(pred)
        new_literal_id = self.get_next_literal_id(clause_id)
        self.literal_id_to_atom[new_literal_id] = atom
        self.literal_id_to_negation[new_literal_id] = neg

        contain_binding: bool = False
        contain_prov: bool = False

        if isinstance(pred, Binding):
            self.binding_enc_to_literal_id[pred.enc] = new_literal_id
            self.binding_str_to_literal_id[str(pred)] = new_literal_id
            contain_binding = True
        elif isinstance(pred, Prov):
            self.prov_op_to_literal_id[pred.beta] = new_literal_id
            self.prov_str_to_literal_id[str(pred)] = new_literal_id
            contain_prov = True
        elif isinstance(pred, RelationPred):
            if pred.op_name == 'coeq':
                self.value_assign = pred.arg2.arg
            else:
                # assumption RelationPred is always in the form Col = Constant
                self.card_col_to_literal_id[str(pred.arg1.arg)] = new_literal_id

        if contain_binding:
            assert isinstance(pred, Binding)
            self.binding_clause_ids.append(clause_id)
        if contain_prov:
            assert isinstance(pred, Prov)
            self.prov_clause_ids.append(clause_id)

        return new_literal_id

    def add_clause_to_id(self, clause: List[Tuple], clause_id: str) -> str:
        """
        c1 -> [l1, l2, l3]  # disjunction
        """
        for lit in clause:
            new_lit_id = self.add_literal_to_clause(lit, clause_id)
            self.clause_id_to_literal_id[clause_id].append(new_lit_id)

        if len(self.clause_id_to_literal_id[clause_id]) > 1:
            self.clause_id_to_contain_disjunction[clause_id] = True

        return clause_id

    def contain_disjunction(self) -> bool:
        return any(cond for cond in self.clause_id_to_contain_disjunction.values())

    def get_binding_formula(self, _id) -> 'CNFFormula':

        assert len(self.binding_clause_ids) > 0

        f_binding = CNFFormula(_id)
        f_binding.read_only = True
        f_binding.clause_id = OrderedDict([(clause_id, '') for clause_id in self.binding_clause_ids])
        f_binding.clause_id_to_literal_id = self.clause_id_to_literal_id.copy()
        f_binding.clause_id_to_contain_disjunction = self.clause_id_to_contain_disjunction.copy()
        f_binding.literal_id_to_atom = self.literal_id_to_atom.copy()
        f_binding.literal_id_to_negation = self.literal_id_to_negation.copy()
        f_binding.binding_clause_ids = self.binding_clause_ids.copy()
        f_binding.binding_enc_to_literal_id = self.binding_enc_to_literal_id.copy()
        f_binding.binding_str_to_literal_id = self.binding_str_to_literal_id.copy()

        f_binding.clause_factory = itertools.tee(self.clause_factory)[0]
        f_binding.literal_factory = itertools.tee(self.literal_factory)[0]

        return f_binding

    def get_prov_formula(self, _id) -> 'CNFFormula':

        # assert len(self.prov_clause_ids) > 0

        if _id == -1:
            f_prov = CNFFormula(self.id)
            f_prov.read_only = True
            f_prov.clause_id = OrderedDict([(clause_id, '') for clause_id in self.prov_clause_ids])
            f_prov.clause_id_to_literal_id = self.clause_id_to_literal_id
            f_prov.clause_id_to_contain_disjunction = self.clause_id_to_contain_disjunction
            f_prov.literal_id_to_atom = self.literal_id_to_atom
            f_prov.literal_id_to_negation = self.literal_id_to_negation
            f_prov.prov_clause_ids = self.prov_clause_ids
            f_prov.prov_op_to_literal_id = self.prov_op_to_literal_id
            f_prov.prov_str_to_literal_id = self.prov_str_to_literal_id
        else:
            f_prov = CNFFormula(_id)
            f_prov.read_only = True
            f_prov.clause_id = OrderedDict([(clause_id, '') for clause_id in self.prov_clause_ids])
            f_prov.clause_id_to_literal_id = self.clause_id_to_literal_id.copy()
            f_prov.clause_id_to_contain_disjunction = self.clause_id_to_contain_disjunction.copy()
            f_prov.literal_id_to_atom = self.literal_id_to_atom.copy()
            f_prov.literal_id_to_negation = self.literal_id_to_negation.copy()
            f_prov.prov_clause_ids = self.prov_clause_ids.copy()
            f_prov.prov_op_to_literal_id = self.prov_op_to_literal_id.copy()
            f_prov.prov_str_to_literal_id = self.prov_str_to_literal_id.copy()

            f_prov.clause_factory = itertools.tee(self.clause_factory)[0]
            f_prov.literal_factory = itertools.tee(self.literal_factory)[0]

        return f_prov

    def negate_negated_prov(self, pred_str: str) -> bool:

        if pred_str.startswith('Prov'):
            if pred_str in self.prov_str_to_literal_id:
                if self.literal_id_to_negation[self.prov_str_to_literal_id[pred_str]]:
                    self.literal_id_to_negation[self.prov_str_to_literal_id[pred_str]] = False
                    return True
        elif pred_str.startswith('Binding'):
            if pred_str in self.binding_str_to_literal_id:
                if self.literal_id_to_negation[self.binding_str_to_literal_id[pred_str]]:
                    self.literal_id_to_negation[self.binding_str_to_literal_id[pred_str]] = False
                    return True

        return False

    def negate_prov(self, pred_str: str) -> bool:

        if pred_str.startswith('Prov'):
            if pred_str in self.prov_str_to_literal_id:
                if not self.literal_id_to_negation[self.prov_str_to_literal_id[pred_str]]:
                    self.literal_id_to_negation[self.prov_str_to_literal_id[pred_str]] = True
                    return True
        elif pred_str.startswith('Binding'):
            if pred_str in self.binding_str_to_literal_id:
                if not self.literal_id_to_negation[self.binding_str_to_literal_id[pred_str]]:
                    self.literal_id_to_negation[self.binding_str_to_literal_id[pred_str]] = True
                    return True

        return False

    def get_neg(self, pred_str: str) -> Optional[bool]:
        if pred_str.startswith('Prov'):
            if pred_str in self.prov_str_to_literal_id:
                return self.literal_id_to_negation[self.prov_str_to_literal_id[pred_str]]
            else:
                return None
        elif pred_str.startswith('Binding'):
            if pred_str in self.binding_str_to_literal_id:
                return self.literal_id_to_negation[self.binding_str_to_literal_id[pred_str]]
            else:
                return None

    def update_card(self, col_name: str, val, op_name: str, containment=False, special=False, neg=False, special_2=False, special_3=False):
        """
        colname means there is only one colname (not a list)
        containment: True: update as long as the colname is a subset
        assumption: our cardinality to literal id must contain this entry, because this is how we derive the input
        Note: this is the highly-optimized implementation that involves the forget operation.
            No predicate is removed from the formula because of efficiency reason, instead, they are replaced with constraints that is trivially correct
        """
        if special:
            # special means we update cardinality for inverse op in summarize
            # TODO: essentially remove all the duplicate constraints
            for lit_id, atom in self.literal_id_to_atom.items():
                pred = atom.predicate
                if isinstance(pred, RelationPred):
                    if neg:
                        if col_name not in pred.arg1.arg:
                            if pred.arg2.arg == 'T':
                                if pred.op_name == 'eq':
                                    self.literal_id_to_atom[lit_id] = create_atom(RelationPred('leq', pred.arg1, pred.arg2))
                    else:
                        if col_name in pred.arg1.arg:
                            if pred.arg2.arg == 'T':
                                if pred.op_name == 'eq':
                                    self.literal_id_to_atom[lit_id] = create_atom(RelationPred('leq', pred.arg1, pred.arg2))

        elif special_3:
            # we need to remove all constraints with that thing
            for lit_id, atom in self.literal_id_to_atom.items():
                pred = atom.predicate
                if isinstance(pred, RelationPred):
                    # print(pred.arg1.arg, pred.arg2.arg)
                    if isinstance(pred.arg1.arg, list):
                        if col_name in pred.arg1.arg:
                            if containment:
                                # that means we remove/relax any cardinality info for any cardinality related to colname (used in filter)
                                self.literal_id_to_atom[lit_id] = create_atom(RelationPred('geq', pred.arg1, create_term(Constant, 1)))
                            else:
                                if len(pred.arg1.arg) == 1:
                                    self.literal_id_to_atom[lit_id] = create_atom(RelationPred('geq', pred.arg1, create_term(Constant, 1)))


        elif special_2:
            # special_2 means we update cardinality for inverse op in bin
            for lit_id, atom in self.literal_id_to_atom.items():
                pred = atom.predicate
                if isinstance(pred, RelationPred):
                    if neg:
                        if col_name not in pred.arg1.arg:
                            if isinstance(pred.arg2, Constant):
                                if pred.op_name == 'leq':
                                    self.literal_id_to_atom[lit_id] = create_atom(RelationPred('geq', pred.arg1, create_term(Constant, 20)))
                    else:
                        if col_name in pred.arg1.arg:
                            if isinstance(pred.arg2, Constant):
                                if pred.op_name == 'leq':
                                    self.literal_id_to_atom[lit_id] = create_atom(RelationPred('geq', pred.arg1, create_term(Constant, 20)))

        else:

            if not containment:
                if col_name == 'T':
                    key = col_name
                else:
                    key = str([col_name])

                atom = self.literal_id_to_atom[self.card_col_to_literal_id[key]]
                assert isinstance(atom.predicate, RelationPred)
                if val != -1:
                    new_atom = create_atom(RelationPred(op_name, atom.predicate.arg1, create_term(Constant, val)))
                    self.literal_id_to_atom[self.card_col_to_literal_id[key]] = new_atom
                else:
                    if isinstance(atom.predicate.arg2, Constant):
                        new_atom = create_atom(RelationPred(op_name, atom.predicate.arg1, create_term(Constant, atom.predicate.arg2.arg)))
                        self.literal_id_to_atom[self.card_col_to_literal_id[key]] = new_atom

            else:
                for lit_id in self.card_col_to_literal_id.values():
                    atom = self.literal_id_to_atom[lit_id]
                    assert isinstance(atom.predicate, RelationPred)
                    if col_name in atom.predicate.arg1.arg:
                        if val != -1:
                            new_atom = create_atom(RelationPred(op_name, atom.predicate.arg1, create_term(Constant, val)))
                            self.literal_id_to_atom[lit_id] = new_atom
                        else:
                            # if val is -1, then value doesn't matter, we just want to flip the symbol
                            if isinstance(atom.predicate.arg2, Constant):
                                new_atom = create_atom(RelationPred(op_name, atom.predicate.arg1, create_term(Constant, atom.predicate.arg2.arg)))
                                self.literal_id_to_atom[lit_id] = new_atom

    def query_card(self, key) -> int:
        """
        just to be safe here
        we know we can directly if our key is a str
        else we need to search through the list
        return -1 if not found
        """
        if isinstance(key, str):
            if key == 'T':
                key = key
            else:
                key = str([key])

            if key in self.card_col_to_literal_id:
                atom: Atom = self.literal_id_to_atom[self.card_col_to_literal_id[key]]
                assert isinstance(atom.predicate, RelationPred)
                return atom.predicate.arg2.arg
            else:
                return -1

        if isinstance(key, list):
            for lit_id in self.card_col_to_literal_id.values():
                atom: Atom = self.literal_id_to_atom[lit_id]
                assert isinstance(atom.predicate, RelationPred)
                if len(key) == len(atom.predicate.arg1.arg):
                    if set(key) == set(atom.predicate.arg1.arg):
                        return atom.predicate.arg2.arg

        return -1

    def query_binding(self, enc: str) -> Optional[Binding]:
        if enc in self.binding_enc_to_literal_id:
            lit_id = self.binding_enc_to_literal_id[enc]
            atom = self.literal_id_to_atom[lit_id]
            assert isinstance(atom.predicate, Binding)
            return atom.predicate
        else:
            return None

    def get_atom(self, cid: str = None, lid: str = None, l_idx: int = 0) -> Atom:

        if lid is None and cid is not None:
            return self.literal_id_to_atom[self.clause_id_to_literal_id[cid][l_idx]]
        else:
            assert lid is not None
            return self.literal_id_to_atom[lid]

    def update(self):
        raise NotImplementedError

    def get_literal_z3_formula(self, literal_id: str):
        atom: Atom = self.literal_id_to_atom[literal_id]
        neg: bool = self.literal_id_to_negation[literal_id]
        if neg:
            return Not(atom.get_z3_formula())
        else:
            return atom.get_z3_formula()

    def get_clause_z3_formula(self, clause_id: str):
        literals = self.clause_id_to_literal_id[clause_id]
        if len(literals) == 1:
            return self.get_literal_z3_formula(literals[0])
        else:
            return Or([self.get_literal_z3_formula(lid) for lid in literals])

    def get_z3_formula(self):
        clauses = list(self.clause_id)
        if len(clauses) == 1:
            return self.get_clause_z3_formula(clauses[0])
        else:
            return And([self.get_clause_z3_formula(cid) for cid in clauses])

    def get_literal_encode(self, literal_id: str, context: Dict):
        atom: Atom = self.literal_id_to_atom[literal_id]
        neg: bool = self.literal_id_to_negation[literal_id]

        if neg:
            return Not(atom.encode(context))
        else:
            return atom.encode(context)

    def get_clause_encode(self, clause_id: str, context: Dict):
        literals = self.clause_id_to_literal_id[clause_id]
        if len(literals) == 1:
            return self.get_literal_encode(literals[0], context)
        else:
            return Or([self.get_literal_encode(lid, context) for lid in literals])

    def encode(self, context: Dict):
        clauses = list(self.clause_id)
        if len(clauses) == 1:
            return self.get_clause_encode(clauses[0], context)
        else:
            return And([self.get_clause_encode(cid, context) for cid in clauses])

    def get_vegalite_format(self) -> Tuple[Dict, Dict]:
        """
        generate the vegalite encoding part according to the refined clause
        i think i can trust the binding info because they are exactly the case
        TODO: might need to return some boolean which means we need to query the program for additional information to finish the vegalite program (e.g. bin, i ignore them for now)
        """
        assert len(self.binding_enc_to_literal_id) > 0
        encoding_dict = {}
        field_to_enc_mapping = {}
        x_field = None
        for enc, lit_id in self.binding_enc_to_literal_id.items():
            atom: Atom = self.literal_id_to_atom[lit_id]
            assert isinstance(atom.predicate, Binding)
            encoding_dict[enc] = {"field": atom.predicate.col}
            field_to_enc_mapping[atom.predicate.col] = enc
            if enc == 'x':
                x_field = atom.predicate.col

        # now loop through the provenance and other stuff
        # NOTE: the final refined type should not contain any disjunction
        for cid in self.clause_id.keys():
            # print(self.__clause_id_to_literals_id[cid])
            assert len(self.clause_id_to_literal_id[cid]) == 1
            literal_id = self.clause_id_to_literal_id[cid][0]
            atom: Atom = self.literal_id_to_atom[literal_id]

            if isinstance(atom.predicate, Prov) and not self.literal_id_to_negation[literal_id]:
                prov_predicate: Prov = atom.predicate
                field = prov_predicate.col
                if prov_predicate.beta == 'bin':
                    encoding_dict[field_to_enc_mapping[field]]['bin'] = True
                elif prov_predicate.beta == 'count':
                    encoding_dict['y'] = {}
                    encoding_dict['y']['aggregate'] = 'count'
                    encoding_dict['y']['field'] = x_field
                    encoding_dict['y']['type'] = 'quantitative'
                else:
                    encoding_dict[field_to_enc_mapping[field]]['aggregate'] = prov_predicate.beta

        return encoding_dict, field_to_enc_mapping

    def repr_literal_helper(self, lid: str) -> str:
        atom_str = self.literal_id_to_atom[lid].__repr__()
        if self.literal_id_to_negation[lid]:
            return '{}{}'.format('¬', atom_str)
        else:
            return atom_str

    def repr_clause_helper(self, cid: str) -> str:
        return ' ∨ '.join([self.repr_literal_helper(lid) for lid in self.clause_id_to_literal_id[cid]])

    def __repr__(self):
        return ' ∧ '.join([self.repr_clause_helper(cid) for cid in self.clause_id])


class XORFormula(Formula):
    """
    A type of formula that specialize in XOR only
    we still keep the over structure of Clause, Literal, Atom, but note that the meaning is different from the CNF formula
    """

    def __init__(self, _id: int, init_clauses: List[List[Tuple[Predicate, bool]]] = None):
        super().__init__(_id)

        if init_clauses is not None:
            self.add_clauses(init_clauses)

    def duplicate(self, _id: int) -> 'Formula':
        raise NotImplementedError

    def add_clause_to_id(self, clause: List[Tuple[Predicate, bool]], clause_id: str):
        """
        c1 -> [l1, l2, l3]  # disjunction
        """
        for lit in clause:
            new_lit_id = self.add_literal_to_clause(lit, clause_id)
            self.clause_id_to_literal_id[clause_id].append(new_lit_id)

        if len(self.clause_id_to_literal_id[clause_id]) > 1:
            self.clause_id_to_contain_disjunction[clause_id] = True

        return clause_id

    def add_literal_to_clause(self, lit: Tuple[Predicate, bool], clause_id: str) -> str:

        pred, neg = lit
        atom = create_atom(pred)
        new_literal_id = self.get_next_literal_id(clause_id)
        self.literal_id_to_atom[new_literal_id] = atom
        self.literal_id_to_negation[new_literal_id] = neg

        return new_literal_id

    def z3_xor_helper(self, clauses: List, caller: Callable, context: Optional[Dict] = None) -> Xor:
        """
            XOR formula is a special case of CNF formula
        """
        assert len(clauses) >= 2

        if len(clauses) == 2:
            if context is None:
                return Xor(caller(clauses[0]), caller(clauses[1]))
            else:
                return Xor(caller(clauses[0], context), caller(clauses[1], context))
        else:
            if context is None:
                arg2 = Xor(caller(clauses[-1]), caller(clauses[-2]))
                for i in range(len(clauses) - 3, -1, -1):
                    arg2 = Xor(caller(clauses[i]), arg2)
                return arg2
            else:
                arg2 = Xor(caller(clauses[-1], context), caller(clauses[-2], context))
                for i in range(len(clauses) - 3, -1, -1):
                    arg2 = Xor(caller(clauses[i], context), arg2)
                return arg2

    def get_literal_z3_formula(self, literal_id: str):
        atom: Atom = self.literal_id_to_atom[literal_id]
        neg: bool = self.literal_id_to_negation[literal_id]
        if neg:
            return Not(atom.get_z3_formula())
        else:
            return atom.get_z3_formula()

    def get_clause_z3_formula(self, clause_id: str):
        literals = self.clause_id_to_literal_id[clause_id]
        # print("literal:", literals)
        if len(literals) == 1:
            return self.get_literal_z3_formula(literals[0])
        else:
            return self.z3_xor_helper(literals, self.get_literal_z3_formula)

    def get_z3_formula(self):
        clauses = list(self.clause_id)
        # print("clauses:", clauses)
        if len(clauses) == 1:
            return self.get_clause_z3_formula(clauses[0])
        else:
            return Or([self.get_clause_z3_formula(cid) for cid in clauses])

    def get_literal_encode(self, literal_id: str, context: Dict):
        atom: Atom = self.literal_id_to_atom[literal_id]
        neg: bool = self.literal_id_to_negation[literal_id]
        if neg:
            return Not(atom.encode(context))
        else:
            return atom.encode(context)

    def get_clause_encode(self, clause_id: str, context: Dict):
        literals = self.clause_id_to_literal_id[clause_id]
        if len(literals) == 1:
            return self.get_literal_encode(literals[0], context)
        else:
            return self.z3_xor_helper(literals, self.get_literal_encode, context)

    def encode(self, context: Dict):
        clauses = list(self.clause_id)
        if len(clauses) == 1:
            return self.get_clause_encode(clauses[0], context)
        else:
            return Or([self.get_clause_encode(cid, context) for cid in clauses])

    def repr_literal_helper(self, literal_id: str) -> str:
        atom_str = self.literal_id_to_atom[literal_id].__repr__()
        if self.literal_id_to_negation[literal_id]:
            return '{}{}'.format('¬', atom_str)
        else:
            return atom_str

    def repr_clause_helper(self, clause_id: str) -> str:
        return ' ⊕ '.join([self.repr_literal_helper(lid) for lid in self.clause_id_to_literal_id[clause_id]])

    def __repr__(self):
        return ' ⊕ '.join([self.repr_clause_helper(cid) for cid in self.clause_id])


class Implication:
    def __init__(self, antecedent: Formula, consequent: Formula):
        self.antecedent: Formula = antecedent
        self.consequent: Formula = consequent

    def get_z3_formula(self):
        return Implies(self.antecedent.get_z3_formula(), self.consequent.get_z3_formula())

    def encode(self):
        context: Dict = {}
        return Implies(self.antecedent.encode(context), self.consequent.encode(context))

    def __repr__(self):
        return '{} -> {}'.format(repr(self.antecedent), repr(self.consequent))


"""
The following are standard formula operations
"""


def conjunct_formula(_id, f1: CNFFormula, f2: CNFFormula) -> CNFFormula:
    """
    create a new formula using conjunction of two formula
    the reason we produce a new formula is because i am not sure if in place update is going to affect anything
    """
    if f1 is not None and f2 is not None:

        if len(f1.literal_id_to_atom) > len(f2.literal_id_to_atom):
            dup_f = f1
            copy_f = f2
        else:
            dup_f = f2
            copy_f = f1

        new_f = dup_f.conjunct_formula(_id, copy_f)
        return new_f
    elif f1 is not None:
        return f1.duplicate(_id)
    elif f2 is not None:
        return f2.duplicate(_id)


def disjunct_formula(f1: CNFFormula, f2: CNFFormula) -> CNFFormula:
    """
    create a new formula using disjunction of two formulas
    TODO: since disjunction of two cnf is no longer cnf. This takes a while to implement if we want to maintain the cnf structure of the formula
    """
    raise NotImplementedError


# cache that stores all the atom/term created in a predicate to a id

atom_id_count = itertools.count(0)
term_id_count = itertools.count(0)
str_to_atom: Dict[str, Atom] = {}
str_to_term: Dict[str, Term] = {}
set_to_term: Dict[FrozenSet, Term] = {}

atom_id_to_atom: Dict[int, Atom] = {}


def reset_atom_id_counter_cache():
    """
    reset the atom_id_counter
    may want to reset per benchmark to save memory (depend how much memory it takes)
    """
    global str_to_atom, term_id_count, atom_id_count, str_to_term, set_to_term
    str_to_atom = {}
    str_to_term = {}
    set_to_term = {}
    atom_id_count = itertools.count(0)
    term_id_count = itertools.count(0)


def create_term(constructor: type, args) -> Term:
    if constructor == Constant:
        return constructor(args)
    elif constructor == Variable:
        new_var_term = constructor(args)
        new_var_id = next(term_id_count)
        new_var_term.id = new_var_id
        return new_var_term
    elif constructor == Card:
        arg_set = frozenset(args)
        if arg_set not in set_to_term:
            new_term = constructor(args)
            new_id = next(term_id_count)
            new_term.id = new_id
            set_to_term[arg_set] = new_term
        return set_to_term[arg_set]
    else:
        new_term = constructor(args)
        if repr(new_term) not in str_to_term:
            new_id = next(term_id_count)
            new_term.id = new_id
            str_to_term[repr(new_term)] = new_term
        return str_to_term[repr(new_term)]


def create_atom(pred: Predicate) -> Atom:
    """
    create a new atom with respect to a predicate
    key thing here is to make sure same predicate get same id
    """
    if isinstance(pred, Prov) or isinstance(pred, Binding):
        if repr(pred) not in str_to_atom:
            new_id = next(atom_id_count)
            new_atom = Atom(new_id, pred)
            atom_id_to_atom[new_id] = new_atom
            str_to_atom[repr(pred)] = new_atom

        return str_to_atom[repr(pred)]
    elif isinstance(pred, RelationPred):
        new_id = next(atom_id_count)
        return Atom(new_id, pred)
    else:
        raise TypeError


def get_atom_pred(atom_id: str) -> Tuple[Atom, Predicate]:
    int_atom_id = int(atom_id[1:])
    if int_atom_id not in atom_id_to_atom:
        raise KeyError("get_atom_pred: Key error")
    return atom_id_to_atom[int_atom_id], atom_id_to_atom[int_atom_id].predicate
