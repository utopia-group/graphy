from typing import List, Union, Optional, Dict

from z3 import Int, SeqRef, ExprRef, Function, IntSort

"""
Class for all our predicate:
Atomic formula:
p1 := Prov(col, beta)
    | Binding(col, vp)
p2 := p2' binop p2'
p2' := Card(table)
        | Max(table)
        | Min(table)
        | const 
table := T | List[col]
I make sure table does not have any recursion here
TODO: need to update this class to fit the new DSL
"""


class Term:
    """
    So all term is in the format of function and argument P(arg)
    """

    def __init__(self, name: str, arg: Optional[Union[int, str, List[str]]]):
        self.name: str = name
        self.arg: Optional[Union[int, str, List[str]]] = arg

        self.id = None

    def get_z3_formula(self) -> ExprRef:
        raise NotImplementedError

    def encode(self, context: Dict):
        raise NotImplementedError

    def arg_hash(self) -> str:
        if isinstance(self.arg, str):
            return '{}'.format(hash(self.arg))
        elif isinstance(self.arg, list):
            arg_sum = sum([hash(a) for a in self.arg])
            return '{}'.format(arg_sum)
        else:
            raise TypeError('{} should not show up here'.format(type(self.arg)))

    def create_proj_encode(self, context: Dict):
        """
        This is the enc function for proj. Since currently we don't have Proj, we temporarily put the encode func here
        """
        assert isinstance(self.arg, str) or isinstance(self.arg, list)

        f_proj = Function('Proj', IntSort(), IntSort(), IntSort())
        if 'T' not in context:
            context['T'] = Int('T')
        cname_id = 'c_{}'.format(self.arg_hash())
        if cname_id not in context:
            context[cname_id] = Int(cname_id)
        f_proj_var = f_proj(context['T'], context[cname_id])

        return f_proj_var

    def __repr__(self):
        return '{}({})'.format(self.name, self.arg)


class Card(Term):
    """
    TODO: we also need additional data constraint here
    I mean what's the benefit????
    """

    def __init__(self, arg: Union[str, List[str]], name: str = 'Card'):
        super().__init__(name, arg)

    def get_z3_formula(self):
        return Int('{}{}'.format('c', self.id))

    def encode(self, context: Dict):
        """
        note that this the arguments of card is doing an implicit projection function, we first need to encode that
        and then encode the entire function into an uninterpreted function
        """

        f_card = Function('Card', IntSort(), IntSort())
        return f_card(self.create_proj_encode(context))


class Max(Term):
    def __init__(self, arg: List[str], name: str = 'Max'):
        super().__init__(name, arg)

    def get_z3_formula(self):
        """
        TODO: this is incorrect
        """
        return Int('{}{}'.format('m', self.id))

    def encode(self, context: Dict):
        f_max = Function('Max', IntSort(), IntSort())
        return f_max(self.create_proj_encode(context))


class Min(Term):
    def __init__(self, arg: List[str], name: str = 'Min'):
        super().__init__(name, arg)

    def get_z3_formula(self):
        """
        TODO: this is incorrect
        """
        return Int('{}{}'.format('m', self.id))

    def encode(self, context: Dict):
        f_min = Function('Min', IntSort(), IntSort())
        return f_min(self.create_proj_encode(context))


class Constant(Term):
    def __init__(self, arg: Union[int, str], name: str = 'Constant'):
        super().__init__(name, arg)

    def get_z3_formula(self):
        if isinstance(self.arg, int):
            return self.arg
        else:
            print('ERROR: not able to generate z3 formula for {}'.format(self.arg))
            raise NotImplementedError

    def encode(self, context: Dict):
        if isinstance(self.arg, int):
            return self.arg
        else:
            print('ERROR: not able to generate z3 formula for {}'.format(self.arg))
            raise NotImplementedError

    def __repr__(self):
        return str(self.arg)


class Variable(Term):
    def __init__(self, name: str = 'v'):
        super().__init__(name, arg=None)

    def get_z3_formula(self):
        return SeqRef('{}{}'.format('v', self.id))

    def encode(self, context: Dict):
        raise NotImplementedError


class Predicate:
    def __init__(self, op_name: str):
        self.op_name: str = op_name

    def encode(self, context: Dict):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class Binding(Predicate):
    """
    binding in p1
    enc := x | y | color | column
    """

    def __init__(self, col: str, enc: str, op_name: str = 'Binding'):
        super().__init__(op_name)
        self.col = col
        self.enc = enc

    def encode(self, context: Dict):
        """
        This is the expected behavior, encode of binding predicate can be handled at the atom level
        """
        raise NotImplementedError

    def __repr__(self):
        return '{}({},{})'.format(self.op_name, self.col, self.enc)


class Prov(Predicate):
    """
    prov in p1
    beta can be: mutate, filter, mean, sum, count, bin
    """

    def __init__(self, col: str, beta: str, op_name: str = 'Prov'):
        super().__init__(op_name)
        self.col = col
        self.beta = beta

    def encode(self, context: Dict):
        """
        This is the expected behavior, encode of prov predicate can be handled at the atom level
        """
        raise NotImplementedError

    def __repr__(self):
        return '{}({},{})'.format(self.op_name, self.col, self.beta)


class RelationPred(Predicate):
    """
    p2
    """

    def __init__(self, op_name: str, arg1: Term, arg2: Term):
        super().__init__(op_name)
        self.arg1: Term = arg1
        self.arg2: Term = arg2

    def get_z3_formula(self):
        if self.op_name == 'eq':
            return self.arg1.get_z3_formula() == self.arg2.get_z3_formula()
        elif self.op_name == 'ge':
            return self.arg1.get_z3_formula() > self.arg2.get_z3_formula()
        elif self.op_name == 'geq':
            return self.arg1.get_z3_formula() >= self.arg2.get_z3_formula()
        elif self.op_name == 'le':
            return self.arg1.get_z3_formula() < self.arg2.get_z3_formula()
        elif self.op_name == 'leq':
            return self.arg1.get_z3_formula() <= self.arg2.get_z3_formula()
        else:
            raise ValueError

    def encode(self, context: Dict):
        """
        semantic_term rule in Fig 22
        """
        arg1_encode = self.arg1.encode(context)
        arg2_encode = self.arg2.encode(context)

        if self.op_name == 'eq':
            return arg1_encode == arg2_encode
        elif self.op_name == 'ge':
            return arg1_encode > arg2_encode
        elif self.op_name == 'geq':
            return arg1_encode >= arg2_encode
        elif self.op_name == 'le':
            return arg1_encode < arg2_encode
        elif self.op_name == 'leq':
            return arg1_encode <= arg2_encode
        else:
            raise ValueError

    def __repr__(self):
        return '{}({},{})'.format(self.op_name, repr(self.arg1), repr(self.arg2))
