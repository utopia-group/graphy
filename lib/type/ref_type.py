from typing import Optional, List

from lib.type.base_type import BaseType
from lib.type.formula import CNFFormula


class RefType:
    """
    abstract class for refinement type
    """

    def __init__(self):
        pass

    def __repr__(self):
        raise NotImplementedError

    def duplicate(self):
        pass

    def duplicate_ignore_base(self):
        pass


class BaseRefType(RefType):
    """
    Basic refinement type: basetype associated with a formula (which is optional )
    """
    def __init__(self, base: BaseType, constraint: Optional[CNFFormula] = None, fields: Optional[List[str]] = None, prob: float = 0.0, is_bot: bool = False):
        super().__init__()
        self.base: BaseType = base
        self.constraint: CNFFormula = constraint
        self.fields: List[str] = fields    # stores all the possible fields so we can init the table type

        # special assignment (for NullOp only), we can optimize this later
        self.enc = None
        self.col = None

        # special field for probability
        self.prob = prob

        # special field for noting if the qualifer is bot
        self.is_bot = is_bot

    def duplicate(self, next_formula_id=None) -> 'BaseRefType':
        new_type = BaseRefType(self.base.duplicate())
        new_type.fields = self.fields
        if self.constraint is not None:
            new_type.constraint = self.constraint.duplicate(next_formula_id)

        new_type.col = self.col
        new_type.enc = self.enc
        return new_type

    def duplicate_ignore_base(self, next_formula_id=None) -> 'BaseRefType':
        new_type = BaseRefType(self.base)
        new_type.fields = self.fields
        if self.constraint is not None:
            new_type.constraint = self.constraint.duplicate(next_formula_id)

        new_type.col = self.col
        new_type.enc = self.enc
        return new_type

    def update_constraint(self):
        """
        Update the type
        """
        raise NotImplementedError

    def __repr__(self):

        if self.is_bot:
            assert self.constraint is None
            return '{{v: {} | _|_ }}'.format(self.base.__repr__())
        else:
            return '{{v: {} | {}}}'.format(self.base.__repr__(), self.constraint.__repr__())


class FunctionType(RefType):
    """
    Function type: from one refinement type to another refinement type
    """

    def __init__(self, _input: BaseRefType, output: BaseRefType):
        super().__init__()
        self.input: BaseRefType = _input
        self.output: BaseRefType = output

    def __repr__(self):
        return '{} -> {}'.format(repr(self.input), repr(self.output))
