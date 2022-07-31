from typing import Optional, List

from lib.grammar.symbol import Symbol
from lib.type.ref_type import RefType, BaseRefType


class Node:
    def __init__(self, _id: int, name: str, sym: Symbol, goal_type: RefType):
        self.id: int = _id
        self.name: str = name
        self.sym: Symbol = sym
        self.__goal_type: RefType = goal_type

    def get_goal_type_restricted(self) -> RefType:
        return self.__goal_type

    def set_goal_type(self, goal_type: RefType):
        assert self.__goal_type is None
        self.__goal_type = goal_type

    def repr_helper(self, children: Optional[List[str]] = None):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class NonterminalNode(Node):
    def __init__(self, _id: int, name: str, sym: Symbol, goal_type: RefType, prod):
        super().__init__(_id, name, sym, goal_type)
        self.prod = prod

    def repr_helper(self, children: Optional[List[str]] = None):

        # print("name:", self.name)
        if self.sym.name == 'pv_func':
            # print("vfunc c:", children)
            return 'lambda {}. {}'.format(children[0], children[1])
        else:
            if children is not None:
                children = [str(child) for child in children]
            return '{}({})'.format(self.name, ', '.join(children))


class VariableNode(Node):
    def __init__(self, _id: int, name: str, sym: Symbol, goal_type: RefType, arg_idx = None):
        super().__init__(_id, name, sym, goal_type)

        # keep track of its argument index
        self.arg_idx = arg_idx

    def repr_helper(self, children: Optional[List[str]] = None):
        return 'v_{}'.format(str(self.id))


class TerminalNode(Node):
    def __init__(self, _id: int, name, sym: Symbol, value: str, goal_type: BaseRefType):
        super().__init__(_id, name, sym, goal_type)
        self.value: str = value

    def repr_helper(self, children: Optional[List[str]] = None):
        return self.value