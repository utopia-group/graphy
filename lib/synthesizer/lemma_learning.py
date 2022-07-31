from typing import Tuple, Optional, List

from lib.type.base_type import BaseType, TableType, ListType, QualitativeType, QuantitativeType, ContinuousType, DiscreteType, CellType
from lib.type.predicate import Predicate, Prov
from lib.type.ref_type import BaseRefType
from lib.type.type_system import generate_base_type_lemma_auto_helper, generate_context_auto_helper


class Lemma:
    def __init__(self, context: BaseRefType, req: BaseRefType):

        self.context: BaseRefType = context
        self.req: BaseRefType = req

        # bookeeping for the base-type only lemmas
        self.base_only: bool = False
        self.base_key: Optional[str] = None
        self.base_str: Optional[str] = None
        self.is_bot: bool = False
        self.additional_prov: List[Prov] = []

        # bookeeping for context base type only but req has ref type
        self.prov_req_only: bool = False
        self.contain_disjunction: bool = False
        self.prov_req: List[List[Tuple[Prov, bool]]] = []  # the outer list is for Or and the inner list is for And, each literal is a tuple of (predicate, is_negated)

    def get_req_str(self) -> str:
        assert self.is_bot and self.base_only
        assert isinstance(self.req.base, TableType)
        return self.base_str

    def __repr__(self):
        if self.base_only:
            return f"{self.context} -> {self.req}"
        else:
            return f"{self.context} -> {self.req} | {self.prov_req}"


def generate_context_auto(arg1: Tuple[str, BaseType], arg2: Tuple[str, BaseType]) -> TableType:
    """
    generate the most general type that can differentiate type of arg1 and arg2 and cast it into a TableType as the context of the lemma
    """

    context_type = generate_context_auto_helper(arg1[1], arg2[1])
    context = TableType()
    assert isinstance(context_type, CellType)
    context.add(arg2[0], context_type)

    return context


def generate_base_type_lemma(arg1: Tuple[str, BaseType], arg2: Tuple[str, BaseType]) -> Optional[Lemma]:
    """
    Lemma generation for table base type as guard only (req can contain qualifiers)
    arg1: src type
    arg2: dst type
    NOTE:   Technically, genreq should be automatically generated. but i am hard code it now.
            For base type guard only, the procedure can actually be exact (i.e. you don't need to take the most general case because the compatibility here is the same as subtyping)
            Since the ultimate goal is still to guarantee subtyping, the following procedure is sound.
            another assumption being made here is that these rules are sound up to 2 operations allowed i think.
    """

    assert isinstance(arg1[1], ListType)
    assert isinstance(arg2[1], ListType)

    context = TableType()
    # context.add(arg2[0], arg2[1].T)

    # Let's hard code some stuff
    if arg1[1].T.name == 'Discrete':  # type: ignore
        if arg2[1].T.name in ['Nominal', 'Ordinal', 'Temporal', 'Qualitative']:  # type: ignore
            context.add(arg2[0], QualitativeType())
            return generate_bot_lemma(context, (arg1[0], ListType(QualitativeType())), [])
        elif arg2[1].T.name == 'Continuous':  # type: ignore
            context.add(arg2[0], ContinuousType())
            return generate_prov_lemma(context, (arg2[0], ListType(ContinuousType()), [[(Prov(arg2[0], 'mean'), False)], [(Prov(arg2[0], 'sum'), False)]]), [])
        elif arg2[1].T.name == 'Discrete':  # type: ignore
            context.add(arg2[0], DiscreteType())
            return generate_prov_lemma(context, (arg2[0], ListType(DiscreteType()), [[(Prov(arg2[0], 'mean'), True), (Prov(arg2[0], 'sum'), True)]]), [])
    elif arg1[1].T.name == 'Continuous':  # type: ignore
        if arg2[1].T.name in ['Nominal', 'Ordinal', 'Temporal', 'Qualitative']:  # type: ignore
            context.add(arg2[0], QualitativeType())
            return generate_bot_lemma(context, (arg1[0], ListType(QualitativeType())), [])
        elif arg2[1].T.name == 'Discrete':  # type: ignore
            context.add(arg2[0], DiscreteType())
            return generate_prov_lemma(context, (arg2[0], ListType(DiscreteType()), [[(Prov(arg2[0], 'count'), False)]]), [])
        elif arg2[1].T.name == 'Continuous':  # type: ignore
            context.add(arg2[0], ContinuousType())
            return generate_prov_lemma(context, (arg2[0], ListType(ContinuousType()), [[(Prov(arg2[0], 'count'), True)]]), [])
    elif arg1[1].T.name in ['Nominal', 'Ordinal', 'Temporal', 'Qualitative']:  # type: ignore
        if arg2[1].T.name == 'Continuous':  # type: ignore
            context.add(arg2[0], ContinuousType())
            return generate_bot_lemma(context, (arg1[0], ListType(QualitativeType())), [Prov(arg1[0], 'sum'), Prov(arg1[0], 'mean')])
        elif arg2[1].T.name == 'Discrete':  # type: ignore
            context.add(arg2[0], QuantitativeType())
            return generate_prov_lemma(context, (arg2[0], ListType(QuantitativeType()), [[(Prov(arg2[0], 'count'), False)]]), [Prov(arg1[0], 'sum'), Prov(arg1[0], 'mean')])
        elif arg2[1].T.name == 'Quantitative':  # type: ignore
            context.add(arg2[0], QuantitativeType())
            return generate_prov_lemma(context, (arg2[0], ListType(QuantitativeType()), [[(Prov(arg2[0], 'count'), False)]]), [Prov(arg1[0], 'sum'), Prov(arg1[0], 'mean')])
        else:
            context.add(arg2[0], arg2[1].T)  # type: ignore
            return generate_none_lemma(context, (arg2[0], ListType(arg2[1].T)))  # type: ignore
    else:
        pass

    return None


def generate_base_type_lemma_auto(arg1: Tuple[str, BaseType], arg2: Tuple[str, BaseType], k=2) -> Optional[Lemma]:
    """
    This is the automated version of genReq as what we described in the paper. Ideally this should support any type system with properly labeled constraint
    argument format: (column_name, base_type of that column)
    arg1: src type
    arg2: dst type
    Note: this function does not support generate addition_prov as generate_base_type_lemma() can do
    """

    src_type = arg1[1]
    dst_type = arg2[1]
    assert isinstance(src_type, ListType)
    assert isinstance(dst_type, ListType)
    context = generate_context_auto((arg1[0], src_type.T), (arg2[0], dst_type.T))

    # call helper
    lemma_output = generate_base_type_lemma_auto_helper(src_type.T, dst_type.T, k)

    # make the output a lemma object
    if lemma_output is None:
        # generate a bot lemma
        # for bot i don't think req is important here
        if isinstance(dst_type.T, QualitativeType):
            return generate_bot_lemma(context, (arg2[0], ListType(QualitativeType())), [])
        else:
            return generate_bot_lemma(context, (arg2[0], ListType(dst_type.T)), [])
    else:
        if len(lemma_output[0]) == 0:
            if len(lemma_output[1]) == 1 and len(lemma_output[1][0]) == 3:
                # if the disallowed operation is all the operations we can predict (which are 3), then we generate a none lemma
                return generate_none_lemma(context, (arg2[0], ListType(dst_type.T)))
            else:
                # otherwise generate a prov operation with the negation predicate enabled
                # the reason we can do this is our specification is actually complete
                assert len(lemma_output[1]) == 1
                return generate_prov_lemma(context, (arg2[0], ListType(dst_type.T), [[(Prov(arg2[0], op_name), True) for op_name in lemma_output[1][0]]]), [])
        else:
            return generate_prov_lemma(context, (arg2[0], ListType(dst_type.T), [[(Prov(arg2[0], op_name), False) for op_name in conj] for conj in lemma_output[0]]), [])


def generate_ref_type_lemma(arg1, arg2) -> Lemma:
    """
    Lemma generation that involves qualifier for guard only (ref can contain qualifiers)
    We skip generating lemma for qualifiers unsat for efficiency reason
    """
    raise NotImplementedError


"""
Following are helper method
"""


def generate_bot_lemma(context: TableType, dst: Tuple[str, BaseType], additional_prov) -> Lemma:
    assert isinstance(dst[1], ListType)

    req_base = TableType()
    req_base.add(dst[0], dst[1].T)
    req = BaseRefType(req_base, is_bot=True)
    new_lemma = Lemma(BaseRefType(context), req)
    new_lemma.base_only = True
    new_lemma.base_key = dst[0]
    new_lemma.base_str = dst[1].T.name
    new_lemma.is_bot = True
    new_lemma.additional_prov = additional_prov
    return new_lemma


def generate_none_lemma(context: TableType, dst: Tuple[str, BaseType]) -> Lemma:
    assert isinstance(dst[1], ListType)

    req_base = TableType()
    req_base.add(dst[0], dst[1].T)
    req = BaseRefType(req_base, is_bot=False)
    new_lemma = Lemma(BaseRefType(context), req)
    new_lemma.base_only = True
    new_lemma.base_key = dst[0]
    new_lemma.base_str = dst[1].T.name
    return new_lemma


def generate_prov_lemma(context: TableType, dst: Tuple[str, BaseType, List[List[Tuple[Predicate, bool]]]], additional_prov) -> Lemma:
    assert isinstance(dst[1], ListType)
    req_base = TableType()
    req_base.add(dst[0], dst[1].T)
    req = BaseRefType(req_base, is_bot=False)
    new_lemma = Lemma(BaseRefType(context), req)
    new_lemma.prov_req_only = True
    new_lemma.contain_disjunction = len(dst[2]) > 1
    new_lemma.prov_req = dst[2]
    new_lemma.additional_prov = additional_prov

    return new_lemma
