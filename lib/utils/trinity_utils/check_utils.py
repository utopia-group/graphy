from typing import Tuple

from lib.utils.trinity_utils.macro_utils import CardinalityOp


def is_subtype(subtype: str, supertype: str):
    """
    Returns True if subtype is a subtype of supertype.
    """
    if subtype == supertype:
        return True
    if subtype == 'Continuous':
        return supertype in ['Quantitative']
    if subtype == 'Discrete':
        return supertype in ['Quantitative']
    if subtype == 'Aggregate':
        return supertype in ['Quantitative']
    if subtype == 'Ordinal':
        return supertype in ['Qualitative']
    if subtype == 'Nominal':
        return supertype in ['Qualitative']
    if subtype == 'Temporal':
        return supertype in ['Qualitative']
    return False


def check_cardinality(fact: Tuple[CardinalityOp, int], requirement: Tuple[CardinalityOp, int]) -> bool:
    """
    Check all the possible combinations of the cardinality operations
    """

    fact_op, fact_val = fact
    req_op, req_val = requirement

    if fact_op == CardinalityOp.EQ:
        if req_op == CardinalityOp.EQ:
            return fact_val == req_val
        if req_op == CardinalityOp.GT:
            return fact_val > req_val
        if req_op == CardinalityOp.GTE:
            return fact_val >= req_val
        if req_op == CardinalityOp.LT:
            return fact_val < req_val
        if req_op == CardinalityOp.LTE:
            return fact_val <= req_val

        raise ValueError(f'Unknown cardinality operation: {req_op}')
    if fact_op == CardinalityOp.GT:
        if req_op == CardinalityOp.EQ:
            return (fact_val + 1) < req_val
        if req_op == CardinalityOp.GT:
            return True
        if req_op == CardinalityOp.GTE:
            return True
        if req_op == CardinalityOp.LT:
            return (fact_val + 1) < req_val     # +1 because we are dealing with integers
        if req_op == CardinalityOp.LTE:
            return (fact_val + 1) <= req_val

        raise ValueError(f'Unknown cardinality operation: {req_op}')
    if fact_op == CardinalityOp.GTE:
        if req_op == CardinalityOp.EQ:
            return fact_val <= req_val
        if req_op == CardinalityOp.GT:
            return True
        if req_op == CardinalityOp.GTE:
            return True
        if req_op == CardinalityOp.LT:
            return fact_val < req_val
        if req_op == CardinalityOp.LTE:
            return fact_val <= req_val

        raise ValueError(f'Unknown cardinality operation: {req_op}')
    if fact_op == CardinalityOp.LT:
        if req_op == CardinalityOp.EQ:
            return (fact_val - 1) >= req_val    # -1 because we are dealing with integers
        if req_op == CardinalityOp.GT:
            return (fact_val - 1) > req_val
        if req_op == CardinalityOp.GTE:
            return (fact_val - 1) >= req_val
        if req_op == CardinalityOp.LT:
            return True
        if req_op == CardinalityOp.LTE:
            return True

        raise ValueError(f'Unknown cardinality operation: {req_op}')
    if fact_op == CardinalityOp.LTE:
        if req_op == CardinalityOp.EQ:
            return fact_val >= req_val
        if req_op == CardinalityOp.GT:
            return fact_val > req_val
        if req_op == CardinalityOp.GTE:
            return fact_val >= req_val
        if req_op == CardinalityOp.LT:
            return True
        if req_op == CardinalityOp.LTE:
            return True

        raise ValueError(f'Unknown cardinality operation: {req_op}')

    raise ValueError(f'Invalid cardinality operation: {fact_op}')
