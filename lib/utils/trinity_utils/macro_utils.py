import enum


class CardinalityOp(enum.Enum):
    """
    Enum for cardinality operations.
    """
    EQ = 0
    GT = 1
    LT = 2
    GTE = 3
    LTE = 4
    NEQ = 5
