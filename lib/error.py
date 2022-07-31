"""
This file defines the exception thrown that can be used for lemma learning
"""
from typing import List, Tuple, Dict

from lib.type.predicate import Predicate

conflict_error_factory: Dict = {}


class ConflictError(Exception):
    def __init__(self, message, conflict_predicate: List[Tuple[Predicate, bool]]):
        super().__init__(message)
        self.conflict_predicate: List[Tuple[Predicate, bool]] = conflict_predicate

    def __repr__(self):
        return f"ConflictError(message={self.args[0]}, conflict_predicate={self.conflict_predicate})"


class ForwardError(Exception):
    def __init__(self, message, cause: Predicate, reason: Predicate):
        super(ForwardError, self).__init__()
        self.cause = cause
        self.reason = reason


def create_conflict_error(message: str, conflict_predicate: List[Tuple[Predicate, bool]]):
    conflict_predicate_str = conflict_predicate.__repr__()
    if conflict_error_factory.get(conflict_predicate_str) is None:
        conflict_error_factory[conflict_predicate] = ConflictError(message, conflict_predicate)
    return conflict_error_factory[conflict_predicate]
