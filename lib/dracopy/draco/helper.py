import json
from typing import Dict, List

import pandas as pd

from lib.dracopy.draco.utils import data_to_asp
from lib.dracopy.draco.run import run_clingo, run_clingo_2, Model, is_satisfiable


def is_valid(draco_query: List[str], debug=False) -> bool:
    """ Check a task.
        Args:
            draco_query: a list of facts
        Returns:
            whether the task is valid
    """
    _, stdout = run_clingo(
        draco_query,
        files=["define.lp", "hard.lp", "hard-integrity.lp"],
        silence_warnings=True,
        debug=debug,
    )

    return json.loads(stdout)["Result"] != "UNSATISFIABLE"


def is_valid2(draco_query: List[str], debug=False) -> bool:
    return is_satisfiable(draco_query, files=["define.lp", "hard.lp", "hard-integrity.lp"])


def get_violation(
    draco_query: List[str],
    constants: Dict[str, str] = None,
    files: List[str] = None,
    relax_hard=False,
    debug=False,
    multiple_solution=False):

    generator = run_clingo_2(
        draco_query, relax_hard=True, get_violation=True)

    model: Model = next(generator)
    violating_constraints = [sym.arguments[0].name for sym in model.answer_set if sym.name == 'hard']
    violating_constraints_str = set([str(c) for c in violating_constraints])

    return violating_constraints_str


def read_data_to_asp(file: str) -> List[str]:
    """ Reads the given JSON file and generates the ASP definition.
        Args:
            file: the json data file
        Returns:
            the asp definition.
    """
    if file.endswith(".json"):
        with open(file) as f:
            data = json.load(f)
            return data_to_asp(data)
    elif file.endswith(".csv"):
        df = pd.read_csv(file)
        df = df.where((pd.notnull(df)), None)
        data = list(df.T.to_dict().values())
        asp = data_to_asp(data)
        return asp
    else:
        raise Exception("invalid file type")
