"""
Run constraint solver to complete spec.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any

import clyngor
from attr import dataclass
from clyngor.answers import Answers
from lib.dracopy.draco.utils import asp_to_vl
import clingo


DEBUG = False

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# --- Logic Programming Constraints --- #
# [*] define.lp: declares the domains to visualization attributes and defines useful helper functions.
#   You almost definitely want this file.
# [*] generate.lp: describe the candidate solution (i.e. search space)
# [*] hard.lp: restricts the search space to only well-formed and expressive specifications.
# soft.lp: defines soft constraints in the form of violation/1 and violation/2 predicates.
#   By themselves, these predicates don't change the search.
# weights.lp: declares default (hand tuned) weights similar to those in CompassQL.
#   There is one constant for each rule in soft.lp. We use this file to generate assign_weights.lp.
# assign_weights.lp: uses violation_weight/2 to assign every violation predicate a weight.
#   These weights usually come from weights.lp. This file is generated from weights.lp.
# optmize.lp: defined the minimization function.
# [*] output.lp: declares which predicates should be shown when an answer set is printed.

DRACO_LP = [
    "define.lp",
    "generate.lp",
    "hard.lp",
    "hard-integrity.lp",
    # "soft.lp",
    # "weights.lp",
    # "assign_weights.lp",
    # "optimize.lp",
    "output.lp",
]
DRACO_LP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../asp")


file_cache: Dict[str, bytes] = {}
file_cache2: Dict[str, str] = {}


class Result:
    props: List[str]
    cost: Optional[int]
    violations: Dict[str, int]

    def __init__(self, answers: Answers, cost: Optional[int] = None) -> None:
        violations: Dict[str, int] = defaultdict(int)
        props: List[str] = []

        for ((head, body),) in answers:
            if head == "cost":
                cost = int(body[0])
            elif head == "soft":
                violations[body[0]] += 1
            else:
                b = ",".join(map(str, body))
                props.append(f"{head}({b}).")

        self.props = props
        self.violations = violations
        self.cost = cost

    def as_vl(self) -> Dict:
        return asp_to_vl(self.props)

    def __repr__(self):
        return str(self.props)

@dataclass
class Model:
    """Class for a model.

    Attributes:
        :answer_set: The answer set of this model.
            An answer set is a list of Clingo Symbols.
        :cost: The cost of this answer set.
        :number: The sequence number of this answer.
    """

    answer_set: List[clingo.Symbol]
    cost: int
    number: int


def load_file(path: str) -> bytes:
    # print(path)
    content = file_cache.get(path)
    if content is not None:
        return content
    with open(path, encoding='utf-8') as f:
        content = f.read().encode("utf8")
        file_cache[path] = content
        return content


def load_file2(path: str) -> str:
    content = file_cache2.get(path)
    if content is not None:
        return content
    with open(path, encoding='utf-8') as f:
        content = f.read()
        file_cache2[path] = content
        return content


def generate_asp_prog(draco_query: List[str],
    constants: Dict[str, str] = None,
    files: List[str] = None,
    relax_hard=False,
    debug=False,
    encode=True,
    get_violation=False,
    soft=False,
    remove_column=False,
    remove_color=False):

    if files is None:
        files = []
        files.extend(DRACO_LP)

    if relax_hard and "hard-integrity.lp" in files:
        files.remove("hard-integrity.lp")

    if get_violation and 'output.lp' in files:
        files.remove('output.lp')

    if soft:
        files.extend([
            "soft.lp",
            "weights.lp",
            # 'weights_learned.lp',
            "assign_weights.lp",
            "optimize.lp",
        ])

    program = "\n".join(draco_query)
    file_names = [os.path.join(DRACO_LP_DIR, f) for f in files]
    if encode:
        asp_program = b"\n".join(map(load_file, file_names)) + program.encode("utf8")

        if not remove_column:
            if not remove_color:
                asp_program = b"single_channel(x;y;color;column;theta).\n" + asp_program
            else:
                asp_program = b"single_channel(x;y;column;theta).\n" + asp_program
        else:
            if not remove_color:
                asp_program = b"single_channel(x;y;color;theta).\n" + asp_program
            else:
                asp_program = b"single_channel(x;y;theta).\n" + asp_program

    else:
        constants_prog = "\n".join(["# const {} = {}.".format(name, value) for name, value in constants]) if \
            constants is not None else ""
        asp_program = constants_prog + "\n".join(map(load_file2, file_names)) + program
        # asp_program = "\n".join(map(load_file2, file_names)) + program

        if not remove_column:
            asp_program = "single_channel(x;y;color;column;theta).\n" + asp_program
        else:
            asp_program = "single_channel(x;y;color;theta).\n" + asp_program



    if debug:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fd:
            fd.write(program)
            if DEBUG:
                logger.info('Debug ASP with "clingo %s %s"', " ".join(file_names), fd.name)

    return asp_program


def run_clingo(
    draco_query: List[str],
    constants: Dict[str, str] = None,
    files: List[str] = None,
    relax_hard=False,
    silence_warnings=False,
    debug=False,
    multiple_solution=False,
    soft=False,
    remove_column=False,
    remove_color=False
) -> Tuple[bytes, bytes]:
    """
    Run draco and return stderr and stdout
    """

    # default args
    constants = constants or {}
    asp_program = generate_asp_prog(draco_query, files=files, relax_hard=relax_hard, debug=debug, soft=soft, remove_column=remove_column, remove_color=remove_color)

    options = ["--outf=2", "--models=0"]
    if multiple_solution:
        # print all solutions
        options += ['--opt-mode=OptN']
        options += ["--quiet=1,2,2"]    # we still only allow optimal solution, but obtain all the optimal ones
    else:
        # only print the optimal solution
        options += ["--quiet=1,2,2"]

    if silence_warnings:
        options.append("--warn=no-atom-undefined")
    for name, value in constants.items():
        options.append(f"-c {name}={value}")

    cmd = ["clingo"] + options
    if DEBUG:
        logger.debug("Command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        args=cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    stdout, stderr = proc.communicate(asp_program)

    return (stderr, stdout)

def run(
    draco_query: List[str],
    constants: Dict[str, str] = None,
    files: List[str] = None,
    relax_hard=False,
    silence_warnings=False,
    debug=False,
    clear_cache=False,
    multiple_solution=False,
    soft=False,
    remove_column=False,
    remove_color=False
) -> Union[Result, List[Result], None]:
    """ Run clingo to compute a completion of a partial spec or violations. """

    # Clear file cache. useful during development in notebooks.
    if clear_cache and file_cache:
        logger.warning("Cleared file cache")
        file_cache.clear()

    stderr, stdout = run_clingo(
        draco_query, constants, files, relax_hard, silence_warnings, debug, multiple_solution, soft, remove_column, remove_color
    )

    # print("stdout:", stdout)

    try:
        json_result = json.loads(stdout)
    except json.JSONDecodeError:
        logger.error("stdout: %s", stdout)
        logger.error("stderr: %s", stderr)
        raise

    if stderr:
        logger.error(stderr)

    result = json_result["Result"]

    if result == "UNSATISFIABLE":
        #print(f'{result}')
        if DEBUG:
            print("unsat")
            print(json.loads(stdout))
            logger.info("Constraints are unsatisfiable.")
        return None
    elif result == "OPTIMUM FOUND":
        # get the last witness, which is the best result
        all_answers = json_result["Call"][0]["Witnesses"]

        for answers in all_answers:
            if DEBUG:
                logger.debug(answers["Value"])
                print("answer:", answers["Value"])

        # print(all_answers)

        results = [Result(
            clyngor.Answers(answers["Value"]).sorted,
            cost=json_result["Models"]["Costs"][0],
        ) for answers in all_answers]

        print(f'{result} | {json_result["Models"]["Costs"][0]} | {len(results)}')

        if multiple_solution:
            # NOTE: the cost of non-optimal ones are not correct
            return results
        else:
            return results[-1]
    elif result == "SATISFIABLE":
        all_answers = json_result["Call"][0]["Witnesses"]

        # print(f'{result} | {json_result["Models"]["Number"]}')

        # assert (
        #     json_result["Models"]["Number"] > 1
        # ), "Should not have more than one model if we don't optimize"

        if DEBUG:
            for answers in all_answers:
                print("answer:", answers["Value"])
                logger.debug(answers["Value"])
                print(type(answers["Value"][0]))
        # print(clyngor.Answers(answers["Value"]))
        return [Result(
            clyngor.Answers(answers["Value"]).sorted,
            cost=-1,
        ) for answers in all_answers]
    else:
        logger.error("Unsupported result: %s", result)
        return None


def run_clingo_2(
        draco_query: List[str],
    constants: Dict[str, str] = None,
    files: List[str] = None,
    relax_hard=False,
    silence_warnings=False,
    debug=False,
    multiple_solution=False,
    get_violation=False):

    asp_program = generate_asp_prog(draco_query, constants=constants, files=files, relax_hard=relax_hard, debug=debug,
                                    encode=False, get_violation=get_violation)

    ctl = clingo.Control()
    ctl.add(
        "base",
        [],
        asp_program,
    )
    ctl.ground([("base", [])])

    config: Any = ctl.configuration

    # if models is not None:
    config.solve.models = str(0)

    config.solve.project = 1

    solve_handle = ctl.solve(yield_=True)
    if isinstance(solve_handle, clingo.solving.SolveHandle):
        with solve_handle as handle:
            for model in handle:
                answer_set = model.symbols(shown=True)
                yield Model(answer_set, model.cost, model.number)


def run2(draco_query: List[str],
    constants: Dict[str, str] = None,
    files: List[str] = None,
    relax_hard=False,
    silence_warnings=False,
    debug=False,
    clear_cache=False,
    multiple_solution=False,
    top_k=10) -> Union[List[Result], None]:

    # Clear file cache. useful during development in notebooks.
    if clear_cache and file_cache2:
        logger.warning("Cleared file cache")
        file_cache2.clear()

    generator = run_clingo_2(
        draco_query, constants, files, relax_hard, silence_warnings, debug, multiple_solution
    )

    solutions = []
    try:
        model: Model = next(generator)
        solutions.append(model)
    except StopIteration:
        return None

    if multiple_solution:
        while len(solutions) <= top_k:
            try:
                model: Model = next(generator)
                # print(model)
                solutions.append(model)
            except StopIteration:
                break

    def answer_set_to_str(answer_set):
        answer_set_str = []
        for sym in answer_set:
            answer_set_str.append(str(sym))
        return answer_set_str

    res = [Result(
            clyngor.Answers(answer_set_to_str(model.answer_set)).sorted,
            cost=model.cost,
        ) for model in solutions]
    # print(res)

    return res


def is_satisfiable(draco_query, files):
    try:
        model = next(run_clingo_2(draco_query, files=files))
        return True
    except StopIteration:
        return False
