import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from lib.eval.benchmark import InputDataset
from lib.grammar.production import Production
from lib.grammar.symbol import NonterminalSymbol, TerminalSymbol
from lib.grammar.typed_cfg import TypedCFG
from lib.program import Program
from lib.node import VariableNode
from lib.solver.solver import SolverOutput
from lib.synthesizer.lemma_learning import Lemma
from lib.type.predicate import Prov
from lib.type.ref_type import BaseRefType, RefType


class ProgramSynthesizer:
    """
    Interface for the program synthesizer with some potential shared functions.
    """

    def __init__(self, grammar: TypedCFG, disable_lemma: bool = False, no_prov: bool = False, no_table: bool = False):

        self.no_provenance = no_prov
        self.no_table = no_table

        self.grammar: TypedCFG = grammar
        self.disable_lemma: bool = disable_lemma
        self.program_counter = itertools.count(0)
        # TODO: need to set a couple hyperparam that reads from some config (either args or config.py)
        self.depth = 3

        # for pruning
        self.explored_progs_types = []
        self.learned_conflicts: Dict[str, Lemma] = {}
        self.learned_conflicts_based: Dict[str, List[Lemma]] = defaultdict(list) # map from a column to a set of lemma, used for top-level pruning
        self.learned_conflicts_prov: Dict[str, Prov] = {}  # map from a prov to a lemma (these are the lemma we learned that is impossible to achieve in any condition), used for top-level pruning

        # for stats
        self.solution_explored: int = 0
        self.partial_prog_visited: int = 0
        self.goal_type_pruned: int = 0

        # for debug
        self.enumerated_progs: List[str] = []

    def synthesize_init(self, _input: InputDataset):
        raise NotImplementedError

    def synthesize(self, spec: BaseRefType, _input: InputDataset) -> List[Program]:
        raise NotImplementedError

    def synthesize_visualization(self, initial_prog: Program, _input: InputDataset = None) -> List[Program]:
        """
        All synthesizer shared the same synthesize visualization function
        Visualization part is a top-down mechanism, the only extra thing need to do here is the strengthening
        """
        synthesized_program: List[Program] = self.enumerate_top_down(
            [initial_prog], 'visualization', input=_input)

        # print("synthesized program in vis:", synthesized_program)
        # do the strengthening
        new_synthesized_program = []
        for prog in synthesized_program:

            if _input is not None:
                strengthened_progs = self.enumerate_strengthen_prog(prog, _input.input_type)
            else:
                strengthened_progs = self.enumerate_strengthen_prog(prog, _input)
            new_synthesized_program.extend(strengthened_progs)

        # print("new_synthesized_prog in vis:", new_synthesized_program)

        return new_synthesized_program

    def enumerate_strengthen_prog(self, prog: Program, input_type: Optional[BaseRefType] = None) -> List[Program]:
        """
        helper function for synthesize_visualization
        once finished synthesize the visualization part,
        we strengthen the program by enumerating all possible types that type check the syntax
        """

        strengthen_progs: List[Program] = []
        plot_func_nid: int = -1
        strengthened_types: List[RefType] = []
        # NOTE: we are doing some goal type pruning here
        #   Premise for pruning: whatever we are prune out here is goal-type directed
        #   (i.e. if one potential goal type is in feasible, then that mean all possible instantiation of the goal type is impossible, up to a certain depth)
        # try:
        plot_func_nid, strengthened_types = prog.viz_table_input_strengthening(input_type=input_type)
        # except ConflictError as e:
        #     self.learned_conflicts[e.conflict_predicate.__repr__()] = e

        for new_type in strengthened_types:
            new_prog = prog.duplicate(next(self.program_counter))
            new_prog.node_to_refined_type[plot_func_nid] = new_type
            strengthen_progs.append(new_prog)

        return strengthen_progs

    def enumerate_top_down(self, worklist: List[Program], component=None, input: InputDataset = None, add_none=False) -> List[Program]:
        """
        a top-down enumeration of the program.
        """

        synthesized_progs: List[Program] = []

        largest_node_id_index = 0

        while len(worklist) > 0:

            curr_prog: Program = worklist.pop()
            # print("partial prog {}: {}".format(self.partial_prog_visited, curr_prog))
            self.partial_prog_visited += 1

            # if self.partial_prog_visited > 10000:
            #     break

            if curr_prog.is_concrete(component):
                curr_largest_idx = next(curr_prog.node_id_counter)
                if curr_largest_idx > largest_node_id_index:
                    largest_node_id_index = curr_largest_idx
                synthesized_progs.append(curr_prog)
                continue

            selected_var: VariableNode = curr_prog.select_var_node()

            if isinstance(selected_var.sym, NonterminalSymbol):
                new_progs = self.expand_nonterminal_sym(curr_prog, selected_var, None) if input is None else self.expand_nonterminal_sym(curr_prog, selected_var, input.input_type)
                worklist.extend(new_progs)
            elif isinstance(selected_var.sym, TerminalSymbol):
                new_progs = self.expand_terminal_sym(
                    curr_prog, selected_var, input, add_none=add_none)
                worklist.extend(new_progs)
            else:
                raise TypeError

        return synthesized_progs

    def expand_nonterminal_sym(self, prog: Program, var_node: VariableNode, input_type: Optional[BaseRefType] = None) -> List[Program]:
        """
        Used in top-down enum: Expand a non-terminal symbol in a partial program.
        """
        raise NotImplementedError

    def expand_terminal_sym(self, prog: Program, var_node: VariableNode, input: InputDataset = None, add_none=False) -> List[Program]:
        """
        Used in top-down enum: Expand a terminal symbol in a partial program.
        """
        raise NotImplementedError

    def synthesize_table_transformation(self, curr_prog: Optional[Program], input_table: InputDataset) -> List[Program]:
        """
        A bottom-up synthesizer that handles the table transformation synthesis procedure
        note that this synthesizer is a naive enumeration with no goal-type directed part
        This method is deprecated I think
        """
        synthesized_program: List[Program] = []

        base_case_program, full_program = self.init_base_case_prog(
            curr_prog, input_table)

        # synthesized_program.extend(full_program)
        # enqueue the base program
        worklist: List[Program] = base_case_program
        candidate_components: List[Program] = []
        curr_depth = 1

        while curr_depth < self.depth and len(worklist) > 0:

            curr_candidate: Program = worklist.pop()

            # expand the current predicate bottom-up and connect with the main program
            new_subprograms, new_programs = self.expand_subprogram(
                curr_prog, curr_candidate, curr_depth, input_table)

            candidate_components.extend(new_subprograms)
            synthesized_program.extend(new_programs)

            if len(worklist) == 0:
                curr_depth += 1
                worklist = candidate_components
                candidate_components = []

        return synthesized_program

    def init_base_case_prog(self, curr_prog: Program, input_table: InputDataset) -> Tuple[List[Program], List[Program]]:
        """
        helper function for synthesize_table_transformation
        initialize the base case for the bottom-down procedure
        """
        raise NotImplementedError

    def instantiate_subprogram(self, parent_prog: Program, subprogram: Program) -> Program:
        """
        helper function for synthesize_table_transformation
        once a subprogram is created, merge the program into its parent_prog
        """
        new_prog: Program = parent_prog.duplicate(next(self.program_counter))
        new_prog.node_id_counter = itertools.tee(subprogram.node_id_counter)[1]
        new_prog.fill_subprog_for_var_node(subprogram)

        return new_prog

    def expand_subprogram(self, curr_prog: Program, curr_candidate: Program, curr_depth: int, input: InputDataset) -> Tuple[List[Program], List[Program]]:
        """
        helper function for synthesize_table_transformation
        expand the partial program based on possible productions
        """
        raise NotImplementedError

    def init_prog(self, goal_type: BaseRefType, input_type: BaseRefType, typed: bool = False) -> Program:
        """
        create a new program with the goal_type and synthesizes the top-level construct
        this function should be used across all sub-classes
        """
        prog = Program(next(self.program_counter), goal_type)
        plot_prod: List[Production] = self.grammar.get_productions(
            self.grammar.start_sym, 'Plot')
        var_node = prog.add_variable_node(
            self.grammar.start_sym, goal_type, None)
        assert len(plot_prod) == 1
        plot_prod[0].apply(prog, var_node, typed, input_type, no_prov=self.no_provenance)
        prog.to_expand_var_nodes = prog.visualization_root_node

        return prog

    def add_lemmas(self, compatibility_check_res: SolverOutput):
        self.learned_conflicts[str(compatibility_check_res.lemma)] = compatibility_check_res.lemma
        if compatibility_check_res.lemma.base_only and compatibility_check_res.lemma.is_bot:
            self.learned_conflicts_based[compatibility_check_res.lemma.base_key].append(compatibility_check_res.lemma)
        for prov_lemma in compatibility_check_res.lemma.additional_prov:
            self.learned_conflicts_prov[str(prov_lemma)] = prov_lemma
