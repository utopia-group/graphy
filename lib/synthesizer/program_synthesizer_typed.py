import itertools
from typing import List, Tuple, Optional

from lib.eval.benchmark import InputDataset
from lib.grammar.production import Production
from lib.grammar.symbol import TerminalSymbol, NonterminalSymbol, Symbol
from lib.grammar.typed_cfg import TypedCFG
from lib.node import VariableNode
from lib.program import Program
from lib.synthesizer.program_synthesizer import ProgramSynthesizer
from lib.type.base_type import TableType, NullType, ConstColType
from lib.type.predicate import Predicate, Prov
from lib.type.ref_type import BaseRefType
from lib.type.type_system_check import check_compatibility, check_lemmas
from lib.utils.misc_utils import printd


class ProgramSynthesizerTyped(ProgramSynthesizer):
    def __init__(self, grammar: TypedCFG, disable_lemma: bool = False):
        super().__init__(grammar, disable_lemma, no_prov=False, no_table=False)
        self.pruned_productions = []

    def synthesize(self, spec: BaseRefType, _input: InputDataset) -> List[Program]:
        """
        synthesize the program with the given spec
        """
        # clear up formula id
        # reset_formula_id()

        col_var_sym = self.grammar.name_to_sym['sym_col']
        col_var_list_sym = self.grammar.name_to_sym['sym_col_list']
        assert isinstance(col_var_sym, TerminalSymbol)
        col_var_sym.values = spec.fields
        col_var_list_sym.values = [spec.fields]

        synthesized_prog: List[Program] = []

        # the top level function here:
        curr_prog: Program = self.init_prog(spec, _input.input_type, typed=True)
        curr_progs = self.synthesize_visualization(curr_prog, _input)
        printd("finished synthesizing visualization")
        printd("len(curr_progs):", len(curr_progs))

        for prog in curr_progs:
            printd('curr_prog:', repr(prog))
            printd("preupdated goal type:", prog.get_goal_type(prog.table_root_node))

            try:
                prog.update_goal_type()
            except TypeError:
                printd("type error, continue to the next goal")
                continue

            # we can't purely use types for filtering, since there might be special cases (x=Origin, color=Origin, for example)
            format_key = '{}:{}'.format(repr(prog), repr(prog.get_goal_type(prog.table_root_node)))
            if format_key in self.explored_progs_types:
                continue
            else:
                self.explored_progs_types.append(format_key)

            printd("updated goal type:", prog.get_goal_type(prog.table_root_node))

            self.pruned_productions = []
            synthesized_prog += self.synthesize_table_transformation_top_down(prog, _input)

        printd("main synthesized_prog: ", synthesized_prog)

        return synthesized_prog

    def enumerate_top_down(self, worklist: List[Program], component=None, input: InputDataset = None, add_none=False) -> List[Program]:
        return super().enumerate_top_down(worklist, component, input, add_none)

    def synthesize_table_transformation_top_down(self, curr_prog: Program, input: InputDataset) -> List[Program]:
        """
        another top-down synthesizer for the table transformation part given the refined goal type from curr_prog
        """
        synthesized_program: List[Program] = []
        worklist: List[Program] = [curr_prog]

        # top-level goal pruning
        goal_type = curr_prog.get_goal_type(curr_prog.table_root_node)
        assert isinstance(goal_type, BaseRefType)
        # print("curr_prog.table_root_node.goal_type:", goal_type)
        if not check_lemmas(goal_type, self.learned_conflicts).res:
            # print("did not pass check_lemmas")
            self.goal_type_pruned += 1
            return synthesized_program

        while len(worklist) > 0:

            curr_candidate: Program = worklist.pop(0)
            # print("partial prog {}: {}".format(self.partial_prog_visited, curr_candidate))
            self.partial_prog_visited += 1

            printd("curr_candidate({}):{}".format(curr_candidate.depth, curr_candidate))

            if curr_candidate.depth > self.depth:
                continue

            if curr_candidate.is_concrete('table'):

                assert curr_candidate.is_concrete()
                self.solution_explored += 1

                # TODO: we need to cache the results of the bottom up inference
                if len(curr_candidate.nodes_pushed) > 1:
                    while len(curr_candidate.nodes_pushed) > 1:
                        curr_node_id = curr_candidate.nodes_pushed[-1]
                        curr_candidate.update_type(input.input_type, curr_node_id)

                        if curr_node_id == curr_candidate.table_root_node.id:
                            break

                # print("curr_candidate:", curr_candidate)
                goal_type = curr_candidate.get_goal_type(curr_candidate.table_root_node)
                refined_type = curr_candidate.node_to_refined_type[curr_candidate.table_root_node.id]
                # print("refined_type2:", refined_type)
                # print("goal_type2:", goal_type)
                subtyping_res = check_compatibility(refined_type, goal_type)

                if subtyping_res.error_code == 1:
                    printd("pass subtyping check")

                    curr_candidate.overall_type_strengthening()
                    synthesized_program.append(curr_candidate)
                    printd("synthesized_program:", synthesized_program)
                else:
                    print("failed subtyping check")
                    # assert False
                continue

            # if the program is not concrete, we need to expand it

            # first, we need to select a variable node to expand
            selected_var: VariableNode = curr_candidate.select_var_node()

            if isinstance(selected_var.sym, NonterminalSymbol):
                new_progs = self.expand_nonterminal_sym_table_transformation(curr_candidate, selected_var, input.input_type)
                worklist.extend(new_progs)
            elif isinstance(selected_var.sym, TerminalSymbol):
                new_progs = self.expand_terminal_sym_table_transformation(curr_candidate, selected_var)
                worklist.extend(new_progs)
            else:
                raise TypeError

        return synthesized_program

    def expand_nonterminal_sym_table_transformation(self, curr_prog: Program, selected_var: VariableNode, input_type: BaseRefType) -> List[Program]:

        new_progs: List[Program] = []

        if curr_prog.depth > self.depth - 1:
            return new_progs

        # first, according to the symbol, find productions with the appropriate goal type
        applicable_productions = self.grammar.select_productions(selected_var, curr_prog.get_goal_type(selected_var), self.pruned_productions)

        prod: Production
        pred: Predicate
        for prod, pred in applicable_productions:

            prog_new: Program = curr_prog.duplicate(next(self.program_counter))
            prod.apply(prog_new, selected_var, True, input_type, self.no_provenance, pred)

            prog_new.depth += 1

            if prod.function_name is 'select':

                if prog_new.depth >= self.depth:
                    printd("depth pruned")
                    continue

                prog_var_node_children = prog_new.get_children(selected_var.id)
                assert isinstance(prog_var_node_children[0], VariableNode)
                assert isinstance(prog_var_node_children[1], VariableNode)

                goal_type = prog_new.get_goal_type(prog_var_node_children[0])

                self.expand_T_in_sym(prog_new, prog_var_node_children[0], input_type=input_type)
                complete_prog = self.expand_terminal_sym(prog_new, prog_var_node_children[1])
                assert len(complete_prog) == 1
                complete_prog = complete_prog[0]
                complete_prog.depth += 1
                complete_prog.update_type(input_type, selected_var.id, no_prov=self.no_provenance)  # TODO: need to cache forward computation results
                # print("complete_prog: ", complete_prog)

                refined_type = complete_prog.node_to_refined_type[selected_var.id]
                # print("goal_type:", goal_type)
                # print("refined_type: ", refined_type)

                compatibility_check_res = check_compatibility(refined_type, goal_type, compute_lemma=True)
                if compatibility_check_res.error_code == 1:
                    print("select pass compatibility check")
                    new_progs.append(complete_prog)
                else:
                    if compatibility_check_res.error_code == 2:
                        print("select refinement compatibility check failed")
                    else:
                        print("select base compatibility check failed ")

                    # print("select failed")
                    # print("compatibility_check_res:", compatibility_check_res)
                    if not self.disable_lemma:
                        if compatibility_check_res.lemma is not None and str(compatibility_check_res.lemma) not in self.learned_conflicts:
                            print("add lemma:", compatibility_check_res.lemma)
                            self.add_lemmas(compatibility_check_res)

            elif prod.function_name is 'summarize' or prod.function_name is 'bin':

                assert isinstance(pred, Prov)

                if prog_new.depth + 1 >= self.depth:
                    printd("depth pruned")
                    continue

                op = pred.beta
                col = pred.col

                # find the children of the appropriate type and fill them in!
                prog_var_node_children = prog_new.get_children(selected_var.id)
                assert isinstance(prog_var_node_children[1], VariableNode)
                assert isinstance(prog_var_node_children[2], VariableNode)

                if prod.function_name is 'summarize':
                    assert prog_var_node_children[1].sym.name == 'alpha'
                    assert prog_var_node_children[2].sym.name == 'sym_col'

                    self.expand_terminal_sym_with_value(prog_new, prog_var_node_children[1], op, duplicate=False)
                    self.expand_terminal_sym_with_value(prog_new, prog_var_node_children[2], col, duplicate=False)

                    # try:
                    new_goal_type = prod.inverse(prog_new.get_goal_type(selected_var), 0, input_type, self.no_provenance, op, col)
                    prog_var_node_children[0].set_goal_type(new_goal_type)

                    assert isinstance(new_goal_type, BaseRefType)
                    check_lemma_output = check_lemmas(new_goal_type, self.learned_conflicts)
                    if not check_lemma_output.res:
                        if check_lemma_output.is_infeasible:
                            # add the provenance constraint to the top-level pruning
                            self.learned_conflicts_prov[str(pred)] = pred

                        self.goal_type_pruned += 1
                        continue
                    else:
                        new_progs.append(prog_new)

                elif prod.function_name is 'bin':
                    assert prog_var_node_children[1].sym.name == 'params'
                    assert prog_var_node_children[2].sym.name == 'sym_col'

                    prog_var_node_children[0].set_goal_type(prod.inverse(prog_new.get_goal_type(selected_var), 0, input_type, self.no_provenance, col))
                    self.expand_terminal_sym_with_value(prog_new, prog_var_node_children[2], col, duplicate=False)
                    new_progs.extend(self.expand_terminal_sym(prog_new, prog_var_node_children[1]))

            else:
                raise ValueError("unknown function name:", prod.function_name)

        return new_progs

    def expand_terminal_sym_table_transformation(self, curr_prog: Program, selected_var: VariableNode) -> List[Program]:
        """
        NOTE: We do not need to implement this now based on the current subset of the grammar we are using
        """
        raise NotImplementedError

    def init_base_case_prog(self, curr_prog: Program, input: InputDataset) -> Tuple[List[Program], List[Program]]:
        """
        Create a program with one single variable T with the appropriate type assignment
        important thing need to ensure here:
        one node get one particular id
        """
        instantiated_programs: List[Program] = []

        base_case_program = Program(next(self.program_counter), None)
        # we need to reset the node counter otherwise we will have same id nodes but are not the same
        base_case_program.node_id_counter = itertools.tee(
            curr_prog.node_id_counter)[1]

        # create a new terminal node of variable T with its type known (which is a refined type)
        t_in_sym: Symbol = self.grammar.name_to_sym['T_in']
        assert isinstance(t_in_sym, TerminalSymbol)
        new_node = base_case_program.add_terminal_node(
            t_in_sym, 'T_in', None, None)
        input_type = input.input_type.duplicate()
        base_case_program.node_to_refined_type[new_node.id] = input_type
        base_case_program.type = input_type

        # check if this satisfy the goal type
        subtyping_check_res = check_compatibility(base_case_program.type, curr_prog.get_goal_type(curr_prog.table_root_node))
        if subtyping_check_res.error_code == 1:
            printd("pass subtyping check")
            instantiated_programs.append(
                self.instantiate_subprogram(curr_prog, base_case_program))

        else:
            printd("fail subtyping check")

        return [base_case_program], instantiated_programs

    def expand_nonterminal_sym(self, curr_prog: Program, selected_var: VariableNode, input_type: Optional[BaseRefType] = None) -> List[Program]:

        assert isinstance(selected_var.sym, NonterminalSymbol)
        assert input_type is not None

        new_progs: List[Program] = []

        # first, according to the symbol, find productions with the appropriate goal type
        applicable_productions = self.grammar.select_productions(selected_var, curr_prog.get_goal_type(selected_var))

        prod: Production
        for prod in applicable_productions:
            # NOTE: I think we don't need to duplicate programs if there is a single prod can be applied, but i will duplicate here anyway
            prog_new: Program = curr_prog.duplicate(next(self.program_counter))
            prod.apply(prog_new, selected_var, True, input_type, self.no_provenance)
            new_progs.append(prog_new)

        return new_progs

    def expand_terminal_sym_with_value(self, curr_prog: Program, selected_var: VariableNode, value, duplicate=True) -> Program:

        var_goal_type = curr_prog.get_goal_type(selected_var)
        assert isinstance(selected_var.sym, TerminalSymbol)
        assert isinstance(var_goal_type, BaseRefType)

        if duplicate:
            prog_new = curr_prog.duplicate(next(self.program_counter))
        else:
            curr_prog.delete_var_node(selected_var)
            prog_new = curr_prog
        prog_new.add_terminal_node(selected_var.sym, value, var_goal_type, curr_prog.get_parent(selected_var.id), sub_id=selected_var.id)

        return prog_new

    def expand_T_in_sym(self, curr_prog: Program, selected_var: VariableNode, input_type: BaseRefType) -> Program:
        """
        handle _in symbol
        """
        var_goal_type = curr_prog.get_goal_type(selected_var)
        assert isinstance(var_goal_type, BaseRefType)
        assert isinstance(var_goal_type.base, TableType)

        curr_prog.delete_var_node(selected_var)

        t_in_sym: Symbol = self.grammar.name_to_sym['T_in']
        assert isinstance(t_in_sym, TerminalSymbol)
        input_type_new = input_type.duplicate()
        curr_prog.node_to_refined_type[selected_var.id] = input_type
        curr_prog.type = input_type
        curr_prog.add_terminal_node(t_in_sym, 'T_in', input_type_new, curr_prog.get_parent(selected_var.id), sub_id=selected_var.id)

        return curr_prog

    def expand_terminal_sym(self, curr_prog: Program, selected_var: VariableNode, input: InputDataset = None, add_none=False) -> List[Program]:

        # print("curr_prog:", curr_prog)

        var_goal_type = curr_prog.get_goal_type(selected_var)
        # print("expand_terminal_sym:", var_goal_type)
        assert isinstance(selected_var.sym, TerminalSymbol)
        assert isinstance(var_goal_type, BaseRefType)


        new_progs: List[Program] = []

        curr_prog.delete_var_node(selected_var)

        if isinstance(var_goal_type.base, NullType):
            """
            this is to handle optional arguments
            """
            new_progs.append(self.expand_terminal_sym_with_value(curr_prog, selected_var, 'None'))

        elif isinstance(var_goal_type.base, ConstColType):
            """
            Here we assume we already know which col assign to which binding
            NOTE: this might need to change in a setting where we don't specify x and y binding
            """
            assert var_goal_type.constraint.value_assign is not None

            # print("selected_var.goal_type.constraint.value_assign: ", selected_var.goal_type.constraint.value_assign)

            new_progs.append(self.expand_terminal_sym_with_value(curr_prog, selected_var, var_goal_type.constraint.value_assign))
        else:
            values = [selected_var.sym.name] if len(selected_var.sym.values) == 0 else selected_var.sym.values
            for value in values:
                new_progs.append(self.expand_terminal_sym_with_value(curr_prog, selected_var, value))

        return new_progs
