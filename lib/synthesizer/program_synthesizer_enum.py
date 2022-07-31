import itertools
from collections import defaultdict
from typing import List, Dict, FrozenSet, Tuple, Optional

from lib.eval.benchmark import InputDataset
from lib.grammar.production import Production
from lib.grammar.symbol import NonterminalSymbol, TerminalSymbol, Symbol
from lib.grammar.typed_cfg import TypedCFG
from lib.node import VariableNode
from lib.program import Program
from lib.synthesizer.program_synthesizer import ProgramSynthesizer
from lib.type.base_type import TableType, CellType
from lib.type.ref_type import BaseRefType, FunctionType, RefType
from lib.type.type_system import get_base_type, get_next_formula_id, allowed_data_type_for_plot
from lib.type.type_system_check import check_bindings, check_compatibility
from lib.utils.misc_utils import printd, powerset


class ProgramSynthesizerEnum(ProgramSynthesizer):
    """
    Note that this class function is replaced by the TrinityProgramSynthesizer
    """

    def __init__(self, grammar: TypedCFG, based_typed: bool = False, colname_hack: bool = False):
        super().__init__(grammar)

        self.visualization_progs: List[Program] = []
        self.table_transformation_prog_start_idx = 0
        self.table_transformation_progs: List[Program] = []

        # class specific fields
        self.complete_progs: List[Program] = []
        self.complete_progs_map: Dict[str, List[Program]] = defaultdict(list)

        # some specific parameter
        self.based_typed = based_typed
        self.colname_hack = colname_hack

    def reinit(self):
        self.complete_progs: List[Program] = []
        self.complete_progs_map: Dict[str, List[Program]] = defaultdict(list)

    def synthesize_init(self, _input: InputDataset):
        """
        synthesize all the syntax valid program in the grammar
        """

        # create a dummy spec
        self.reinit()
        spec: BaseRefType = BaseRefType(get_base_type('Plot'), fields=_input.colnames)

        # map from column list that is displayed to its components
        collist_to_vis_table_maps: Dict[FrozenSet[str], Tuple[List[Program], List[Program]]] = {}

        visualization_progs: List[Program] = self.enumerate_top_down([self.init_prog(spec, _input.input_type)], 'visualization', input=_input)

        # some post processing
        for prog in visualization_progs:

            if prog.x_axis is None:
                continue
            elif prog.x_axis == 'None':
                continue

            if prog.x_axis == prog.y_axis:
                continue

            # if prog.color == prog.column and prog.color != 'None':
            #     continue

            if prog.color == prog.y_axis and prog.plot_type != 'histogram':
                continue

            if prog.column == prog.y_axis and prog.plot_type != 'histogram':
                continue

            if prog.plot_type == 'histogram':
                if prog.y_axis is not None:
                    if prog.y_axis != 'None':
                        continue

            self.visualization_progs.append(prog)

            col_list = frozenset([col for col in [prog.x_axis, prog.y_axis, prog.color, prog.column] if col != 'None'])
            if col_list not in collist_to_vis_table_maps:
                collist_to_vis_table_maps[col_list] = ([], [])
            collist_to_vis_table_maps[col_list][0].append(prog)

        self.table_transformation_progs: List[Program] = self.synthesize_table_transformation(None, _input)

        print('len(self.visualization_progs):', len(self.visualization_progs))
        print('len(self.table_transformation_progs):', len(self.table_transformation_progs))

        for prog in self.table_transformation_progs:
            collist_to_vis_table_maps[prog.select][1].append(prog)

        # for each prog, we need to attach the table transformation prog to the visualization prog
        for key, (v_prog_list, t_prog_list) in collist_to_vis_table_maps.items():
            for v_prog in v_prog_list:
                for t_prog in t_prog_list:
                    new_v_prog = v_prog.duplicate(next(self.program_counter))
                    new_v_prog.select = t_prog.select
                    new_prog = self.instantiate_subprogram(new_v_prog, t_prog)
                    new_prog.visualization_root_node = new_prog.get_children(new_prog.start_node.id)[0]
                    new_prog.table_root_node = new_prog.get_children(new_prog.start_node.id)[1]
                    self.complete_progs.append(new_prog)
                    self.complete_progs_map[v_prog.plot_type].append(new_prog)
                    # if 'Origin, Origin' in new_prog.__repr__():
                    # print("complete prog {}: {}".format(len(self.complete_progs), new_prog))

        print("====== finished initializing all the complete program ======")
        print("len(self.complete_progs):", len(self.complete_progs))

    def synthesize(self, spec: BaseRefType, _input: InputDataset) -> List[Program]:
        """
        this should be a mainly type inference and checking loop
        """

        # truncate csv file
        f = open('prog_synth_enum.csv', "w+")
        f.close()

        if self.based_typed:
            print(spec.base.__repr__())
            plot_type: str = spec.base.__repr__()
            if plot_type == 'BarPlot':
                enumerate_progs = self.complete_progs_map['bar']
            elif plot_type == 'Histogram':
                enumerate_progs = self.complete_progs_map['histogram']
            elif plot_type == 'ScatterPlot':
                enumerate_progs = self.complete_progs_map['scatter']
            elif plot_type == 'LinePlot':
                enumerate_progs = self.complete_progs_map['line']
            elif plot_type == 'AreaPlot':
                enumerate_progs = self.complete_progs_map['area']
            else:
                raise ValueError("plot type {} not supported".format(plot_type))
        else:
            enumerate_progs = self.complete_progs

        if self.colname_hack:
            # quickly prune those that does not type check using the column names mentioned in the spec
            enumerate_progs = [prog for prog in enumerate_progs if set(prog.select) == set(spec.fields)]
            # print("spec:", spec)
            # print("enumerate_progs:", enumerate_progs)

        print('len of enumerated progs:', len(enumerate_progs))
        # print("enumerate_progs:", enumerate_progs)

        valid_synthesized_progs = []

        for prog in enumerate_progs:
            decomposed_viz_goal_type = BaseRefType(
                spec.base, spec.constraint.get_binding_formula(get_next_formula_id()))
            decomposed_table_goal_type = BaseRefType(
                TableType(spec.fields), spec.constraint.get_prov_formula(get_next_formula_id()))
            input_type = _input.input_type.duplicate()

            try:
                inferred_type = prog.infer_type(prog.start_node, input_type, _input)
            except TypeError:
                # print("Type inference failed. Go to the next program.")
                continue

            inferred_viz_types: List[FunctionType] = self.enumerate_viz_plot_type(prog, inferred_type[0], _input)
            inferred_table_type = inferred_type[1]
            assert isinstance(inferred_table_type, BaseRefType)
            assert isinstance(inferred_table_type.base, TableType)

            for viz_type in inferred_viz_types:
                # infer type of prog and check against spec
                # print("viz_type:", viz_type)
                # print("decomposed_viz_goal_type:", decomposed_viz_goal_type)
                inferred_table_record = inferred_table_type.base.get_record()
                inferred_viz_base_type = viz_type.output.base.name
                if check_compatibility(viz_type.output, decomposed_viz_goal_type) == 1 and check_compatibility(inferred_table_type, decomposed_table_goal_type) == 1 \
                        and check_bindings(viz_type.output.constraint, decomposed_viz_goal_type.constraint, inferred_table_record, inferred_viz_base_type) \
                        and check_compatibility(inferred_table_type, viz_type.input) == 1:
                    # print("type check")
                    # set the node_to_refinement_type field
                    # refine the type of the synthesized program

                    prog.node_to_refined_type[prog.start_node.id] = viz_type.output
                    prog.node_to_refined_type[prog.visualization_root_node.id] = viz_type
                    prog.node_to_refined_type[prog.table_root_node.id] = inferred_table_type
                    prog.overall_type_strengthening()
                    # print("prog strengthened:", prog.node_to_refined_type[prog.start_node.id])
                    valid_synthesized_progs.append(prog)

                    # print("valid prog:", prog)
                    break
                else:
                    # NOTE: Enable the following when debugging, otherwise too much info printed
                    # print("VIZ GOAL TYPE")
                    # print(decomposed_viz_goal_type)
                    # print("TABLE GOAL TYPE")
                    # print(decomposed_table_goal_type)
                    # print("PROGRAM")
                    # print(prog)
                    # print("GOAL TYPE")
                    # print(spec)
                    # print("INFERRED VIZ TYPE")
                    # print(viz_type.output)
                    # print("INFERRED_VIZ_TYPE_INPUT")
                    # print(viz_type.input)
                    # print("INFERRED TABLE TYPE")
                    # print(inferred_table_type)
                    continue
                printd("Subtyping check failed. Go to the next type.")

            printd("All possible subtyping check failed. Go to the next program")

        # printd("synthesized_prog: ", synthesized_prog)
        # print("num progs enumerated: ", len(synthesized_progs))
        # print("size:", len(valid_synthesized_progs))
        print("valid progs:", valid_synthesized_progs)
        # for prog in valid_synthesized_progs:
        #     print(prog.node_to_refined_type[prog.start_node.id])

        return valid_synthesized_progs

    def enumerate_viz_plot_type(self, prog: Program, inferred_type: RefType, input_data: InputDataset) -> List[FunctionType]:
        """
        Given a program, enumerate all possible plot types according to the rules in the type system
        """
        # NOTE: let me write a dumb one

        assert isinstance(inferred_type, FunctionType)

        table_input_assignment: Dict = {}

        plot_type = prog.get_plot_type_name()
        assert plot_type is not None
        # print("plot type:", plot_type)

        if prog.x_axis is not None:
            table_input_assignment['x'] = prog.x_axis
        if prog.y_axis is not None:
            table_input_assignment['y'] = prog.y_axis
        if prog.color is not None:
            table_input_assignment['color'] = prog.color
        if prog.column is not None:
            table_input_assignment['column'] = prog.column

        # enumerate all possible input base types
        fields = list(table_input_assignment.values())
        worklist: List[TableType] = [TableType()]
        for key, value in table_input_assignment.items():
            if value == 'None':
                continue
            new_worklist: List[TableType] = []
            while len(worklist) > 0:
                curr_type = worklist.pop()
                for data_type in allowed_data_type_for_plot[plot_type][key]:
                    new_type = curr_type.duplicate()
                    new_data_type = get_base_type(data_type)
                    assert isinstance(new_data_type, CellType)
                    new_type.add(value, new_data_type)
                    new_worklist.append(new_type)
            worklist = new_worklist

        all_input_base_types: List[TableType] = worklist
        # print all possible types
        # print("enumerated_input_base_types: ", all_input_base_types)

        # combined the enumerated table base types and the inferred table ref type
        # also combined the input type with the inferred output type to get the final function type
        all_enumerated_types: List[FunctionType] = []
        for input_base_type in all_input_base_types:
            # infer another basetype
            # for turn input base type to a refinement type
            input_ref_type = BaseRefType(input_base_type, fields=fields)
            inferred_type_2 = prog.infer_type(prog.visualization_root_node, input_ref_type, input_data)
            assert isinstance(inferred_type_2, FunctionType)
            all_enumerated_types.append(FunctionType(BaseRefType(input_base_type, inferred_type_2.input.constraint), inferred_type.output))

        return all_enumerated_types

    def enumerate_top_down(self, worklist: List[Program], component=None, input: InputDataset = None, add_none=True) -> List[Program]:
        return super(ProgramSynthesizerEnum, self).enumerate_top_down(worklist, component, input, add_none)

    def expand_nonterminal_sym(self, curr_prog: Program, selected_var: VariableNode, input_type: Optional[BaseRefType] = None) -> List[Program]:

        assert isinstance(selected_var.sym, NonterminalSymbol)

        new_progs: List[Program] = []

        # first, according to the symbol, find productions with the appropriate goal type
        applicable_productions = self.grammar.select_productions_without_pruning(
            selected_var)
        prod: Production
        for prod in applicable_productions:
            # print("production applied:", prod)
            prog_new: Program = curr_prog.duplicate(next(self.program_counter))
            prod.apply(prog_new, selected_var, False, input_type)
            new_progs.append(prog_new)

        return new_progs

    def expand_terminal_sym(self, curr_prog: Program, selected_var: VariableNode, input: InputDataset = None, add_none=True) -> List[Program]:

        # print("expand_terminal_sym:", selected_var.goal_type)
        assert isinstance(selected_var.sym, TerminalSymbol)

        new_progs: List[Program] = []

        curr_prog.delete_var_node(selected_var)

        if selected_var.sym == self.grammar.name_to_sym['sym_col_list']:
            values = powerset(input.colnames)
            # values = powerset(['Cylinders', 'Model'])
        elif selected_var.sym == self.grammar.name_to_sym['sym_col']:
            values = input.colnames.copy()
            # values = ['Cylinders', 'Model']
            if add_none and 'None' not in values:
                values.append('None')

        else:
            values = [selected_var.sym.name] if len(
                selected_var.sym.values) == 0 else selected_var.sym.values

        prog_str = repr(curr_prog)
        for value in values:
            # print("value:", value)
            # when enumerating bin/summarize, disallow any columns that have not been selected
            if selected_var.sym == self.grammar.name_to_sym['sym_col']:
                if 'select' in prog_str and value not in prog_str:
                    continue

            prog_new = curr_prog.duplicate(next(self.program_counter))

            if selected_var.sym == self.grammar.name_to_sym['sym_col']:
                if 'plot' in prog_str:
                    if selected_var.arg_idx == 0:
                        prog_new.x_axis = value
                    elif selected_var.arg_idx == 1:
                        prog_new.y_axis = value
                    elif selected_var.arg_idx == 2:
                        prog_new.color = value
                    elif selected_var.arg_idx == 3:
                        prog_new.column = value
            elif selected_var.sym == self.grammar.name_to_sym['sym_col_list']:
                if 'select' in prog_str:
                    prog_new.select = frozenset(value)

            prog_new.add_terminal_node(selected_var.sym, value, prog_new.get_goal_type(selected_var), curr_prog.get_parent(
                selected_var.id), sub_id=selected_var.id)
            new_progs.append(prog_new)

            # print("new_prog:", prog_new)
            # print("children:", prog_new.to_children)

        return new_progs

    def init_base_case_prog(self, curr_prog: Optional[Program], input: InputDataset) -> Tuple[List[Program], List[Program]]:
        """
        Create a program with one single variable T with the appropriate type assignment
        important thing need to ensure here:
        one node get one particular id
        """
        instantiated_programs: List[Program] = []

        base_case_program = Program(next(self.program_counter), None)
        # we need to reset the node counter otherwise we will have same id nodes but are not the same
        base_case_program.node_id_counter = itertools.count(10)

        # create a new terminal node of variable T with its type known (which is a refined type)
        t_in_sym: Symbol = self.grammar.name_to_sym['T_in']
        assert isinstance(t_in_sym, TerminalSymbol)
        base_case_program.add_terminal_node(
            t_in_sym, 'T_in', None, None)

        # print("base_case_program:", base_case_program.type)

        return [base_case_program], instantiated_programs

    def expand_subprogram(self, parent_prog: Program, curr_candidate: Program, curr_depth: int, input: InputDataset) -> Tuple[List[Program], List[Program]]:
        """
        given the current candidate, expand it, check feasibility, and assign to the parent program if type check (fill var hole that's left basically)
        """
        pt_sym: Symbol = self.grammar.name_to_sym['pt']
        applicable_productions = self.grammar.get_productions(pt_sym, "Table")
        # In the first round, only apply select production
        if curr_depth == 1:
            applicable_productions = applicable_productions[:1]
        elif curr_depth > 1:
            # if the select operation does consider all the possible operations, then any depth more than 1 don't have to consider it
            applicable_productions = applicable_productions[1:]
        else:
            raise ValueError("curr_depth value is invalid")

        new_subprograms = []
        for prod in applicable_productions:
            new_subprog: Program = curr_candidate.duplicate(
                next(self.program_counter))
            # make sure new subprogram doesn't assign node ids that already appear in curr_candidate
            new_subprog.node_id_counter = itertools.tee(
                curr_candidate.node_id_counter)[1]
            prod.reverse_apply(new_subprog, curr_candidate, input.input_type)

            # invoke a mini top-down enumerator to fill the rest of the holes of the new sub-programs
            # print("new_subprog:", new_subprog)
            new_subprogs: List[Program] = self.enumerate_top_down(
                [new_subprog], input=input, add_none=False)

            new_subprograms.extend(new_subprogs)

        return new_subprograms, new_subprograms