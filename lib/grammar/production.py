from typing import List, Union, Dict, Optional

from lib.grammar.symbol import NonterminalSymbol, Symbol
from lib.program import Program, Node
from lib.node import VariableNode
from lib.type.base_type import TableType, BarPlot, ScatterPlot, LinePlot, AreaPlot, Histogram, NullType, ListType, DiscreteType, AggregateType, ContinuousType, QuantitativeType, \
    BaseType, NullOpType, ConstColType
from lib.type.formula import Prov, conjunct_formula, create_term, CNFFormula, XORFormula
from lib.type.predicate import RelationPred, Variable, Constant, Card, Binding
from lib.type.ref_type import BaseRefType, FunctionType, RefType
from lib.type.type_system import get_base_type, create_cnf_formula_instance, create_list_type, allowed_data_type_for_plot, get_next_formula_id, create_xor_formula_instance
from lib.utils.misc_utils import printd

"""
Production class for each of the production, includes its unique inverse/forward semantics 
"""

"""
some helper function here
"""


def plot_inverse_semantics(goal_type: RefType, arg_index: int, input_type: BaseRefType, no_prov=False) -> RefType:
    """
    inverse semantic for all the plot functions
    """
    assert isinstance(goal_type, BaseRefType)

    if arg_index == 0:
        return BaseRefType(ConstColType(col=goal_type.constraint.query_binding('x').col),
                           create_cnf_formula_instance([[(RelationPred('coeq', Variable('v'), Constant(goal_type.constraint.query_binding('x').col)), False)]]))
    elif arg_index == 1:
        if goal_type.base.name == "Histogram":
            return BaseRefType(get_base_type('Null'))
        if goal_type.constraint.query_binding('y') is not None:
            return BaseRefType(ConstColType(col=goal_type.constraint.query_binding('y').col),
                               create_cnf_formula_instance([[(RelationPred('coeq', Variable('v'), Constant(goal_type.constraint.query_binding('y').col)), False)]]))
        else:
            # Note that for histogram we can not specifying y axis because it does not matter
            assert goal_type.base.name == 'Histogram'
            return BaseRefType(get_base_type('Null'))
    elif arg_index == 2:
        if goal_type.constraint.query_binding('color') is not None:
            return BaseRefType(ConstColType(col=goal_type.constraint.query_binding('color').col),
                               create_cnf_formula_instance([[(RelationPred('coeq', Variable('v'), Constant(goal_type.constraint.query_binding('color').col)), False)]]))
        elif no_prov:
            ret_type = BaseRefType(get_base_type('NullOp'), fields=goal_type.fields)
            ret_type.enc = 'color'
            return ret_type
        else:
            return BaseRefType(get_base_type('Null'))
    elif arg_index == 3:
        if goal_type.constraint.query_binding('column') is not None:
            return BaseRefType(ConstColType(col=goal_type.constraint.query_binding('column').col),
                               create_cnf_formula_instance([[(RelationPred('coeq', Variable('v'), Constant(goal_type.constraint.query_binding('column').col)), False)]]))
        elif no_prov:
            ret_type = BaseRefType(get_base_type('NullOp'), fields=goal_type.fields)
            ret_type.enc = 'column'
            return ret_type
        else:
            return BaseRefType(get_base_type('Null'))


def plot_forward_semantics(plot_str: str, arg_types: List[BaseRefType]) -> BaseRefType:
    base_type = get_base_type(plot_str)

    formula = []
    for i, (enc, arg) in enumerate(zip(['x', 'y', 'color', 'column'], arg_types)):
        if isinstance(arg.base, NullType):
            continue
        elif isinstance(arg.base, NullOpType):
            # create forward encoding for the plot function
            assert arg.col is not None
            assert arg.enc is not None
            if arg.col != 'None':
                formula.append([(Binding(arg.col, arg.enc), False)])
        else:

            # extract the relationpred
            # print(arg.constraint)
            assert len(arg.constraint.clause_id_to_literal_id) == 1
            lit_id = list(arg.constraint.clause_id_to_literal_id.values())[0][0]
            pred = arg.constraint.get_atom(lid=lit_id).predicate
            assert isinstance(pred, RelationPred)
            assert pred.op_name == 'coeq'
            assert isinstance(pred.arg2, Constant)
            col = pred.arg2.arg
            formula.append([(Binding(col, enc), False)])

    constraint = create_cnf_formula_instance(formula)
    return BaseRefType(base_type, constraint)


def add_duplicate_constraint(formula: List, non_empty_enc_except_y: Dict) -> List:
    """
    append the avoid duplicate constraint to the formula
    """
    formula.append([(RelationPred('eq', create_term(Card, list(
        non_empty_enc_except_y.values())), create_term(Card, 'T')), False)])

    return formula


def add_shared_cardinality_constraints(formula: List, non_empty_enc_except_y: Dict, input_type: BaseRefType) -> List:
    """
    append the shared cardinality constraints (|color| < 10 and |column| < 10 and |T| > 1) to the formula
    """
    formula.append(
        [(RelationPred('ge', create_term(Card, 'T'), create_term(Constant, 1)), False)])

    input_type_base: BaseType = input_type.base
    assert isinstance(input_type_base, TableType)

    if 'color' in non_empty_enc_except_y:
        formula.append([(RelationPred('leq', create_term(
            Card, [non_empty_enc_except_y['color']]), create_term(Constant, 20)), False)])
    if 'column' in non_empty_enc_except_y:
        formula.append([(RelationPred('leq', create_term(
            Card, [non_empty_enc_except_y['column']]), create_term(Constant, 10)), False)])

    return formula


def add_qualitative_cardinality_constraints(formula: List, nominal_ordinal_col: List) -> List:
    """
    append the avoid super high cardinality constraints for nominal or ordinal variable used in this plot (given we learned the binding now)
    """

    for col in nominal_ordinal_col:
        formula.append(
            [(RelationPred('leq', create_term(Card, [col]),
                           create_term(Constant, 50)), False)]
        )

    return formula


def enumerate_new_type(input_table_type: BaseRefType, non_empty_enc: Dict, plot_type: BaseRefType, has_input_type=False) -> List[RefType]:
    """
    we need a worklist-like algorithm to enumerate new type
    """

    return_types: List[RefType] = []

    worklist: List[BaseRefType] = [input_table_type]

    for enc, col in non_empty_enc.items():

        if has_input_type:
            candidates = allowed_data_type_for_plot[plot_type.base.name][enc]
        else:
            candidates = ['Cell']
        new_incomplete_types: List[BaseRefType] = []

        while len(worklist) > 0:
            curr_type = worklist.pop()

            for cell_type in candidates:

                new_input_type_new: BaseRefType = curr_type.duplicate(
                    get_next_formula_id())
                assert isinstance(new_input_type_new.base, TableType)

                try:
                    new_input_type_new.base.update_record(
                        col, create_list_type(cell_type))
                except TypeError:
                    continue

                if cell_type in ['Qualitative', 'Nominal', 'Ordinal']:
                    new_constraint = create_cnf_formula_instance(
                        add_qualitative_cardinality_constraints([], [col]))
                    new_input_type_new.constraint = conjunct_formula(
                        get_next_formula_id(), new_input_type_new.constraint, new_constraint)

                new_incomplete_types.append(new_input_type_new)

        worklist = new_incomplete_types

    for candidate_type in worklist:
        return_types.append(FunctionType(candidate_type, plot_type))

    return return_types


"""
Here start the production classes
"""


class Production:
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):

        self.function_name: str = function_name

        self.ret_sym: NonterminalSymbol = ret_sym
        self.ret_type: RefType = ret_type
        self.arg_syms: List[Symbol] = arg_syms
        self.arg_types: List[RefType] = arg_types

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):

        prog.delete_var_node(expand_var)

        # generate f(v1, v2) where f is the parent_node, [v1,v2] are the children node
        parent_node = prog.add_nonterminal_node(
            self.function_name, expand_var.sym, prog.get_goal_type(expand_var), prog.get_parent(expand_var.id), self, sub_id=expand_var.id)
        # need to keep track of the last node pushed to the program
        prog.nodes_pushed.append(parent_node.id)
        if typed:
            parent_node_children = [prog.add_variable_node(arg_sym, self.inverse(
                prog.get_goal_type(expand_var), i, input_type, no_prov=no_prov), parent_node, arg_idx=i) for i, arg_sym in enumerate(self.arg_syms)]
        else:
            parent_node_children = [prog.add_variable_node(
                arg_sym, None, parent_node, arg_idx=i) for i, arg_sym in enumerate(self.arg_syms)]
        printd("new children created with goal type: ", [
            prog.get_goal_type(node) for node in parent_node_children])
        prog.set_children(parent_node, parent_node_children)

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        # Create new parent node that curr_prog's start node will be a child of
        parent_node = new_prog.add_nonterminal_node(
            self.function_name, self.ret_sym, curr_prog.type, None, self, sub_id=next(new_prog.node_id_counter))
        # Update new_prog to have same nodes and edges as curr_prog

        children = []
        # for each arg in production, check if it has base type Table. If it does, slot curr_prog into this arg
        for i, (arg_sym, arg_type) in enumerate(zip(self.arg_syms, self.arg_types)):
            assert isinstance(arg_type, BaseRefType)
            if isinstance(arg_type.base, TableType):
                # assert isinstance(curr_prog.type.base, TableType)
                new_prog.set_parent(parent_node, curr_prog.start_node)
                children.append(curr_prog.start_node)
            else:
                # TODO: i don't think you should be using inverse semantics here
                var_node = new_prog.add_variable_node(arg_sym, self.inverse(
                    curr_prog.type, i, input_type, no_prov=False), parent_node, sub_id=next(new_prog.node_id_counter))
                children.append(var_node)
        new_prog.set_children(parent_node, children)

    def forward(self, arg_types: List[RefType], additional_args: List[Union[str, Node, List[str]]], input_type: BaseRefType, no_prov: bool, **args) -> Union[RefType, List[RefType]]:
        """
        encode forward semantic for type transformation (refine the type of the synthesized program )
        it can return multiple refined type because we are adding additional constraints
        """
        raise NotImplementedError

    def inverse(self, goal_type: RefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> Optional[RefType]:
        """
        encode inverse semantic for type transformation (propagate goal type top down)
        """
        raise NotImplementedError

    def get_prov_formula(self, field_names) -> XORFormula:
        raise NotImplementedError

    def __repr__(self):
        return self.function_name


class PlotProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)

        new_created_children_node = prog.get_children(expand_var.id)
        assert len(new_created_children_node) == 2
        prog.visualization_root_node = new_created_children_node[0]
        prog.table_root_node = new_created_children_node[1]

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        super().reverse_apply(new_prog, curr_prog, input_type)

    def forward(self, arg_types: List[RefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool, **args) -> Union[RefType, List[RefType]]:
        return arg_types

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        """
        need to split the type into
        """
        assert goal_type.fields is not None
        if arg_index == 0:
            decomposed_viz_goal_type = BaseRefType(
                goal_type.base, goal_type.constraint.get_binding_formula(get_next_formula_id()))
            return FunctionType(BaseRefType(TableType(goal_type.fields)), decomposed_viz_goal_type)
        else:
            decomposed_table_goal_type = BaseRefType(
                TableType(goal_type.fields), goal_type.constraint.get_prov_formula(get_next_formula_id()), fields=goal_type.fields)
            # print("new decomposed_table_goal_type:", decomposed_table_goal_type)
            return decomposed_table_goal_type


class PlotFuncProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        raise NotImplementedError

    def update_table_type(self, table_type: TableType, viz_constraint: CNFFormula) -> TableType:
        fields = []

        for literal_id in viz_constraint.binding_enc_to_literal_id.values():
            binding_atom = viz_constraint.literal_id_to_atom[literal_id]
            assert isinstance(binding_atom.predicate, Binding)
            field = binding_atom.predicate.col
            if field != 'None':
                fields.append(field)
        new_table_type = TableType(fields)
        for field in fields:
            old_type = table_type.get_record()[field]
            new_table_type.update(field, old_type)
        return new_table_type

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool, inferring_type: bool = False, has_input_type=True, **args) -> Union[RefType, List[RefType]]:
        """
        this enumerates the forward semantic for different plot_func depend on the arg_types
        this is a long function lol
        """
        assert len(arg_types) == 2

        input_table_type = arg_types[0]
        assert isinstance(input_table_type.base, TableType)
        plot_type = arg_types[1]
        if inferring_type or no_prov:
            input_table_type.base = self.update_table_type(
                input_table_type.base, plot_type.constraint)
        # first check consistency between col in binding and col in table input
        if not no_prov:
            if not inferring_type and not (set(plot_type.constraint.query_binding(enc).col for enc in plot_type.constraint.binding_enc_to_literal_id.keys()) == (set(input_table_type.base.fields))):
                print("inconsistency between col in binding and col in input table")
                return []

        # first find out the non-empty enc (e.g. sometimes subplot might be omitted) other than 'y'
        non_empty_enc_except_y = dict([(enc, plot_type.constraint.query_binding(enc).col) for enc in [
            'x', 'color', 'column'] if plot_type.constraint.query_binding(enc) is not None and plot_type.constraint.query_binding(enc).col != 'None'])

        if not inferring_type:
            assert 'x' in non_empty_enc_except_y

        if has_input_type:
            formula = add_shared_cardinality_constraints(
                [], non_empty_enc_except_y, input_type)
        else:
            formula = []

        if isinstance(plot_type.base, BarPlot):
            if has_input_type:
                formula = add_duplicate_constraint(
                    formula, non_empty_enc_except_y)
            input_table_type.constraint = create_cnf_formula_instance(formula)

        elif isinstance(plot_type.base, Histogram):
            if has_input_type:
                formula = add_duplicate_constraint(
                    formula, non_empty_enc_except_y)
            if not inferring_type:
                input_table_type.constraint = create_cnf_formula_instance(
                    formula)
            else:
                # for histogram, we require only count is used on the x value, nothing else
                if 'x' in non_empty_enc_except_y:
                    formula.append(
                        [(Prov(non_empty_enc_except_y['x'], 'count'), False)])
                for enc, col in non_empty_enc_except_y.items():
                    formula.extend([[(Prov(col, 'mean'), True)],
                                    [(Prov(col, 'sum'), True)]])
                input_table_type.constraint = create_cnf_formula_instance(
                    formula)
        elif isinstance(plot_type.base, ScatterPlot):
            input_table_type.constraint = create_cnf_formula_instance(formula)

        elif isinstance(plot_type.base, LinePlot):
            if has_input_type:
                formula = add_duplicate_constraint(
                    formula, non_empty_enc_except_y)
            input_table_type.constraint = create_cnf_formula_instance(formula)

        elif isinstance(plot_type.base, AreaPlot):
            if has_input_type:
                formula = add_duplicate_constraint(
                    formula, non_empty_enc_except_y)
            input_table_type.constraint = create_cnf_formula_instance(formula)

        else:
            printd("TypeError: the type {} does not support.".format(
                plot_type.base.name))
            raise TypeError

        if inferring_type:
            return FunctionType(input_table_type, plot_type)

        if isinstance(plot_type.base, Histogram):
            return_types = enumerate_new_type(
                input_table_type, non_empty_enc_except_y, plot_type, has_input_type)
        else:
            return_types = enumerate_new_type(input_table_type, {
                **non_empty_enc_except_y, 'y': plot_type.constraint.query_binding('y').col}, plot_type, has_input_type)
        printd("return_types:", return_types)

        return return_types

    def inverse(self, goal_type: FunctionType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        if arg_index == 0:
            return goal_type.input
        else:
            return goal_type.output


class BarProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)
        prog.plot_type = 'bar'

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        raise NotImplementedError

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool, **args) -> RefType:
        return plot_forward_semantics('BarPlot', arg_types)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov, *args) -> RefType:
        return plot_inverse_semantics(goal_type, arg_index, input_type, no_prov=no_prov)


class HistogramProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)
        prog.plot_type = 'histogram'

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        raise NotImplementedError

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool, **args) -> RefType:
        return plot_forward_semantics('Histogram', arg_types)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        return plot_inverse_semantics(goal_type, arg_index, input_type, no_prov)


class ScatterProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)
        prog.plot_type = 'scatter'

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        raise NotImplementedError

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool, **args) -> RefType:
        return plot_forward_semantics('ScatterPlot', arg_types)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        return plot_inverse_semantics(goal_type, arg_index, input_type, no_prov)


class LineProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)
        prog.plot_type = 'line'

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        raise NotImplementedError

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool,  **args) -> RefType:
        return plot_forward_semantics('LinePlot', arg_types)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        return plot_inverse_semantics(goal_type, arg_index, input_type, no_prov)


class AreaProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)
        prog.plot_type = 'area'

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        raise NotImplementedError

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool, **args) -> RefType:
        return plot_forward_semantics('AreaPlot', arg_types)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        return plot_inverse_semantics(goal_type, arg_index, input_type, no_prov)


class BinProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        # before we do reverse_apply, we need to first make sure that curr_prog (bottom-up prog) col_target is a continuous type
        # we don't know col_target yet, but we just have to check if there exists a column
        super().reverse_apply(new_prog, curr_prog, input_type)

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Union[Node, str]], input_type: BaseRefType, no_prov: bool,
                inferring_type: bool = False, has_input_type: bool = True, **args) -> RefType:

        cur_type = arg_types[0]
        curr_base_type = cur_type.base
        assert isinstance(curr_base_type, TableType)
        curr_ref_type = cur_type.constraint
        param = int(additional_args[0])
        col_name = additional_args[1]

        # now we can further check if col_name is a Continuous type
        if not isinstance(curr_base_type.get_record()[col_name].T, ContinuousType) and has_input_type:
            raise TypeError(f'{col_name} is not a Continuous type but a {curr_base_type.get_record()[col_name].T}')

        new_base_type = curr_base_type.duplicate()
        assert isinstance(new_base_type, TableType)
        # Binned column now has discrete type
        new_base_type.update(col_name, ListType(DiscreteType()), False)
        new_formula = create_cnf_formula_instance(
            [[(Prov(col_name, "bin"), False)]])
        new_ref_type = conjunct_formula(
            get_next_formula_id(), curr_ref_type, new_formula)
        # Binned column now has cardinality equal to param
        if has_input_type:
            new_ref_type.update_card(col_name, param, "eq")
            new_ref_type.update_card(col_name, param, "eq")
        return BaseRefType(new_base_type, new_ref_type, cur_type.fields)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> Optional[RefType]:

        if arg_index == 0:
            if len(args) == 0:
                return None
            else:
                assert len(args) == 1
                col_name = args[0]

                assert isinstance(goal_type.base, TableType)
                new_base_type = goal_type.base.duplicate()
                new_base_type.update(
                    col_name, create_list_type('Continuous'), False)

                new_formula = goal_type.constraint.duplicate(
                    get_next_formula_id())
                to_be_update_prov = [Prov(col_name, "bin")]
                to_be_update_prov.extend(
                    [Prov(col_name, op) for op in ['sum', 'mean', 'count']])
                new_prov_list = []
                for prov in to_be_update_prov:
                    prov_str = repr(prov)
                    if new_formula.get_neg(prov_str) is None:
                        new_prov_list.append([(prov, True)])
                    else:
                        new_formula.negate_prov(prov_str)

                new_prov_formula = create_cnf_formula_instance(new_prov_list)
                new_formula = conjunct_formula(
                    get_next_formula_id(), new_formula, new_prov_formula)

                # print("new_formula:", new_formula)
                new_formula.update_card(col_name, None, 'leq', containment=False,
                                        special=False, special_2=True)

                return BaseRefType(new_base_type, new_formula, goal_type.fields)
        else:
            return BaseRefType(get_base_type('Const'))

    def get_prov_formula(self, field_names: List[str]) -> XORFormula:
        """
        encode the require in order for a bin operation to be valid
        Prov(colA, "bin") XOR Prov(colB, "bin") XOR Prov(colC, "bin")
        """

        def bin_gen(x): return create_xor_formula_instance(
            [[(Prov(f, 'bin'), False)] for f in x])

        return bin_gen(field_names)


class FilterProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed, input_type: BaseRefType, no_prov: bool,  *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        super().reverse_apply(new_prog, curr_prog, input_type)

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Union[Node, str]], input_type: BaseRefType, no_prov: bool, has_input_type: bool = True, **args) -> RefType:
        # base type stays the similar, need to update the provenance and the cardinality
        cur_type = arg_types[0]
        curr_base_type = cur_type.base.duplicate()
        assert isinstance(curr_base_type, TableType)
        curr_ref_type = cur_type.constraint

        col_name = additional_args[0]
        assert isinstance(col_name, str)

        new_formula = create_cnf_formula_instance([[(Prov(col_name, 'filter'), False)]])
        new_ref_type = conjunct_formula(get_next_formula_id(), curr_ref_type, new_formula)

        if has_input_type:
            # update cardinality: filter affects all columns cardinality
            for col in curr_base_type.get_record().keys():
                new_ref_type.update_card(col, -1, "leq", containment=True)
            new_ref_type.update_card("T", -1, "leq")

        return BaseRefType(curr_base_type, new_ref_type, cur_type.fields)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        if arg_index == 0:
            assert len(args) == 1
            col_name = args[0]

            new_base_type = goal_type.base.duplicate()
            new_formula = goal_type.constraint.duplicate(get_next_formula_id())
            to_be_remove_prov = Prov(col_name, 'filter')
            prov_str = repr(to_be_remove_prov)

            # we don't allow the same column to get filtered again
            if new_formula.get_neg(prov_str) is None:
                new_prov_formula = create_cnf_formula_instance([[(to_be_remove_prov, True)]])
                new_formula = conjunct_formula(get_next_formula_id(), new_formula, new_prov_formula)
            else:
                new_formula.negate_prov(prov_str)

            # also need to remove the cardinality constraints since
            new_formula.update_card(col_name, None, '', containment=True, special_3=True)

            return BaseRefType(new_base_type, new_formula, goal_type.fields)
        else:
            return BaseRefType(get_base_type('Const'))


class SelectProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        super().reverse_apply(new_prog, curr_prog, input_type)

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Union[List[str], Node]], input_type: BaseRefType, no_prov: bool, **args) -> RefType:
        cur_type = arg_types[0]
        curr_base_type = cur_type.base
        # print("curr_base_type:", curr_base_type)
        assert isinstance(curr_base_type, TableType)
        col_list = additional_args[0]
        # print("col_list:", col_list)
        new_base_type = TableType(col_list)
        # Table only has columns listed in col_list
        # print("additional_args:", additional_args)
        for col_name in col_list:
            new_base_type.update(
                col_name, curr_base_type.get_record()[col_name], True)

        return BaseRefType(new_base_type, cur_type.constraint, col_list)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        if arg_index == 0:
            return goal_type
        else:
            return BaseRefType(get_base_type('Const'))


class MutateProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        super().reverse_apply(new_prog, curr_prog, input_type)

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool, **args) -> RefType:
        raise NotImplementedError

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        raise NotImplementedError


class SummarizeProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

        self.properties = {'qualitative_to_quantitative': 'count'}

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        super().reverse_apply(new_prog, curr_prog, input_type)

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Union[str, Node]], input_type: BaseRefType, no_prov: bool, has_input_type=True, **args) -> RefType:
        cur_type = arg_types[0]
        curr_base_type = cur_type.base
        assert isinstance(curr_base_type, TableType)
        curr_ref_type = cur_type.constraint
        alpha = additional_args[0]
        col_name = additional_args[1]
        new_base_type = curr_base_type.duplicate()
        # Target column now has aggregate type except count
        # need to check the type of the column before change its type
        if alpha == 'mean' or alpha == 'sum':
            if not issubclass(curr_base_type.get_record()[col_name].T.__class__, QuantitativeType) and has_input_type:
                # Note: this exception should be unreachable
                raise TypeError("Cannot summarize a qualitative column")
        if alpha is not 'count':
            new_base_type.update(col_name, ListType(AggregateType()), False)
        else:
            # if no_prov:
            new_base_type.update(col_name, ListType(DiscreteType()), False)

        to_update_prov = Prov(col_name, alpha)
        new_formula = curr_ref_type.duplicate(get_next_formula_id())
        if not new_formula.negate_negated_prov(repr(to_update_prov)):
            new_prov_formula = create_cnf_formula_instance(
                [[(to_update_prov, False)]])
            new_formula = conjunct_formula(
                get_next_formula_id(), new_formula, new_prov_formula)
        # Target column and table must now have cardinality <= Card(T)
        if alpha is not 'count' and has_input_type:
            # Get the current cardinality of T
            new_card_val = new_formula.query_card('T')
            new_formula.update_card(col_name, new_card_val, "leq")
            new_formula.update_card("T", new_card_val, "leq")
        elif alpha is 'count' and has_input_type:
            try:
                new_card_val = new_formula.query_card([col_name])
                new_formula.update_card(col_name, new_card_val, "eq")
                new_formula.update_card(
                    col_name, new_card_val, "leq", containment=True)
                new_formula.update_card("T", new_card_val, "eq")
            except KeyError:
                pass
        else:
            pass

        return BaseRefType(new_base_type, new_formula, cur_type.fields)

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> Optional[RefType]:

        # infer the type for the first argument from the goal type
        if arg_index == 0:
            if len(args) == 0:
                return None
            else:
                assert len(args) == 2

                # TODO: here we should also check if the goal type also specifies that some other construct to do with this
                #   if it does, we should return a exception that saying we cannot apply two aggregation on the same field at the same time

                op = args[0]
                col_name = args[1]
                assert isinstance(goal_type.base, TableType)
                new_base_type = goal_type.base.duplicate()
                if op is not 'count':
                    # TODO: here we can check if there is any possibility to make this column a quantitative column
                    #   if not, we should return a exception that explains the cause and the reason

                    input_base_type = input_type.base
                    assert isinstance(input_base_type, TableType)

                    new_base_type.update(col_name, create_list_type('Quantitative'), False)
                else:
                    # if no_prov:
                    new_base_type.update(col_name, create_list_type('Cell'), False)

                new_formula = goal_type.constraint.duplicate(
                    get_next_formula_id())
                new_prov = []
                for beta in ['sum', 'mean', 'count']:
                    curr_prov = Prov(col_name, beta)
                    check_neg = new_formula.get_neg(repr(curr_prov))
                    if check_neg is not None:
                        if check_neg:
                            pass
                        else:
                            new_formula.negate_prov(repr(curr_prov))
                    else:
                        new_prov.append([(curr_prov, True)])

                # if the op is a count operation, we need to add additional constraint so that no more count can be enumerated on any field
                # NOTE: in the current parser model, we don't need the following rules since count will only assigned to one field (save me a for loop lol)
                # if op is 'count':
                #     new_prov.extend([[(Prov(field, 'count'), True)]for field in goal_type.fields if field is not col_name])

                if len(new_prov) > 0:
                    new_prov_formula = create_cnf_formula_instance(new_prov)
                    new_formula = conjunct_formula(
                        get_next_formula_id(), new_formula, new_prov_formula)

                if op == 'mean' or op == 'sum':
                    new_formula.update_card(
                        col_name, None, 'leq', containment=False, special=True, neg=True)
                    new_formula.update_card(
                        col_name, None, 'leq', containment=False, special=False, neg=False, special_3=True)

                    # print(new_formula)
                    # assert False
                elif op == 'count':
                    new_formula.update_card(
                        col_name, None, 'leq', containment=False, special=True)
                return BaseRefType(new_base_type, new_formula, goal_type.fields)
        else:
            return BaseRefType(get_base_type('Const'))

    def get_prov_formula(self, field_names: List[str]) -> XORFormula:
        """
        encode the predicate that will be used to prove the correctness of the production
        maybe i should make this a lambda thing
        suppose we have filed name A, B , C, then formula should be
        ( Prov(A, mean) XOR Prov(A, sum) XOR Prov(A, count) ) XOR ( Prov(B, mean) XOR Prov(B, sum) XOR Prov(B, count) ) XOR ( Prov(C, mean) XOR Prov(C, sum) XOR Prov(C, count) )
        """

        def summarize_gen(x): return create_xor_formula_instance(
            [[(Prov(f, op), False) for op in ['mean', 'sum', 'count']] for f in x])

        return summarize_gen(field_names)


class ConstProduction(Production):
    def __init__(self, function_name: str, ret_sym: NonterminalSymbol, ret_type: RefType, arg_syms: List[Symbol], arg_types: List[RefType]):
        super().__init__(function_name, ret_sym, ret_type, arg_syms, arg_types)

    def apply(self, prog: Program, expand_var: VariableNode, typed: bool, input_type: BaseRefType, no_prov: bool, *args):
        super().apply(prog, expand_var, typed, input_type, no_prov)

    def reverse_apply(self, new_prog: Program, curr_prog: Program, input_type: BaseRefType):
        super().reverse_apply(new_prog, curr_prog, input_type)

    def forward(self, arg_types: List[BaseRefType], additional_args: List[Node], input_type: BaseRefType, no_prov: bool, **args) -> RefType:
        raise NotImplementedError

    def inverse(self, goal_type: BaseRefType, arg_index: int, input_type: BaseRefType, no_prov: bool, *args) -> RefType:
        return BaseRefType(get_base_type('Const'))
