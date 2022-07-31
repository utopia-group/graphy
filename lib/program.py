import itertools
from collections import OrderedDict
from typing import Dict, List, Optional, Union, Tuple

from lib.eval.benchmark import InputDataset
from lib.grammar.symbol import Symbol, TerminalSymbol
from lib.node import Node, NonterminalNode, VariableNode, TerminalNode
from lib.type.base_type import TableType, ContinuousType, NullType, NullOpType, ConstColType
from lib.type.formula import conjunct_formula, CNFFormula
from lib.type.predicate import Prov, RelationPred, Variable, Constant
from lib.type.ref_type import BaseRefType, RefType, FunctionType
from lib.type.type_system import get_next_formula_id, get_base_type, create_cnf_formula_instance
from lib.utils.misc_utils import printd


class Program:
    def __init__(self, _id: int, goal_type: Optional[BaseRefType]):
        """
        Program data structure
        """

        # Classic fields
        self.id: int = _id
        self.start_node: Optional[NonterminalNode] = None
        self.nodes: Dict[int, Node] = {}
        self.to_children: Dict[int, List[int]] = {}
        self.to_parent: Dict[int, int] = {}

        # For type-directed synthesis
        # note that it is possible to have one reftype or multiple ned refined type when strengthening since we are doing some guessing here
        self.node_to_refined_type: Dict[int,
                                        Union[RefType, List[RefType]]] = {}
        self.nodes_pushed: List[int] = []

        # domain-specific stuff
        self.visualization_root_node: Optional[Node] = None
        self.table_root_node: Optional[Node] = None  # Note that node should never get modified once created since it is shared across all programs
        self.table_root_node_goal_type: Optional[RefType] = None  # this can be modified

        # Keep track for var nodes (although not sure if this is what we need)
        self.var_nodes: OrderedDict[int, str] = OrderedDict()
        self.to_expand_var_nodes: Optional[VariableNode] = None

        self.type: BaseRefType = goal_type

        self.node_id_counter = itertools.count(0)

        # the min depth of the ast tree
        self.depth: int = 0

        # plot info to better keep track of enumeration
        self.plot_type = None
        self.x_axis = None
        self.y_axis = None
        self.color = None
        self.column = None
        self.select = None

    def duplicate(self, _id: int) -> 'Program':
        ret = Program(_id, self.type)
        ret.start_node = self.start_node
        ret.nodes = self.nodes.copy()
        ret.to_children = self.to_children.copy()
        ret.to_parent = self.to_parent.copy()
        ret.node_to_refined_type = self.node_to_refined_type.copy()
        ret.nodes_pushed = self.nodes_pushed.copy()
        ret.visualization_root_node = self.visualization_root_node
        ret.table_root_node = self.table_root_node
        ret.var_nodes = self.var_nodes.copy()
        ret.to_expand_var_nodes = self.to_expand_var_nodes
        ret.node_id_counter = itertools.tee(self.node_id_counter)[1]

        ret.table_root_node_goal_type = self.table_root_node_goal_type
        ret.depth = self.depth

        ret.plot_type = self.plot_type
        ret.x_axis = self.x_axis
        ret.y_axis = self.y_axis
        ret.color = self.color
        ret.column = self.column
        ret.select = self.select

        return ret

    def is_concrete(self, component=None) -> bool:
        if component is None:
            return len(self.var_nodes) == 0

        assert component is 'visualization' or component is 'table'

        filter_str = 'pv' if component is 'visualization' else 'pt'
        for var_node_id in self.var_nodes:
            if filter_str in self.nodes[var_node_id].sym.name:
                return False
            if len(self.var_nodes) > 1:
                return False
        return True

    def add_variable_node(self, sym: Symbol, goal_type: Optional[RefType], parent_node: Optional[NonterminalNode], sub_id=None, arg_idx=None) -> VariableNode:

        new_id = next(self.node_id_counter) if sub_id is None else sub_id
        new_node = VariableNode(new_id, 'v', sym, goal_type, arg_idx)
        self.nodes[new_node.id] = new_node

        if parent_node is None:
            self.start_node = new_node
        else:
            self.set_parent(parent_node, new_node)

        self.var_nodes[new_node.id] = ''

        return new_node

    def add_terminal_node(self, sym: TerminalSymbol, value: str, goal_type: Optional[BaseRefType], parent_node: Optional[NonterminalNode], sub_id=None) -> TerminalNode:

        new_id = next(self.node_id_counter) if sub_id is None else sub_id
        new_node = TerminalNode(new_id, value, sym, value, goal_type)

        if parent_node is None:
            self.start_node = new_node
        else:
            self.set_parent(parent_node, new_node)

        self.nodes[new_node.id] = new_node

        return new_node

    def add_parent_node(self, name: str, sym: Symbol, goal_type: Optional[RefType], curr_node: Node):

        raise NotImplementedError
        # # create id for new node
        # new_id = next(self.node_id_counter)
        # # create new node
        # new_node = NonterminalNode(new_id, name, sym, goal_type)
        # self.start_node = new_node

    def fill_subprog_for_var_node(self, subprog: 'Program'):
        """
         we need to merge the following:
        - update the counter
        - update the nodes
        - update the node_to_refined_type
        - update the parent/children relationship too because subprogram can be a tree
        - remove the var to be filled in the program
        - establish the parent/children relationship
        """

        assert len(self.var_nodes) == 1
        var_node_id, _ = self.var_nodes.popitem(last=False)

        self.nodes.update(subprog.nodes)
        self.node_to_refined_type.update(subprog.node_to_refined_type)
        self.to_parent.update(subprog.to_parent)
        self.to_children.update(subprog.to_children)

        var_node_parent_id = self.to_parent[var_node_id]
        self.to_parent[subprog.start_node.id] = var_node_parent_id
        self.to_children[var_node_parent_id] = self.to_children[var_node_parent_id].copy(
        )
        self.to_children[var_node_parent_id][self.to_children[var_node_parent_id].index(
            var_node_id)] = subprog.start_node.id
        del self.to_parent[var_node_id]

    def add_nonterminal_node(self, name: str, sym: Symbol, goal_type: Optional[RefType], parent_node: Optional[NonterminalNode], prod: Optional[object], sub_id=None) -> NonterminalNode:
        new_id = next(self.node_id_counter) if sub_id is None else sub_id
        new_node = NonterminalNode(new_id, name, sym, goal_type, prod)

        if parent_node is None:
            self.start_node = new_node
        else:
            self.set_parent(parent_node, new_node)

        self.nodes[new_node.id] = new_node

        return new_node

    def select_var_node(self) -> VariableNode:
        if self.to_expand_var_nodes is not None:
            ret_node = self.to_expand_var_nodes
            self.to_expand_var_nodes = None
            return ret_node
        else:
            # get the first var node
            ret_node_id, _ = next(iter(reversed(self.var_nodes.items())))
            ret_node = self.nodes[ret_node_id]
            assert isinstance(ret_node, VariableNode)
            return ret_node

    def delete_var_node(self, node: VariableNode):
        del self.var_nodes[node.id]

    def set_children(self, parent_node: NonterminalNode, children_node: List[Node]):
        self.to_children[parent_node.id] = [c.id for c in children_node]

    def set_parent(self, parent_node: NonterminalNode, child_node: Node):
        self.to_parent[child_node.id] = parent_node.id

    def get_parent(self, child_node_id: int) -> Optional[Node]:
        if child_node_id == self.start_node.id:
            return None
        else:
            return self.nodes[self.to_parent[child_node_id]]

    def get_children(self, parent_node_id: int) -> Optional[List[Node]]:
        if parent_node_id not in self.to_children:
            return None
        else:
            return [self.nodes[c_id] for c_id in self.to_children[parent_node_id]]

    """
    Type related
    """

    def update_type(self, input_type: BaseRefType, node_id: int = -1, has_input_type: bool = True, no_prov: bool = False) -> BaseRefType:

        if node_id == -1:
            node_id = self.start_node.id
        else:
            assert node_id == self.nodes_pushed[-1]

        begin_node = self.nodes[node_id]
        assert begin_node.sym.name == 'pt'
        assert isinstance(begin_node, NonterminalNode)
        # assert self.start_node.sym.name == 'pt'

        # first node corresponds to input table
        child_type = [
            self.node_to_refined_type[self.to_children[node_id][0]]]
        # additional args needed for forward semantics
        additional_nodes = [self.nodes[i]
                            for i in self.to_children[node_id][1:]]

        additional_args = [node.value for node in additional_nodes]

        updated_type = begin_node.prod.forward(child_type, additional_args, input_type, no_prov, has_input_type=has_input_type)
        self.node_to_refined_type[node_id] = updated_type
        self.type = self.node_to_refined_type[node_id]

        if node_id != self.start_node.id:
            self.nodes_pushed.pop(-1)

        return updated_type

    def overall_type_strengthening(self):
        table_type = self.node_to_refined_type[self.table_root_node.id]
        plot_func_type = self.node_to_refined_type[self.visualization_root_node.id]
        assert isinstance(table_type, BaseRefType)
        assert isinstance(plot_func_type, FunctionType)
        self.node_to_refined_type[self.start_node.id] = BaseRefType(plot_func_type.output.base, conjunct_formula(get_next_formula_id(), table_type.constraint, plot_func_type.output.constraint))
        # print("overall_refined_type:", overall_refined_type)

    def viz_table_input_strengthening(self, has_input_type: bool = True, input_type: Optional[BaseRefType] = None) -> Tuple[int, List[RefType]]:
        """
        We are trying to strengthen the visualization based on the current partial program
        the results can have multiple new types -> that's why we stop tracing the stack at the pv_func symbol, we might need to create new partial program based on it
        return the node id for plotfunc too since we need it

        loop from leaf, root (the trace is obtained from the node_pushed construct)
        a couple step is omitted:
        - the refined type of plot should be derived from the refined type of its argument -> we don't have refinement information
        - most of the code is trying to refine input table by calling the forward semantic of plotfunc()
        - the refined type of the function is a trivial step
        """

        while len(self.nodes_pushed) > 0:
            curr_node = self.nodes[self.nodes_pushed.pop()]
            assert isinstance(curr_node, NonterminalNode)
            # print("curr_node:", curr_node)
            # print("curr_node.sym.name:", curr_node.sym.name)
            if curr_node.sym.name == 'program':
                # we don't refine anything at the top level because we only finishes one of the node
                pass
            elif curr_node.sym.name == 'pv':

                # if it is pv symbol, then nothing to be propagated
                # TODO: need to change this
                # self.node_to_refined_type[curr_node.id] = curr_node.goal_type
                children_type: List[RefType] = []
                for cid in self.to_children[curr_node.id]:
                    children_type.append(self.get_goal_type(self.nodes[cid]))
                self.node_to_refined_type[curr_node.id] = curr_node.prod.forward(children_type, None, input_type, False)
                # print("curr_node.goal_type:", self.node_to_refined_type[curr_node.id])
            else:
                # we only using the information of non terminal node when strengthening (basically we need to propose what is the refined type for T in plotfunc)
                children_type: List[RefType] = []
                for cid in self.to_children[curr_node.id]:
                    if self.node_to_refined_type.get(cid) is None:
                        children_type.append(self.get_goal_type(self.nodes[cid]))
                    else:
                        children_type.append(self.node_to_refined_type[cid])
                # print("here children_type:", children_type)
                return curr_node.id, curr_node.prod.forward(children_type, [], input_type, False, has_input_type=has_input_type)

    def viz_table_input_strengthening_2(self, has_input_type: bool = True, input_type: Optional[BaseRefType] = None, no_prov=False) -> Tuple[int, List[RefType]]:
        """
        Same as viz_table_input_strengthening, but i create another function here to avoid baseline mess up with the tool code
        """

        while len(self.nodes_pushed) > 0:
            curr_node = self.nodes[self.nodes_pushed.pop()]
            assert isinstance(curr_node, NonterminalNode)
            # print("curr_node:", curr_node)
            # print("curr_node.sym.name:", curr_node.sym.name)
            if curr_node.sym.name == 'program':
                # we don't refine anything at the top level because we only finishes one of the node
                pass
            elif curr_node.sym.name == 'pv':

                # if it is pv symbol, then nothing to be propagated
                # TODO: need to change this
                # self.node_to_refined_type[curr_node.id] = curr_node.goal_type
                children_type: List[RefType] = []
                for cid in self.to_children[curr_node.id]:
                    children_type.append(self.get_goal_type(self.nodes[cid]))
                self.node_to_refined_type[curr_node.id] = curr_node.prod.forward(children_type, None, input_type, no_prov)
                # print("curr_node.goal_type:", self.node_to_refined_type[curr_node.id])
            else:
                # we only using the information of non terminal node when strengthening (basically we need to propose what is the refined type for T in plotfunc)
                children_type: List[RefType] = []
                for cid in self.to_children[curr_node.id]:
                    if self.node_to_refined_type.get(cid) is None:
                        children_type.append(self.get_goal_type(self.nodes[cid]).duplicate_ignore_base())
                    else:
                        children_type.append(self.node_to_refined_type[cid].duplicate_ignore_base())
                # print("here children_type:", children_type)
                return curr_node.id, curr_node.prod.forward(children_type, [], input_type, no_prov, has_input_type=has_input_type)

    def update_goal_type(self, no_prov=False):
        """
        TODO: update the goal type of the table transformation part according to the results of strengthening. Hack for now to run test
        the refined type should take base type and data constraints in strengthened input table type, conjunct with the provenance constraint in decomposed input table type
        TODO: issue here is that we cannot simply conjunct the strengthed input table type and decomposed input table type
        """

        refined_vis_type = self.node_to_refined_type[self.visualization_root_node.id]
        decomposed_input_table_type = self.get_goal_type(self.table_root_node)

        assert isinstance(refined_vis_type, FunctionType)
        assert isinstance(decomposed_input_table_type, BaseRefType)
        assert isinstance(decomposed_input_table_type.base, TableType)

        strengthened_input_table_type = refined_vis_type.input
        if not no_prov:
            strengthened_input_table_type.base.fields = decomposed_input_table_type.base.fields
            strengthened_input_table_type.fields = decomposed_input_table_type.base.fields
        else:
            fields = list(set(strengthened_input_table_type.base.fields))
            strengthened_input_table_type.base.fields = fields
            strengthened_input_table_type.fields = fields

        printd("decomposed_input_table_type:", decomposed_input_table_type)
        printd("strengthened_input_table_type:", strengthened_input_table_type)

        refined_table_type = BaseRefType(strengthened_input_table_type.base, constraint=conjunct_formula(get_next_formula_id(),
                                                                                                         strengthened_input_table_type.constraint, decomposed_input_table_type.constraint))
        refined_table_type.fields = decomposed_input_table_type.base.fields

        # check plot type
        # if the plot is a histogram, append more constraint
        # histogram requires the goal type has a count operation on the x axis, if not, throw a type error
        # TODO: maybe make this more efficient?
        if refined_vis_type.output.base.name == 'Histogram':
            # get binding x
            assert isinstance(refined_vis_type.output.constraint, CNFFormula)
            x_binding = refined_vis_type.output.constraint.literal_id_to_atom[refined_vis_type.output.constraint.binding_enc_to_literal_id['x']].predicate.col

            count_prov = Prov(x_binding, 'count')
            count_prov_neg = refined_table_type.constraint.get_neg(repr(count_prov))
            if count_prov_neg is not None:
                if count_prov_neg:
                    raise TypeError

            # also check for the same column, and other column, if mean and sum operation has been applied too
            for op in ['mean', 'sum']:
                for col in decomposed_input_table_type.base.fields:
                    check_neg = refined_table_type.constraint.get_neg(repr(Prov(col, op)))
                    if check_neg is not None:
                        if not check_neg:
                            raise TypeError

        self.table_root_node_goal_type = refined_table_type
        printd("refined_table_type:", refined_table_type)

    def get_goal_type(self, node: Node) -> RefType:
        """
        Get the most recent derived goal type of a given node
        """
        assert node is not None

        if self.table_root_node is not None and node.id == self.table_root_node.id:
            if self.table_root_node_goal_type is not None:
                return self.table_root_node_goal_type
            else:
                return node.get_goal_type_restricted()

        if self.node_to_refined_type.get(node.id) is not None:
            return self.node_to_refined_type[node.id]
        else:
            return node.get_goal_type_restricted()

    """
    Visualization related
    """

    def get_plot_type_name(self) -> Optional[str]:
        """
        return the proper name (scatter -> ScatterPlot)
        """
        if self.plot_type is None:
            return None
        elif self.plot_type == 'scatter':
            return 'ScatterPlot'
        elif self.plot_type == 'histogram':
            return 'Histogram'
        elif self.plot_type == 'bar':
            return 'BarPlot'
        elif self.plot_type == 'line':
            return 'LinePlot'
        elif self.plot_type == 'area':
            return 'AreaPlot'
        else:
            raise Exception("Unknown plot type: " + self.plot_type)

    def to_vega_lite(self) -> Dict:
        """
        To generate the vega lite dict, I think we just need to use the refined synthesized type
        TODO: need to add more stuff if we start to incorporate mutate/filter
                also may need to ask the constraint.get_vegalite_format to return more information so we know what to look for in the program
        """
        # print("to_vega_lite: ", self)

        vega_dict = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                     "data": {"url": "data/cars.json"},
                     "mark": None,
                     "encoding": {}}

        refined_output_type = self.node_to_refined_type[self.start_node.id]
        assert isinstance(refined_output_type, BaseRefType)
        # print("refined_output_type: ", refined_output_type)

        vega_dict["mark"] = refined_output_type.base.get_vegalite_name()

        vegalite_enc, field_to_enc_map = refined_output_type.constraint.get_vegalite_format()

        # update the enc with the datatype
        if self.table_root_node is not None and self.node_to_refined_type.get(self.table_root_node.id) is not None:
            table_transformation_type = self.node_to_refined_type[self.table_root_node.id]
            assert isinstance(table_transformation_type, BaseRefType)
            assert isinstance(table_transformation_type.base, TableType)
            table_base_type_record = table_transformation_type.base.get_record()

            for enc, values in vegalite_enc.items():
                if 'field' in values:
                    field_type = table_base_type_record[values['field']]
                    values['type'] = field_type.T.get_vegalite_name()

        vega_dict['encoding'] = vegalite_enc

        return vega_dict

    """
    Additional helpers
    """

    def get_known_enc_col_assignment(self, node: NonterminalNode) -> Dict[str, str]:

        enc_col_mapping = {}

        def get_col_for_enc(child_node: Node) -> Optional[str]:

            goal_type = child_node.get_goal_type_restricted()
            assert isinstance(goal_type, BaseRefType)
            if isinstance(child_node, VariableNode):
                if isinstance(goal_type.base, NullType):
                    return 'None'
                elif isinstance(goal_type.base, NullOpType):
                    return None
                elif isinstance(goal_type.base, ConstColType):
                    assert goal_type.base.col is not None
                    return goal_type.base.col
            elif isinstance(child_node, TerminalNode):
                if child_node.value == 'None':
                    return None
                return child_node.value
            else:
                assert False

            return None

        assert node is not None

        for i, child_node in enumerate(self.get_children(node.id)):
            col_for_enc = get_col_for_enc(child_node)
            if i == 0:
                if col_for_enc is not None:
                    enc_col_mapping['x'] = col_for_enc
            elif i == 1:
                if col_for_enc is not None:
                    enc_col_mapping['y'] = col_for_enc
            elif i == 2:
                if col_for_enc is not None:
                    enc_col_mapping['color'] = col_for_enc
            elif i == 3:
                if col_for_enc is not None:
                    enc_col_mapping['column'] = col_for_enc

        return enc_col_mapping

    def __eq__(self, other):
        if isinstance(other, Program):
            return self.__repr__() == other.__repr__()
        else:
            return False

    def repr_helper(self, node: Node):
        if isinstance(node, TerminalNode) or isinstance(node, VariableNode):
            return node.repr_helper(None)
        else:
            children = [self.repr_helper(self.nodes[cnode_id])
                        for cnode_id in self.to_children[node.id]]
            return node.repr_helper(children)

    def infer_type(self, node: Node, input_type: BaseRefType, input_data: InputDataset) -> Union[RefType, List[RefType]]:
        if isinstance(node, TerminalNode) or isinstance(node, VariableNode):
            if node.sym.name in {"T", "T_in"}:
                input_data.update_input_type(None, all_columns=True)
                input_table_type = input_data.input_type
                return input_table_type
            assert isinstance(node, TerminalNode)
            if node.value is 'None':
                return BaseRefType(get_base_type('Null'), create_cnf_formula_instance([[(RelationPred('coeq', Variable('v'), Constant(node.value)), False)]]))
            else:
                return BaseRefType(get_base_type('ConstCol'), create_cnf_formula_instance([[(RelationPred('coeq', Variable('v'), Constant(node.value)), False)]]))
        else:
            assert isinstance(node, NonterminalNode)
            # print("prod:", node.prod)
            if node.prod.function_name in {"select", "bin", "summarize"}:
                child_nodes = [self.nodes[cnode_id]
                               for cnode_id in self.to_children[node.id]]
                child_type = [self.infer_type(child_nodes[0], input_type, input_data)]
                # additional args needed for forward semantics
                additional_nodes = child_nodes[1:]
                # The values of the child nodes
                additional_args = [node.value for node in additional_nodes]
                if node.prod.function_name == 'select':
                    # ask the input_data to directly give you a new type, this will be much easier than filtering out
                    input_data.update_input_type(additional_args[0])
                    return input_data.input_type
                else:
                    return node.prod.forward(child_type, additional_args, input_data.input_type, False, inferring_type=True)
            else:
                child_nodes = [self.nodes[cnode_id]
                               for cnode_id in self.to_children[node.id]]
                child_types = [self.infer_type(node, input_type, input_data) for node in child_nodes]
                child_vals = [(node.value if isinstance(node, TerminalNode) else None)
                              for node in child_nodes]
                return node.prod.forward(child_types, child_vals, input_data.input_type, False, inferring_type=True)

    def __repr__(self):
        """
        return a string representation of the program
        """
        return 'lambda T. {}'.format(self.repr_helper(self.start_node))

    def __hash__(self):
        return hash(self.__repr__())
