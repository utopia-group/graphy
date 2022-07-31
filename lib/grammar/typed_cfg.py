from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple
from lib.grammar.production import Production, PlotProduction, PlotFuncProduction, BarProduction, ScatterProduction, LineProduction, AreaProduction, BinProduction, FilterProduction, SelectProduction, \
    MutateProduction, SummarizeProduction, ConstProduction, HistogramProduction
from lib.grammar.symbol import TerminalSymbol, NonterminalSymbol, Symbol
from lib.node import VariableNode
from lib.type.base_type import TableType, DiscreteType, AggregateType, QuantitativeType, CellType, ContinuousType
from lib.type.formula import CNFFormula
from lib.type.predicate import Predicate, Prov
from lib.type.ref_type import BaseRefType, FunctionType, RefType
from lib.type.type_system import create_ref_type, create_func_type, subtyping_relation
from lib.type.type_system_check import check_implication, get_implication_model
from lib.utils.misc_utils import printd

"""
So, this is a typed CFG 
"""


class TypedCFG:
    def __init__(self):

        self.terminal_syms: List[TerminalSymbol] = []
        self.nonterminal_syms: List[NonterminalSymbol] = []
        self.start_sym: NonterminalSymbol = None

        self.name_to_sym: Dict[str, Symbol] = {}
        self.name_to_prod: Dict[str, Production] = {}

        # symbol -> output type -> production
        self.symbol_to_out_production: defaultdict[Symbol,
                                                   Dict[str, List[Production]]] = defaultdict(dict)

        self.parse()

        assert self.start_sym is not None

    def parse(self):
        """
        generate the grammar
        """
        # nonterminals
        prog_sym = self.add_nonterminal_sym('program')
        pv_func_sym = self.add_nonterminal_sym('pv_func')
        pv_sym = self.add_nonterminal_sym('pv')
        pt_sym = self.add_nonterminal_sym('pt')
        v = self.add_nonterminal_sym('v')

        self.start_sym = prog_sym

        # terminals
        # the values are empty because we don't know that yet
        t_in_sym = self.add_terminal_sym(
            'T_in', [])  # used specifically for input
        # used for the input T in the lambda
        t_sym = self.add_terminal_sym('T', [])
        const_sym = self.add_terminal_sym('const', [])
        # placeholders
        # params_sym = self.add_terminal_sym('params', ["5", "10"])
        params_sym = self.add_terminal_sym('params', ["5"])
        # used for the reference in the plot functions
        symbolic_col_sym = self.add_terminal_sym('sym_col', [])
        symbolic_col_list_sym = self.add_terminal_sym('sym_col_list', [])
        col_sym = self.add_terminal_sym('col', [])
        # placeholders
        op_sym = self.add_terminal_sym('op', ['<', '=', '>'])
        alpha_sym = self.add_terminal_sym('alpha', ['mean', 'sum', 'count'])

        """
        Intuition: we are able to find 'relatively' type-checked program because the return type in the grammar does not have the notion about the refinement predicate, 
        even it has, we still need bi-directional type checking to confirm it so during enumeration stage we can make a good guess but not a certain guess
        therefore here i think it is sufficient to use base type to index the productions
        for efficiency reason we should just use string. (but this is tentative)
        """
        prog_sym_productions = {
            'Plot': [PlotProduction('plot', prog_sym, create_ref_type('Plot'), [pv_func_sym, pt_sym], [create_ref_type('Table'), create_func_type('Table', 'Plot')])]
        }
        self.symbol_to_out_production[prog_sym] = prog_sym_productions

        pv_func_sym_productions = {
            'Table|Plot': [PlotFuncProduction('v_func', pv_func_sym, create_func_type('Table', 'Plot'), [t_sym, pv_sym], [create_ref_type('Table'), create_ref_type('Plot')])]
        }
        self.symbol_to_out_production[pv_func_sym] = pv_func_sym_productions

        pv_sym_productions = {
            'BarPlot': [BarProduction('bar', pv_sym, create_ref_type('BarPlot'),
                                      [symbolic_col_sym, symbolic_col_sym,
                                          symbolic_col_sym, symbolic_col_sym],
                                      [create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol')])],
            'Histogram': [HistogramProduction('histogram', pv_sym, create_ref_type('Histogram'),
                                              [symbolic_col_sym, symbolic_col_sym,
                                                  symbolic_col_sym, symbolic_col_sym],
                                              [create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol')])],
            'ScatterPlot': [ScatterProduction('scatter', pv_sym, create_ref_type('ScatterPlot'),
                                              [symbolic_col_sym, symbolic_col_sym,
                                                  symbolic_col_sym, symbolic_col_sym],
                                              [create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol')])],
            'LinePlot': [LineProduction('line', pv_sym, create_ref_type('LinePlot'),
                                        [symbolic_col_sym, symbolic_col_sym,
                                            symbolic_col_sym, symbolic_col_sym],
                                        [create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol')])],
            'AreaPlot': [AreaProduction('area', pv_sym, create_ref_type('AreaPlot'),
                                        [symbolic_col_sym, symbolic_col_sym,
                                            symbolic_col_sym, symbolic_col_sym],
                                        [create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol'), create_ref_type('ConstCol')])],
        }
        self.symbol_to_out_production[pv_sym] = pv_sym_productions

        pt_sym_productions = {
            'Table': [SelectProduction('select', pt_sym, create_ref_type('Table'), [pt_sym, symbolic_col_list_sym],
                                       [create_ref_type('Table'), create_ref_type('ConstCol')]),
                      BinProduction('bin', pt_sym, create_ref_type('Table'), [pt_sym, params_sym, symbolic_col_sym],
                                    [create_ref_type('Table'), create_ref_type('Const'), create_ref_type('ConstCol')]),
                      # FilterProduction('filter', pt_sym, create_ref_type('Table'), [pt_sym, v, op_sym, v],
                      #                  [create_ref_type('Table'), create_ref_type('Const'), create_ref_type('Const'), create_ref_type('Const')]),
                      SummarizeProduction('summarize', pt_sym, create_ref_type('Table'), [pt_sym, alpha_sym, symbolic_col_sym],
                                          [create_ref_type('Table'), create_ref_type('ConstCol'), create_ref_type('Alpha'), create_ref_type('ConstCol')]),
                      # MutateProduction('mutate', pt_sym, create_ref_type('Table'), [pt_sym, symbolic_col_sym, op_sym, symbolic_col_sym],
                      #                  [create_ref_type('Table'), create_ref_type('ConstCol'), create_ref_type('Const'), create_ref_type('ConstCol')]),
                      ]
        }
        self.symbol_to_out_production[pt_sym] = pt_sym_productions

        v_productions = {
            'Const': [ConstProduction('const', v, create_ref_type('Const'), [const_sym], [create_ref_type('Const')]),
                      ConstProduction('col', v, create_ref_type('Const'), [symbolic_col_sym], [create_ref_type('ConstCol')])],
        }

        self.symbol_to_out_production[v] = v_productions

    def add_terminal_sym(self, name, values) -> TerminalSymbol:
        sym = TerminalSymbol(name, values)
        self.name_to_sym[name] = sym
        return sym

    def add_nonterminal_sym(self, name) -> NonterminalSymbol:
        sym = NonterminalSymbol(name)
        self.name_to_sym[name] = sym

        return sym

    def get_productions(self, sym: Symbol, base_type_name: str) -> List[Production]:
        """
        get production according to the base type
        """
        # print("Sym:", sym)
        # print("sym=>prods:", self.symbol_to_out_production[sym])
        return self.symbol_to_out_production[sym][base_type_name]

    def select_productions_without_pruning(self, var: VariableNode) -> List[Production]:
        productions: List[Production] = []
        prod_dict = self.symbol_to_out_production[var.sym]
        for prods in prod_dict.values():
            productions.extend(prods)
        return productions

    def select_productions(self, var: VariableNode, goal_type: RefType, pruned_production: Optional[List] = None) -> List[Union[Production, Tuple[Production, Predicate]]]:
        """
        Technically, we need to loop through the production list to see which ones fit the subtyping relationship
        but this is not very efficient
        to achieve constant time lookup it seems the only solution here is to hard code all the subtyping relationship
        """
        var_goal_type = goal_type

        if 'func' in var.sym.name:
            # need a special case for function type
            assert isinstance(var_goal_type, FunctionType)
            goal_base_str = '{}|{}'.format(var_goal_type.input.base.name, var_goal_type.output.base.name)
            if goal_base_str in self.symbol_to_out_production[var.sym]:
                return self.symbol_to_out_production[var.sym][goal_base_str]
            else:
                # handle Table -> Plot situation
                input_super_class = var_goal_type.input.base.name \
                    if var_goal_type.input.base.name not in subtyping_relation \
                    else subtyping_relation[var_goal_type.input.base.name]
                output_super_class = var_goal_type.output.base.name \
                    if var_goal_type.output.base.name not in subtyping_relation \
                    else subtyping_relation[var_goal_type.output.base.name]

                goal_base_super_str = '{}|{}'.format(input_super_class, output_super_class)
                if goal_base_super_str == goal_base_str:
                    raise TypeError

                if goal_base_super_str in self.symbol_to_out_production[var.sym]:
                    return self.symbol_to_out_production[var.sym][goal_base_super_str]
                else:
                    raise TypeError('(goal_base_str, goal_base_super_str):', goal_base_str, goal_base_super_str)
        else:
            assert isinstance(var_goal_type, BaseRefType)
            # print("var_goal_type:", var_goal_type)
            # print("var.sym:", var.sym)
            if var_goal_type.base.name in self.symbol_to_out_production[var.sym]:
                if var_goal_type.base.name == 'Table':

                    assert isinstance(var_goal_type.base, TableType)
                    assert len(var_goal_type.base.fields) > 0
                    assert pruned_production is not None

                    return_prod = []

                    curr_prod: Production
                    for curr_prod in self.symbol_to_out_production[var.sym][var_goal_type.base.name]:

                        if curr_prod.function_name in pruned_production:
                            continue

                        if curr_prod.function_name is 'select':

                            if len(var_goal_type.constraint.prov_clause_ids) == 0:
                                return_prod.append((curr_prod, None))
                            else:
                                # core idea: we know that when select operation is applied, there should be no more true prov predicate in the goal type for the var
                                printd('var.goal_type:', var_goal_type)
                                prov_predicates: CNFFormula = var_goal_type.constraint.get_prov_formula(-1)

                                # get all literal ids in prov_predicates and check if it contains not neg
                                # if it does, then we can't select this production
                                all_prov_literal_idx = [literal_id for clause_id in prov_predicates.clause_id.keys() for literal_id in prov_predicates.clause_id_to_literal_id[clause_id]]
                                if all([prov_predicates.literal_id_to_negation[literal_id] for literal_id in all_prov_literal_idx]):
                                    return_prod.append((curr_prod, None))
                                else:
                                    printd("avoid enum select")

                        elif curr_prod.function_name is 'bin' or curr_prod.function_name is 'summarize':

                            # ideally, this branch should only be reachable for the no prov one
                            if len(var_goal_type.constraint.prov_clause_ids) == 0:
                                for field in var_goal_type.base.fields:

                                    if issubclass(DiscreteType, var_goal_type.base.get_record()[field].T.__class__):
                                        if curr_prod.function_name == 'bin':
                                            return_prod.append((curr_prod, Prov(field, 'bin')))
                                        if curr_prod.function_name == 'summarize':
                                            return_prod.append((curr_prod, Prov(field, 'count')))

                                    if curr_prod.function_name == 'summarize' and issubclass(ContinuousType, var_goal_type.base.get_record()[field].T.__class__):
                                        return_prod.append((curr_prod, Prov(field, 'mean')))
                                        return_prod.append((curr_prod, Prov(field, 'sum')))

                            else:

                                prov_predicates = var_goal_type.constraint.get_prov_formula(-1)
                                printd("negated_prov_predicates:", prov_predicates)
                                prod_predicates = curr_prod.get_prov_formula(var_goal_type.base.fields)
                                printd("prod_predicates:", prod_predicates)

                                # first optimization: check if prod_predicates and prov_predicates are completely disjoint.
                                # if so, we don't have to call the solver

                                # conjunct two formula's mentioned variable
                                prov_predicates_atom_id = [atom.id for _, atom in prov_predicates.literal_id_to_atom.items()]
                                prod_predicates_atom_id = [atom.id for _, atom in prod_predicates.literal_id_to_atom.items()]
                                disjoint = set(prov_predicates_atom_id).isdisjoint(set(prod_predicates_atom_id))
                                printd("disjoint:", disjoint)
                                if disjoint:
                                    assert curr_prod.function_name is 'bin'  # there is no way this can be summarize

                                    for field in var_goal_type.base.fields:
                                        if issubclass(DiscreteType, var_goal_type.base.get_record()[field].T.__class__):
                                            # need to check one more premise
                                            # if there is a prov predicate related to this column that is true, then we cannot proceed
                                            # because once the bin operation is applied, this prov will be false
                                            # TODO: there must be a better way to do this. i am rush on time now lol
                                            try:
                                                if all([prov_predicates.literal_id_to_negation[prov_predicates.prov_str_to_literal_id[str(Prov(field, op))]] for op in ['mean', 'sum', 'count']]):
                                                    return_prod.append((curr_prod, Prov(field, 'bin')))
                                            except KeyError:
                                                # this branch for the case that the prov constraint is not complete
                                                prov_str = []
                                                for op in ['mean', 'sum', 'count']:
                                                    prov = Prov(field, op)
                                                    if str(prov) in prov_predicates.prov_str_to_literal_id:
                                                        prov_str.append(str(prov))
                                                if all([prov_predicates.literal_id_to_negation[prov_predicates.prov_str_to_literal_id[prov_str_]] for prov_str_ in prov_str]):
                                                    return_prod.append((curr_prod, Prov(field, 'bin')))
                                else:

                                    # get model and assign it to proper production
                                    # print("prov_predicates:", prov_predicates)
                                    # print("prod_predicates:", prod_predicates)
                                    get_model = get_implication_model(prov_predicates, prod_predicates)
                                    model_count = 0
                                    try:
                                        while True:
                                            atom, prov = next(get_model)
                                            model_count += 1

                                            # NOTE: this test pass on the assumption that only create_atom is called to create atom
                                            if atom in prod_predicates.literal_id_to_atom.values() and issubclass(var_goal_type.base.get_record()[prov.col].T.__class__, CellType):
                                                return_prod.append((curr_prod, prov))
                                            # else:
                                            #     print("debugging")
                                            #     print(var_goal_type.base.get_record()[prov.col])

                                    except StopIteration:
                                        if model_count == 0:
                                            printd("{} is pruned".format(curr_prod.function_name))
                                            pruned_production.append(curr_prod.function_name)

                        else:
                            raise ValueError("unknown function name:", curr_prod.function_name)

                    printd("return_prod:", return_prod)
                    return return_prod
                else:
                    return self.symbol_to_out_production[var.sym][var_goal_type.base.name]
            else:
                # get super type
                if var_goal_type.base.name not in subtyping_relation:
                    printd("TypeError: no subtyping relation for {}".format(var_goal_type.base.name))
                    raise TypeError

                if subtyping_relation[var_goal_type.base.name] in self.symbol_to_out_production[var.sym]:
                    return self.symbol_to_out_production[var.sym][subtyping_relation[var_goal_type.base.name]]
                else:
                    printd("TypeError: wrong subtyping relation for {}".format(var_goal_type.base.name))
                    raise TypeError
