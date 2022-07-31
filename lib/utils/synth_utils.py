import os
from typing import Dict, List

from lib.eval.benchmark import Benchmark
from lib.falx.visualization.chart import VisDesign
from lib.program import Program
from lib.synthesizer.lemma_learning import Lemma
from lib.type.base_type import BaseType
from lib.type.formula import CNFFormula
from lib.type.predicate import Prov
from lib.type.ref_type import BaseRefType
from lib.type.type_system import allowed_data_type_for_plot, subtyping_relation, create_cnf_formula_instance
from lib.type.type_system_check import check_compatibility
from lib.utils.data_utils import get_data


def check_goal_type(goal_type: BaseRefType, conflicts_based: Dict[str, List[Lemma]]) -> bool:
    """
    Check if a given goal type is valid of not (according to the conflicts)
    we only focus on the binding of the goal type
    for instance, if a variable is assigned to x of a line plot and you know that both ordinal and temporal is not allowed. you can prune it early.
    """
    goal_type_base: BaseType = goal_type.base
    plot_type = goal_type_base.name
    goal_type_f: CNFFormula = goal_type.constraint

    # we do not check for any constraint unless it is line or area because in that case you can prune easily
    if plot_type != 'LinePlot' and plot_type != 'AreaPlot':
        return True

    # gather all the encoding
    encoding_dict = {}
    encoding_dict_inverse = {}
    for binding_str in goal_type_f.binding_str_to_literal_id.keys():
        # print("binding_str: ", binding_str)
        split_binding_str = binding_str.split(',')
        column_name = split_binding_str[0][8:]
        enc = split_binding_str[1][:-1]

        encoding_dict[column_name] = enc
        encoding_dict_inverse[enc] = column_name

    if encoding_dict_inverse['x'] not in conflicts_based:
        return True

    lemmas = conflicts_based[encoding_dict_inverse['x']]
    forbidden_type = [l.get_req_str() for l in lemmas]
    # print("forbidden_type: ", forbidden_type)
    if 'Temporal' in forbidden_type and 'Ordinal' in forbidden_type:
        return False

    return True


def check_disallow_prov(goal_type: BaseRefType, conflicts_prov: Dict[str, Prov]) -> bool:
    goal_type_f: CNFFormula = goal_type.constraint

    for prov_str, lit_id in goal_type_f.prov_str_to_literal_id.items():
        if not goal_type_f.literal_id_to_negation[lit_id] and prov_str in conflicts_prov:
            return False

    return True


def create_pseudo_goal_type(goal_type: BaseRefType) -> BaseRefType:
    """
    We create a fake goal type that trim all the constraints other than the x, y -binding constraints and the fields
    """
    # output_type = BaseRefType(get_base_type(plot_str), constraint=constraint, fields=field_used, prob=self.score)
    # we extract the x, y binding because we have already synthesized that
    constraints = []
    for binding_str, lit_id in goal_type.constraint.binding_str_to_literal_id.items():
        if ',x' in binding_str or ',y' in binding_str:
            pred = goal_type.constraint.literal_id_to_atom[lit_id].predicate
            neg = goal_type.constraint.literal_id_to_negation[lit_id]
            constraints.append([(pred, neg)])
    constraint_f = create_cnf_formula_instance(constraints)

    new_goal_type = BaseRefType(goal_type.base, constraint=constraint_f, fields=goal_type.fields, prob=goal_type.prob)

    return new_goal_type


def check_prov_constraint(b: Benchmark, prog: Program, goal_type: BaseRefType) -> bool:
    """
    check the synthesized program against the prov constraints
    """
    # print("check_prov_constraint:", prog.to_vega_lite())
    actual_type = prog.node_to_refined_type[prog.start_node.id]
    assert isinstance(actual_type, BaseRefType)
    # print(goal_type)
    # print(actual_type)

    # check if binding is consistent
    # check if the binding strings are set equivalent
    if set(goal_type.constraint.binding_str_to_literal_id.keys()) != set(actual_type.constraint.binding_str_to_literal_id.keys()):
        # print("binding not right")
        return False

    # then we check if table transformation is consistent, this can be viewed as a compatibility check
    if not check_compatibility(actual_type, goal_type).res:
        # assert False
        # print("table transformation not right")
        return False
    else:
        return True


def compile_and_check_table(b: Benchmark, prog: Program, no_table: bool = False) -> bool:
    """
    If no_table is True, then that means we need to check the cardinality information
    If no_table is False, then that means we only need to check the base type information (i.e. we actually don't need to compile the graph)
    """

    # print("check prog:", prog)

    is_valid = True

    data_dir = os.path.join("eval", "data")
    _, _, falx_data = get_data(data_dir, b, mode='synth', generate_synthesis_constraint=True)

    vl_spec = prog.to_vega_lite()
    vl_prog = VisDesign.load_from_vegalite(vl_spec, falx_data)

    plot_type_name = prog.get_plot_type_name()
    for encoding, val in vl_prog.chart.encodings.items():
        if encoding == 'order':
            continue
        allowed_types = allowed_data_type_for_plot[plot_type_name][encoding]
        cur_type = b.data.datatype_properties[val.field]['type']
        if val.aggregate:
            if val.aggregate in {'sum', 'mean'} and subtyping_relation[cur_type] != 'Quantitative':
                is_valid = False
                break
            cur_type = 'Aggregate'
        elif val.bin:
            if subtyping_relation[cur_type] != 'Quantitative':
                is_valid = False
                break
            cur_type = 'Discrete'
        if cur_type not in allowed_types and subtyping_relation[cur_type] not in allowed_types:
            is_valid = False
            break

    binned_fields = [item.field for item in vl_prog.chart.encodings.values() if item.bin]
    # additional constraint for the binning field
    for binned_field in binned_fields:
        # 1. the binned field must be quantitative
        if b.data.datatype_properties[binned_field]['type'] != 'Discrete' or b.data.datatype_properties[binned_field]['type'] != 'Continuous':
            is_valid = False
            break

        # TODO: 2. the initial cardinality to perform a binning should be greater than 20 (otherwise what is the point?)

    if no_table:

        try:
            vl_output = vl_prog.eval(falx_data, )
        except TypeError:
            return False

        assert vl_output is not None

        if len(vl_output) <= 1:
            is_valid = False

        color_card = len(
            set([vl_output[i].color for i in range(len(vl_output))]))
        column_card = len(
            set([vl_output[i].column for i in range(len(vl_output))]))

        # shared cardinality constraint
        if color_card > 20 or column_card > 10:
            x_field = vl_prog.chart.encodings['x'].field
            y_field = vl_prog.chart.encodings['y'].field
            if not (plot_type_name in {'ScatterPlot', 'LinePlot', 'AreaPlot'} and (x_field in binned_fields or y_field in binned_fields)):
                is_valid = False

        # duplicate constraint
        if plot_type_name != 'ScatterPlot' and plot_type_name != 'Histogram':
            all_enc_except_y_card = len(set(
                [(vl_output[i].x, vl_output[i].color, vl_output[i].column) for i in range(len(vl_output))]))
            # all_enc_card = len(set([tuple(vl_output[i])
            #                         for i in range(len(vl_output))]))
            if hasattr(vl_output[0], 'y'):
                y_card = len(set([vl_output[i].y for i in range(len(vl_output))]))
            else:
                y_card = len(set([vl_output[i].y1 for i in range(len(vl_output))]))
            # print(y_card, all_enc_except_y_card)
            if all_enc_except_y_card < y_card:
                is_valid = False

        # qualitative cardinality constraint
        nominal_ordinal_col = [encoding for encoding, item in vl_prog.chart.encodings.items(
        ) if (b.data.datatype_properties[item.field]['type'] in {'Nominal', 'Ordinal'}) and not item.aggregate]
        col_card = 0
        for encoding in nominal_ordinal_col:
            # print("enc:", encoding)
            if encoding == 'x':
                col_card = len(
                    set([vl_output[i].x for i in range(len(vl_output))]))
            elif encoding == 'y':
                if hasattr(vl_output[0], 'y'):
                    col_card = len(set([vl_output[i].y for i in range(len(vl_output))]))
                elif hasattr(vl_output[0], 'y1'):
                    col_card = len(set([vl_output[i].y1 for i in range(len(vl_output))]))
            else:
                continue
            # print("col_card:", col_card)
            if col_card > 50:
                is_valid = False
                break

    return is_valid


def get_curr_enc_assignment(fields: List[str], known_enc_column_assignment: Dict[str, str], color_or_column: bool = False) -> List[str]:

    if color_or_column:
        # color or column assignment should satsify the following:
        # avoid the y-assignment
        # avoid assign the same value as color/column
        ret_assignment = []
        for field in fields:
            if 'y' in known_enc_column_assignment and field == known_enc_column_assignment['y']:
                continue
            # if 'color' in known_enc_column_assignment and field == known_enc_column_assignment['color']:
            #     continue
            # if 'column' in known_enc_column_assignment and field == known_enc_column_assignment['column']:
            #     continue

            ret_assignment.append(field)

        return ret_assignment

    else:
        raise NotImplementedError
