"""
Utilities for running evaluation such as
- parsing the spec/sql/vlspec/synthesized_spec language
- clean the visualization spec
- generate variants of plot to help evaluate the ground truth
"""
import re
import traceback
from io import UnsupportedOperation
from typing import Tuple, Union, Dict, List

from lib.falx.visualization.chart import VisDesign
from lib.neural_parser.labels import CLASSES
from lib.nlp.TabularSemanticParsing.moz_sp import parse
from lib.utils.prev_synth_utils import parse_spec

tregex1 = re.compile(r"lower[(]datum[[]\"(.+)\"[]][)] (<=|>=|<|>|!=|==) (.+)")
parse_field_1 = re.compile(r"'field': '([a-zA-Z0-9_-]+)'")
clean_spec_1 = r"'tooltip': [{][^}]+[}](,)?"
parse_task_1 = re.compile(r"'aggregate': '([a-zA-Z]+)'")
parse_task_2 = re.compile(r"'sort': [{][^}]+'order': '([a-zA-Z]+)'[^}]+}")
parse_vis_1 = re.compile(r"'mark': [{][^}]+'type': '([a-zA-Z]+)'[^}]+[}]")


def parse_single_spec(spec: str) -> Tuple[str, Union[str, None]]:

    if '(' not in spec:
        return spec, None

    intent = spec[:(spec.find('('))]
    field = spec[(spec.find('(') + 1):spec.find(')')]

    return intent, field


def clean_vlspec(spec, change_field=False):

    def change_to_underscore(d:dict):

        if not change_field:
            return d

        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = change_to_underscore(v)
            elif isinstance(v, list):
                new_l = []
                for e in v:
                    if isinstance(e, dict):
                        new_l.append(change_to_underscore(e))
                    else:
                        new_l.append(e)
                d[k] = new_l
            else:
                if k == "field":
                    d[k] = v.replace(" ", "_")
        return d

    if spec is None:
        return None
    
    # I filter out all the table visualization
    if "hconcat" in spec:
        return None

    # only clean top-level transform so far
    if "transform" not in spec:
        return change_to_underscore(spec)
    if len(spec["transform"]) == 0:
        return change_to_underscore(spec)
    
    spec["transform"] = parse_vlspec_transform(spec["transform"])
    
    return change_to_underscore(spec)

def generate_all_valid_variants(spec: dict):

    if "ignore_orientation" in spec:
        new_spec = {}
        for k, v in spec.items():
            if not k == "encoding":
                new_spec[k] = v
                continue
            if "x" in v and "y" in v:
                new_encoding_spec = {}
                for k1, v1 in v.items():
                    if k1 == "x":
                        new_encoding_spec["y"] = v1
                    elif k1 == "y":
                        new_encoding_spec["x"] = v1
                    else:
                        new_encoding_spec[k1] = v1
                new_spec["encoding"] = new_encoding_spec
    
        return [spec, new_spec]
    else:
        return [spec]


# suppose no and/or 
def parse_vlspec_transform(trans):
    
    new_transform = []
    for t in trans:
        if type(t['filter']) == dict:
            new_transform.append(t)
        else:
            assert type(t['filter']) == str
            preds_str = t['filter']
            if '&' in preds_str or '|' in preds_str:
                print("parsing expression involving & or | not supported")
                raise UnsupportedOperation
            parsed_str = re.match(tregex1, preds_str)
            field = parsed_str.group(1)
            op = parsed_str.group(2)
            value = parsed_str.group(3)
            if op == '>=':
                new_t = {"filter": {"field": field, "gte": eval(value)}}
                new_transform.append(new_t)
            elif op == '>':
                new_t = {"filter": {"field": field, "gt": eval(value)}}
                new_transform.append(new_t)
            elif op == '<=':
                new_t = {"filter": {"field": field, "lte": eval(value)}}
                new_transform.append(new_t)
            elif op == '<':
                new_t = {"filter": {"field": field, "lt": eval(value)}}
                new_transform.append(new_t)
            elif op == '!=':
                new_t = {"filter": {"not": {"field": field, "equal": eval(value)}}}
                new_transform.append(new_t)
            elif op == '==':
                new_t = {"filter": {"field": field, "equal": eval(value)}}
                new_transform.append(new_t)
    return new_transform


def parse_vlspec_fields(vlspec):
    vlspec = re.sub(clean_spec_1, "", str(vlspec))
    return re.findall(parse_field_1, str(vlspec))


def parse_asp_fields(asp_prog: list):
    fields = []
    for spec in asp_prog:
        if "field(" in spec:
            fields.append(parse_spec(spec, asp=True)[1].split(',')[1].strip()[1:-1])
    return fields


def parse_vlspec_task(vlspec):
    task_list = []
    task_list.extend(re.findall(parse_task_1, str(vlspec)))
    task_list.extend(re.findall(parse_task_2, str(vlspec)))
    if "filter" in str(vlspec):
        task_list.append("filter")
    return task_list


def parse_vlspec_visualization(vlspec):
    vis_list = []
    vis_list.extend(re.findall(parse_vis_1, str(vlspec)))
    if "color" in str(vlspec):
        vis_list.append("color")
    if "column" in str(vlspec):
        vis_list.append("column")
    return vis_list


def parse_sql_fields(query, dataname):
    regex = re.compile(r"{}[.]([A-Za-z0-9_]+)".format(dataname.lower()))
    return set(re.findall(regex, query))


def parse_sql_predicates(query, dataname):

    try:
        parsed_sql = parse(query)
    except Exception:
        traceback.print_exc()
        return None
    return parsed_sql.get('where')


def check_spec_equiv(data, vlspec_gt, vlspec_synth: Union[Dict, List[Dict]]) -> List[bool]:
    """
    TODO: this is not measuring top-1 acc
    """

    if vlspec_synth is None:
        return False

    if all(vlspec_synth_i is None for vlspec_synth_i in vlspec_synth):
        return [False for _ in vlspec_synth]

    # print("vlspec_synth:", vlspec_synth)

    ignore_orientation = True if "ignore_orientation" in vlspec_gt else False
    vis_gts = [VisDesign.load_from_vegalite(spec, data).eval() for spec in generate_all_valid_variants(vlspec_gt)]
    vis_gt_str = set([repr(e) for e in vis_gts[0]])

    # NOTE: i am removing all sort-related spec in the gt now, since we can't deal with sort
    for enc in vlspec_gt['encoding'].values():
        if 'sort' in enc:
            del enc['sort']

    equiv_list : List[bool] = []
    if isinstance(vlspec_synth, list):
        for vlspec in vlspec_synth:
            # print("current vlspec:", vlspec)

            if vlspec is None or vlspec['encoding'] is None:
                continue

            # TODO: we are going to do a bit hack here (for eval only)
            if vlspec['encoding'].get('color') is not None and vlspec['encoding'].get('column') is not None \
                and vlspec['encoding']['color'].get('field') is not None:
                if vlspec['encoding']['color']['field'] == vlspec['encoding']['column']['field']:
                    del vlspec['encoding']['color']
            try:
                vis_synth = VisDesign.load_from_vegalite(vlspec, data).eval()
            except AssertionError as e:
                print('AssertionError')
                continue
            except SyntaxError as e:
                print("SyntaxError")
                continue
            vis_synth_str = set([repr(e) for e in vis_synth])

            # print(vis_gt_str)
            # print(vis_synth_str)
            if any([set([repr(e) for e in vis_gt]) == vis_synth_str for vis_gt in vis_gts]):
                equiv_list.append(True)
            else:
                equiv_list.append(False)
        return equiv_list

    else:
        vis_synth = VisDesign.load_from_vegalite(vlspec_synth, data).eval()

        vis_synth_str = set([repr(e) for e in vis_synth])

        # print(vis_gt_str)
        # print(vis_synth_str)

        return any([set([repr(e) for e in vis_gt]) == vis_synth_str  for vis_gt in vis_gts])


def weighted_size_of_vis(vl: dict) -> int:
    num_encoding_score = 0
    deatil_score = 0
    for e in vl["encoding"].values():
        num_encoding_score += 10
        if "aggregate" in e:
            deatil_score += 1
        if "bin" in e:
            deatil_score += 1
        if "axis" in e:
            deatil_score += 1

    return num_encoding_score + deatil_score


def parse_keyword_based_synthesized_spec(spec: str) -> dict:
    parsed_spec = {}

    to_be_parse_intent_type = list(CLASSES.keys()) + ['field']

    spec_split = spec.split(', ')

    for sp in spec_split:
        sp = sp[1:-1]
        if 'transform' in sp:
            continue

        intent_type = sp[:(sp.find('('))]
        field = sp[(sp.find('(') + 1) : sp.find(')')]

        if intent_type not in to_be_parse_intent_type:
            continue

        if intent_type == 'field':
            if 'field' not in parsed_spec:
                parsed_spec['field'] = []
            parsed_spec['field'].append(field)
        elif intent_type == 'plot':
            parsed_spec['plot'] = (field, 'null')
        else:
            if '*' in field:
                if intent_type in parsed_spec:
                    pass
                else:
                    parsed_spec[intent_type] = (intent_type, 'null')
            else:
                parsed_spec[intent_type] = (intent_type, field)

    for intent_type in to_be_parse_intent_type:
        if intent_type not in parsed_spec:
            parsed_spec[intent_type] = ('null', 'null')

    return parsed_spec
