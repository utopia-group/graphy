import re
from io import UnsupportedOperation
from typing import List, Dict

tregex1 = re.compile(r"lower[(]datum[[]\"(.+)\"[]][)] (<=|>=|<|>|!=|==) (.+)")


def check_spec_equiv(gt_spec: Dict, synth_specs: List[Dict]) -> List[bool]:
    if synth_specs is None:
        return []

    if all(vlspec_synth_i is None for vlspec_synth_i in synth_specs):
        return [False for _ in synth_specs]

    # The following is a quick way to check equivalence in the chi21 dataset
    check = []
    for vlspec_synth_i in synth_specs:
        # mark:
        if vlspec_synth_i['mark']['type'] != gt_spec['mark']['type']:
            check.append(False)
            continue

        if vlspec_synth_i['encoding']['x']['field'] != gt_spec['encoding']['x']['field']:
            check.append(False)
            continue

        if vlspec_synth_i['encoding']['y']['field'] != gt_spec['encoding']['y']['field']:
            check.append(False)
            continue

        if 'aggregate' in gt_spec['encoding']['y']:
            if 'aggregate' not in vlspec_synth_i['encoding']['y']:
                check.append(False)
                continue
            else:
                if vlspec_synth_i['encoding']['y']['aggregate'] != gt_spec['encoding']['y']['aggregate']:
                    check.append(False)
                    continue
        else:
            if 'aggregate' in vlspec_synth_i['encoding']['y']:
                check.append(False)
                continue

        if vlspec_synth_i['encoding'].get('color') is not None and vlspec_synth_i['encoding'].get('column') is not None \
                and vlspec_synth_i['encoding']['color'].get('field') is not None:
            if vlspec_synth_i['encoding']['color']['field'] == vlspec_synth_i['encoding']['column']['field']:
                del vlspec_synth_i['encoding']['color']

        if 'color' in gt_spec['encoding']:
            if 'color' not in vlspec_synth_i['encoding']:
                check.append(False)
                continue
            else:
                if vlspec_synth_i['encoding']['color']['field'] != gt_spec['encoding']['color']['field']:
                    check.append(False)
                    continue
        else:
            if 'color' in vlspec_synth_i['encoding']:
                check.append(False)
                continue

        if 'column' in gt_spec['encoding']:
            if 'column' not in vlspec_synth_i['encoding']:
                check.append(False)
                continue
            else:
                if vlspec_synth_i['encoding']['column']['field'] != gt_spec['encoding']['column']['field']:
                    check.append(False)
                    continue
        else:
            if 'column' in vlspec_synth_i['encoding']:
                check.append(False)
                continue

        check.append(True)

    return check


def extract_plot_type_from_gt_name(gt_name):
    gt_name = gt_name.lower()
    if 'histogram' in gt_name or 'attrbar' in gt_name:
        return 'Histogram'
    elif 'bar' in gt_name:
        return 'Bar'
    elif 'line' in gt_name:
        return 'Line'
    elif 'scatter' in gt_name:
        return 'Scatter'
    elif 'area' in gt_name:
        return 'Area'
    else:
        raise ValueError('Unknown plot type: {}'.format(gt_name))
