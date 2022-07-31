"""
utils for parsing vegalite
"""
from typing import List, Dict, Union


def parse_filter_condition(fc: Union[dict, str]) -> str:

    if isinstance(fc, str):
        if '"' in fc:
            split_fc = fc.split('"')
            if len(split_fc) < 3:
                pass
            else:
                field = split_fc[1]
                rest_fc = split_fc[2]

                if rest_fc[0] == ']':
                    rest_fc = rest_fc[1:]
                if rest_fc[0] == ')':
                    rest_fc = rest_fc[1:]

                return "{}{}".format(field, rest_fc)

        print("WARNING: {} cannot handle".format(fc))
        assert False

    field = fc['field']

    if 'range' in fc:
        return '{} BETWEEN {} AND {}'.format(field, fc['range'][0], fc['range'][1])

    if 'oneOf' in fc:
        oneOf_conditions = []
        for ins in fc['oneOf']:
            oneOf_conditions.append('{} = {}'.format(field, ins))
        return ' OR '.join(oneOf_conditions)

    if 'gte' in fc:
        return '{} >= {}'.format(field, fc['gte'])

    if 'lte' in fc:
        return '{} <= {}'.format(field, fc['lte'])

    if 'gt' in fc:
        return '{} > {}'.format(field, fc['gt'])

    if 'lt' in fc:
        return '{} < {}'.format(field, fc['lt'])

    if 'equal' in fc:
        return '{} = {}'.format(field, fc['equal'])

    pass


def parse_vl_spec(vl_spec: Dict) -> List:
    parsed_spec = []

    if 'hconcat' in vl_spec:
        return parsed_spec

    if 'mark' in vl_spec:
        plot_type = vl_spec['mark']['type'] if not vl_spec['mark']['type'] == 'point' else 'scatter'
        parsed_spec.append('plot({})'.format(plot_type))

    if 'encoding' in vl_spec:
        encodings: Dict = vl_spec['encoding']
        for key, enc in encodings.items():

            if key == 'tooltip':
                continue

            field = enc['field']

            parsed_spec.append('field({})'.format(field))
            parsed_spec.append('{}({})'.format(key, field))

            if 'aggregate' in enc:
                if enc['aggregate'] is None:
                    pass
                else:
                    parsed_spec.append('{}({})'.format(enc['aggregate'], field))

    if 'transform' in vl_spec:
        if vl_spec['transform'] is None or len(vl_spec['transform']) == 0:
            pass
        else:
            transforms: List[Dict] = vl_spec['transform']
            for t in transforms:
                for key, value in t.items():
                    if not key == 'filter' and not key == 'aggregate':
                        print('WARNING: {} cannot handle'.format(key))

                    if key == 'filter':
                        parsed_condition = parse_filter_condition(value)
                        parsed_spec.append('filter({})'.format(parsed_condition))

                    if key == 'aggregate':
                        parsed_spec.append('{}({})'.format(value[0]['op'], value[0]['field']))

    return list(set(parsed_spec))
