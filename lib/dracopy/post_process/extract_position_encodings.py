import copy
import json


def adjust_position_encodings(obj):
    position_prior = {
        'nominal': 5,
        'ordinal': 4,
        'temporal': 3,
        'quantitative': 2
    }

    encoding = obj['encoding']
    if 'x' in encoding and 'y' in encoding:
        x_enc, y_enc = encoding['x'], encoding['y']
        x_field_type = x_enc['type']
        y_field_type = y_enc['type']
        # prefer binning on x-axis
        binned_x = 10 if ('bin' in x_enc and x_enc['bin']) else 1
        binned_y = 10 if ('bin' in y_enc and y_enc['bin']) else 1
        # prefer aggregating on y-axis
        agg_x = 0.1 if 'aggregate' in x_enc else 1
        agg_y = 0.1 if 'aggregate' in y_enc else 1
        x_prior = position_prior[x_field_type] * binned_x * agg_x
        y_prior = position_prior[y_field_type] * binned_y * agg_y
        if x_prior < y_prior:
            # reverse x and y field
            temp = encoding['y']
            encoding['y'] = encoding['x']
            encoding['x'] = temp
        elif x_prior == y_prior:
            # sort x and y field in alphabetical order
            x_field = x_enc['field']
            y_field = y_enc['field']
            if x_field >= y_field:
                temp = encoding['y']
                encoding['y'] = encoding['x']
                encoding['x'] = temp


def extract_position_encodings(obj):
    sig = copy.deepcopy(obj)
    encoding = sig['encoding']
    if 'x' in encoding:
        encoding['x'] = [encoding['x']]
    if 'y' in encoding:
        encoding['x'].append(encoding['y'])
        del encoding['y']
    encoding['x'] = sorted([json.dumps(e) for e in encoding['x']])
    return json.dumps(sig)