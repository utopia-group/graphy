import json
import os
import shutil
import sys


def extract_plot_semantics(obj):
    binning, aggregate, filtering = [], [], []
    semantic_id = ''

    encoding = obj['encoding']
    encoding_ids = []
    for enc_type in encoding.keys():
        encoding_id = ''
        field = encoding[enc_type]
        field_name = field['field'] if 'field' in field else ''
        encoding_id += (field_name + '_')
        if 'bin' in field and field['bin']:
            binning.append(field['field'])
            encoding_id += 'binned_'
        if 'aggregate' in field:
            if 'field' in field:
                aggregate.append((field['aggregate'], field['field']))
            encoding_id += 'agg_{}_'.format(field['aggregate'])
        if encoding_id.endswith('_'):
            encoding_id = encoding_id[:-1]
        encoding_ids.append(encoding_id)
    semantic_id += ('_'.join(sorted(encoding_ids)) + '_')

    if 'transform' in obj:
        for transform in obj['transform']:
            if 'filter' in transform:
                filtering.append(transform['filter'])
                field_name, op, val = transform['filter'].split()
                semantic_id += 'filter_{}_{}'.format(field_name, op)
    return semantic_id


def extract_plot_semantics_with_types(obj):
    binning, aggregate, filtering = [], [], []
    semantic_id = ''

    encoding = obj['encoding']
    encoding_ids = []
    for enc_type in encoding.keys():
        encoding_id = ''
        field = encoding[enc_type]
        field_type = field['type'] if 'type' in field else ''
        encoding_id += (field_type + '_')
        if 'bin' in field and field['bin']:
            binning.append(field_type)
            encoding_id += 'binned_'
        if 'aggregate' in field:
            if 'type' in field:
                aggregate.append((field['aggregate'], field['type']))
            encoding_id += 'agg_{}_'.format(field['aggregate'])
        if encoding_id.endswith('_'):
            encoding_id = encoding_id[:-1]
        encoding_ids.append(encoding_id)
    semantic_id += ('_'.join(sorted(encoding_ids)) + '_')

    if 'transform' in obj:
        for transform in obj['transform']:
            if 'filter' in transform:
                filtering.append(transform['filter'])
                field_name, op, val = transform['filter'].split()
                semantic_id += 'filter_{}_{}'.format(field_name, op)
    return semantic_id


def run():
    in_dir = sys.argv[1]
    for dataset in os.listdir(in_dir):
        subdir = os.path.join(in_dir, dataset)
        if os.path.isdir(subdir) and dataset not in ['__stats', '.']:
            for field_comb in os.listdir(subdir):
                plot_dir = os.path.join(subdir, field_comb)
                print('processing {}'.format(plot_dir))
                if os.path.isdir(plot_dir) and field_comb not in ['__stats', '.']:
                    for in_json in os.listdir(plot_dir):
                        if in_json.endswith('.vl.json'):
                            in_json = os.path.join(plot_dir, in_json)
                            out_json = in_json + '.cp'
                            with open(in_json) as f:
                                plot_obj = json.load(f)
                                new_plot_obj = extract_plot_semantics(plot_obj)
                            with open(out_json, 'w') as o_f:
                                json.dump(new_plot_obj, o_f)
                            shutil.move(out_json, in_json)


if __name__ == '__main__':
    run()
