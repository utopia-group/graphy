import pandas as pd
from pprint import pprint
import numpy as np
import copy

from lib.dracopy.draco.run import DEBUG


BANNED_FIELD_TYPES = {
    'COORDINATE',
    'GEO_LATITUDE',
    'GEO_LONGITUDE',
    'STRING',
    'ORDINAL_INDEX'
}


BANNED_TUPLE_FIELD_TYPES = {
}


BONUS_TUPLE_FIELD_TYPES = {
    frozenset({'COUNT', 'GEO_CITY'}): 3.0,
    frozenset({'COUNT', 'GEO_STATE'}): 3.0,
    frozenset({'COUNT', 'GEO_COUNTRY'}): 3.0,
    frozenset({'COUNT', 'TEMPORAL_YEAR'}): 3.0,
    frozenset({'COUNT', 'TEMPORAL_MONTH'}): 3.0,
    frozenset({'COUNT', 'TEMPORAL'}): 3.0,
    frozenset({'QUANT_MONEY', 'GEO_CITY'}): 3.0,
    frozenset({'QUANT_MONEY', 'GEO_STATE'}): 3.0,
    frozenset({'QUANT_MONEY', 'GEO_COUNTRY'}): 3.0,
    frozenset({'QUANT_MONEY', 'TEMPORAL_YEAR'}): 3.0,
    frozenset({'QUANT_MONEY', 'TEMPORAL_MONTH'}): 3.0,
    frozenset({'QUANT_MONEY', 'TEMPORAL'}): 3.0,
    frozenset({'QUANT_WITH_UNIT', 'GEO_CITY'}): 3.0,
    frozenset({'QUANT_WITH_UNIT', 'GEO_STATE'}): 3.0,
    frozenset({'QUANT_WITH_UNIT', 'GEO_COUNTRY'}): 3.0,
    frozenset({'QUANT_WITH_UNIT', 'TEMPORAL_YEAR'}): 3.0,
    frozenset({'QUANT_WITH_UNIT', 'TEMPORAL_MONTH'}): 3.0,
    frozenset({'QUANT_WITH_UNIT', 'TEMPORAL'}): 3.0,
    frozenset({'RATIO', 'GEO_CITY'}): 3.0,
    frozenset({'RATIO', 'GEO_STATE'}): 3.0,
    frozenset({'RATIO', 'GEO_COUNTRY'}): 3.0,
    frozenset({'RATIO', 'TEMPORAL_YEAR'}): 3.0,
    frozenset({'RATIO', 'TEMPORAL_MONTH'}): 3.0,
    frozenset({'RATIO', 'TEMPORAL'}): 3.0,
    frozenset({'RATIO_PERCENT', 'GEO_CITY'}): 3.0,
    frozenset({'RATIO_PERCENT', 'GEO_STATE'}): 3.0,
    frozenset({'RATIO_PERCENT', 'GEO_COUNTRY'}): 3.0,
    frozenset({'RATIO_PERCENT', 'TEMPORAL_YEAR'}): 3.0,
    frozenset({'RATIO_PERCENT', 'TEMPORAL_MONTH'}): 3.0,
    frozenset({'RATIO_PERCENT', 'TEMPORAL'}): 3.0,
    frozenset({'RAW_NUMBER', 'GEO_CITY'}): 3.0,
    frozenset({'RAW_NUMBER', 'GEO_STATE'}): 3.0,
    frozenset({'RAW_NUMBER', 'GEO_COUNTRY'}): 3.0,
    frozenset({'RAW_NUMBER', 'TEMPORAL_YEAR'}): 3.0,
    frozenset({'RAW_NUMBER', 'TEMPORAL_MONTH'}): 3.0,
    frozenset({'RAW_NUMBER', 'TEMPORAL'}): 3.0,
    frozenset({'RAW_INTEGER', 'GEO_CITY'}): 2.0,
    frozenset({'RAW_INTEGER', 'GEO_STATE'}): 2.0,
    frozenset({'RAW_INTEGER', 'GEO_COUNTRY'}): 2.0,
    frozenset({'RAW_INTEGER', 'TEMPORAL_YEAR'}): 2.0,
    frozenset({'RAW_INTEGER', 'TEMPORAL_MONTH'}): 2.0,
    frozenset({'RAW_INTEGER', 'TEMPORAL'}): 2.0,
    frozenset({'COUNT', 'CATEGORICAL'}): 2.0,
    frozenset({'QUANT_MONEY', 'CATEGORICAL'}): 2.0,
    frozenset({'QUANT_WITH_UNIT', 'CATEGORICAL'}): 2.0,
    frozenset({'RATIO', 'CATEGORICAL'}): 2.0,
    frozenset({'RATIO_PERCENT', 'CATEGORICAL'}): 2.0,
    frozenset({'RAW_NUMBER', 'CATEGORICAL'}): 2.0,
    frozenset({'RAW_INTEGER', 'CATEGORICAL'}): 1.0,
    frozenset({'COUNT', 'NOMINAL'}): 2.0,
    frozenset({'QUANT_MONEY', 'NOMINAL'}): 2.0,
    frozenset({'QUANT_WITH_UNIT', 'NOMINAL'}): 2.0,
    frozenset({'RATIO', 'NOMINAL'}): 2.0,
    frozenset({'RATIO_PERCENT', 'NOMINAL'}): 2.0,
    frozenset({'RAW_NUMBER', 'NOMINAL'}): 2.0,
    frozenset({'RAW_INTEGER', 'NOMINAL'}): 1.0,
    frozenset({'COUNT', 'ORDINAL'}): 2.0,
    frozenset({'QUANT_MONEY', 'ORDINAL'}): 2.0,
    frozenset({'QUANT_WITH_UNIT', 'ORDINAL'}): 2.0,
    frozenset({'RATIO', 'ORDINAL'}): 2.0,
    frozenset({'RATIO_PERCENT', 'ORDINAL'}): 2.0,
    frozenset({'RAW_NUMBER', 'ORDINAL'}): 2.0,
    frozenset({'RAW_INTEGER', 'ORDINAL'}): 1.0,
    frozenset({'ORDINAL', 'GEO_CITY'}): 1.0,
    frozenset({'ORDINAL', 'GEO_STATE'}): 1.0,
    frozenset({'ORDINAL', 'GEO_COUNTRY'}): 1.0,
    frozenset({'ORDINAL', 'TEMPORAL_YEAR'}): 1.0,
    frozenset({'ORDINAL', 'TEMPORAL_MONTH'}): 1.0,
    frozenset({'ORDINAL', 'TEMPORAL'}): 1.0,
    frozenset({'ORDINAL', 'CATEGORICAL'}): 1.0,
    frozenset({'ORDINAL', 'NOMINAL'}): 0.5,
}


BANNED_TRIPLE_FIELD_TYPES = {
    frozenset({'NOMINAL', 'NOMINAL', 'NOMINAL'})
}


def select_fields_of_interests(columns, fine_types):
    assert(len(columns) == len(fine_types))
    """Given a list of columns, decides which fields can be interesting for visualization """
    return zip(*[(f, t) for f, t in zip(columns, fine_types) if t not in BANNED_FIELD_TYPES])


POSITIONAL_CHANNELS = ["x", "y", "column", "row"]
NON_POSITIONAL_CHANNELS = ["size", "color"]


def gen_visual_trace_with_mult(_data, _fields):
    """given a dataset and a few fields, returns the projection of the table on the fields
        the result is represented as a map that maps each tuple to its multiplicity
    """
    visual_trace = [tuple([t[f] for f in _fields]) for t in _data]
    count_map = {}
    for tr in visual_trace:
        if tr not in count_map:
            count_map[tr] = 0
        count_map[tr] += 1
    return count_map


def enum_filters(df, enc_types, limit=2):
    """enrich dataset by adding filters """
    candidates = []
    for i, c in enumerate(df.columns):
        enc_ty = enc_types[i]
        if enc_ty == "nominal":
            v_candidates = np.random.choice(list(set(df[c])), 5)
            for vc in v_candidates:
                filtered_df = df[df[c] == vc]
                expr = {"col": c, "op": "==", "val": vc}
                if filtered_df.shape[0] == 1  or filtered_df.shape[0] == df.shape[0] - 1:
                    # ignore filters such that the filtered output size is 1
                    continue
                candidates.append((filtered_df, expr))
        if enc_ty == "quantitative":
            v_candidates = np.random.choice(list(set(df[c])), 5)
            for vc in v_candidates:
                for op in [">=", "<="]:
                    filtered_df = df[df[c] >= vc] if op == ">=" else df[df[c] <= vc]
                    expr = {"col": c, "op": op, "val": vc}
                    if filtered_df.shape[0] == 1 or filtered_df.shape[0] == df.shape[0] - 1:
                        continue
                    candidates.append((filtered_df, expr))
        if len(candidates) >= limit:
            break
    return candidates

def add_sorting(spec):
    """enrich sorting """
    # add sorting on x-axis values
    if "x" not in spec["encoding"]:
        return [spec]
    
    params = []
    if spec["encoding"]["x"]["type"] in ["quantitative", "ordinal"]:
        params += ["ascending", "descending"]

    if (spec["encoding"]["x"]["type"] == "nominal" 
        and "y" in spec["encoding"] 
        and spec["encoding"]["y"]["type"] in ["quantitative", "ordinal"]):
        params += [{"encoding": "y", "order": "ascending"}, {"encoding": "y", "order": "descending"}]

    result = []
    for sorting_param in params:
        temp_spec = copy.deepcopy(spec)
        temp_spec["encoding"]["x"]["sort"] = sorting_param
        result.append(temp_spec)

    return result


def process_final_data(spec, raw_data):
    """Given the raw data and the spec, extract the pre visualization data"""
    inv_map = {}
    for ch in spec["encoding"]:
        enc = spec["encoding"][ch]
        field_name = enc["field"] if "field" in enc else None
        aggregate = enc["aggregate"] if "aggregate" in enc else None
        bin_size = enc["bin"] if "bin" in enc else None
        if field_name is not None:
            inv_map[field_name] = (ch, field_name, aggregate, bin_size)
        else:
            assert (aggregate == "count")
            inv_map["__count__"] = (ch, field_name, aggregate, bin_size)

    df = pd.DataFrame.from_dict(raw_data)
    # TODO: @clwang to confirm the fix is right
    df = df[[inv_map[f][1] for f in inv_map if f != "__count__"]]

    # note this binning is different from vega lite binning
    for f in inv_map:
        if inv_map[f][3] == True:
            df[f] = pd.cut(df[f], bins=10 if inv_map[f][0] in ["x", "y", "row", "column"] else 6)

    # there is something to aggregate
    if any([inv_map[f][2] != None for f in inv_map]):
        gb_fields = [inv_map[f][1] for f in inv_map if inv_map[f][2] is None]
        aggregates = {f if f != "__count__" else gb_fields[0] : inv_map[f][2] for f in inv_map if inv_map[f][2] != None}

        if gb_fields != []:
            df = df.groupby(by=gb_fields)
            df = df.agg(aggregates)
            df = df.rename(columns={gb_fields[0]: "__count__"}, level=0).reset_index()
        else:
            # in this branch, there is no field to perform group by
            # this could not be count, since count requires another field
            series = df.agg(aggregates)
            df = series.to_frame()
            df = df.rename(columns = {0: series.index[0]})

    return df, inv_map


def filter_broken_chart_and_process(spec, raw_data):

    pre_vis_df, inv_map = process_final_data(spec, raw_data)
    pre_vis_data = pre_vis_df.to_dict(orient="records")

    if spec["mark"] in ["line", "area"]:
        if is_broken_line_area_charts(pre_vis_data, inv_map):
            if DEBUG:
                print(" [Remove broken chart] Line / area chart contains multiple y values with the same x value.")
            return None

    # checks if there exists a field with cardinality 1
    for f in inv_map:
        # if f == "internationalaffiliation4":
        #   print(pre_vis_df[f])
        #   print(pd.Series.nunique(pre_vis_df[f]))
        #   sys.exit(-1)
        cardinality = pd.Series.nunique(pre_vis_df[f])
        if cardinality == 1 or cardinality == 0:
            if DEBUG:
                print(" [Remove broken chart] Cardinality of field {} = 1.".format(f))
            return None

    # print("#####")
    # print(inv_map)
    # print(spec["encoding"])

    # process the input data to handle zero
    for f in inv_map:
        ch = inv_map[f][0]
        if spec["encoding"][ch]["type"] == "quantitative" and ("bin" not in spec["encoding"][ch]):
            col_data = pre_vis_df[f]

            # remove zero is the distance is too big
            if ((col_data.max() - col_data.min()) * 5 < col_data.min() - 0 and col_data.min() > 10):
                spec["encoding"][ch]["scale"] = { "zero": False }
            else:
                if "scale" in spec["encoding"][ch]:
                    del spec["encoding"][ch]["scale"]

    return spec


def is_broken_line_area_charts(pre_vis_data, inv_map):
    """Given the spec and its corresponding data, filter undesirable data."""

    non_pos_cols = [f for f in inv_map if inv_map[f][0] not in ["x", "y"]]
    x_col, y_col = [f for f in inv_map if inv_map[f][0] == "x"][0], [f for f in inv_map if inv_map[f][0] == "y"][0]
    
    partitions = {}
    for r in pre_vis_data:
        key = tuple([r[f] for f in non_pos_cols])
        if key not in partitions:
            partitions[key] = {}
        if r[x_col] not in partitions[key]:
            partitions[key][r[x_col]] = []
        partitions[key][r[x_col]].append(r[y_col])

    if any([any([len(ys) > 2 for x, ys in p.items()]) for k1, p in partitions.items()]):
        # contain points that has the same x index
        
        # full_spec = copy.deepcopy(spec)
        # full_spec["data"] = {"values": raw_data}
        # print(full_spec)
        return True

    return False


def post_process_vl_spec(spec, data, try_add_new_encoding_when_overlapping=False, try_fix_overlapping=False):
    """Given a vega-lite spec, analyze it and process it"""
    if not spec["mark"] in ["point", "rect", "circle"]:
        return spec

    # extend one field or add an aggregation 
    # if there exists positional overlapping for discrete or low cardinality fields
    contains_bin_in_pos_channel = False

    positional_channel_all_discrete = True
    for ch in spec["encoding"]:
        enc = spec["encoding"][ch]
        if ch in POSITIONAL_CHANNELS and enc["type"] in ["quantitative", "temporal"]:
            enc = spec["encoding"][ch]
            if "aggregate" in enc:
                # positional is not discrete if aggregation is used
                positional_channel_all_discrete = False
                break

            if "bin" in enc and enc["bin"]:
                # it is discrete if binned
                contains_bin_in_pos_channel = True
                continue
            else:
                cardinality = len(set([t[enc["field"]] for t in data]))
                if cardinality > 20:
                    positional_channel_all_discrete = False
                    break

    if positional_channel_all_discrete:

        existing_pos_fields = [spec["encoding"][ch]["field"] 
                                for ch in spec["encoding"] 
                                    if ch in POSITIONAL_CHANNELS and "field" in spec["encoding"][ch]]

        count_map = gen_visual_trace_with_mult(data, existing_pos_fields)
        exists_duplicates = True if contains_bin_in_pos_channel or any([count_map[x] > 1 for x in count_map]) else False        
        
        if not contains_bin_in_pos_channel:
            avg_duplicate_ratio = np.average([count_map[x] for x in count_map])

            if avg_duplicate_ratio < 2:
                # duplicate ratio is pretty low, ignore it
                return spec

            if try_add_new_encoding_when_overlapping:

                # contains unhandeled bin or contains high overlapping
                # try to add one additional field to resolve overlaps 
                potential_fields = [f for f in data[0] if len(set([t[f] for t in data])) <= 20]

                candidate_field = None
                current_duplicate_ratio = avg_duplicate_ratio
                for f in potential_fields:
                    count_map = gen_visual_trace_with_mult(data, existing_pos_fields + [f])
                    rt = np.average([count_map[x] for x in count_map])
                    if rt < current_duplicate_ratio:
                        candidate_field = f
                        current_duplicate_ratio = rt

                if current_duplicate_ratio < 1.2:
                    if DEBUG:
                        print("[post_process] >> add new channels")
                        print("  {} -> {}".format(avg_duplicate_ratio, current_duplicate_ratio))
                    # we can add this field to the 
                    candidate_channels = [ch for ch in POSITIONAL_CHANNELS if ch not in spec["encoding"]]
                    if len(candidate_channels) > 0:
                        spec["encoding"][candidate_channels[0]] = {"type": "nominal", "field": f}
                        return spec

        non_pos_channels = [ch for ch in spec["encoding"] 
                                if ch in NON_POSITIONAL_CHANNELS 
                                    and spec["encoding"][ch]["type"] == "quantitative"]
        
        if len(non_pos_channels) == 0:
            # use the first channel not used by the current chart
            if try_fix_overlapping:
                if DEBUG:
                    print("[post_process] >> add count")
                candidate_channels = [ch for ch in NON_POSITIONAL_CHANNELS if ch not in spec["encoding"]]
                spec["encoding"][candidate_channels[0]] = {"type": "quantitative", "aggregate": "count"}
            else:
                return None
        else:
            if try_fix_overlapping:
                fixed = False
                for ch in non_pos_channels:
                    if "aggregate" not in spec["encoding"][ch] and "bin" not in spec["encoding"][ch]:
                        if DEBUG:
                            print("[post_process] >> update aggregate")
                        spec["encoding"][ch]["aggregate"] = "average"
                        fixed = True
                if fixed == False:
                    return None
            else:
                return None

    return spec
