from typing import Tuple, List

import torch

from lib.config.config import CACHED_COLUMN_DIR
from lib.nlp.sql_bridge import RunNL2SQL

from lib.eval.benchmark import Benchmark
from lib.utils.nlp_utils import get_tokenized_str
from lib.utils.eval_utils import parse_sql_fields, parse_sql_predicates
from lib.utils.misc_utils import printd
from lib.utils.nl2sql_parser_utils import filter_invalid_pred


class QueryParser:
    def __init__(self):
        self.nl2sql_parser = RunNL2SQL() if torch.cuda.is_available() else None
        self.nl2sql_topk = 1
        # TODO: need to redo the caching here
        self.colname_cache = {"fields": {}, "predicates": {}}
        self.colname_cache_path = '{}/cache.pkl'.format(CACHED_COLUMN_DIR)

    def load_cache(self):
        raise NotImplementedError

    def save_cache(self):
        raise NotImplementedError

    def parse(self, benchmark: Benchmark, field_only=False):
        raise NotImplementedError

    def parse_nl2sql_result(self, benchmark: Benchmark, sql_queries: list, dataname: str) -> Tuple[list, list]:

        if sql_queries is None:
            return [], []
        if sql_queries[0] is None:
            return [], []
        if self.nl2sql_topk == 1:
            query_fields = parse_sql_fields(sql_queries[0], dataname)
            # query_fields_formatted = ["^".join(['field({})'.format(field) for field in query_fields])]
            query_fields_formatted = ['field({})'.format(field) for field in query_fields]
            predicates = filter_invalid_pred(benchmark, parse_sql_predicates(sql_queries[0], dataname))
            predicates_formatted = ["transform({})".format(str(predicates))] if predicates is not None else []
            return query_fields_formatted, predicates_formatted
        else:
            all_possible_query_fields_comb = []

            # NOTE: for predicate let's just extract one because I don't want to deal with the engineering
            predicates = filter_invalid_pred(benchmark, parse_sql_predicates(sql_queries[1][0][1], dataname))
            predicates_formatted = ["transform({})".format(str(predicates))] if predicates is not None else []

            for i in range(self.nl2sql_topk):
                if i >= len(sql_queries[1][0]):
                    break
                query_fields = parse_sql_fields(sql_queries[1][0][i], dataname)
                # all_possible_query_fields_comb.append("^".join(['field({})'.format(field) for field in query_fields]))
                all_possible_query_fields_comb.extend(['field({})'.format(field) for field in query_fields])

            return all_possible_query_fields_comb, predicates_formatted

    def parse_colnames(self, b: Benchmark) -> Tuple[List, List]:
        def find_colname_exact_match(nl, colnames) -> List:

            exact_match_coln = []
            nl = nl.lower()
            colnames = sorted(colnames, key=str.lower, reverse=True)
            for coln in colnames:
                if coln.lower() in nl:
                    exact_match_coln.append(coln.replace(" ", "_"))
                    nl = nl.replace(coln.lower(), "")
                if '_' in coln:
                    coln_space = coln.replace('_', ' ')
                    if coln_space.lower() in nl:
                        exact_match_coln.append(coln.replace(' ', '_'))
                        nl = nl.replace(coln_space.lower(), "")

            return exact_match_coln

        if b.get_id() in self.colname_cache['fields']:
            return self.colname_cache['fields'][b.get_id()], self.colname_cache['predicates'][b.get_id()]

        query = get_tokenized_str(b.nl)

        if self.nl2sql_parser is not None and b.dataname.lower() != 'superstore':
            bench_set = b.benchmark_set if b.dataname.lower() == 'movies' else None
            _, queries, confusion_span, replacement_span = self.nl2sql_parser.run_nl2sql(b.dataname, query, bench_set)
            print("queries: ", queries)
            print("confusion_span: ", confusion_span)
            print("replacement_span: ", replacement_span)

            nl2sql_parsed_res = self.parse_nl2sql_result(b, queries, b.dataname)
        else:
            nl2sql_parsed_res = ([], [])
        # print("nl2sql_parsed_res:", nl2sql_parsed_res)

        exact_match_field_name = ['field({})'.format(field) for field in find_colname_exact_match(b.nl, b.data.colnames)]
        # print("exact_match_field_name:", exact_match_field_name)

        # we need to filter out non-real-column name from nl2sql_parsed_res[0]
        filtered_nl2sql_parsed_res = []
        assert len(b.data.colnames) > 0
        for colname in nl2sql_parsed_res[0]:
            if colname[6:-1] in b.data.colnames:
                filtered_nl2sql_parsed_res.append(colname)
        # print("filtered_nl2sql_parsed_res:", filtered_nl2sql_parsed_res)

        colnames = list(set(exact_match_field_name + filtered_nl2sql_parsed_res))
        predicates = nl2sql_parsed_res[1]

        self.colname_cache['fields'][b.get_id()] = colnames
        self.colname_cache['predicates'][b.get_id()] = predicates
        return colnames, predicates
