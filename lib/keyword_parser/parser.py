import os
import pickle
import re
from typing import List

from lib.eval.benchmark import Benchmark
from lib.utils.nlp_utils import get_tokenized_str
from lib.keyword_parser.lexicons import Lexicons, NL4DVLexicons
from lib.keyword_parser.derivations import Derivations, NL4DVDerivations
from lib.keyword_parser.context import DerivTriggerContext, ParseContext
from lib.query_parser import QueryParser
from lib.utils.misc_utils import printd


class SemanticParser:
    def __init__(self, lexicons: Lexicons, derivations: Derivations) -> None:
        self.lexicons = lexicons
        self.derivations = derivations
    
    def parse(self, query, colnames) -> List:
        raise NotImplementedError


class NL4DVParser(SemanticParser):
    def __init__(self, lexicons: Lexicons, derivations: Derivations) -> None:
        super(NL4DVParser, self).__init__(lexicons, derivations)

    def parse(self, query, colnames) -> List:

        query = get_tokenized_str(query)

        query_context = ParseContext(query)

        parsed_res = []

        for lexicon_type, lexicons in self.lexicons.rules.items():
            for triggered_text, deriv_rules in lexicons:
                triggered = True
                if type(triggered_text) == str:
                    if triggered_text not in query:
                        triggered = False
                else:
                    # assume it's regex since it is ugly to type check
                    if len(re.findall(triggered_text[0], query)) == 0:
                        triggered = False
                    else:
                        triggered_text = triggered_text[1]

                if not triggered:
                    continue

                # complex_rules_applied = False
                for rule in deriv_rules:
                    # if rule == deriv_rules[-1] and complex_rules_applied:
                    #     continue
                    deriv_res = self.derivations.run_deriv(lexicon_type, rule, DerivTriggerContext(query_context, triggered_text, deriv_rules[-1]), colnames)
                    if not deriv_res == '':
                        parsed_res.append(deriv_res)
                        # complex_rules_applied = True

        return parsed_res


class QueryKeywordParser(QueryParser):
    def __init__(self, lexicons: Lexicons, derivations: Derivations):
        super().__init__()
        if isinstance(lexicons, NL4DVLexicons) and isinstance(derivations, NL4DVDerivations):
            self.semantic_parser: SemanticParser = NL4DVParser(lexicons, derivations)
        else:
            raise NotImplementedError("lexicon and derivation types not supported")

    def load_cache(self):
        if os.path.isfile(self.colname_cache_path):
            with open(self.colname_cache_path, 'rb') as f:
                self.colname_cache = pickle.load(f)

    def save_cache(self):
        with open(self.colname_cache_path, 'wb') as f:
            pickle.dump(self.colname_cache, f)

    def parse(self, b: Benchmark, field_only=False) -> List[str]:

        # clean find lexicon and parse
        query = get_tokenized_str(b.nl)
        print("tokenized_str: ", query)

        colnames, predicates = self.parse_colnames(b)

        if not field_only:
            semantic_parsed_res = self.semantic_parser.parse(query, b.data.colnames)
            return ['data(\"{}\")'.format(b.data.data_path)] + list(set(colnames + predicates + semantic_parsed_res))
        else:
            return colnames