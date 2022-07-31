import re
import inflect
from typing import Callable, Dict
from spacy.matcher import DependencyMatcher, Matcher
from lib.keyword_parser.context import DerivTriggerContext, get_dependency_matcher, get_matcher

inflect = inflect.engine()


class Derivations:
    def __init__(self) -> None:
        self.rules: Dict[str, Dict[str, Callable]] = {}
    
    def init(self):
        raise NotImplementedError
    
    def run_deriv(self, lexicon_type: str, deriv_rule: str, x: DerivTriggerContext, colnames: list):
        raise NotImplementedError


def field_name_dep_match_helper(x: DerivTriggerContext, matcher: DependencyMatcher) -> str:

    matches = matcher(x.parse_context.spacy_context)
        
    if len(matches) == 0:
        return ""
    
    # TODO: currently only take one res, generalize to a list later
    _, token_ids = matches[0]
    parse_res = {}
    for i in range(len(token_ids)):
        parse_res['field_name'] = x.parse_context.spacy_context[token_ids[i]].text
    
    if not 'field_name' in parse_res:
        return ''

    return parse_res['field_name']


def field_name_match_helper(x: DerivTriggerContext, matcher: Matcher) -> str:
    
    matches = matcher(x.parse_context.spacy_context, as_spans=True)

    if len(matches) ==  0:
        return ''

    # TODO: curently only take the last results (match the most)
    matched_text = matches[-1].text
    
    # remove the keyword
    insensitive_word_replace = re.compile(re.escape(x.triggered_word), re.IGNORECASE)
    field_name = insensitive_word_replace.sub('', matched_text)

    return field_name.strip()


def dep_pattern_generation_helper(triggered_word: str, dep_tag: str, output_name="field_name") -> list:
    pattern = [
        {
            "RIGHT_ID": "founded",
            "RIGHT_ATTRS": {"LOWER": triggered_word}
        },
        {
            "LEFT_ID": "founded",
            "REL_OP": ">>",
            "RIGHT_ID": output_name,
            "RIGHT_ATTRS": {"DEP": dep_tag}
        }
    ]

    return pattern


def find_triggered_word(triggered_text):
    return triggered_text if len(triggered_text.split()) == 1 else triggered_text.split()[0]


def standardize_field_name(field, colnames):

    if field == 'mpg' or field == 'Mpg':
        field = 'MPG'
    elif field.islower():
        field = field[0].upper() + field[1:]
    elif field.isupper():
        field = field[0] + field[1:].lower()

    # ground to column name here
    colname_lower_to_colname_map = {c.lower(): c for c in colnames}
    
    if field.lower() in colname_lower_to_colname_map:
        return field

    elif ' ' in field:
        field_new = field.lower().replace(' ', '_')
        # print("field_new:", field_new)
        if field_new in colname_lower_to_colname_map:
            return colname_lower_to_colname_map[field_new]
        
        # one of the str is in colname
        field_split = field.lower().split(' ')
        # TODO: only take the first one that matches otherwise i don't know how to resolve
        for f_split in field_split:
            if f_split in colname_lower_to_colname_map:
                return colname_lower_to_colname_map[f_split]
    
    else:
        if not inflect.singular_noun(field):
            pass
        else:
            singular_field = inflect.singular_noun(field).lower()
            if singular_field in colname_lower_to_colname_map:
                return colname_lower_to_colname_map[singular_field]

        # str is part of the colname
        for cn in colnames:
            if '_' in cn:
                cn_split = [c.lower() for c in cn.split('_')]
                if field.lower() in cn_split:
                    return cn

    return ""


def task_attr_pobj(x: DerivTriggerContext, colnames: list):

    triggered_word = find_triggered_word(x.triggered_text)
    pattern = dep_pattern_generation_helper(triggered_word, 'pobj')

    matcher = get_dependency_matcher(x.triggered_func, triggered_word, pattern)
    field_name = standardize_field_name(field_name_dep_match_helper(x, matcher), colnames)

    
    return "{}({})".format(x.triggered_func, field_name) if not field_name == "" else ""


def task_attr_dir(x: DerivTriggerContext, colnames: list):
    triggered_word = find_triggered_word(x.triggered_text)
    x.triggered_word = triggered_word

    # TODO: match the first consecutive (ADJ)? NN+ occurence after this word
    pattern = [
        {"LOWER": triggered_word},
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN", "OP": "+"}
    ]
    matcher = get_matcher(x.triggered_func, triggered_word, pattern)
    field_name = standardize_field_name(field_name_match_helper(x, matcher), colnames)

    return "{}({})".format(x.triggered_func, field_name) if not field_name == "" else ""


def channel_attr_dobj(x: DerivTriggerContext, colnames: list):

    triggered_word = find_triggered_word(x.triggered_text)
    pattern = dep_pattern_generation_helper(triggered_word, 'dobj')

    matcher = get_dependency_matcher(x.triggered_func, triggered_word, pattern)
    field_name = standardize_field_name(field_name_dep_match_helper(x, matcher), colnames)
    
    return "{}({})".format(x.triggered_func, field_name) if not field_name == "" else ""


def channel_attr_pobj(x: DerivTriggerContext, colnames: list):
    triggered_word = find_triggered_word(x.triggered_text)
    pattern = dep_pattern_generation_helper(triggered_word, 'pobj')

    matcher = get_dependency_matcher(x.triggered_func, triggered_word, pattern)
    field_name = standardize_field_name(field_name_dep_match_helper(x, matcher), colnames)
    
    return "{}({})".format(x.triggered_func, field_name) if not field_name == "" else ""


class NL4DVDerivations(Derivations):
    def __init__(self) -> None:
        super(NL4DVDerivations, self).__init__()
        self.init()
    
    def init(self):
        self.rules['plot'] = {
            'scatter': lambda x, c: 'plot(scatter)',
            'bar': lambda x, c: 'plot(bar)',
            'strip': lambda x, c: 'plot(strip)',
            'pie': lambda x, c: 'plot(pie)',
            'box': lambda x, c: 'plot(box)',
            'line': lambda x, c: 'plot(line)',
            'area': lambda x, c: 'plot(area)',
            # 'stacked': lambda x: 'plot(bar) ^ plot(stacked)'
            'stacked': lambda x, c: 'plot(bar)'
        }

        self.rules['task'] = {
            'count': lambda x, c: 'count(*)',
            'mean': lambda x, c: 'mean(*)',
            'median': lambda x, c: 'median(*)',
            'sum': lambda x, c: 'sum(*)',
            'trend': lambda x, c: 'trend(*)',
            'task_attr_pobj': lambda x, c: task_attr_pobj(x, c),
            'task_attr_dir': lambda x, c: task_attr_dir(x, c)
        }

        self.rules['channel'] = {
            'color': lambda x, c: 'color(*)',
            'channel': lambda x, c: 'channel(*)',
            'column': lambda x, c: 'column(*)',
            'channel_attr_pobj': lambda x, c: channel_attr_pobj(x, c),
            'channel_attr_dobj': lambda x, c: channel_attr_dobj(x, c)
        }

    def run_deriv(self, lexicon_type: str, deriv_rule: str, x: DerivTriggerContext, colnames: list):

        return self.rules[lexicon_type][deriv_rule](x, colnames)