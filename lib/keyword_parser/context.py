from spacy.matcher import DependencyMatcher, Matcher
from lib.nlp.spacy import Spacy

spacy_model = Spacy()


def get_dependency_matcher(rule_name, trigger_word, pattern) -> DependencyMatcher:
    return spacy_model.get_dependency_matcher(rule_name, trigger_word, pattern)


def get_matcher(rule_name, trigger_word, pattern) -> Matcher:
    return spacy_model.get_matcher(rule_name, trigger_word, pattern)


class ParseContext:
    def __init__(self, str_context: str) -> None:
        self.str_context = str_context
        self.spacy_context = spacy_model.init_text(str_context)


class DerivTriggerContext:
    def __init__(self, parse_context: ParseContext, triggered_text: str, triggered_func: str) -> None:
        self.parse_context: ParseContext = parse_context
        self.triggered_text: str = triggered_text
        self.triggered_func: str = triggered_func
        self.triggered_word: str = None
