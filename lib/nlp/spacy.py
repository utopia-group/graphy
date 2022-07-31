import spacy
from spacy.matcher import DependencyMatcher, Matcher
from spacy.symbols import NOUN, PROPN, PRON
from spacy.language import Language

@Language.component("merge_noun_chunks_new")
def merge_noun_chunks_new(doc):
    if not doc.has_annotation("DEP"):
        return doc
    with doc.retokenize() as retokenizer:
        for np in doc.noun_chunks:
            # add a check so that we follow the rules
            pass_check = True
            for t in np:
                if t.text.lower() in ("sum", "average"):
                    pass_check = False
                    break
                if t.pos not in (NOUN, PROPN, PRON):
                    pass_check = False
                    break
                if "." in t.text:
                    pass_check = False
                    break
            if pass_check:
                attrs = {"tag": np.root.tag, "dep": np.root.dep}
                retokenizer.merge(np, attrs=attrs)
    return doc


class Spacy:
    def __init__(self) -> None:
        self.model = spacy.load("en_core_web_trf")

        ruler = self.model.get_pipe("attribute_ruler")
        ruler.add(patterns=[[{"LOWER": "plot"}]], attrs={"POS": "VERB", "TAG": "VBG"})
        ruler.add(patterns=[[{"LOWER": "color"}, {"LOWER": "by"}]], attrs={"POS": "VERB", "TAG": "VBG"}, index=0)

        self.model.add_pipe("merge_subtokens")
        self.model.add_pipe("merge_noun_chunks_new")
    
    def get_dependency_matcher(self, rule_name, trigger_word, pattern):
        matcher = DependencyMatcher(self.model.vocab)
        matcher.add("{}_{}".format(rule_name, trigger_word), [pattern])

        return matcher
    
    def get_matcher(self, rule_name, trigger_word, pattern):
        matcher = Matcher(self.model.vocab)
        matcher.add("{}_{}".format(rule_name, trigger_word), [pattern])

        return matcher
    
    def init_text(self, text: str):
        return self.model(text)