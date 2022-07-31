"""
nl process utils
- standardize keyword
- standardize neural
"""
import re
from typing import List, Callable

replace_rules_keyword: List[Callable] = [
    lambda x: re.sub(r'(scatter|strip)plot[s]?', r'\1 plot', x),
    lambda x: re.sub(r'(bar|line|area)chart[s]?', r'\1 chart', x),
    lambda x: re.sub('x axis', 'x-axis', x),
    lambda x: re.sub('y axis', 'y-axis', x),
    lambda x: re.sub(r'x[ ]?=', 'x-axis=', x),
    lambda x: re.sub(r'y[ ]?=', 'y-axis=', x),
    lambda x: re.sub(r'(avg|sum)[ ]?[(](\s+)[)]', '\1 of \2', x),
    lambda x: re.sub('avg', 'average', x),
    lambda x: re.sub(r'v[/]?s[.]?', 'versus', x),
    lambda x: re.sub(r'group(ed )?by', 'group by', x),
    lambda x: re.sub(r'color(ed )?by', 'color by', x),
    lambda x: re.sub(r'numbers of', 'number of', x)
]


replace_rules_neural: List[Callable] = [
    lambda x: re.sub(r'(scatter|strip)plot[s]?', r'\1 plot', x, flags=re.I),
    lambda x: re.sub(r'(bar|line|area)chart[s]?', r'\1 chart', x, flags=re.I),
    lambda x: re.sub('x[-]axis[ ]?=[ ]?', 'x-axis = ', x, flags=re.I),
    lambda x: re.sub('y[-]axis[ ]?=[ ]?', 'y-axis = ', x, flags=re.I),
    lambda x: re.sub('x axis[ ]?=[ ]?', 'x-axis = ', x, flags=re.I),
    lambda x: re.sub('y axis[ ]?=[ ]?', 'y-axis = ', x, flags=re.I),
    lambda x: re.sub('x axis', 'x-axis', x, flags=re.I),
    lambda x: re.sub('y axis', 'y-axis', x, flags=re.I),
    lambda x: re.sub(r'x[ ]?=[ ]?', 'x-axis = ', x, flags=re.I),
    lambda x: re.sub(r'y[ ]?=[ ]?', 'y-axis = ', x, flags=re.I),
    lambda x: re.sub(r'(\w+)[ ]?[(](\w+)[)]', r'\1 ( \2 )', x, flags=re.I),
    lambda x: re.sub(r'v[/]?s[.]?', 'versus', x, flags=re.I),
    lambda x: re.sub(r'groupby', 'group by', x),
    lambda x: re.sub(r'colorby', 'color by', x),
]


def standardize_keyword(s: str) -> str:
    s = s.lower()
    for rule in replace_rules_keyword:
        s = rule(s)

    return s


def standardize_neural(s: str) -> str:
    for rule in replace_rules_neural:
        s = rule(s)

    return s


def clean(s: str) -> str:
    s = re.sub(r'[|]|[(]|[)]', ' ', s)
    s = " ".join(s.split())

    return s


def tokenize(s: str) -> list:
    return standardize_keyword(clean(s)).split()


def get_tokenized_str(s: str) -> str:
    return " ".join(tokenize(s))