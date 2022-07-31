import re
from typing import Dict, Tuple, List

class Lexicons:
    def __init__(self) -> None:
        self.rules: Dict[str, List[Tuple]] = {}
    
    def init(self):
        raise NotImplementedError
        

class NL4DVLexicons(Lexicons):
    def __init__(self) -> None:
        super(NL4DVLexicons, self).__init__()
        self.init()
    
    def init(self):

        self.rules['plot'] = [
            ('scatter', ['scatter']),
            ('point', ['scatter']),
            ('correlation', ['scatter']),
            ('correlate', ['scatter']),
            ('bar', ['bar']),
            ('strip', ['strip']),
            ('pie', ['pie']),
            ('box', ['box']),
            ('line', ['line']),
            ('area', ['area']),
            ('stacked', ['stacked']),
            ('histogram', ['bar'])
        ]

        self.rules['task'] = [
            ('histogram', ['task_attr_pobj', 'count']),
            ('distribution', ['task_attr_pobj', 'count']),
            ('number of', ['task_attr_pobj', 'count']),
            ((re.compile(r'count[^\w]'), 'count'), ['task_attr_pobj', 'count']),
            ('mean', ['task_attr_pobj', 'task_attr_dir', 'mean']),
            ('average', ['task_attr_pobj', 'task_attr_dir', 'mean']),
            ('median', ['task_attr_pobj', 'task_attr_dir', 'median']),
            ('sum', ['task_attr_pobj', 'task_attr_dir', 'sum']),
            ('total', ['task_attr_pobj','task_attr_dir', 'sum']),
            ('by year', ['trend']),
            ((re.compile(r'over (\w+ )?year'), 'over time'), ['trend']),
            ((re.compile(r'over (\w+ )?time'), 'over year'), ['trend'])
        ]


        self.rules['channel'] = [
            ('color', ['channel_attr_pobj', 'channel_attr_dobj', 'color']),
            ('coloring by', ['channel_attr_pobj', 'color']),
            ('group by', ['channel_attr_pobj', 'color']),
            ('across', ['channel_attr_pobj', 'channel']),
            ('from', ['channel_attr_pobj', 'channel']),
            ('split by', ['channel_attr_pobj', 'column']),
            ('segregate', ['channel_attr_pobj', 'column']),
            ('separated by', ['channel_attr_pobj', 'column']),
            ((re.compile(r'per (?!year)'), 'per'), ['channel_attr_pobj', 'column'])
        ]
