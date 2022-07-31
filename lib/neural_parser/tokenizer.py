from typing import List, Tuple
from transformers import BertTokenizer, RobertaTokenizer


class Tokenizer:
    def __init__(self, model_type) -> None:
        self.model_type = model_type

        if model_type.startswith('bert'):
            self.tokenizer = BertTokenizer.from_pretrained(model_type)
        elif model_type.startswith('roberta'):
            self.tokenizer = RobertaTokenizer.from_pretrained(model_type)
    
    def tokenize(self, query, new_word=False) -> List:
        if new_word and self.model_type.startswith('roberta'):
            tokenized = self.tokenizer.tokenize(query)
            tokenized[0] = 'Ġ' + tokenized[0]
            return tokenized
        else:
            return self.tokenizer.tokenize(query)
    
    def tokenize_with_sep_pos(self, query, new_word=False) -> Tuple[List, int]:
        tokenized = self.tokenize(query, new_word=new_word)
        if self.model_type.startswith('bert'):
            sep_pos = len(tokenized) + 1
        elif self.model_type.startswith('roberta'):
            sep_pos = len(tokenized) + 2
        
        return tokenized, sep_pos
    
    def encode_plus(self, *args, **kwargs):
        # print(args)
        return self.tokenizer.encode_plus(*args, **kwargs)

    def convert_ids_to_tokens(self, input_ids):
        return self.tokenizer.convert_ids_to_tokens(input_ids)
    
    def recover_origin_field_from_tokens(self, tokens: list):
        # print("tokens:",tokens)
        
        if self.model_type.startswith('bert'):
            recovered_str = '_'.join(tokens)    
            return recovered_str.replace('_##', '').replace('_-_', '-')
        
        elif self.model_type.startswith('roberta'):
            recovered_str = ''.join(tokens)
            if recovered_str.startswith('Ġ'):
                return recovered_str[1:].replace('Ġ', '_')
            else:
                return recovered_str.replace('Ġ', '_')
