import pickle

import torch
from torch.utils.data import DataLoader, Dataset

from lib.utils.misc_utils import get_processed_data_path


class DatavisDataset(Dataset):

    def __init__(self, ids, queries, fields_vocabs, intent_targets, field_targets, fields_masks, fields_indices, fields_token_masks, tokenizer, max_len, encodings=None):
        self.ids = ids
        self.queries = queries
        self.intent_targets = intent_targets
        self.fields_vocabs = fields_vocabs
        self.field_targets = field_targets
        self.fields_masks = fields_masks
        self.fields_indices = fields_indices
        self.fields_token_masks = fields_token_masks
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = encodings

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, item):
        
        _id = str(self.ids[item])
        query = str(self.queries[item])
        # print("query:", query)
        fields_vocab = self.fields_vocabs[item]
        intent_target = self.intent_targets[item]
        field_target = self.field_targets[item]
        field_mask = self.fields_masks[item]
        field_indices = self.fields_indices[item]
        field_token_masks = self.fields_token_masks[item]
        encoding = self.encodings[item] if self.encodings is not None else \
            self.tokenizer.encode_plus(
                query,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
        return {
            'id': _id,
            'query_text': query,
            'fields_vocab': ','.join(fields_vocab),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'intent_targets': torch.tensor(intent_target, dtype=torch.long),
            'field_targets': torch.tensor(field_target, dtype=torch.long),
            'field_masks': torch.gt(torch.tensor(field_mask, dtype=torch.long), 0),
            'field_indices': torch.tensor(field_indices, dtype=torch.long),
            'field_token_masks': torch.tensor(field_token_masks, dtype=torch.long)
        }


def load_processed_data(args):
    in_pkl = get_processed_data_path(args)
    print('loading processed data: {}'.format(in_pkl))

    with open(in_pkl, 'rb') as f:
        return pickle.load(f)


def create_data_loader(df, tokenizer, max_len, batch_size, gt_type):
    ds = DatavisDataset(
        ids=df['id'].to_numpy(),
        queries=df['query'].to_numpy(),
        fields_vocabs=df['fields'].to_numpy(),
        intent_targets=df[gt_type].to_numpy(),
        field_targets=df['{}-{}'.format(gt_type, 'field')].to_numpy(),
        fields_masks=df['fields_masks'].to_numpy(),
        fields_indices=df['fields_indices'].to_numpy(),
        fields_token_masks=df['fields_token_masks'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
        encodings=df['encoding'].to_numpy()
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)
