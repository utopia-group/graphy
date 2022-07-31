import torch

from torch import nn
from transformers import BertModel


class IntentClassifier(nn.Module):
    def __init__(self, args, n_classes):
        super(IntentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.drop = nn.Dropout(args.emb_dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, field_indices, field_token_masks):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = bert_output['pooled_output']
        _, pooled_output = bert_output
        output = self.drop(pooled_output)
        return self.out(output)

      
class IntentFieldFinder(nn.Module):
    def __init__(self, args, n_classes):
        super(IntentFieldFinder, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_model)
        self.drop_intent = nn.Dropout(args.emb_dropout)
        self.drop_field = nn.Dropout(args.emb_dropout)
        self.out_intent = nn.Linear(self.bert.config.hidden_size, n_classes)
        # self.field1 = nn.Linear(self.bert.config.hidden_size, n_classes)
        # self.field2 = nn.Linear(n_classes, self.bert.config.hidden_size)
        self.field3 = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)

        self.max_seq_len = args.input_dim
        self.max_field_num = args.max_field_num
        self.max_field_token_len = args.max_token_num_per_field
        self.hidden_size = self.bert.config.hidden_size
        self.batch_size = args.batch_size
        self.device = args.device


    def forward(self, input_ids, attention_mask, field_indices, field_token_masks):

        """
        input:
        - input_ids: batch_size * seq_len
        - attention_mask: batch_size * seq_len
        - field_indices: batch_size * max_field_num * max_token_num
        - field_masks: batch_size * max_field_num
        - field_token_masks: batch_size * max_field_num * max_token_num * 1
        output:
        - intent_output: batch_size * n_classes
        - field_output: batch_size * max_field_num
        """
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        # pooled_output = bert_output['pooled_output']
        last_hidden_state, pooled_output, output_attention = bert_output

        bert_pooled_output_for_intent = self.drop_intent(pooled_output)
        intent_output = self.out_intent(bert_pooled_output_for_intent)

        # extract field token hidden state
        curr_batch_size = field_indices.size(0)
        global_indices_padding = torch.tensor([[self.max_seq_len * b] * (self.max_field_token_len * self.max_field_num) for b in range(curr_batch_size)], dtype=torch.long).to(self.device)
        global_indices_padding = torch.flatten(global_indices_padding)
        # print(self.batch_size)
        # print(field_indices.shape)
        # print(global_indices_padding.shape)
        flatten_field_indices = field_indices.flatten() + global_indices_padding
        flatten_hidden_state = last_hidden_state.flatten(end_dim=1)
        # print('flatten_field_indices:', flatten_field_indices)
        # print('type:', type(flatten_field_indices))
        flatten_global_field_token_hidden_state = flatten_hidden_state.index_select(dim=0, index=flatten_field_indices)
        field_token_hidden_state = flatten_global_field_token_hidden_state.reshape(curr_batch_size, self.max_field_num, self.max_field_token_len, self.hidden_size)

        # mask the unused token and sum all the token in a field to obtain the field embedding
        # print(field_token_hidden_state.shape)
        # print(field_token_masks.shape)
        field_token_hidden_state *= field_token_masks
        field_hidden_state = field_token_hidden_state.sum(dim=2)

        # out_field(pooled_output): batch_size * hidden_state_size
        # field_hidden_state: batch_size * num_field * hidden_state_size
        bert_pooled_output_for_field = self.drop_field(pooled_output)
        # print('field_hidden_state:', field_hidden_state.shape)
        # f2_f1_out = self.field2(self.field1(bert_pooled_output_for_field))
        f2_f1_out = self.field3(bert_pooled_output_for_field)
        # print('bert_pooled_output_for_field:', bert_pooled_output_for_field.shape)
        # print('f1_out:', self.field1(bert_pooled_output_for_field).shape)
        # print('f2_f1_out:', f2_f1_out.shape)
        field_output = torch.bmm(field_hidden_state, f2_f1_out.reshape(curr_batch_size, self.hidden_size, 1)).reshape(curr_batch_size, self.max_field_num)
        # field_output_dot = f2_f1_out * field_hidden_state
        # field_output_dot = self.field2(self.field1(bert_pooled_output_for_field)) * field_hidden_state
        # print('field_output_dot:', field_output.shape)
        # field_output = field_output_dot.sum(dim=2)

        return intent_output, field_output, output_attention
