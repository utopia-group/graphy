"""
utils for all the neural model part including:
- compute field loss/pred/prob
- eval model for inference
"""

from typing import Union, Tuple, Any, Dict

import numpy as np
import torch
from torch import nn

from lib.eval.output import NeuralParserOutput
from lib.neural_parser.labels import CLASSES


def compute_field_loss(device, field_outputs, field_masks, field_targets, field_loss_fn):
    batch_field_losses = torch.tensor([0.0], requires_grad=True).to(device)
    batch_num_entry = 0
    for field_output, field_mask, field_targets in zip(field_outputs, field_masks, field_targets):
        unmask_field_output = torch.masked_select(field_output, field_mask)
        # print('unmask_field_output:', unmask_field_output)
        # print('field_mask:', field_mask)
        # print('field_targets:', field_targets)
        batch_field_losses += field_loss_fn(unmask_field_output.unsqueeze(0), field_targets.unsqueeze(0))
        batch_num_entry += 1
    field_loss = batch_field_losses / batch_num_entry

    return field_loss


def compute_field_pred(field_masks, field_outputs):
    field_masks_int = field_masks.long()
    # print('field_masks_int:', field_masks_int)
    field_padding = (1 - field_masks_int) * -float('inf')
    # print('field_padding:', field_padding)
    field_padding[field_padding != field_padding] = 0
    # print('field_padding:', field_padding)
    # print('temp:', field_outputs + field_padding)
    # print('max:', torch.max(field_outputs * field_padding, dim=1))
    return torch.max(field_outputs + field_padding, dim=1)


def compute_field_prob(softmax_f, field_masks, field_outputs):
    output = []
    for field_mask, field_output in zip(field_masks, field_outputs):
        unmask_field_output = torch.masked_select(field_output, field_mask)
        output.append(softmax_f(unmask_field_output, dim=0))

    return output


def eval_model(args, gt_type, model, data_loader, intent_loss_fn, field_loss_fn, n_examples, inference=False, overall_output=None) -> \
        Union[Tuple[Any, Any, Any], Tuple[Any, Any, Any, Dict]]:
    """
    Assumption:
    - inference mode always have batch size of 1
    - inference mode may not know the gt, if not knowning the gt, the gt value is -1
    """
    if inference:
        assert data_loader.batch_size == 1
        assert overall_output is not None

    model = model.eval()

    test_iter = iter(data_loader)

    intent_losses = []
    field_losses = []
    comb_losses = []
    intent_correct_predictions = 0
    field_correct_predictions = 0
    softmax = nn.functional.softmax

    wrong_instance = {'intent': [], 'field': []}

    with torch.no_grad():
        for _, batch_data in enumerate(test_iter):
            bids = batch_data['id']
            input_ids = batch_data["input_ids"].to(args.device)
            attention_mask = batch_data["attention_mask"].to(args.device)
            intent_targets = batch_data["intent_targets"].to(args.device)
            field_targets = batch_data["field_targets"].to(args.device)
            field_masks = batch_data['field_masks'].to(args.device)
            field_indices = batch_data['field_indices'].to(args.device)
            field_token_masks = batch_data['field_token_masks'].to(args.device)

            intent_outputs, field_outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask, field_indices=field_indices, field_token_masks=field_token_masks)

            intent_probs = softmax(intent_outputs, dim=1)
            _, intent_preds = torch.max(intent_outputs, dim=1)

            field_probs = compute_field_prob(softmax, field_masks, field_outputs)
            _, field_preds = compute_field_pred(field_masks, field_outputs)

            # skip loss computation (not important anyway) since the ground truth is not in the class available
            if inference and intent_targets.item() == -1:
                intent_loss = torch.tensor([0]).to(args.device)
            else:
                intent_loss = intent_loss_fn(intent_outputs, intent_targets)

            if inference and field_targets.item() == -1:
                field_loss = torch.tensor([0]).to(args.device)
            else:
                field_loss = compute_field_loss(args.device, field_outputs, field_masks, field_targets, field_loss_fn)

            comb_loss = intent_loss + field_loss

            intent_correct_predictions += torch.sum(intent_preds == intent_targets)
            field_correct_predictions += torch.sum(field_preds == field_targets)
            comb_losses.append(comb_loss.item())

            if inference and overall_output is not None:
                # in-place update for overall_output
                # note that i have to do this because the inference works by model but not benchmark

                if overall_output.get(bids[0]) is None:
                    overall_output[bids[0]] = NeuralParserOutput(bids[0], batch_data['query_text'][0])

                current_output: NeuralParserOutput = overall_output.get(bids[0])
                if intent_targets[0].item == -1:
                    current_output.add_output(gt_type, 'intent', CLASSES[gt_type][intent_preds[0].item()], 'unknown', intent_probs, None)
                else:
                    current_output.add_output(gt_type, 'intent', CLASSES[gt_type][intent_preds[0].item()], CLASSES[gt_type][intent_targets[0].item()], intent_probs, None)
                fields_vocab = batch_data['fields_vocab'][0].split(',')
                if field_targets[0].item() == -1:
                    current_output.add_output(gt_type, 'field', fields_vocab[field_preds[0].item()], 'unknown', field_probs, fields_vocab)
                else:
                    current_output.add_output(gt_type, 'field', fields_vocab[field_preds[0].item()], fields_vocab[field_targets[0].item()], field_probs, fields_vocab)

            if inference and current_output.contain_gt:
                # print("preds:", preds)
                if not intent_preds[0].item() == intent_targets[0].item():
                    wrong_instance['intent'].append(bids[0])
                if not field_preds[0].item() == field_targets[0].item():
                    wrong_instance['field'].append(bids[0])

    if inference:
        return (intent_correct_predictions / n_examples), (field_correct_predictions / n_examples), np.mean(comb_losses), wrong_instance
    else:
        return (intent_correct_predictions / n_examples), (field_correct_predictions / n_examples), np.mean(comb_losses)
