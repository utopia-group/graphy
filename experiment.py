import os
import random
from typing import Dict

import wandb
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup

from lib.eval.output import NeuralParserOutput
from parse_args import args
from preprocess import generate_training_data
from lib.neural_parser.labels import CLASSES
from lib.neural_parser.models import IntentFieldFinder
from lib.neural_parser.tokenizer import Tokenizer
from lib.neural_parser.data_processor import load_processed_data, create_data_loader
from lib.utils.neural_model_utils import eval_model, compute_field_loss, compute_field_pred
from lib.utils.csv_utils import read_csv_to_dict
from lib.utils.eval_utils import parse_keyword_based_synthesized_spec
from lib.utils.misc_utils import get_model_dir, get_model_path, get_saved_prediction_pkl_path, get_train_val_test_file_name, get_processed_data_path

"""
This is the file for neural parser training and inference and producing metrics
"""


def train(args):
    wandb.init(config=args)
    print("args: ", args)
    print("wandb id: ", wandb.run.id)
    wandb.run.name = wandb.run.id

    wandb.save(get_processed_data_path(args))

    tokenizer = Tokenizer(args.bert_model)
    dataset = load_processed_data(args)
    train_data = dataset['train']
    print('{} training examples loaded'.format(len(train_data)))
    val_data = dataset['val']
    test_data = dataset['test']
    print('{} validation examples loaded'.format(len(val_data)))

    for task_type in CLASSES.keys():

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        print("Training {} ...".format(task_type))

        model = IntentFieldFinder(args, len(CLASSES[task_type]))
        model = model.to(args.device)

        wandb.watch(model)

        curr_epochs, curr_lr, curr_wr = args.params[task_type]

        print("curr_epochs:", curr_epochs)
        print("curr_lr:", curr_lr)
        print("curr_wr:", curr_wr)

        train_data_loader = create_data_loader(train_data, tokenizer, args.input_dim, args.batch_size, task_type)
        val_data_loader = create_data_loader(val_data, tokenizer, args.input_dim, args.batch_size, task_type)
        test_data_loader = create_data_loader(test_data, tokenizer, args.input_dim, 1, task_type)
        num_entry = len(train_data)

        optimizer = optim.AdamW(model.parameters(), lr=curr_lr)
        total_steps = len(train_data_loader) * curr_epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=round(curr_wr * total_steps), num_training_steps=total_steps)
        intent_loss_fn = nn.CrossEntropyLoss().to(args.device)
        field_loss_fn = nn.CrossEntropyLoss().to(args.device)

        best_val_acc = 0
        best_val_loss = 0

        for epoch in range(1, curr_epochs + 1):

            model.train()

            print('epoch {}'.format(epoch))
            intent_losses = []
            field_losses = []
            comb_losses = []
            intent_correct_predictions = 0
            field_correct_predictions = 0

            for batch_idx, batch_data in enumerate(tqdm(train_data_loader)):

                input_ids = batch_data["input_ids"].to(args.device)
                attention_mask = batch_data["attention_mask"].to(args.device)
                intent_targets = batch_data["intent_targets"].to(args.device)
                field_targets = batch_data["field_targets"].to(args.device)
                field_masks = batch_data['field_masks'].to(args.device)
                field_indices = batch_data['field_indices'].to(args.device)
                field_token_masks = batch_data['field_token_masks'].to(args.device)

                intent_outputs, field_outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask, field_indices=field_indices, field_token_masks=field_token_masks)

                # print('field_masks:', field_masks.shape)
                # print('field_outputs:', field_outputs)

                _, intent_preds = torch.max(intent_outputs, dim=1)
                intent_loss = intent_loss_fn(intent_outputs, intent_targets)

                _, field_preds = compute_field_pred(field_masks, field_outputs)
                field_loss = compute_field_loss(args.device, field_outputs, field_masks, field_targets, field_loss_fn)

                comb_loss = intent_loss + field_loss

                intent_correct_predictions += torch.sum(intent_preds == intent_targets)
                field_correct_predictions += torch.sum(field_preds == field_targets)

                intent_losses.append(intent_loss.item())
                field_losses.append(field_loss.item())
                comb_losses.append(comb_loss.item())

                comb_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()

                if batch_idx % args.log_interval == 0:
                    wandb.log({"batch {} intent_losses".format(task_type): intent_loss.item(), "batch {} field_losses".format(task_type): field_loss.item(),
                               "batch {} combined_loss".format(task_type): comb_loss.item()})

            print("epoch {} train loss: {}".format(epoch, np.mean(comb_losses)))
            print("epoch {} train correct intent predictions: {}".format(epoch, (intent_correct_predictions / num_entry)))
            print("epoch {} train correct field predictions: {}".format(epoch, (field_correct_predictions / num_entry)))
            wandb.log({"{} epoch train loss".format(task_type): np.mean(comb_losses),
                       "{} epoch train correct intent predictions".format(task_type): (intent_correct_predictions / num_entry),
                       "{} epoch train correct field predictions".format(task_type): (field_correct_predictions / num_entry)})

            # eval on val data
            val_intent_acc, val_field_acc, val_loss = eval_model(args, task_type, model, val_data_loader, intent_loss_fn, field_loss_fn, len(val_data))
            print("epoch {} val loss: {}".format(epoch, val_loss))
            print("epoch {} val correct intent predictions: {}".format(epoch, val_intent_acc))
            print("epoch {} val correct field predictions: {}".format(epoch, val_field_acc))
            wandb.log({"{} epoch val loss".format(task_type): np.mean(val_loss),
                       "{} epoch val correct intent predictions".format(task_type): val_intent_acc,
                       "{} epoch val correct field predictions".format(task_type): val_field_acc})

            # eval on test data
            overall_output = defaultdict(dict)
            test_intent_acc, test_field_acc, test_loss, _ = eval_model(args, task_type, model, test_data_loader, intent_loss_fn, field_loss_fn, len(test_data), inference=True,
                                                                       overall_output=overall_output)
            print("epoch {} test loss: {}".format(epoch, test_loss))
            print("epoch {} test correct intent predictions: {}".format(epoch, test_intent_acc))
            print("epoch {} test correct field predictions: {}".format(epoch, test_field_acc))
            wandb.log({"{} epoch test loss".format(task_type): np.mean(test_loss),
                       "{} epoch test correct intent predictions".format(task_type): test_intent_acc,
                       "{} epoch test correct field predictions".format(task_type): test_field_acc})

            mean_val_acc = (val_intent_acc + val_field_acc) / 2

            if task_type == 'sum' or task_type == 'column':
                if mean_val_acc >= best_val_acc:
                    if mean_val_acc == best_val_acc:
                        if val_loss < best_val_loss:
                            torch.save(model.state_dict(), get_model_path(task_type, args))
                            best_val_loss = val_loss
                            print("model save")
                    else:
                        torch.save(model.state_dict(), get_model_path(task_type, args))
                        best_val_loss = val_loss
                        print("model save")
                    best_val_acc = mean_val_acc
            else:
                if mean_val_acc > best_val_acc:
                    torch.save(model.state_dict(), get_model_path(task_type, args))
                    best_val_acc = mean_val_acc
                    print("model save")


def inference(args):
    tokenizer = Tokenizer(args.bert_model)
    dataset = load_processed_data(args)
    test_data = dataset['test']
    print('{} test examples loaded'.format(len(test_data)))

    # create a dict check if the x-y field gt in the field vocab
    test_selected = test_data[['id', 'x-field', 'y-field']].values.tolist()
    test_id_x_y_gt = {item[0]: (item[1], item[2]) for item in test_selected}

    all_wrong_instances = []
    all_test_acc = {}
    all_predictions: Dict[str, NeuralParserOutput] = {}

    for task_type in CLASSES.keys():
        model_path = get_model_path(task_type, args)
        if 'cpu' in str(args.device):
            checkpoint = torch.load(model_path, map_location=args.device)
        else:
            checkpoint = torch.load(model_path)

        model = IntentFieldFinder(args, len(CLASSES[task_type]))
        model.load_state_dict(checkpoint)
        model = model.to(args.device)
        model.eval()
        intent_loss_fn = nn.CrossEntropyLoss().to(args.device)
        field_loss_fn = nn.CrossEntropyLoss().to(args.device)

        test_data_loader = create_data_loader(test_data, tokenizer, args.input_dim, 1, task_type)
        intent_test_acc, field_test_acc, loss, wrong_instances = eval_model(args, task_type, model, test_data_loader, intent_loss_fn, field_loss_fn, len(test_data), inference=True,
                                                                            overall_output=all_predictions)
        all_wrong_instances.extend(wrong_instances)
        all_test_acc[task_type] = intent_test_acc
        all_test_acc['{}-{}'.format(task_type, 'field')] = field_test_acc

    print('single task_type test_acc: {}'.format(str(all_test_acc)))

    correct = 0
    correct_with_field = 0
    total = 0
    wrong_instance_output = []

    all_results = {}
    intent_field_joint_acc = defaultdict(int)

    print(" ==== all predictions ====")
    for bid, pred in all_predictions.items():

        output_str = repr(pred)
        print(output_str)

        # update a couple metric at the same time here
        pred.update_intent_field_joint_acc(intent_field_joint_acc)

        if pred.top_1_check_res == 'CORRECT':
            correct += 1
            wrong_instance_output.append(output_str)
        else:
            wrong_instance_output.append(output_str)

        if pred.top_1_check_res == 'CORRECT' and not (test_id_x_y_gt[bid][0] == -1) and not (test_id_x_y_gt[bid][1] == -1):
            correct_with_field += 1

        total += 1

        all_results[bid] = pred.prob_output

    print(" ====  error ====")
    open('output.txt', 'w').close()
    for wo in wrong_instance_output:
        print(wo)
        with open('output.txt', 'a') as f:
            f.write(wo + "\n\n\n")

    print("overall correct rate (em): {} / {} = {}".format(correct, total, str(correct / total)))
    print("overall correct rate (with fields em): {} / {} = {}".format(correct_with_field, total, str(correct_with_field / total)))

    intent_field_joint_acc_percent = {k: v / len(all_predictions) for k, v in intent_field_joint_acc.items()}
    print('intent + field joint acc by intent: ', intent_field_joint_acc_percent)

    # save all_results as a pickle file
    with open(get_saved_prediction_pkl_path(args), 'wb') as f:
        pickle.dump(all_results, f)


def check_baseline(args):
    def read_previous_log(fname):

        previous_parse = {}
        parsed_data = read_csv_to_dict(fname)

        for d in parsed_data:
            if not d['spec'] == '':
                previous_parse[d['bid']] = d['spec']

        return previous_parse

    def check_correct(parsed_spec, gt_spec):

        intent_correct = False
        field_correct = False

        if parsed_spec[0] == gt_spec[0]:
            intent_correct = True

        if parsed_spec[1] == gt_spec[1]:
            field_correct = True

        return intent_correct, field_correct, (intent_correct and field_correct)

    # all_chi_previous_parse = {**read_previous_log('eval_output/EvalSynth_chi21_run_parsing_res_cars.csv'),
    #                           **read_previous_log('eval_output/EvalSynth_chi21_run_parsing_res_movie.csv'),
    #                           **read_previous_log('eval_output/EvalSynth_chi21_run_parsing_res_superstore.csv')}
    all_chi_previous_parse = {**read_previous_log('eval_output/EvalSynthPrev_chi21_movies_run_parsing_only_res.csv')}

    _, _, test_file_name = get_train_val_test_file_name(args)
    test_data = read_csv_to_dict('eval/datavis/{}'.format(test_file_name))

    correct = 0
    correct_with_field = 0
    total = 0

    partial_correct = {'plot': 0, 'color': 0, 'column': 0, 'mean': 0, 'sum': 0, 'count': 0, 'field': 0, 'field_containment': 0}
    partial_correct_intent = {'plot': 0, 'color': 0, 'column': 0, 'mean': 0, 'sum': 0, 'count': 0}
    partial_correct_field = {'plot': 0, 'color': 0, 'column': 0, 'mean': 0, 'sum': 0, 'count': 0}

    for d in test_data:

        fields_vocab = [item[1:-1] for item in d['fields'][1:-1].split(', ')]
        parsed_spec = parse_keyword_based_synthesized_spec(all_chi_previous_parse['{}-{}'.format(d['data'].lower(), d['id'])][1:-1])  # format {intnet: (intent, field), ...}
        gt_spec = dict([(task_type, (CLASSES[task_type][int(d[task_type])], fields_vocab[int(d['{}-field'.format(task_type)])])) for task_type in CLASSES.keys()])

        # print(set(['{}({})'.format(values[0], values[1]) for intent_type, values in parsed_spec.items() if not intent_type == 'field']))
        # print(set(['{}({})'.format(intent, field) for intent, field in gt_spec.values()]))

        if set(['{}({})'.format(values[0], values[1]) for intent_type, values in parsed_spec.items() if not intent_type == 'field']) \
                == set(['{}({})'.format(intent, field) for intent, field in gt_spec.values()]):
            # print("Correct")
            correct += 1

            if set(fields_vocab[:-1]).issubset(set(parsed_spec['field'])):
                correct_with_field += 1

        for task_type in CLASSES.keys():
            if task_type == 'plot':
                if gt_spec[task_type][0] == parsed_spec[task_type][0]:
                    partial_correct['plot'] += 1
                    partial_correct_intent['plot'] += 1
            else:
                intent_correct, field_correct, both_correct = check_correct(parsed_spec[task_type], gt_spec[task_type])
                if intent_correct:
                    partial_correct_intent[task_type] += 1
                if field_correct:
                    partial_correct_field[task_type] += 1
                if both_correct:
                    partial_correct[task_type] += 1

        if set(parsed_spec['field']) == set(fields_vocab[:-1]):
            partial_correct['field'] += 1

        if set(fields_vocab[:-1]).issubset(set(parsed_spec['field'])):
            partial_correct['field_containment'] += 1

        total += 1

    print("total_correct: {} / {} = {}".format(correct, total, (correct / total)))
    print("total_correct_with_field: {} / {} = {}".format(correct_with_field, total, (correct_with_field / total)))
    print("partial spec correct:")
    for key, value in partial_correct.items():
        print("{}_correct: {} / {} = {}".format(key, value, total, (value / total)))
    print("partial intent correct:")
    for key, value in partial_correct_intent.items():
        print("{}_intent_correct: {} / {} = {}".format(key, value, total, (value / total)))
    print("partial field correct:")
    for key, value in partial_correct_field.items():
        print("{}_field_correct: {} / {} = {}".format(key, value, total, (value / total)))


if __name__ == '__main__':

    print(args)

    if args.gpu is not None:
        args.device = torch.device(('cuda:' + args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        args.device = 'cpu'

    print("Pytorch using device: ", args.device)

    if args.process_data:
        generate_training_data(args)
    elif args.train:
        model_dir = get_model_dir(args)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        train(args)
        # upload all trained models to wandb
        for task_type in CLASSES:
            wandb.save(get_model_path(task_type, args))
        print("model_path: ", model_dir)
    elif args.inference:
        print("args: ", args)
        inference(args)
        # inference_prev(args)
    elif args.check_baseline:
        # get metrics fot the keyword-based approach
        check_baseline(args)
    else:
        pass
