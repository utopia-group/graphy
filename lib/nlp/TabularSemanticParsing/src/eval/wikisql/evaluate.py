"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
#!/usr/bin/env python
import json
from argparse import ArgumentParser
from tqdm import tqdm
from lib.nlp.TabularSemanticParsing.src.eval.wikisql.lib.dbengine import DBEngine
from lib.nlp.TabularSemanticParsing.src.eval.wikisql.lib.query import Query
from lib.nlp.TabularSemanticParsing.src.eval.wikisql.lib.common import count_lines


def get_evaluation_metrics(fs, fp, engine):
    assert(len(fs) == len(fp))
    for ls, lp in tqdm(zip(fs, fp), total=len(fs)):
        correct, match = eval_fun(ls, lp, engine)
        grades.append(correct)
        exact_match.append(match)
    lf_acc = sum(exact_match) / len(exact_match)
    ex_acc = sum(grades) / len(grades)
    return lf_acc, ex_acc


def eval_fun(ls, lp, engine):
    eg = json.loads(ls) if isinstance(ls, str) else ls
    ep = json.loads(lp) if isinstance(lp, str) else lp
    # print(ls)
    # print(lp)
    # import pdb
    # pdb.set_trace()
    qg = Query.from_dict(eg['sql'], ordered=False)
    if engine:
        gold = engine.execute_query(eg['table_id'], qg, lower=True)
    pred = ep.get('error', None)
    qp = None
    if not ep.get('error', None):
        try:
            qp = Query.from_dict(ep['query'], ordered=False)
            if engine:
                # print(eg['table_id'])
                # print(ep['query'])
                # import pdb
                # pdb.set_trace()
                pred = engine.execute_query(eg['table_id'], qp, lower=True)
        except Exception as e:
            pred = repr(e)
    correct = (pred == gold) if engine else False
    match = qp == qg
    return correct, match


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('source_file', help='source file for the prediction')
    parser.add_argument('db_file', help='source database for the prediction')
    parser.add_argument('pred_file', help='predictions by the model')
    parser.add_argument('--ordered', action='store_true', help='whether the exact match should consider the order of conditions')
    args = parser.parse_args()

    engine = DBEngine(args.db_file)
    exact_match = []
    with open(args.source_file) as fs, open(args.pred_file) as fp:
        grades = []
        for ls, lp in tqdm(zip(fs, fp), total=count_lines(args.source_file)):
            eg = json.loads(ls)
            ep = json.loads(lp)
            qg = Query.from_dict(eg['sql'], ordered=args.ordered)
            gold = engine.execute_query(eg['table_id'], qg, lower=True)
            pred = ep.get('error', None)
            qp = None
            if not ep.get('error', None):
                try:
                    qp = Query.from_dict(ep['query'], ordered=args.ordered)
                    pred = engine.execute_query(eg['table_id'], qp, lower=True)
                except Exception as e:
                    pred = repr(e)
            correct = pred == gold
            match = qp == qg
            grades.append(correct)
            exact_match.append(match)
        print(json.dumps({
            'ex_accuracy': sum(grades) / len(grades),
            'lf_accuracy': sum(exact_match) / len(exact_match),
            }, indent=2))
