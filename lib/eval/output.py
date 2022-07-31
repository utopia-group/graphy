import copy

import numpy as np

from collections import defaultdict
from typing import Dict, List, Union, Iterator

from lib.eval.benchmark import InputDataset
from lib.neural_parser.labels import CLASSES
from lib.prev_synthesizer.spec import SpecProg, prog_hash
from lib.type.ref_type import BaseRefType
from lib.utils.enum_utils import get_score, get_pred_and_diff
from lib.utils.misc_utils import printc
from lib.utils.pq import PriorityQueue


class Output:
    """
    Wrapper for output instance
    """

    def __init__(self):
        pass


class NeuralParserOutput(Output):
    """
    Output that neural parser produces for one benchmark
    """

    def __init__(self, bid, query, cached_res=None):
        super().__init__()
        self.bid = bid
        self.query = query
        self.output: Dict[str, Dict] = {}  # stores {pred_type: {query: query, ...}}

        # we pickle the prob_output to be used in the future
        # stores {'task_type': {'intent': {label: prob}}, 'field': ...}
        # NOTE: we will use this for enumeration
        self.joint_prob_output: Dict[str, List] = {}
        if cached_res is None:
            self.prob_output: Dict[str, Dict[str, Dict]] = {} if cached_res is None else cached_res
        else:
            self.prob_output = cached_res
            self.postprocess()

        self.top_1_all_pred = []  # top 1 pred across all intents
        self.all_gt = []  # gt across all intents

        # eval book-keeping
        self.top_1_check_res = 'UNKNOWN'
        self.contain_gt: bool = True

        self.added_count = 0  # count the amount of task already added

    def format_output_key(self, task_type: str, pred_type: str) -> str:
        return '{}-{}'.format(task_type, pred_type)

    def get_next_goal_type(self, data: InputDataset, limit=-1) -> Iterator[BaseRefType]:
        """
        enumerate the parsed results from high to low probability and parse it into a refinement goal type
        i adopt the same algorithm from the enumerator.py
        the high level idea is keep track of what is the next highest spec across all categories using the difference (we are doing a sum of probabilities here)
        """

        worklist = PriorityQueue(get_score)
        enumerate_count = 0
        enumerated_specs = {}

        print("processed_prediction:", self.joint_prob_output)

        def create_spec(pred, score, spec) -> Union[SpecProg, None]:
            """
            create a SpecProg, we do cache the results to avoid duplicate spec enumerated
            """

            spec_id = prog_hash(pred)
            if spec_id in enumerated_specs:
                return None

            new_spec = SpecProg(pred=pred, score=score, spec=spec)
            enumerated_specs[spec_id] = ''
            return new_spec

        def init_spec(processed_prediction: dict):
            """
            create the highest SpecProg as the first one to start with
            """
            all_classes_preds = dict([(class_type, get_pred_and_diff(processed_prediction, class_type)) for class_type in CLASSES])
            spec_init = create_spec(pred=dict([(class_type, v[0][0]) for class_type, v in all_classes_preds.items()]),
                                    score=sum([v[0][0][1] for v in all_classes_preds.values()]),
                                    spec=dict([(class_type, (v[0][1:], v[1])) for class_type, v in all_classes_preds.items()]))

            assert spec_init is not None
            return spec_init

        spec_init = init_spec(self.joint_prob_output)
        worklist.put(spec_init)

        while not worklist.is_empty():
            curr_state = worklist.pop()

            if limit > -1:
                if enumerate_count > limit:
                    raise StopIteration

            for full_spec in curr_state.get_pred_type(self.prob_output['field'][:-1], data):
                enumerate_count += 1
                yield full_spec

            # print("curr_state:", curr_state)
            curr_score = curr_state.score
            curr_state_spec = list(curr_state.spec.items())

            # preview the enum order here
            # the order is determined by who has the minimum difference
            curr_state_spec.sort(key=lambda e: e[1][1][0])

            for spec_idx, spec_entry in enumerate(curr_state_spec):

                class_type, (current_focus_pred, current_focus_diff) = spec_entry

                pred = copy.deepcopy(curr_state.pred)
                spec = {}

                # print("curr_state_spec:", curr_state_spec)

                # current_focus_pred, current_focus_diff = curr_state.spec[class_type]
                pred[class_type] = current_focus_pred[0]
                updated_score = curr_score - current_focus_diff[0]

                # print("updated_score:", updated_score)

                if len(current_focus_pred) == 1:
                    if not (spec_idx + 1) >= len(curr_state.spec):
                        spec = dict(curr_state_spec[(spec_idx + 1):])
                else:
                    spec = dict(curr_state_spec[spec_idx:])
                    spec[class_type] = (current_focus_pred[1:], current_focus_diff[1:])

                new_spec = create_spec(pred=pred, score=updated_score, spec=spec)

                # print("new_spec:", new_spec)
                if new_spec is not None:
                    worklist.put(new_spec)

    def query_output(self, task_type, pred_type, key) -> str:
        return self.output[self.format_output_key(task_type, pred_type)][key]

    def print_probs(self, task_type: str, pred_type: str) -> str:
        key = self.format_output_key(task_type, pred_type)
        return ','.join(['{}({:.5f})'.format(label, prob) for label, prob in self.prob_output[task_type][pred_type].items()])

    def add_output(self, task_type: str, pred_type: str, pred: str, gt, probs, vocab=None):
        """
        task_type has to be one of the CLASSES category
        pred_type can either be intent or field
        """
        # print("task_type:", task_type)
        # print("pred_type:", pred_type)
        # we know this benchmark output does not have any gt if any of the intent gt is unknown (note that field can be unknown)
        if pred_type == 'intent' and gt == 'unknown':
            self.contain_gt = False

        key = self.format_output_key(task_type, pred_type)
        probs = np.array(probs[0].cpu())
        so = {'pred': pred, 'gt': gt, 'probs': probs, 'vocab': vocab}
        self.output[key] = so

        if self.prob_output.get(task_type) is None:
            self.prob_output[task_type] = {}

        if pred_type == 'intent':
            self.prob_output[task_type][pred_type] = {label: prob for label, prob in zip(CLASSES[task_type], list(probs))}
        else:
            self.prob_output[task_type][pred_type] = {label: prob for label, prob in zip(vocab, list(probs))}
            self.prob_output['field'] = vocab
            self.added_count += 1

        # print(self.added_count)
        # print(len(CLASSES))

        # not able to compute this until we get all the results
        if self.added_count == len(CLASSES):
            self.postprocess()
            self.top_1_all_pred = ['{}({})'.format(self.query_output(intent_type, 'intent', 'pred'),
                                                   self.query_output(intent_type, 'field', 'pred'))
                                   for intent_type in CLASSES.keys()]
            if self.contain_gt:
                self.all_gt = ['{}({})'.format(self.query_output(intent_type, 'intent', 'gt'),
                                               self.query_output(intent_type, 'field', 'gt'))
                               for intent_type in CLASSES.keys()]
                self.top_1_check_res = 'CORRECT' if set(self.top_1_all_pred) == set(self.all_gt) else 'WRONG'

    def check_top_1(self) -> str:
        return self.top_1_check_res

    def update_intent_field_joint_acc(self, intent_field_joint_acc: Dict[str, int]):
        """
        perform in-place update for the intent_field_joint_acc
        """
        if self.contain_gt:
            for task_type in CLASSES.keys():
                if self.query_output(task_type, 'intent', 'pred') == self.query_output(task_type, 'intent', 'gt') and \
                        self.query_output(task_type, 'field', 'pred') == self.query_output(task_type, 'field', 'gt'):
                    intent_field_joint_acc[task_type] += 1

    def postprocess(self):
        """
        postprocess the self.prob_output
        produce {task_type: [ranked list of intent * field]}
        """
        # print(list(self.prob_output.items()))
        for task_type, preds in self.prob_output.items():
            if task_type == 'field':
                continue

            if task_type == 'plot':
                preds = list(preds['intent'].items())
                preds.sort(key=lambda v: v[1], reverse=True)
                self.joint_prob_output[task_type] = preds
            else:
                new_preds = []
                # print("preds:", preds)
                for intent, prob_i in preds['intent'].items():
                    for field, prob_f in preds['field'].items():
                        if intent == 'null' and not field == 'null':
                            continue
                        new_preds.append(('{}({})'.format(intent, field), prob_i * prob_f))
                new_preds.sort(key=lambda v: v[1], reverse=True)
                self.joint_prob_output[task_type] = new_preds

    def __repr__(self):
        """
        override the origin print method
        """
        output_str = ''
        output_str += "{} {}:\n".format(self.bid, self.top_1_check_res)
        output_str += ">> query: {}\n".format(self.query)
        output_str += ">> pred: {}\n".format(str(self.top_1_all_pred))

        if self.contain_gt:
            output_str += ">> gt: {}".format(str(self.all_gt))
        for task_type in CLASSES.keys():
            output_str += '\n>>> {}: {}'.format(task_type, self.print_probs(task_type, 'intent'))
            output_str += '\n>>> {}-field: {}'.format(task_type, self.print_probs(task_type, 'field'))

        return output_str


class EvalOutput(Output):
    """
    Output object that EvalEngine output (I think this seems out dated)
    """

    def __init__(self, bid, nl, output, correct=None, cost=None, gt=None):
        super().__init__()
        self.bid: str = bid
        self.query: str = nl
        self.output = output
        self.correct: Union[bool, List[bool]] = correct
        self.cost: int = cost
        self.gt = gt

        self.spec = None
        self.asp_prog = None
        self.field_match = None
        self.time = None
        self.goal_type_enumerated = None

        # ablation stuff
        self.new_partial_prog_explored = None
        self.new_solution_explored = None
        self.old_partial_prog_explored = None
        self.old_time = None
        self.old_solution_explored = None

    def check_correct(self, _print=True) -> str:
        if self.correct is None:
            ret, correct = 'OMITTED', False
            printc(_print, self.display_output(correct, omitted=True))

            return ret
        if isinstance(self.correct, bool):
            ret, correct = ('CORRECT', True) if self.correct else ('WRONG', False)
            printc(_print, self.display_output(correct))

            return ret
        else:
            assert isinstance(self.correct, list)
            if len(self.correct) == 0:
                return ('WRONG', False, False)
            ret, correct, topk = ('CORRECT', True, False) if self.correct[0] else \
                ('CORRECT-5', False, True) if any(self.correct[:5]) else \
                    ('CORRECT-10', False, True) if any(self.correct[:10]) else \
                        ('CORRECT-K', False, True) if any(self.correct) else \
                            ('WRONG', False, False)

            printc(_print, self.display_output(correct, topk))

            return ret

    def display_output(self, eval_res, topk=False, omitted=False) -> str:

        output_str = ''

        if omitted:
            output_str += '>> {} omitted\n'.format(self.bid)
        elif not eval_res:
            output_str += '>> {} wrong\n'.format(self.bid)
        elif topk:
            output_str += '>> {} top k correct\n'.format(self.bid)
        else:
            assert eval_res
            output_str += '>> {} top 1 correct\n'.format(self.bid)
        output_str += '>>> gt: {}\n'.format(self.gt)

        if isinstance(self.output, list):
            if len(self.output) == 0:
                output_str += '>>> output: empty\n'
            else:
                output_str += '>>> output: {}\n'.format(self.output[0])
        else:
            output_str += '>>> output: {}\n'.format(self.output)

        return output_str

    def __repr__(self):
        repr_str = ""
        repr_str += "{} {}:\n".format(self.bid, self.check_correct())
        repr_str += ">>> time: {}\n".format(self.time)
        repr_str += ">>> old time: {}\n".format(self.old_time)

        return repr_str


class DatavisOutputPrev(EvalOutput):
    def __init__(self, bid, nl, output):
        super().__init__(bid, nl, output)

        # extra stuff for EvalSynthPrev
        self.shortest_output = None
        self.shortest_output_correct = None

        self.soft_output = None
        self.soft_output_correct = None


class DatavisOutput(EvalOutput):
    def __init__(self, bid, nl, output):
        super().__init__(bid, nl, output)

        # TODO additional information that we want to analyse


class TrinityOutput(EvalOutput):
    """
    Output object that EvalEngine output (I think this seems out dated)
    """

    def __init__(self, bid, nl, output, correct=None, cost=None, gt=None):
        super().__init__(bid, nl, output)
        self.bid: str = bid
        self.query: str = nl
        self.output = output
        self.correct: Union[bool, List[bool]] = correct
        self.cost: int = cost
        self.gt = gt

        self.spec = None
        self.field_match = None
        self.time = None
        self.goal_type_enumerated = None

    def display_output(self, eval_res, topk=False, omitted=False) -> str:

        output_str = ''

        if omitted:
            output_str += '>> {} omitted\n'.format(self.bid)
        elif not eval_res:
            output_str += '>> {} wrong\n'.format(self.bid)
        elif topk:
            output_str += '>> {} top k correct\n'.format(self.bid)
        else:
            assert eval_res
            output_str += '>> {} top 1 correct\n'.format(self.bid)
        output_str += '>>> gt: {}\n'.format(self.gt)

        if isinstance(self.output, list):
            if len(self.output) == 0:
                output_str += '>>> output: empty\n'
            else:
                output_str += '>>> output: {}\n'.format(self.output[0])
        else:
            output_str += '>>> output: {}\n'.format(self.output)

        return output_str
