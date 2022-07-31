import copy
from typing import Union, Iterator

from lib.neural_parser.labels import CLASSES
from lib.prev_synthesizer.spec import SpecProg, prog_hash
from lib.utils.enum_utils import get_score, get_pred_and_diff
from lib.utils.pq import PriorityQueue


class Enumerator:
    def __init__(self):
        self.worklist = PriorityQueue(get_score)
        self.enumerated_progs = {}

    def reinit(self):
        self.worklist = PriorityQueue(get_score)
        self.enumerated_progs = {}

    def create_prog(self, pred, score, spec) -> Union[SpecProg, None]:

        prog_id = prog_hash(pred)
        if prog_id in self.enumerated_progs:
            return None

        new_prog = SpecProg(pred=pred, score=score, spec=spec)
        self.enumerated_progs[prog_id] = ''
        return new_prog

    def init_prog(self, processed_prediction: dict):
        all_classes_preds = dict([(class_type, get_pred_and_diff(processed_prediction, class_type)) for class_type in CLASSES])
        prog_init = self.create_prog(pred=dict([(class_type, v[0][0]) for class_type, v in all_classes_preds.items()]),
                                     score=sum([v[0][0][1] for v in all_classes_preds.values()]),
                                     spec=dict([(class_type, (v[0][1:], v[1])) for class_type, v in all_classes_preds.items()]))

        assert prog_init is not None

        return prog_init

    def enumerate(self, processed_prediction: dict, limit: int = -1) -> list:

        enumerated_list = []
        enumerator = self.enumerate_yield(processed_prediction, limit)

        try:
            prog = next(enumerator)
            enumerated_list.append(prog)
        except StopIteration:
            pass

        return enumerated_list

    def enumerate_yield(self, processed_prediction: dict, limit: int = -1) -> Iterator[SpecProg]:

        enumerated_list = []

        prog_init = self.init_prog(processed_prediction)
        self.worklist.put(prog_init)

        while not self.worklist.is_empty():
            curr_state = self.worklist.pop()
            enumerated_list.append(curr_state)
            yield curr_state

            # print("curr_state:", curr_state)

            if limit > -1:
                if len(enumerated_list) > limit:
                    raise StopIteration

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
                        spec = dict(curr_state_spec[(spec_idx+1):])
                else:
                    spec = dict(curr_state_spec[spec_idx:])
                    spec[class_type] = (current_focus_pred[1:], current_focus_diff[1:])

                new_prog = self.create_prog(pred=pred, score=updated_score, spec=spec)

                # print("new_prog:", new_prog)
                if new_prog is not None:
                    self.worklist.put(new_prog)

        # return enumerated_list

    def enumerate_slow(self, processed_prediction: dict, limit: int = -1) -> list:

        enumerated_list = []

        prog_init = self.init_prog(processed_prediction)
        self.worklist.put(prog_init)

        while not self.worklist.is_empty():
            curr_state = self.worklist.pop()
            enumerated_list.append(curr_state)

            if limit > -1:
                if len(enumerated_list) > limit:
                    break

            curr_score = curr_state.score

            for class_type in curr_state.spec.keys():
                pred = copy.deepcopy(curr_state.pred)
                spec = copy.deepcopy(curr_state.spec)

                current_focus_pred, current_focus_diff = curr_state.spec[class_type]
                pred[class_type] = current_focus_pred[0]
                updated_score = curr_score - current_focus_diff[0]

                if len(current_focus_pred) == 1:
                    del spec[class_type]
                else:
                    spec[class_type] = (current_focus_pred[1:], current_focus_diff[1:])

                new_prog = self.create_prog(pred=pred, score=updated_score, spec=spec)
                if new_prog is not None:
                    self.worklist.put(new_prog)

        return enumerated_list
