import time
from typing import List, Dict

from lib import nl4dv
from lib.config.config import DEPENDENCY_PARSER_CONFIG
from lib.eval.eval import EvalEngine
from lib.eval.output import EvalOutput
from lib.utils.eval_utils import clean_vlspec, check_spec_equiv


class EvalNL4DV(EvalEngine):
    """
    EvalEngine for the NL4DV baseline
    """
    def __init__(self):
        super().__init__()
        self.nl4dv_instance = nl4dv.NL4DV(verbose=False, debug=False)
        self.nl4dv_instance.set_dependency_parser(config=DEPENDENCY_PARSER_CONFIG)

        self.incorrect_attributes = []
        self.incorrect_task = []
        self.incorrect_visualization = []
        self.output_none = []
        self.k = 10

    def run_nl4dv(self, data_space, benchmark) -> List:
        if not self.nl4dv_instance.data_genie_instance.dataname == benchmark.dataname:
            self.nl4dv_instance.data_genie_instance.set_data(data_value=data_space, dataname=benchmark.dataname)
        response = self.nl4dv_instance.analyze_query(query=benchmark.nl)
        # take top-1
        if len(response['visList']) == 0:
            vlspec = None
        else:
            vlspec = []
            for i in range(self.k):
                if i >= len(response['visList']):
                    break
                vlspec.append(clean_vlspec(response['visList'][i]['vlSpec'], change_field=True))

        # print("vlspec before cleaning:", vlspec)
        return vlspec

    def eval(self, benchmark, new_old_ablation=False) -> EvalOutput:
        data_underscore, data_space = self.get_data(benchmark)

        # print(data_underscore[0])
        # print(data_space[0])

        vlspec_gt = self.read_gt(benchmark)
        start = time.time()
        vlspec_synth : List = self.run_nl4dv(data_space, benchmark)
        end = time.time()
        # print("vlspec_synth:", vlspec_synth)

        bid = "{}-{}".format(benchmark.dataname, benchmark.bname)

        if vlspec_synth is None:
            self.output_none.append(bid)
            eval_output = EvalOutput(bid, benchmark.nl, vlspec_synth, correct=[False], gt=vlspec_gt, cost=0)
            eval_output.time = end - start
            return eval_output

        output = EvalOutput(bid, benchmark.nl, vlspec_synth)
        output.correct = check_spec_equiv(data_underscore, vlspec_gt, vlspec_synth)
        output.gt = vlspec_gt
        output.time = end-start
        output.cost = 0

        # The following are evaluate some sub-task, we comment it out since it is hard to adapt to top-k setting
        # gt_fields = parse_vlspec_fields(vlspec_gt)
        # output_fields = parse_vlspec_fields(vlspec_synth)
        # field_eval_res = (set(gt_fields) == set(output_fields))
        # # overall_results[bid]["eval_nl4dv_fields"] = field_eval_res
        # if not field_eval_res:
        #     self.incorrect_attributes.append(bid)

        # gt_tasks = parse_vlspec_task(vlspec_gt)
        # output_tasks = parse_vlspec_task(vlspec_synth)
        # task_eval_res = (set(gt_tasks) == set(output_tasks))
        # # overall_results[bid]["eval_nl4dv_task"] = task_eval_res
        # if not task_eval_res:
        #     self.incorrect_task.append(bid)

        # gt_vis = parse_vlspec_visualization(vlspec_gt)
        # output_vis = parse_vlspec_visualization(vlspec_synth)
        # vis_eval_res = (set(gt_vis) == set(output_vis))
        # # overall_results[bid]["eval_nl4dv_vis"] = vis_eval_res
        # if not vis_eval_res:
        #     self.incorrect_visualization.append(bid)

        return output

    def write_to_overall_output(self, res: EvalOutput) -> Dict:
        output_to_write = {}
        output_to_write['query'] = res.query
        output_to_write['spec'] = res.spec
        output_to_write['gt'] = str(res.gt)
        output_to_write['output({})'.format(self.get_name())] = str('\n'.join([str(o) for o in res.output])) \
            if res.output is not None else "None"
        output_to_write['top-1 correct'] = res.correct[0]
        output_to_write['top-5 correct'] = any(res.correct[:5])
        output_to_write['top-10 correct'] = any(res.correct[:10])
        output_to_write['time({})'.format(self.get_name())] = res.time
        output_to_write['cost({})'.format(self.get_name())] = res.cost
        return output_to_write

    def print_additional_output(self):
        print("incorrect attributes: ", self.incorrect_attributes)
        print("incorrect tasks: ", self.incorrect_task)
        print("incorrect visualization: ", self.incorrect_visualization)
        print("output none: ", self.output_none)
