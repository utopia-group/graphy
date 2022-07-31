import os
import pickle
from typing import List, Dict

import torch
from torch import nn

from lib.config.config import GT_DIR, CACHED_NEURAL_PREDICTION_DIR, get_cached_prediction_ts
from lib.eval.benchmark import Benchmark
from lib.eval.output import NeuralParserOutput
from lib.neural_parser.data_processor import create_data_loader
from lib.neural_parser.labels import CLASSES
from lib.neural_parser.models import IntentFieldFinder
from lib.neural_parser.tokenizer import Tokenizer
from lib.query_parser import QueryParser
from lib.utils.misc_utils import get_model_path, get_saved_prediction_pkl_path
from lib.utils.neural_model_utils import eval_model
from lib.utils.preprocess_utils import read_spec_gt, format_single_data, df_data


class QueryNeuralParser(QueryParser):
    """
    NeuralQueryParser that uses a slot-filling model
    If there is gpu present, we run the model, if no gpu present, we run the cached results locally (for testing purpose)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.cached_predictions: Dict[str, NeuralParserOutput] = {}
        if args.device is not 'cpu':
            self.all_parser_models = {}
            self.tokenizer = Tokenizer(self.args.bert_model)
            for task_type in CLASSES.keys():
                model_path = get_model_path(task_type, args)
                if 'cpu' in str(args.device):
                    checkpoint = torch.load(
                        model_path, map_location=args.device)
                else:
                    checkpoint = torch.load(model_path)
                model = IntentFieldFinder(self.args, len(CLASSES[task_type]))
                model.load_state_dict(checkpoint)
                self.all_parser_models[task_type] = model
            print('Query parser model loaded')

            # load previously cached results if there exists one
            self.cached_predictions_path = '{}/cache.pkl'.format(
                CACHED_NEURAL_PREDICTION_DIR)
            # NOTE: I decide not to use this because this mess up with the cache
            # args.ts = CACHED_PREDICTION_TS
            # with open(get_saved_prediction_pkl_path(args), 'rb') as f:
            #     self.cached_predictions = pickle.load(f)
        else:
            # read my local predicted file
            args.ts = get_cached_prediction_ts(args.test_set)

            # let's first read the prediction file
            with open(get_saved_prediction_pkl_path(args), 'rb') as f:
                test_predictions = pickle.load(f)
                print("test predictions loaded")

            # convert all instance of test_predictions to neuralparseroutput format
            self.cached_predictions = {}
            for bid, pred in test_predictions.items():
                self.cached_predictions[bid] = NeuralParserOutput(
                    bid, 'nl omitted', pred)

    def load_cache(self):
        if os.path.isfile(self.colname_cache_path):
            with open(self.colname_cache_path, 'rb') as f:
                self.colname_cache = pickle.load(f)
        if os.path.isfile(self.cached_predictions_path):
            with open(self.cached_predictions_path, 'rb') as f:
                self.cached_predictions = pickle.load(f)

    def save_cache(self):
        with open(self.colname_cache_path, 'wb') as f:
            pickle.dump(self.colname_cache, f)

        with open(self.cached_predictions_path, 'wb') as f:
            pickle.dump(self.cached_predictions, f)

    def parse(self, b: Benchmark, field_only=False) -> NeuralParserOutput:
        """
        parse the benchmark
        technically we can run in batch, but now let's make this a TODO
        if the benchmark result is already in the cached value, we just return the value directly
        """
        if b.bname in self.cached_predictions:
            return self.cached_predictions[b.bname]

        # assert not self.args.device == 'cpu'

        # TODO: i think it would be more efficient to separate this method (also all other methods) out
        def preprocess(b: Benchmark, colnames: List):
            def format_data():
                benchmark_dict = b.get_neural_format()
                if not b.bname.startswith('user'):
                    # we can actually read the gt
                    gtname = '{}.spec'.format(benchmark_dict['gtname']) if not benchmark_dict['gtname'] == '' else '{}-{}.spec'.format(
                        benchmark_dict['data'], benchmark_dict['id'])
                    gt_info = read_spec_gt(GT_DIR, gtname, self.args.bert_model)
                else:
                    gt_info = None
                return [format_single_data(benchmark_dict, colnames, gt_info=gt_info, model_name=self.args.bert_model)]

            fd = format_data()
            fd = df_data(self.args, fd, self.tokenizer)

            return fd
        colnames, _ = self.parse_colnames(b)

        # preprocess the query
        formatted_data = preprocess(b, colnames)
        all_predictions: Dict[str, NeuralParserOutput] = {}
        for task_type in CLASSES.keys():
            model = self.all_parser_models[task_type].to(self.args.device)
            model.eval()
            intent_loss_fn = nn.CrossEntropyLoss().to(self.args.device)
            field_loss_fn = nn.CrossEntropyLoss().to(self.args.device)

            test_data_loader = create_data_loader(
                formatted_data, self.tokenizer, self.args.input_dim, 1, task_type)
            _, _, _, _ = eval_model(self.args, task_type, model, test_data_loader, intent_loss_fn, field_loss_fn, len(formatted_data),
                                    inference=True, overall_output=all_predictions)
        for bid, pred in all_predictions.items():
            print(pred)

        assert len(all_predictions) == 1
        self.cached_predictions[b.bname] = all_predictions[b.bname]
        return all_predictions[b.bname]
