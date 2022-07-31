import argparse
from lib.neural_parser.params import CAR_PARAMS, SUPERSTORE_PARAMS, MOVIE_PARAMS

parser = argparse.ArgumentParser(description='main.py')

"""
The following are all the neural-model related arguments 
"""
parser.add_argument("--data_path", type=str,
                    default="eval/datavis", help="path to data")
parser.add_argument("--dataset", type=str,
                    default="datavis", help="path to data")
parser.add_argument("--test_set", type=str,
                    default="cars", help="the test set")

parser.add_argument("--process_data", action='store_true', default=False)
parser.add_argument("--held_out", action='store_true', default=True)
parser.add_argument("--oracle_field_test", action='store_true', default=False)
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--inference", action='store_true', default=False)
parser.add_argument('--check_baseline', action='store_true', default=True)

parser.add_argument("--gpu", type=str, default="0", help="gpu id")
parser.add_argument('--seed', type=int, default=233,
                    help='RNG seed (default = 0)')
parser.add_argument('--epochs', type=int, default=30,
                    help='num epochs to train for')
parser.add_argument('--lr', type=float, default=.00002)
parser.add_argument('--warm_up_ratio', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--clip_grad', type=float, default=1.0)
parser.add_argument('--bert_model', type=str, default='bert-large-uncased')

parser.add_argument('--model_params', type=dict, default=CAR_PARAMS)

# regarding model saving
parser.add_argument('--model_id', type=str, default=None,
                    help='model identifier')
parser.add_argument('--saving_from', type=int, default=20,
                    help='saving from - epoch')
parser.add_argument('--saving_interval', type=int,
                    default=10, help='saving iterval')
parser.add_argument('--ts', type=str, default=None,
                    help='timestamp for output saving')

parser.add_argument('--decoder_len_limit', type=int,
                    default=100, help='output length limit of the decoder')
parser.add_argument('--input_dim', type=int, default=100,
                    help='input vector dimensionality')
parser.add_argument('--output_dim', type=int, default=100,
                    help='output vector dimensionality')
parser.add_argument('--hidden_size', type=int, default=150,
                    help='hidden state dimensionality')

parser.add_argument('--max_field_num', type=int, default=10,
                    help='max number of field allowed per benchmark')
parser.add_argument('--max_token_num_per_field', type=int,
                    default=5, help='max number of token per field allowed')

# Hyperparameters for the encoder -- feel free to play around with these!
parser.add_argument('--no_bidirectional', dest='bidirectional',
                    default=True, action='store_false', help='bidirectional LSTM')
parser.add_argument('--reverse_input', dest='reverse_input',
                    default=False, action='store_true')
parser.add_argument('--emb_dropout', type=float,
                    default=0.2, help='input dropout rate')
parser.add_argument('--rnn_dropout', type=float, default=0.2,
                    help='dropout rate internal to encoder RNN')

parser.add_argument('--no_early_stop', dest='early_stop',
                    default=True, action='store_false', help='early stopping')
parser.add_argument('--early_stop_patience', type=int,
                    default=5, help='early stopping')

parser.add_argument('--log_interval', type=int,
                    default=5, help='wandb log interval')

args, unknown = parser.parse_known_args()


def set_args(test_set):
    if test_set == 'cars':
        args.test_set = 'car'
        args.model_params = CAR_PARAMS
    elif test_set == 'movies':
        args.test_set = 'movies'
        args.model_params = MOVIE_PARAMS
    else:
        assert test_set == 'superstore'
        args.test_set = 'superstore'
        args.model_params = SUPERSTORE_PARAMS

    args.params = {'plot': (args.model_params["plot_epochs"], args.model_params["plot_lr"], args.model_params["plot_wr"]),
                   'color': (args.model_params["color_epochs"], args.model_params["color_lr"], args.model_params["color_wr"]),
                   'column': (args.model_params["column_epochs"], args.model_params["column_lr"], args.model_params["column_wr"]),
                   'count': (args.model_params["count_epochs"], args.model_params["count_lr"], args.model_params["count_wr"]),
                   'mean': (args.model_params["mean_epochs"], args.model_params["mean_lr"], args.model_params["mean_wr"]),
                   'sum': (args.model_params["sum_epochs"], args.model_params["sum_lr"], args.model_params["sum_wr"])}
