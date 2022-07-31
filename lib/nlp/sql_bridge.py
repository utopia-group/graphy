import os

from lib.nlp.TabularSemanticParsing.src.parse_args import args
from lib.nlp.TabularSemanticParsing.src.trans_checker.args import args as cs_args
from lib.nlp.TabularSemanticParsing.src.demos.demos import Text2SQLWrapper
from lib.nlp.TabularSemanticParsing.src.data_processor.schema_graph import SchemaGraph
import lib.nlp.TabularSemanticParsing.src.utils.utils as utils

class RunNL2SQL(object):
    def __init__(self):
        self.args = args_init()
        self.data_dir = 'eval/data'
        self.data_to_nl2sql_instance = {}

    def load_nl2sql_instance(self, db_name, benchmark_set=None):
        
        if benchmark_set is not None:
            db_folder_path = os.path.join(self.data_dir, "{}_{}".format(db_name, benchmark_set))
        else:
            db_folder_path = os.path.join(self.data_dir, db_name)

        db_path = os.path.join(db_folder_path, '{}.sqlite'.format(db_name))
        schema = SchemaGraph(db_name, db_path=db_path)
        in_csv = os.path.join(db_folder_path, '{}.csv'.format(db_name))
        in_type = os.path.join(db_folder_path, '{}.types'.format(db_name))
        schema.load_data_from_csv_file(in_csv, in_type)
        schema.pretty_print()

        t2sql = Text2SQLWrapper(self.args, cs_args, schema)

        return t2sql
    
    def run_nl2sql(self, db_name, nl, benchmark_set=None):

        db_name = db_name.lower()

        if not db_name in self.data_to_nl2sql_instance:
            self.data_to_nl2sql_instance[db_name] = self.load_nl2sql_instance(db_name, benchmark_set)
        
        nl2sql_instance = self.data_to_nl2sql_instance[db_name]
        output = nl2sql_instance.process(nl, db_name)
        translatable = output['translatable']
        sql_query = output['sql_query']
        confusion_span = output['confuse_span']
        replacement_span = output['replace_span']

        return translatable, sql_query, confusion_span, replacement_span
        

def args_init():
    args.data_dir = "data/spider"
    args.data_dir = "data/spider/database"
    args.dataset_name = "spider"
    args.question_split = True
    args.question_only = True
    args.denormalize_sql = True
    args.table_shuffling = True
    args.use_lstm_encoder = True
    args.use_meta_data_encoding = True
    args.sql_consistency_check = True
    args.use_picklist = True
    args.anchor_text_match_threshold = 0.85
    args.top_k_picklist_matches = 0.85
    args.process_sql_in_execution_order = True
    args.num_random_tables_added = 0
    args.save_best_model_only = True
    args.schema_augmentation_factor = 1
    args.data_augmentation_factor = 1
    args.vocab_min_freq = 0
    args.text_vocab_min_freq = 0
    args.program_vocab_min_freq = 0
    args.num_values_per_field = 0
    args.max_in_seq_len = 152
    args.max_out_seq_len = 60
    args.model = 'bridge'
    args.num_steps = 100000
    args.curriculum_interval = 0
    args.num_peek_steps  = 1000     
    args.num_accumulation_steps = 2     
    args.train_batch_size = 16
    args.dev_batch_size  = 24
    args.encoder_input_dim  = 1024
    args.encoder_hidden_dim = 400
    args.decoder_input_dim = 400
    args.num_rnn_layers = 1
    args.num_const_attn_layers = 0
    args.emb_dropout_rate = 0.3
    args.pretrained_lm_dropout_rate = 0
    args.rnn_layer_dropout_rate = 0
    args.rnn_weight_dropout_rate = 0
    args.cross_attn_dropout_rate = 0
    args.cross_attn_num_heads = 8
    args.res_input_dropout_rate = 0.2
    args.res_layer_dropout_rate = 0
    args.ff_input_dropout_rate = 0.4
    args.ff_hidden_dropout_rate = 0.0
    args.pretrained_transformer = 'bert-large-uncased'
    args.bert_finetune_rate = 0.00006
    args.learning_rate = 0.0005
    args.learning_rate_scheduler = 'inverse-square'
    args.trans_learning_rate_scheduler = 'inverse-square'
    args.warmup_init_lr = 0.0005
    args.warmup_init_ft_lr = 0.00003
    args.num_warmup_steps = 4000
    args.grad_norm = 0.3    
    args.decoding_algorithm = 'beam-search'
    args.beam_size = 16
    args.bs_alpha = 1.05
    args.gpu = 0
    args.demo_db = 'movies'
    args.checkpoint_path = 'model/test/model-best.tar'
    args.model_id = utils.model_index[args.model]

    return args
