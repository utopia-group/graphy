"""
This file stores some general config values for scripts
"""
import os

DEBUG_PRINT = False

PROFILING = False

GT_DIR = 'eval/gt'

CACHE_PATH = '.cache'
CACHE_IN_MEMORY = True
IN_MEMORY_CACHE_SIZE = 500000

CAR_CACHED_PREDICTION_TS = '20210729-095213'
SUPERSTORE_CACHED_PREDICTION_TS = '20211108-104600'
MOVIES_CACHED_PREDICTION_TS = 'temp'
CACHED_COLUMN_DIR = '{}/query_parser'.format(CACHE_PATH)
CACHED_NEURAL_PREDICTION_DIR = '{}/query_neural_parser'.format(CACHE_PATH)

if not os.path.isdir(CACHED_COLUMN_DIR):
    os.makedirs(CACHED_COLUMN_DIR)

if not os.path.isdir(CACHED_NEURAL_PREDICTION_DIR):
    os.makedirs(CACHED_NEURAL_PREDICTION_DIR)

DEPENDENCY_PARSER_CONFIG = {'name': 'corenlp', 'model': os.path.join("jars", "stanford-english-corenlp-2018-10-05-models.jar"),
                            'parser': os.path.join("jars", "stanford-parser.jar")}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_cache_path(new_cache_path):
    global CACHE_PATH
    CACHE_PATH = new_cache_path


def set_debug_print(b: bool):
    global DEBUG_PRINT
    DEBUG_PRINT = b


def get_cached_prediction_ts(dataname):
    if dataname.lower() == 'car':
        return CAR_CACHED_PREDICTION_TS
    elif dataname.lower() == 'movies':
        return MOVIES_CACHED_PREDICTION_TS
    elif dataname.lower() == 'superstore':
        return SUPERSTORE_CACHED_PREDICTION_TS
    else:
        raise FileNotFoundError
