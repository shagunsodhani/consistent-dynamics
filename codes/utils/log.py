import json
import logging
from pathlib import Path

import numpy as np

from codes.utils.config import get_config
from codes.utils.util import running_mean

TIME = "time"
CONFIG = "config"
METRIC = "metric"
METADATA = "metadata"
TYPE = "type"
MESSAGE = "message"
PRINT = "print"
LOSS = "loss"
BATCH_SIZE = "batch_size"
TIME_TAKEN = "time_taken"
BATCH_INDEX = "batch_index"
MODE = "mode"
EPOCH_INDEX = "epoch_index"
BEST_EPOCH_INDEX = "best_epoch_index"
ITERATION_INDEX = "iteration_index"
TRAIN = "train"
VAL = "val"
TEST = "test"
LOG = "log"
BLEU = "bleu"
ENTOVERLAP = "entity_overlap"
RELOVERLAP = "rel_overlap"
FILE_PATH = "file_path"
DEFAULT_LOGGER = "defualt_logger"
IMAGINATION_LOG_LIKLIHOOD = "imagination_log_likelihood"
IMITATION_LEARNING_LOSS = "imitation_learning_loss"
CONSISTENCY_LOSS = "consistency_loss"
DISCRIMINATOR_LOSS = "discriminator_loss"

log_types = [TIME, CONFIG, METRIC, METADATA]


def _format_log(log):
    return json.dumps(log)


def write_log(log):
    '''This is the default method to write a log. It is assumed that the log has already been processed
     before feeding to this method'''
    get_logger().warning(log)


def read_log(log):
    '''This is the single point to read any log message from the file since all the log messages are persisted as jsons'''
    try:
        data = json.loads(log)
    except json.JSONDecodeError as e:
        data = {
        }
    if (data["type"] == "print"):
        data = {

        }
    return data


def _format_custom_logs(keys=[], raw_log={}, _type=METRIC):
    log = {}
    if (keys):
        for key in keys:
            if key in raw_log:
                log[key] = raw_log[key]
    else:
        log = raw_log
    log[TYPE] = _type
    return _format_log(log), log


def write_message_logs(message):
    kwargs = {MESSAGE: message}
    log, _ = _format_custom_logs(keys=[], raw_log=kwargs, _type=PRINT)
    write_log(log)


def write_config_log(config):
    config[TYPE] = CONFIG
    log = _format_log(config)
    write_log(log)


def write_metric_logs(**kwargs):
    log, log_d = _format_custom_logs(keys=[LOSS, BATCH_SIZE, IMAGINATION_LOG_LIKLIHOOD,
                                           IMITATION_LEARNING_LOSS,
                                           BATCH_INDEX, EPOCH_INDEX, TIME_TAKEN, MODE,
                                           DISCRIMINATOR_LOSS, CONSISTENCY_LOSS], raw_log=kwargs, _type=METRIC)
    write_log(log)


def write_metadata_logs(**kwargs):
    log, _ = _format_custom_logs(keys=[BEST_EPOCH_INDEX], raw_log=kwargs, _type=METADATA)
    write_log(log)


def pprint(config):
    print(json.dumps(config, indent=4))


def parse_log_file(log_file_path):
    logs = {}
    running_metrics = {}
    metric_keys = [LOSS, TIME_TAKEN, IMAGINATION_LOG_LIKLIHOOD, IMITATION_LEARNING_LOSS, DISCRIMINATOR_LOSS, CONSISTENCY_LOSS]
    top_level_keys = [CONFIG, METADATA]
    for key in top_level_keys:
        logs[key] = []
    for mode in [TRAIN, VAL, TEST]:
        logs[mode] = {}
        running_metrics[mode] = {}
        for key in metric_keys:
            logs[mode][key] = []
            running_metrics[mode][key] = []

    with open(log_file_path, "r") as f:
        for line in f:
            data = read_log(line)
            if (data):
                _type = data[TYPE]
                if (_type == CONFIG):
                    logs[_type].append(data)
                elif (_type == METADATA):
                    logs[METADATA].append(data)
                else:
                    if (not (BATCH_INDEX in data)):
                        epoch_index = data[EPOCH_INDEX]
                        mode = data[MODE]
                        for key in metric_keys:
                            if key in data:
                                logs[mode][key].append(data[key])
                    # epoch_index = data[EPOCH_INDEX]
                    # batch_index = data[BATCH_INDEX]
                    # mode = data[MODE]
                    # if (batch_index == 0 and epoch_index > 0):
                    # new epoch
                    # for key in metric_keys:
                    #     if(key==TIME_TAKEN):
                    #         logs[mode][key].append(sum(running_metrics[mode][key]))
                    #     else:
                    #         logs[mode][key].append(np.mean(np.asarray(running_metrics[mode][key])))
                    #     running_metrics[mode][key] = []
                    # for key in metric_keys:
                    #     running_metrics[mode][key].append(data[key])
    logs = _transform_logs(logs)
    return logs


def _transform_logs(logs):
    keys_to_transform = set([TRAIN, VAL, TEST])
    for key in logs:
        if (key in keys_to_transform):
            metric_dict = {}
            for metric in logs[key]:
                if (logs[key][metric]):
                    metric_dict[metric] = running_mean(np.asarray(logs[key][metric]), 100)
            logs[key] = metric_dict
    return logs


def get_config_from_appid(app_id):
    config = get_config(read_cmd_args=False)
    log_file_path = config[LOG][FILE_PATH]
    logs_dir = "/".join(log_file_path.split("log.txt")[0].split("/")[:-2])
    log_file_path = Path(logs_dir, app_id, "log.txt")
    logs = parse_log_file(log_file_path)
    return logs[CONFIG][0]


def get_logger():
    return logging.getLogger(DEFAULT_LOGGER)


def set_logger(config):
    '''Modified from https://docs.python.org/3/howto/logging-cookbook.html'''
    logger = logging.getLogger(DEFAULT_LOGGER)
    logger.setLevel(logging.INFO)
    # create file handler which logs all the messages
    fh = logging.FileHandler(config.log.file_path)
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
