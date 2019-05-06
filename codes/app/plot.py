import matplotlib

# matplotlib.use('agg')
matplotlib.use('TkAgg')

import numpy as np

import matplotlib.pyplot as plt
# from pylab import *
plt.isinteractive()
import json

from codes.utils.util import set_seed, timing, make_dir

from codes.utils.config import get_config, get_sample_config
from codes.utils.log import parse_log_file

from codes.utils.argument_parser import argument_parser
from copy import deepcopy
from functional import seq

import os
from time import sleep

USE_DATABASE = False
PROJECT = "interactive-generative-agents"

if USE_DATABASE:
    import sys
    from database.db import Database
    from codes.utils.spreadsheet import log_to_spreadsheet

def plot(data, mode, key, plot_dir):
    plt.plot(data)
    xlabel = "Number of Batches"
    ylabel = key
    title = key + " for {} mode".format(mode)
    if (key == "time_taken"):
        title = key + " for {} mode".format(mode)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    # plt.show()
    path = os.path.join(plot_dir, "_".join(title.split()) + ".png")
    plt.savefig(path)
    plt.clf()


def plot_multiple(data_list, legend_list, mode, key):
    for data, legend in zip(data_list, legend_list):
        plt.plot(data, label=legend)
    xlabel = "Number of Batches"
    ylabel = key
    title = key + " for {} mode".format(mode)
    if (key == "time_taken"):
        title = key + " for {} mode".format(mode)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(0, 6)
    plt.legend()
    plt.title(title)
    # plt.show()
    plt.savefig("temp.png")
    # path = os.path.join(plot_dir, "_".join(title.split()) + ".png")
    # plt.savefig(path)
    # plt.clf()

def get_logs_from_config_id(config_id):
    config = get_config(config_id=config_id, should_make_dir=False)
    bootstrap(config)
    log_file_path = config.log.file_path
    logs = parse_log_file(log_file_path=log_file_path)
    return logs

def run(config_dict=None, db=None, config_id=None):
    if not config_dict:
        config_dict = get_config(config_id)
        bootstrap(config_dict)

    logs = get_logs_from_config_id(config_dict.general.id)

    if not logs["config"]:
        raise ValueError("Empty log file")
    metric_keys = set(["imagination_log_likelihood", "loss", "time_taken"])
    plot_dir = config_dict.plot.base_path
    for mode in ["train", "val"]:
        for key in logs[mode]:
            if key in metric_keys:
                plot(logs[mode][key], mode, key, plot_dir)
    if(USE_DATABASE):
        best_metric_logs = log_to_spreadsheet(logs)
        try:
            if (not db):
                db = Database(connect_to_firebase=False)
            db.update_job(job_id=config_dict.general.id,
                          project_id=PROJECT,
                          data_to_update={
                              "status": "recorded"
                          })
        except FileNotFoundError as f:
            print("Could not log results to journal")
        return best_metric_logs
    else:
        return None


def bootstrap(config_dict):
    print(config_dict.log)
    set_seed(seed=config_dict.general.seed)

def make_remote_config(sample_config, app_id):
    remote_config = deepcopy(sample_config)
    remote_config.general.id = app_id
    # Log Params
    key = "file_path"
    remote_config.log[key] = os.path.join(remote_config.general.base_path,
                                           "logs", remote_config.general.id)
    remote_config.log[key] = os.path.join(remote_config.log[key], "log.txt")

    # Plot Params
    remote_config.plot.base_path = os.path.join(remote_config.general.base_path,
                                                      "plots", remote_config.general.id)

    make_dir(remote_config.plot.base_path)

    return remote_config


def run_all_synced():
    '''Method to run all the tasks that have been completed'''
    sample_config = get_sample_config()
    db = Database(connect_to_firebase=False)
    appid_list = map(
        lambda x: x["id"], db.list_jobs(status="synced", project=PROJECT))
    metrics = []
    flag = False
    appid_list_to_print = []
    appid_list = list(map(lambda x: str(x), appid_list))
    for app_id in appid_list:
        flag = True
        print(app_id)
        remote_config = make_remote_config(sample_config, app_id)
        # print(app_id)
        try:
            best_metric_logs = run(config_dict=remote_config, db=db)
            metrics.append(best_metric_logs)
        except ValueError as e:
            print("Error for {}. Message: {}".format(app_id, e))
            continue
        appid_list_to_print.append(app_id)
        sleep(2)
    if (flag):
        print(appid_list_to_print)
        summary_file_path = os.path.join(sample_config.general.base_path, "summary.json")
        with open(summary_file_path, "w") as f:
            f.write(json.dumps(metrics, indent=4))


def run_multiple(config_dicts=None, db=None, config_ids=None):
    if not config_dicts:
        config_dicts = list(map(lambda config_id: get_config(config_id=config_id, should_make_dir=False), config_ids))
    list(map(lambda _dict: bootstrap(_dict), config_dicts))
    legend_list = list(map(lambda _dict: _dict.general.id, config_dicts))
    log_file_paths = list(map(lambda _dict: _dict.log.file_path, config_dicts))
    logs_list = list(map(parse_log_file, log_file_paths))
    metric_keys = set(["imagination_log_likelihood"])
    for mode in ["val"]:
        for key in logs_list[0][mode]:
            if key in metric_keys:
                plot_multiple(map(lambda x: x[mode][key], logs_list),
                              legend_list,mode, key)

def make_variance_plot(grouped_config_ids):
    def _plot_variance_curve(data, label):
        for i in data:
            print(i.shape)
        mean_data = np.mean(data, axis=0)
        std_data = np.std(data, axis=0)
        x = range(len(mean_data))
        plt.plot(x, mean_data, label=label)
        plt.fill_between(x, mean_data + std_data, mean_data - std_data, alpha=0.2)

    mode = "val"
    metric_keys = set(["imagination_log_likelihood", "imitation_learning_loss", "discriminator_loss", "consistency_loss"])
    for metric in metric_keys:
        for grouped_config in grouped_config_ids:
            grouped_logs = np.asarray(list((seq(grouped_config)
                            .map(get_logs_from_config_id)
                            .map(lambda logs: logs[mode][metric], ))))
            alpha = get_logs_from_config_id(grouped_config[0])['config'][-1]['model']['imagination_model']['consistency_model']['alpha']
            _plot_variance_curve(grouped_logs, label=alpha)
        title = metric
        plot_dir = "."
        path = os.path.join(plot_dir, "_".join(title.split()) + ".png")
        plt.legend()
        plt.savefig(path)
        plt.clf()

if __name__ == "__main__":
    config_id = argument_parser()
    # for config_id in ['arxiv52', 'arxiv53', 'arxiv54', 'arxiv55', 'arxiv56', 'arxiv57', 'arxiv58', 'arxiv59', 'arxiv60']:
    #     run(config_id=config_id)
    run_multiple(config_ids=['arxiv85', 'arxiv93', 'arxiv94'])
    run_all_synced()
    # pusher_grouped_configs = [
        # ['arxiv62', 'arxiv66', 'arxiv70'],
    # ['arxiv63', 'arxiv67', 'arxiv71'],
    # ['arxiv64', 'arxiv68', 'arxiv72'],]
    # ['arxiv65', 'arxiv69', 'arxiv73']]
    # make_variance_plot(pusher_grouped_configs)

