from time import time

from addict import Dict

from codes.experiment.bootstrap import _setup_experiment
from codes.experiment.util import move_data_to_device
from codes.metric.metric_registry import get_metrics_dict
from codes.utils.log import write_metric_logs, write_config_log, write_metadata_logs, write_message_logs
import torch

def run_experiment(config):
    write_config_log(config)
    experiment = _setup_experiment(config)
    _run_epochs(experiment)


def _run_epochs(experiment):
    validation_metrics_dict = experiment.validation_metrics_dict
    metric_to_perform_early_stopping = experiment.metric_to_perform_early_stopping
    config = experiment.config
    for key in validation_metrics_dict:
        validation_metrics_dict[key].reset()
    while experiment.epoch_index < config.model.num_epochs:
        _run_one_epoch_all_modes(experiment)
        for scheduler in experiment.schedulers:
            if (config.model.scheduler_type == "exp"):
                scheduler.step()
            elif (config.model.scheduler_type == "plateau"):
                scheduler.step(validation_metrics_dict[metric_to_perform_early_stopping].current_value)
        if (config.model.persist_per_epoch > 0):
            if (experiment.epoch_index % config.model.persist_per_epoch == 0):
                experiment.model.save_model(epochs=experiment.epoch_index, optimizers=experiment.optimizers)

        if(config.model.persist_best_model):
            if (validation_metrics_dict[metric_to_perform_early_stopping].is_best_so_far()):
                experiment.model.save_model(epochs=experiment.epoch_index, optimizers=experiment.optimizers,
                                            is_best_model=True)

        if (validation_metrics_dict[metric_to_perform_early_stopping].should_stop_early()):
            best_epoch_index = experiment.epoch_index - validation_metrics_dict[
                metric_to_perform_early_stopping].time_span
            write_metadata_logs(best_epoch_index=best_epoch_index)
            write_message_logs("Early stopping after running {} epochs".format(experiment.epoch_index))
            write_message_logs("Best performing model corresponds to epoch id {}".format(best_epoch_index))
            for key, value in validation_metrics_dict.items():
                write_message_logs("{} of the best performing model = {}".format(
                    key, value.get_best_so_far()))
            break
        experiment.epoch_index += 1
    else:
        best_epoch_index = experiment.epoch_index - validation_metrics_dict[metric_to_perform_early_stopping].counter
        write_metadata_logs(best_epoch_index=best_epoch_index)
        write_message_logs("Early stopping after running {} epochs".format(experiment.epoch_index))
        write_message_logs("Best performing model corresponds to epoch id {}".format(best_epoch_index))
        for key, value in validation_metrics_dict.items():
            write_message_logs("{} of the best performing model = {}".format(
                key, value.get_best_so_far()))


def _run_one_epoch_all_modes(experiment):
    for mode in experiment.support_modes:
        if mode in experiment.dataloaders:
            _run_one_epoch(experiment, mode)


def _run_one_epoch(experiment, mode):
    model = experiment.model
    device = experiment.device
    dataloader = experiment.dataloaders[mode]
    dataset = dataloader.dataset
    dataset.load_next_buffer()

    optimizers = experiment.optimizers

    if (mode == "train"):
        model.train()
    else:
        model.eval()

    aggregated_metrics = get_metrics_dict()
    current_input = Dict()

    epoch_start_time = time()
    for batch_idx, data in enumerate(dataloader):
        start_time = time()
        data, batch_size = move_data_to_device(data, device)
        data = experiment.data_container_cls._make(data)
        if (mode == "train"):
            for optimizer in optimizers:
                optimizer.zero_grad()

        current_input.data = data
        current_input.batch_size = batch_size
        current_input.mode = mode

        experiment, aggregated_metrics, current_metrics = compute_loss_and_metrics(experiment, current_input,
                                                                                   aggregated_metrics)
        current_metrics = current_metrics.to_dict()
        write_metric_logs(time_taken=time() - start_time,
                          batch_index=batch_idx,
                          effective_batch_size=batch_size,
                          epoch_index=experiment.epoch_index,
                          mode=mode,
                          **current_metrics)

    num_examples = aggregated_metrics.num_examples * 1.0
    aggregated_metrics = aggregated_metrics.to_dict()
    aggregated_metrics.pop("num_examples")
    for key in aggregated_metrics.keys():
        aggregated_metrics[key] = aggregated_metrics[key] / num_examples
    write_metric_logs(time_taken=time() - epoch_start_time,
                      epoch_index=experiment.epoch_index,
                      mode=mode,
                      **aggregated_metrics)

    if (mode == "val"):
        experiment.validation_metrics_dict["loss"].update(aggregated_metrics["loss"])
        experiment.validation_metrics_dict["imagination_log_likelihood"].update(
            aggregated_metrics["imagination_log_likelihood"])

    return experiment


def compute_loss_and_metrics(experiment, current_input, aggregated_metrics):
    if (experiment.is_single_computational_graph or False):
        return _compute_loss_and_metrics_single_computational_graph(
            experiment, current_input, aggregated_metrics)
    else:
        return _compute_loss_and_metrics_multiple_computational_graphs(
            experiment, current_input, aggregated_metrics)


def _compute_loss_and_metrics_single_computational_graph(experiment, current_input, aggregated_metrics):
    '''Note: This method is deprectared as of now. Method to compute the relevant losses and metrics'''
    model = experiment.model
    optimizers = experiment.optimizers
    data = current_input.data
    batch_size = current_input.batch_size
    mode = current_input.mode

    output = model(data)
    loss = model.loss(output, data)
    current_metrics = get_metrics_dict()

    if not isinstance(loss, tuple):
        # We are casting the loss scalar to a tuple so that we can have a common interface for all the models
        loss = (loss,)
    current_loss = tuple(map(lambda x: x.item(), loss))
    current_metrics.loss = sum(current_loss)
    aggregated_metrics.loss += (current_metrics.loss * batch_size)
    aggregated_metrics.num_examples += batch_size
    aggregated_metrics.imagination_log_likelihood += output.reporting_metrics.log_likelihood * batch_size
    current_metrics.imagination_log_likelihood = output.reporting_metrics.log_likelihood

    if (mode == "train"):
        retain_graph = False
        if (experiment.config.model.name == "imagination_model" and len(loss) > 2):
            retain_graph = True
        for loss_component in loss:
            loss_component.backward(retain_graph=retain_graph)
        for optimizer in optimizers:
            optimizer.step()

    return experiment, aggregated_metrics, current_metrics


def _compute_loss_and_metrics_multiple_computational_graphs(experiment, current_input, aggregated_metrics):
    '''Method to compute the relevant losses and metrics in the case of multiple computation graphs'''
    model = experiment.model
    optimizers = experiment.optimizers
    data = current_input.data
    batch_size = current_input.batch_size
    mode = current_input.mode

    current_metrics = get_metrics_dict()

    for output in model(data):
        if (mode == "train"):
            output.loss.backward(retain_graph=output.retain_graph)
        current_metrics.loss += output.loss.item()
        if (output.description == "close_loop"):
            current_metrics.imagination_log_likelihood += output.imagination_log_likelihood
            current_metrics.consistency_loss +=output.consistency_loss
        elif (output.description == "imitation_learning"):
            current_metrics.imitation_learning_loss += output.imitation_learning_loss
        elif (output.description=="discriminator"):
            current_metrics.discriminator_loss+=output.discriminator_loss

    aggregated_metrics.loss += (current_metrics.loss * batch_size)
    aggregated_metrics.num_examples += batch_size
    aggregated_metrics.imagination_log_likelihood += current_metrics.imagination_log_likelihood * batch_size
    aggregated_metrics.imitation_learning_loss += current_metrics.imitation_learning_loss * batch_size
    aggregated_metrics.consistency_loss += current_metrics.consistency_loss * batch_size
    aggregated_metrics.discriminator_loss+=current_metrics.discriminator_loss*batch_size


    if (mode == "train"):
        for optimizer in optimizers:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

    return experiment, aggregated_metrics, current_metrics
