from addict import Dict

from codes.metric.trackable_metric import TrackableMetric


def get_trackable_metric_dict(time_span=100):
    trackable_metric = {
        "loss": TrackableMetric(name="loss", default_value=1e6, time_span=time_span, mode="min"),
        "imagination_log_likelihood": TrackableMetric(name="imagination_log_likelihood",
                                                      default_value=-1e6,
                                                      time_span=time_span,
                                                      mode="max"),
        "imitation_learning_loss": TrackableMetric(name="imitation_learning_loss",
                                                      default_value=1e6,
                                                      time_span=time_span,
                                                      mode="min")
    }
    return trackable_metric

def get_metrics_dict():
    # Empty metrics dict (but with keys) to hold the different metrics
    # Why is it initilaized with keys? Well this dict is fed directly to write_metric_log function.
    # So any key that could be here, needs to be registered there as well.
    metrics = Dict()
    metrics.loss = 0.0
    metrics.imagination_log_likelihood = 0.0
    metrics.num_examples = 0.0
    metrics.imitation_learning_loss = 0.0
    metrics.discriminator_loss = 0.0
    metrics.consistency_loss = 0.0
    return metrics