class TrackableMetric():
    """Implementation of a simple class to track value of a metric over multiple timesteps"""

    def __init__(self, name, default_value, time_span, mode="max"):
        """Initializes a metric object

        Args:
            default_value: float, the default value of the metric when it has not been observed experimentally.
            For instance, the default value of loss is +inf and that of accuracy is 0.
            time_span: int, the number of time steps for which we would retain the value of the metric to compute
            quantities like running mean.
            mode: str, support values (max, min). Whether we want to maximise or minimise this metric.
        """

        self._name = name
        self._default_value = default_value
        self._value = default_value
        self._best_value = default_value
        self._current_value = default_value
        self._want_max = False
        if (mode == "max"):
            self._want_max = True
        self._time_span = time_span
        self._counter = None
        self._reset_counter()

    def _reset_counter(self):
        """Method to reset the counter to 0"""
        self._counter = 0
        # print("reset")

    def _increment_counter(self):
        """Method to increment the counter"""
        self._counter += 1

    def update(self, new_value):
        """Update the value of the trackable metric
        Args:
            neW_value: float, the new value of the metric being tracked
        """
        self._current_value = new_value

        if (self._want_max):
            if (new_value > self._value):
                self._reset_counter()
                self._best_value = max(new_value, self._best_value)
            else:
                self._increment_counter()
        else:
            if (new_value > self._value):
                self._increment_counter()
            else:
                self._reset_counter()
                # self._value = new_value
                self._best_value = min(new_value, self._best_value)
        self._value = new_value

    def should_stop_early(self):
        """Method to determine if early stopping should be performed based on the metric value that has been
        tracked so far"""
        return self._counter >= self._time_span

    def reset(self):
        self._reset_counter()
        self._value = self._default_value
        self._current_value = self._default_value
        self._best_value = self._default_value

    @property
    def time_span(self):
        return self._time_span

    @property
    def counter(self):
        return self._counter

    def is_best_so_far(self):
        return self._counter == 0

    def get_best_so_far(self):
        return self._value

    @property
    def current_value(self):
        return self._current_value