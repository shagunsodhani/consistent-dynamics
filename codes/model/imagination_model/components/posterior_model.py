from codes.model.imagination_model.components.variational_encoder import Model as VariationalEncoder
from codes.model.imagination_model.util import get_action_space_size


class Model(VariationalEncoder):
    """ Posterior Model"""

    def __init__(self, config):
        super().__init__(config=config)

    def _get_input_size(self):
        _action_size = get_action_space_size(self.config)
        _input_size = 2 * self.config.model.imagination_model.hidden_state_size + _action_size
        return _input_size
