# This is the file where we define which keys are needed by which models

from collections import namedtuple

RecurrentEnvironmentSimulatorDataKeys = ["obs", "action", "next_obs"]
RecurrentEnvironmentSimulatorData = namedtuple("RecurrentEnvironmentSimulatorData",
                                               RecurrentEnvironmentSimulatorDataKeys)

RolloutSequenceDataKeys = ["obs", "env_state", "action", "rewards",
                           "done", "next_obs", "next_env_state"]
RolloutSequenceData = namedtuple("RolloutSequenceData", RolloutSequenceDataKeys)


RolloutObservationDataKeys = ["obs"]
RolloutObservationData = namedtuple("RolloutObservationData", RolloutObservationDataKeys)

LearningToQueryDataKeys = ["obs", "action", "next_obs"]
LearningToQueryData = namedtuple("RecurrentEnvironmentSimulatorData",
                                               RecurrentEnvironmentSimulatorDataKeys)

def get_model_data_spec_from_config(config):
    if (config.model.name == "imagination_model" and
            config.model.imagination_model.name.startswith("recurrent_environment_simulator")):
        return {"keys":RecurrentEnvironmentSimulatorDataKeys,
                "container": RecurrentEnvironmentSimulatorData
                }
    elif (config.model.name == "imagination_model" and
            config.model.imagination_model.name.startswith("learning_to_query")):
        return {"keys":LearningToQueryDataKeys,
                "container": LearningToQueryData
                }
    else:
        return {"keys": RolloutSequenceDataKeys,
                "container": RolloutSequenceData
                }
