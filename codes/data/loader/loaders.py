# Note that most of this code is modified from https://github.com/ctallec/world-models/blob/c27fe8c202c653f552bf15113ec1d456fb7ac3ba/data/loaders.py
import os
from bisect import bisect
from os import listdir
from os.path import join

import numpy as np
import torch.utils.data
from attr import attrs, attrib
from torchvision import transforms

from codes.data.datastructure import RolloutObservationData, get_model_data_spec_from_config
from codes.data.util import decode_trajectory_from_nparray


@attrs
class _RolloutDataset(torch.utils.data.Dataset):
    config = attrib()
    mode = attrib()
    _transform = attrib(init=False)
    _cum_size = None
    _buffer = None
    _buffer_fnames = None
    _buffer_index = attrib(init=False, default=0)
    _buffer_size = attrib(init=False)
    _files = attrib(init=False)
    _model_data_spec = attrib(init=False)
    _trajectory_length = attrib(init=False)

    @_transform.default
    def get_transform(self):
        if (self.mode == "train"):
            return transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize((self.config.env.height, self.config.env.width)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize((self.config.env.height, self.config.env.width)),
                transforms.ToTensor(),
            ])

    @_files.default
    def get_files(self):
        directory = os.path.join(self.config.dataset.base_path,
                                 self.config.dataset.name)
        files = [
            join(directory, file_name)
            for file_name in listdir(directory) if file_name.endswith(".npy")]

        effective_file_count = self.config.dataset.num_trajectories

        if self.mode == "train":
            train_index = int(self.config.dataset.split.train * effective_file_count)
            files = files[:train_index]
        elif self.mode == "val":
            train_index = int(self.config.dataset.split.train * effective_file_count)
            val_index = int((self.config.dataset.split.train + self.config.dataset.split.val) * effective_file_count)
            files = files[train_index:val_index]
        else:
            val_index = int((self.config.dataset.split.train + self.config.dataset.split.val) * effective_file_count)
            test_index = int((self.config.dataset.split.train +
                              self.config.dataset.split.val +
                              self.config.dataset.split.test) * effective_file_count)
            files = files[val_index:test_index]
        return files

    @_buffer_size.default
    def get_buffer_size(self):
        buffer_size = self.config.dataset.buffer_size
        if (self.mode != "train"):
            buffer_size = int((buffer_size + 1) / 2)
        return buffer_size

    @_model_data_spec.default
    def get_model_data_spec(self):
        return get_model_data_spec_from_config(self.config)

    @_trajectory_length.default
    def get_trajectory_length(self):
        return self.config.dataset.sequence_length + self.config.dataset.imagination_length + 1

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        # pbar = tqdm(total=len(self._buffer_fnames),
        #             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        # pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            data = np.load(f)
            trajectory = decode_trajectory_from_nparray(data)
            self._buffer.append(trajectory)
            self._cum_size += [self._cum_size[-1] +
                               self._len_data_per_trajectory(len(trajectory))]
            # pbar.update(1)
        # pbar.close()

    def __len__(self):
        # to have a full trajecotry, you need num_actions_per_trajectory + 1 obs, as
        # you must produce both an num_actions_per_trajectory obs and num_actions_per_trajectory next_obs sequences
        return self._cum_size[-1]

    def __getitem__(self, i):
        # binary search through cum_size
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _len_data_per_trajectory(self, data_length):
        return int(data_length / 2)

    def _transform_frame(self, frame):
        return torch.cat(
            list(
                map(lambda chunk:
                    self._transform(chunk.squeeze(0)).unsqueeze(0),
                    np.split(frame, self.config.env.num_stack))), dim=0)


@attrs
class RolloutSequenceDataset(_RolloutDataset):
    """ Encapsulates rollouts.

    Rollouts should be stored in seperate files in the root directory, in the form of npy files.
    Each file contains information on one trajectory
    Each trajectory is a sequence of state, action pairs.
    Each state contains the following:
        - pixel state observation
        - reward
        - done
        - info (which also contains the state as returned by mujoco)

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of tuples (obs, action, reward, done, next_obs):
    - obs: (num_actions_per_trajectory, *obs_shape)
    - env_state: (num_actions_per_trajectory, *env_state_shape)
    - actions: (num_actions_per_trajectory, action_size)
    - reward: (num_actions_per_trajectory,)
    - done: (num_actions_per_trajectory,) boolean
    - next_obs: (num_actions_per_trajectory, *obs_shape)
    - next_env_state: (num_actions_per_trajectory, *nextenv_state_shape)

    """

    pass

    def _get_data(self, data, seq_index):
        # Note that in our case, seq_index is guaranteed to be 0.
        _data = []
        state_data = data[2 * seq_index::2]
        for key in self._model_data_spec["keys"]:
            if (key == "action"):
                _data.append(self._get_action_data(data, seq_index))
            elif (key == "reward"):
                _data.append(self._get_reward_data(state_data))
            elif (key == "done"):
                _data.append(self._get_done_data(state_data))
            elif (key == "obs"):
                _data.append(self._get_obs_data(state_data))
            elif (key == "next_obs"):
                _data.append(self._get_next_obs_data(state_data))
            elif (key == "env_state"):
                _data.append(self._get_env_state_data(state_data))
            elif (key == "next_env_state"):
                _data.append(self._get_next_env_state_data(state_data))
        _data = list(map(self._filter_data, _data))
        return self._model_data_spec["container"]._make(_data)

    def _filter_data(self, data):
        return data[:self._trajectory_length]

    def _get_action_data(self, data, seq_index):
        action = data[2 * seq_index + 1::2]
        action = np.asarray(action[:self._trajectory_length], dtype=np.float32)
        return action

    def _get_reward_data(self, state_data):
        rewards = np.asarray(list(map(lambda state: state[1], state_data[1:])), dtype=np.float32)
        return rewards

    def _get_done_data(self, state_data):
        done = np.asarray(list(map(lambda state: state[2], state_data[1:])), dtype=np.uint8)
        return done

    def _get_obs_data(self, state_data):
        obs_data = list(
            map(
                lambda state: self._transform_frame(state[0]).unsqueeze(0), state_data))[:self._trajectory_length]
        obs = torch.cat(
            obs_data,
            dim=0)
        return obs

    def _get_next_obs_data(self, state_data):
        next_obs_data = list(
            map(
                lambda state: (self._transform_frame(state[0])[3]).unsqueeze(0), state_data))[
                        1:self._trajectory_length + 1]
        # Note that unlike the input obs, the output obs has only 1 frame as that is all that we want to predict.
        next_obs = torch.cat(
            next_obs_data,
            dim=0)
        return next_obs

    def _get_env_state_data(self, state_data):
        env_state_data = torch.cat(
            list(map(lambda state: (torch.from_numpy(state[3]["state"])).unsqueeze(0), state_data)),
            dim=0)

        env_state = env_state_data[:-1, :]

        return env_state

    def _get_next_env_state_data(self, state_data):
        env_state_data = torch.cat(
            list(map(lambda state: (torch.from_numpy(state[3]["state"])).unsqueeze(0), state_data)),
            dim=0)

        next_env_state = env_state_data[1:, :]

        return next_env_state

    def _len_data_per_trajectory(self, data_length):
        return 1


class RolloutObservationDataset(_RolloutDataset):
    """ Encapsulates rollouts.

        The only different between this and RolloutSequenceDataset is that here we return just the current
        observation and nothing else. This is useful for learning an encoder-decoder for the images.

        """

    def _get_data(self, data, seq_index):
        return RolloutObservationData._make(self._transform_frame(data[2 * seq_index][0]))

    def _len_data_per_trajectory(self, data_length):
        return int(data_length / 2)
