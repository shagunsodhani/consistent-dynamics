import time
from itertools import chain

import torch
from addict import Dict
from torch import nn

from codes.model.base_model import BaseModel
from codes.model.imagination_model.util import get_component, merge_first_and_second_dim, \
    unmerge_first_and_second_dim, sample_zt_from_distribution, clamp_mu_logsigma
from codes.utils.util import get_product_of_iterable, log_pdf

class Model(BaseModel):
    """Learning to Query w/o Imitation Learning
    This model uses the observation-dependent path"""

    def __init__(self, config):
        super(Model, self).__init__(config=config)
        self.convolutional_encoder = get_component("convolutional_encoder", config)
        self.state_transition_model = get_component("stochastic_state_transition_model", config)
        self.convolutional_decoder = get_component("stochastic_convolutional_decoder", config)
        self.prior_model = get_component("prior_model", config)
        self.posterior_model = get_component("posterior_model", config)
        self.use_consistency_model = False
        if(self.config.model.imagination_model.consistency_model.alpha!=0.0):
            self.use_consistency_model = True
            _consistency_model_name = self.config.model.imagination_model.consistency_model.name
            self.is_consistency_model_metric_based = False
            if _consistency_model_name in ("euclidean", "cosine"):
                self.is_consistency_model_metric_based = True
            self.consistency_model = get_component("consistency_model.{}".format(_consistency_model_name), config)

        self.use_imitation_learning_model = False
        if self.config.model.imagination_model.imitation_learning_model.should_train:
            self.use_imitation_learning_model = True

        if(self.use_imitation_learning_model):
            self.imitation_learning_model = get_component("imitation_learning_model.{}".
                format(
                self.config.model.imagination_model.imitation_learning_model.name),
                config)


    def encode_obs(self, obs):
        obs_shape = obs.shape
        per_image_shape = obs_shape[-3:]
        batch_size = obs_shape[0]
        trajectory_length = obs_shape[1]
        num_frames = obs_shape[2]
        h_t = self.convolutional_encoder(obs.view(-1, *per_image_shape)).view(batch_size, trajectory_length, num_frames,
                                                                              -1)
        h_t = torch.mean(h_t, dim=2)
        return h_t, trajectory_length

    def decode_obs(self, output, trajectory_length):
        reconstructed_obs = self.convolutional_decoder(output)
        per_image_shape = reconstructed_obs.shape[-3:]
        batch_size = int(reconstructed_obs.shape[0] / trajectory_length)
        return reconstructed_obs.view(batch_size, trajectory_length, *per_image_shape)

    def forward(self, x):
        # not that x is same as x_(t-1)

        sequence_length = self.config.dataset.sequence_length
        imagination_length = self.config.dataset.imagination_length

        h, _ = self.encode_obs(obs=x.obs)
        output_obs = x.next_obs
        output_obs_encoding, _ = self.encode_obs(obs=output_obs.unsqueeze(2))
        output_obs_encoding = output_obs_encoding
        action = x.action

        open_loop_data = Dict()

        # Preparing input for open_loop by using a seperate namespace called as input
        index_to_select_till = sequence_length + imagination_length

        open_loop_data.input = Dict()
        open_loop_data.input.unroll_length = index_to_select_till
        open_loop_data.input.output_obs_encoding = output_obs_encoding[:, :index_to_select_till, :]


        open_loop_data.input.output_obs = output_obs[:, :index_to_select_till, :]

        if (self.use_imitation_learning_model):
            open_loop_data.input.action = action[:, 0, :]
        else:
            open_loop_data.input.action = action[:, :index_to_select_till+1, :]

        open_loop_data.input.h_t = h[:, 0, :]

        open_loop_data, imitation_learning_data = self._vectorized_open_loop_prediction(open_loop_data)

        if(self.use_imitation_learning_model):

            imitation_learning_data.action = action
            imitation_learning_output = self._prepare_imitation_learning_result_to_yield(
                self._imitation_learning_prediction(imitation_learning_data))

            yield imitation_learning_output
            del imitation_learning_output


        to_yield = Dict()
        to_yield.loss = open_loop_data.output.loss
        to_yield.retain_graph = False
        to_yield.description = "open_loop"

        yield to_yield
        del to_yield

        close_loop_data = Dict()
        close_loop_data.input = Dict()

        # Note that this is not a bug, we are purposefully not making the prediction over the entire sequence length
        if(sequence_length >= 2*imagination_length):
            close_loop_data.input.sequence_length = 2 * imagination_length
        elif(sequence_length >= imagination_length):
            close_loop_data.input.sequence_length = imagination_length
        else:
            close_loop_data.input.sequence_length = 1
        close_loop_data.input.imagination_length = imagination_length
        index_to_start_with = max(0, sequence_length - close_loop_data.input.sequence_length)
        index_to_select_till = index_to_start_with + close_loop_data.input.sequence_length + close_loop_data.input.imagination_length

        close_loop_data.input.h_t = unmerge_first_and_second_dim(open_loop_data.h_t,
                                                                 first_dim=action.shape[0])[:,index_to_start_with:index_to_select_till,:]

        close_loop_data.input.action = action[:, index_to_start_with:index_to_select_till, :]

        close_loop_data.input.output_obs = output_obs[:, index_to_start_with:index_to_select_till, :]

        close_loop_output, discriminator_output = self._vectorized_closed_loop_prediction(close_loop_data)

        output = Dict()
        output.open_loop = open_loop_data.output
        output.close_loop = close_loop_output
        output.reporting_metrics.log_likelihood = close_loop_output.likelihood.item()
        output.reporting_metrics.consistency_loss = close_loop_output.consistency_loss.item()
        output.discriminator = discriminator_output

        alpha = self.config.model.imagination_model.consistency_model.alpha
        loss_tuple = (output.close_loop.loss + alpha * output.close_loop.consistency_loss,
                      output.discriminator.loss)
        loss_tuple = tuple(filter(lambda _loss: _loss.requires_grad, loss_tuple))

        to_yield = Dict()
        to_yield.loss = loss_tuple[0]
        to_yield.imagination_log_likelihood = output.reporting_metrics.log_likelihood
        to_yield.consistency_loss = output.reporting_metrics.consistency_loss
        to_yield.retain_graph = False
        to_yield.description = "close_loop"

        if (len(loss_tuple) == 1):
            yield to_yield

        else:
            to_yield.retain_graph = True
            yield to_yield

            to_yield = Dict()
            to_yield.loss = loss_tuple[1]
            to_yield.discriminator_loss = loss_tuple[1].item()
            to_yield.retain_graph = False
            to_yield.description = "discriminator"
            yield to_yield

    def _imitation_learning_prediction(self, imitation_learning_data):
        true_output = merge_first_and_second_dim(imitation_learning_data.action[:,1:,:].contiguous())
        predicted_output = imitation_learning_data.prediction
        imitation_learning_output = Dict()
        loss = self.imitation_learning_model.loss(predicted_output, true_output)
        imitation_learning_output.loss = self.config.model.imagination_model.imitation_learning_model.alpha * loss

        imitation_learning_output.imitation_learning_loss = loss.item()
        return imitation_learning_output

    def _prepare_imitation_learning_result_to_yield(self, imitation_learning_output):
        imitation_learning_output.retain_graph = True
        imitation_learning_output.imagination_log_likelihood = 0.0
        imitation_learning_output.description = "imitation_learning"
        return imitation_learning_output

    def _vectorized_open_loop_prediction(self, open_loop_data):
        # This is a simple implementation of the open loop prediction. This function pulls some operations outside the
        #  for-loop and vectorizes them. This is meant as the primary function for doing open-loop prediction.
        # Open loop

        unroll_length = open_loop_data.input.unroll_length
        output_obs_encoding = open_loop_data.input.output_obs_encoding
        output_obs = open_loop_data.input.output_obs
        action = open_loop_data.input.action
        if(self.use_imitation_learning_model):
            # First action
            a_t = action

        else:
            a_t = action[:, 0, :]

        # First state encoding
        h_t = open_loop_data.input.h_t

        self.state_transition_model.set_state(h_t)

        # Note that this datastructure is used as a container for variables to track. It helps to avoid writing multiple
        # statements.
        temp_data = Dict()

        vars_to_track = ["h_t", "z_t", "posterior_mu", "posterior_sigma"]

        if(self.use_imitation_learning_model):
            vars_to_track.append("a_t")

        for name in vars_to_track:
            key = name + "_list"
            temp_data[key] = []

        for t in range(0, unroll_length):
            current_output_obs_encoding = output_obs_encoding[:, t, :]

            posterior = self.sample_zt_from_posterior(h=h_t, a=a_t, o=current_output_obs_encoding)
            z_t = posterior.z_t
            inp = torch.cat((z_t, a_t), dim=1)
            h_t = self.state_transition_model(inp.unsqueeze(1)).squeeze(1)
            if(self.use_imitation_learning_model):
                a_t = self.imitation_learning_model(torch.cat((h_t, z_t), dim=1))
            else:
                a_t = action[:, t+1, :]
            posterior_mu = posterior.mu
            posterior_sigma = posterior.sigma

            for name in vars_to_track:
                key = name + "_list"
                temp_data[key].append(eval(name).unsqueeze(1))

        for name in vars_to_track:
            key = name + "_list"
            temp_data[name] = merge_first_and_second_dim(torch.cat(temp_data[key], dim=1))

        if not self.use_imitation_learning_model:
            temp_data.a_t = merge_first_and_second_dim(action[:, :unroll_length, :].contiguous())

        temp_data.prior = self.sample_zt_from_prior(
            h=temp_data.h_t,
            a=temp_data.a_t)

        likelihood_mu, likelihood_sigma = self.convolutional_decoder(
            torch.cat((temp_data.h_t,
                       temp_data.z_t), dim=1))

        elbo_prior = log_pdf(temp_data.z_t, temp_data.prior.mu, temp_data.prior.sigma)
        elbo_q_likelihood = log_pdf(temp_data.z_t, temp_data.posterior_mu,
                                    temp_data.posterior_sigma)
        elbo_likelihood = log_pdf(merge_first_and_second_dim(output_obs.contiguous()),
                                  likelihood_mu, likelihood_sigma)
        elbo = sum([torch.mean(x) for x in (
            elbo_likelihood, elbo_prior, -elbo_q_likelihood)])
        open_loop_data.output = Dict()
        open_loop_data.output.loss = -elbo
        open_loop_data.output.log_likelihood = torch.mean(elbo_likelihood)
        open_loop_data.h_t = temp_data.h_t.detach()

        imitation_learning_data = Dict()
        imitation_learning_data.prediction = temp_data.a_t

        return open_loop_data, imitation_learning_data

    def _vectorized_closed_loop_prediction(self, close_loop_data):
        # This is a simple implementation of the close loop prediction. This function pulls some operations outside the
        # for-loop and vectorizes them. This is meant as the primary function for doing close-loop prediction.
        # Closed Loop

        sequence_length = close_loop_data.input.sequence_length
        imagination_length = close_loop_data.input.imagination_length
        output_obs = close_loop_data.input.output_obs
        action = close_loop_data.input.action.contiguous()
        a_t = merge_first_and_second_dim(action[:, :sequence_length, :].contiguous())
        true_h_t = close_loop_data.input.h_t
        h_t = true_h_t[:, :sequence_length, :]
        h_t = merge_first_and_second_dim(h_t.contiguous())

        self.state_transition_model.set_state(h_t)
        elbo_likelihood = []
        consistency_loss = Dict()
        consistency_loss.discriminator = []
        consistency_loss.close_loop = []
        h_t_from_close_loop = None


        for t in range(0, imagination_length):

            prior = self.sample_zt_from_prior(h=h_t, a=a_t)
            z_t = prior.z_t
            inp = torch.cat((z_t, a_t), dim=1)
            h_t = self.state_transition_model(inp.unsqueeze(1)).squeeze(1)
            if(self.use_imitation_learning_model):
                a_t = self.imitation_learning_model(torch.cat((h_t, z_t), dim=1))

            else:
                a_t = merge_first_and_second_dim(action[:, t+1:t + sequence_length+1, :].contiguous())

            h_t_from_open_loop = true_h_t[:, t + 1:t + sequence_length + 1, :]
            h_t_from_close_loop = h_t

            if(self.use_consistency_model):

                if (self.is_consistency_model_metric_based):
                    h_t_from_open_loop = merge_first_and_second_dim(h_t_from_open_loop.contiguous())

                else:
                    h_t_from_close_loop = unmerge_first_and_second_dim(h_t_from_close_loop,
                                                                       first_dim=-1,
                                                                       second_dim=sequence_length)

                loss_close_loop, loss_discriminator = self.consistency_model((h_t_from_open_loop, h_t_from_close_loop))
                consistency_loss.close_loop.append(loss_close_loop)
                consistency_loss.discriminator.append(loss_discriminator)

            likelihood_mu, likelihood_sigma = self.convolutional_decoder(
                torch.cat((h_t, z_t), dim=1))

            elbo_likelihood.append(
                log_pdf(merge_first_and_second_dim(output_obs[:, t:t + sequence_length, :].contiguous()),
                        likelihood_mu,
                        likelihood_sigma))

        elbo_likelihood = list(map(lambda x: torch.mean(x).unsqueeze(0), elbo_likelihood))
        elbo_likelihood = torch.mean(torch.cat(elbo_likelihood))

        for key in consistency_loss:
            if consistency_loss[key]:
                # Checking if the list is non-empty
                consistency_loss[key] = torch.mean(torch.cat(consistency_loss[key]))
            else:
                consistency_loss[key] = torch.tensor(0.0).to(device=elbo_likelihood.device)


        close_loop_output = Dict()
        close_loop_output.loss = -elbo_likelihood
        close_loop_output.likelihood = elbo_likelihood
        close_loop_output.consistency_loss = consistency_loss.close_loop
        discriminator_output = Dict()
        discriminator_output.loss = consistency_loss.discriminator

        return close_loop_output, discriminator_output


    def sample_zt_from_prior(self, h, a):
        mu, logsigma = self.prior_model(torch.cat((h, a), dim=1))
        mu, logsigma = clamp_mu_logsigma(mu, logsigma)
        return sample_zt_from_distribution(mu, logsigma)

    def sample_zt_from_posterior(self, h, a, o):
        mu, logsigma = self.posterior_model(torch.cat((h, a, o), dim=1))
        mu, logsigma = clamp_mu_logsigma(mu, logsigma)
        return sample_zt_from_distribution(mu, logsigma)

    def get_optimizers(self):
        '''Method to return the list of optimizers for the model'''
        optimizers = []
        model_params = []
        if(self.use_imitation_learning_model):
            imitation_learning_model_params = list(self.get_imitation_learning_model_params())
            model_params.append(imitation_learning_model_params)
        open_loop_params = list(self.get_open_loop_params())
        model_params.append(open_loop_params)
        close_loop_params = list(self.get_close_loop_params())
        model_params.append(close_loop_params)
        if(self.use_consistency_model):
            consistency_model_params = list(self.get_consistency_model_params())
            model_params.append(consistency_model_params)
        optimizers = tuple(map(self._register_params_to_optimizer, filter(lambda x: x, model_params)))
        if (optimizers):
            return optimizers
        return None

    def get_open_loop_params(self):
        # Method to get params which are to be updated with the open loop
        open_loop_models = (self.convolutional_encoder,
                            self.state_transition_model,
                            self.convolutional_decoder,
                            self.prior_model,
                            self.posterior_model)
        open_loop_params = tuple(map(lambda model: model.get_model_params(), open_loop_models))
        return chain(*open_loop_params)

    def get_close_loop_params(self):
        # Method to get params which are to be updated with the close loop
        close_loop_models = [self.state_transition_model,]
        if(self.use_imitation_learning_model):
            close_loop_models.append(self.imitation_learning_model)
        close_loop_params = tuple(map(lambda model: model.get_model_params(), close_loop_models))
        return chain(*close_loop_params)

    def get_consistency_model_params(self):
        # Method to get params which are to be updated with the consistency model
        consistency_models = (self.consistency_model,)
        consistency_model_params = tuple(map(lambda model: model.get_model_params(), consistency_models))
        return chain(*consistency_model_params)

    def get_imitation_learning_model_params(self):
        # Method to get params which are to be updated with the imitation learning model
        imitation_learning_models = (self.imitation_learning_model,)
        imitation_learning_model_params = tuple(map(lambda model: model.get_model_params(), imitation_learning_models))
        return chain(*imitation_learning_model_params)

