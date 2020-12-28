import gym
import random
import pybullet
import numpy as np
import pybullet_envs
import scipy.stats as ss
from copy import deepcopy
from multiprocessing import Pool


MAX_SEED = 2**16 - 1


def compute_weight_decay(weight_decay, model_param_list):
    """
    Compute weight decay penalty
    :param weight_decay: (float) weight decay coefficient
    :param model_param_list: (ndarray) weight parameters
    :return: (float) weight decay penalty
    """
    return -weight_decay * np.mean(np.square(model_param_list))


class FixedWeightModule:
    def __init__(self, input_dim, output_dim, bias=False, recurrent=False):
        self.bias = bias
        self.parameters = list()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if self.bias:
            self.bias_param = np.zeros((1, self.output_dim))
            self.parameters.append((self.bias_param, "bias_param"))

        self.recurrent = recurrent
        if self.recurrent:
            self.r_weight = np.zeros((self.output_dim, self.output_dim))
            self.parameters.append((self.r_weight, "r_weight"))
            self.recurrent_trace = np.zeros((1, self.output_dim))

        k = np.sqrt(1/self.input_dim)*0.1
        self.weights = np.random.uniform(
            low=-k, high=k, size=(self.input_dim, self.output_dim))
        self.parameters.append((self.weights, "weights"))

        self.param_ref_list = self.parameters
        self.parameters = np.concatenate([_p[0].flatten() for _p in self.param_ref_list])

    def forward(self, spikes, func=None):
        weighted_spikes = np.matmul(spikes, self.weights) + (self.bias_param if self.bias else 0.0)
        if self.recurrent:
            weighted_spikes += np.matmul(self.recurrent_trace, self.r_weight)
        post_synaptic = weighted_spikes
        if func is not None:
            post_synaptic = func(post_synaptic)
        weighted_spikes = post_synaptic
        if self.recurrent:
            self.recurrent_trace = weighted_spikes
        return weighted_spikes

    def reset(self):
        if self.recurrent:
            self.recurrent_trace = np.zeros((1, self.output_dim))

    def params(self):
        return self.parameters

    def update_params(self, eps, add_eps=True):
        eps_index = 0
        for _ref in range(len(self.param_ref_list)):
            _val, _str_ref = self.param_ref_list[_ref]
            pre_eps = eps_index
            eps_index = eps_index + _val.size
            if add_eps:
                new_val = _val.flatten() + eps[pre_eps:eps_index]
            else:
                new_val = eps[pre_eps:eps_index]
            new_val = new_val.reshape(self.param_ref_list[_ref][0].shape)

            self.param_ref_list[_ref] = new_val, _str_ref
            setattr(self, _str_ref, new_val)
        self.parameters = np.concatenate([_p[0].flatten() for _p in self.param_ref_list])



class GANetworkDef:
    def __init__(self, params, noise_std=0.01, num_eps_samples=64):
        """
        Genetic Algorithm Network Definition
        :param noise_std: (float) noise perturbation standard deviation
        :param num_eps_samples: (int) number of perturbation samples
        """
        self.params = params
        self.ga_optim = GAParamSampler(
            noise_std=noise_std, num_eps_samples=num_eps_samples)

    def parameters(self):
        """
        Return list of network parameters
        :return: (ndarray) network parameters
        """
        params = list()
        for _param in range(len(self.params)):
            params.append(self.params[_param].params())
        return np.concatenate(params, axis=0)

    def generate_eps_samples(self, seed):
        """
        Generate noise samples for list of parameters
        :param seed: (int) random number seed
        :return: (ndarray) parameter noise
        """
        params = self.parameters()
        sample = self.ga_optim.sample(params, seed)
        return sample

    def update_params(self, eps_sample, add_eps=True):
        """
        Update internal network parameters
        :param eps_sample: (ndarray) noise sample
        :param add_eps (bool)
        :return: None
        """
        param_itr = 0
        for _param in range(len(self.params)):
            pre_param_itr = param_itr
            param_itr += self.params[_param].parameters.size
            param_sample = eps_sample[pre_param_itr:param_itr]
            self.params[_param].update_params(param_sample, add_eps=add_eps)


class GANetwork(GANetworkDef):
    def __init__(self, input_size, output_size, noise_std=0.01, num_eps_samples=64):
        """
        Genetic Algorithm Network
        :param input_size: (int) input dimensionality
        :param output_size: (int) output/action dimensionality
        :param noise_std: (float) noise perturbation standard deviation
        :param num_eps_samples: (int) number of perturbation samples
        """
        self.params = list()
        self.recurrent = True

        self.hidden_base_1 = 16
        self.hidden_base_2 = 16
        self.input_dim = input_size
        self.output_dim = output_size
        self.neuron_module_1 = FixedWeightModule(
            self.input_dim, self.hidden_base_1
        )
        self.neuron_module_2 = FixedWeightModule(
            self.hidden_base_1, self.hidden_base_2
        )
        self.action_module = FixedWeightModule(
            self.hidden_base_2, self.output_dim
        )

        self.params.append(self.neuron_module_1)
        self.params.append(self.neuron_module_2)
        self.params.append(self.action_module)

        super(GANetwork, self).__init__(noise_std=noise_std,
                params=self.params, num_eps_samples=num_eps_samples)

    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def forward(self, fp_input):
        neural_activity = self.neuron_module_1.forward(
            fp_input + np.random.normal(loc=0.0, scale=0.01, size=(self.input_dim,)),)
        neural_activity = self.neuron_module_2.forward(
            neural_activity + np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_base_1,)))
        action = self.action_module.forward(
            neural_activity + np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_base_2,))
        )
        action = (action + np.random.normal(loc=0.0, scale=0.01, size=(self.output_dim,))).squeeze()

        return action

    def reset(self):
        self.neuron_module_1.reset()
        self.neuron_module_2.reset()
        self.action_module.reset()


def compute_returns(seed, noise_std):
    """
    :param seed:
    :param noise_std:
    :return:
    """
    avg_stand = 0
    returns = list()
    total_env_interacts = 0
    max_env_interacts = 100

    local_env = gym.make("Ant-v2")

    network = GANetwork(
        local_env.observation_space.shape[0],
        local_env.action_space.shape[0])
    network.ga_optim.noise_std = noise_std

    weight_decay_term = 0.01
    ga_samples = network.generate_eps_samples(seed)

    avg_action = 0.0
    num_env_rollouts = 1
    for _sample in range(ga_samples.shape[0]):
        return_avg = 0.0

        network.update_params(ga_samples[_sample], add_eps=False)

        # iterate through the target number of env rollouts
        for _roll in range(num_env_rollouts):
            network.reset()
            state = local_env.reset()
            neg_count = 0.0
            for _inter in range(max_env_interacts):
                # forward propagate using noisy weights and plasticity values, also update trace
                state = state.reshape((1, state.size))
                action1 = network.forward(state)
                avg_action += action1
                state, reward, game_over, _info = local_env.step(action1)
                reward = _info['reward_forward']
                if _inter > 200:
                    neg_count = neg_count + 1 if reward < 0.0 else 0
                return_avg += reward
                if game_over or neg_count > 30:#reward != 0.0:
                    break
                avg_stand += 1
                total_env_interacts += 1
        weight_penalty = num_env_rollouts*compute_weight_decay(weight_decay_term, network.parameters())
        return_avg += weight_penalty
        returns.append(return_avg / num_env_rollouts)
    return np.array(returns), total_env_interacts / num_env_rollouts, 0



class GAParamSampler:
    def __init__(self, num_eps_samples, noise_std=0.01):
        """
        Genetic Algorithm Parameter Sampler
        :param num_eps_samples: (int) number of noise samples to generate
        :param noise_std: (float) noise standard deviation
        """
        self.noise_std = noise_std
        self.num_eps_samples = num_eps_samples

    def sample(self, params, seed):
        """
        Sample noise for network parameters
        :param params: (ndarray) network parameters
        :param seed: (int) random seed used to sample
        :return: (ndarray) sampled noise
        """
        p_size = params.size
        noise_vectors = list()
        for _seed in seed:
            noise_vector = np.zeros((1, p_size))
            for _sub_seed in _seed:
                rand_m = np.random.RandomState(seed=_sub_seed)
                noise_vector += \
                    rand_m.randn(1, p_size) * self.noise_std
            noise_vectors.append(noise_vector)
        noise_vector = np.concatenate(noise_vectors, axis=0)
        return noise_vector


class GAOptimizer:
    def __init__(self, noise_std, num_workers=1,
            epsilon_samples=48, weight_decay=0.01, noise_std_decay=None):
        """
        :param noise_std:
        :param num_workers:
        :param epsilon_samples: (int) number of perturbation samples
        :param weight_decay:
        :param noise_std_decay: (float) noise std decay factor ep(t+1) = ep(t)*decay
        """
        assert (epsilon_samples % num_workers == 0), "Epsilon sample size not divis num workers"
        self.noise_std = noise_std
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.noise_std_decay = noise_std_decay
        self.epsilon_samples = epsilon_samples

        self.elites = 10
        self.elite_saves = 2
        self.network_seed = [0,]
        self.previous_elites = [
            (-1000, self.network_seed) for _ in range(self.elite_saves)]
        self.elite_candidates = self.previous_elites

    def parallel_returns(self, x):
        """
        Function call for collecting parallelized rewards
        :param x: (tuple(int, int)) worker id and seed
        :return: (list) collected returns
        """
        worker_id, noise_std, seed = x
        return compute_returns(seed=seed, noise_std=noise_std)

    def update(self, iteration):
        """
        Update weights of given model using OpenAI-ES
        :param iteration: (int) iteration number used for random seed sampling
        :return: None
        """
        net_seeds = list()
        timestep_list = list()
        sample_returns = list()
        mutation_seeds = [(_w, self.noise_std,
            [self.elite_candidates[np.random.choice(
            np.array(list(range(len(self.elite_candidates)))))][1] +
            [(iteration*self.epsilon_samples) +_k + _w*self.num_workers]
            for _k in range(self.epsilon_samples//self.num_workers)])
                for _w in range(self.num_workers)]
        with Pool(self.num_workers) as p:
            values = p.map(func=self.parallel_returns, iterable=mutation_seeds)
        # todo: we dont need pool if we return seed or net id
        num_corr_tot = 0
        total_timesteps = 0
        for _worker in range(self.num_workers):
            _, _, net_seed = mutation_seeds[_worker]
            returns, timesteps, num_corr = values[_worker]
            num_corr_tot += num_corr
            timestep_list.append(timesteps)
            total_timesteps += timesteps

            sample_returns.append(returns)
            net_seeds.append(net_seed)
        sample_returns = np.concatenate(sample_returns)

        net_seeds = [_k for _s in net_seeds for _k in _s]
        net_rew = [(sample_returns[_k],
            net_seeds[_k]) for _k in range(len(net_seeds))]
        net_rew.sort(key=lambda x: x[0], reverse=True)
        if iteration == 0:
            self.elite_candidates = net_rew[:self.elites]
        else:
            self.elite_candidates = net_rew[:self.elites-self.elite_saves] + self.previous_elites

        # todo: evaluate each elite over 30 ts and set new prev elite
        self.previous_elites = sorted(
            self.elite_candidates, key=lambda x: x[0], reverse=True)[:self.elite_saves]

        if self.noise_std_decay is not None:
            self.noise_std = self.noise_std*self.noise_std_decay

        avg_return_rec = sum(sample_returns) / len(sample_returns)
        return avg_return_rec, total_timesteps, num_corr_tot



envrn = gym.make("Ant-v2")
envrn.reset()

if __name__ == "__main__":
    t_time = 0.0

    workers = 8
    max_itr = 3000

    eps_samples            = 40
    param_noise_std        = 0.01
    param_noise_std_decay_ = 0.999


    n_type = "linear"

    es_optim = GAOptimizer(
        weight_decay        = 0.01,
        num_workers         = workers,
        epsilon_samples     = eps_samples,
        noise_std           = param_noise_std,
        noise_std_decay     = param_noise_std_decay_,
    )

    import pickle
    top_reward = -100000.0
    reward_list = list()
    for _i in range(max_itr):
        r, t, pr = es_optim.update(_i)
        t_time += t
        reward_list.append((r, _i, t_time))

        with open("save_{}_net_rew_{}.pkl".format(n_type, 0), "wb") as f:
            pickle.dump(reward_list, f)

        if r >= top_reward:
            print("New Best Performance!", round(r, 5),
                  _i, round(t/eps_samples, 5), param_noise_std, eps_samples)
            #with open("save_{}_net_{}.pkl".format(n_type, 0), "wb") as f:
            #    pickle.dump(net, f)
            top_reward = r
        else:
            print(round(r, 5), _i, round(t/eps_samples, 5),
                    param_noise_std, eps_samples)
    print("~~~~~~~~~~~~~~~~~~~~")












