import gym
import random
import pybullet
import numpy as np
import pybullet_envs
import scipy.stats as ss
from copy import deepcopy
from multiprocessing import Pool


class Adam:
    def __init__(self, params, stepsize, epsilon=1e-08, beta1=0.99, beta2=0.999):
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = params
        self.dim = params.size
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.m = np.zeros(params.size, dtype=np.float32)
        self.v = np.zeros(params.size, dtype=np.float32)

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.params
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        new_theta = self.params + step
        self.params = new_theta
        return ratio, new_theta

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step



MAX_SEED = 2**16 - 1



class FixedWeightModule:
    def __init__(self, input_dim, output_dim, plastic=False, bias=False, recurrent=False):
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

        self.plastic = plastic
        if self.plastic:
            self.weights = np.random.uniform(
                low=-0.1, high=0.1, size=(self.input_dim, self.output_dim))
            self.alpha_hebb = np.zeros((self.input_dim, self.output_dim))
            self.parameters.append((self.alpha_hebb, "alpha_hebb"))
            self.hebb_bias = np.zeros((self.input_dim, self.output_dim))
            self.parameters.append((self.hebb_bias, "hebb_bias"))
            self.hebb_trace = np.zeros((self.input_dim, self.output_dim))
        else:
            k = np.sqrt(1/self.input_dim)*0.1
            self.weights = np.random.uniform(
                low=-k, high=k, size=(self.input_dim, self.output_dim))
            self.parameters.append((self.weights, "weights"))

        self.param_ref_list = self.parameters
        self.parameters = np.concatenate([_p[0].flatten() for _p in self.param_ref_list])

    def forward(self, spikes, func=None):
        if self.plastic:
            weighted_spikes = np.matmul(spikes,
                self.weights + self.hebb_trace) + self.bias_param if self.bias else 0.0
            if self.recurrent:
                weighted_spikes += np.matmul(self.recurrent_trace, self.r_weight)
            pre_synaptic = spikes
            post_synaptic = weighted_spikes
            if func is not None:
                post_synaptic = func(post_synaptic)
            weighted_spikes = post_synaptic
            if self.recurrent:
                self.recurrent_trace = weighted_spikes
            H = np.matmul(pre_synaptic.T, post_synaptic)
            self.hebb_trace = self.hebb_trace + self.alpha_hebb*(H + self.hebb_bias)#, a_max=1, a_min=-1)
        else:
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
        if self.plastic:
            self.weights = np.random.uniform(
                low=-0.1, high=0.1, size=(self.input_dim, self.output_dim))
            self.hebb_trace = np.zeros((self.input_dim, self.output_dim))
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





class PlasticWeightModule:
    def __init__(self, input_dim, output_dim, bias=False, recurrent=False, modulatory=False):
        self.bias = bias
        self.parameters = list()
        self.recurrent = recurrent
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modulatory = modulatory
        self.modulatory_signal = None
        self.recur_modulatory_signal = None

        if self.modulatory:
            self.modulatory_signal = np.zeros((1, 1))
            if self.recurrent:
                self.recur_modulatory_signal = np.zeros((1, 1))

        self.learning_rate = np.e**(1/-20)
        self.eta = np.random.uniform(
            low=-0.1, high=0.1, size=(self.input_dim, self.output_dim))
        self.parameters.append((self.eta, "eta"))
        self.A = np.random.uniform(
            low=-0.1, high=0.1, size=(self.input_dim, self.output_dim))
        self.parameters.append((self.A, "A"))
        self.B = np.random.uniform(
            low=-0.1, high=0.1, size=(self.input_dim, self.output_dim))
        self.parameters.append((self.B, "B"))
        self.C = np.random.uniform(
            low=-0.1, high=0.1, size=(self.input_dim, self.output_dim))
        self.parameters.append((self.C, "C"))
        self.D = np.random.uniform(
            low=-0.1, high=0.1, size=(self.input_dim, self.output_dim))
        self.parameters.append((self.D, "D"))
        self.weights = np.random.uniform(
            low=-0.1, high=0.1, size=(self.input_dim, self.output_dim))
        self.weight_save = deepcopy(self.weights)
        self.parameters.append((self.weights, "weights"))

        if self.recurrent:
            self.recur_eta = np.random.uniform(
                low=-0.1, high=0.1, size=(self.output_dim, self.output_dim))
            self.parameters.append((self.recur_eta, "recur_eta"))
            self.recur_A = np.random.uniform(
                low=-0.1, high=0.1, size=(self.output_dim, self.output_dim))
            self.parameters.append((self.recur_A, "recur_A"))
            self.recur_B = np.random.uniform(
                low=-0.1, high=0.1, size=(self.output_dim, self.output_dim))
            self.parameters.append((self.recur_B, "recur_B"))
            self.recur_C = np.random.uniform(
                low=-0.1, high=0.1, size=(self.output_dim, self.output_dim))
            self.parameters.append((self.recur_C, "recur_C"))
            self.recur_D = np.random.uniform(
                low=-0.1, high=0.1, size=(self.output_dim, self.output_dim))
            self.parameters.append((self.recur_D, "recur_D"))
            self.recur_weights = np.random.uniform(
                low=-0.1, high=0.1, size=(self.output_dim, self.output_dim))
            self.parameters.append((self.recur_weights, "recur_weights"))
            self.recur_trace = np.zeros((1, self.output_dim))
            self.recur_weight_save = deepcopy(self.recur_weights)

        self.param_ref_list = self.parameters
        self.parameters = np.concatenate([_p[0].flatten() for _p in self.param_ref_list])

    def relu(self, x):
        x[x < 0] = 0
        return x

    def hard_tanh(self, x, gain=0.2):
        return np.clip(gain * x, a_min=-1.0, a_max=1.0)

    def hard_sigmoid(self, x, gain=0.2):
        return np.clip((gain * x) + 0.5, a_min=0.0, a_max=1.0)

    def zero_centered_sigmoid(self, x, gain=0.2):
        return np.clip((gain * x), a_min=0.0, a_max=1.0)

    def softmax(self, x):
        """ Compute softmax values for each sets of scores in x """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def forward(self, spikes, func=None, activity=None, action_layer=False):
        post_activity = None
        pre_synaptic_act = spikes
        post_synaptic = np.matmul(pre_synaptic_act, self.weight_save)

        if activity is not None:
            post_synaptic += activity
        if self.recurrent:
            post_synaptic += np.matmul(self.recur_trace, self.recur_weight_save)
        if action_layer:
            post_activity = deepcopy(post_synaptic)
        if func is not None:
            post_synaptic = func(post_synaptic)

        h = np.matmul(pre_synaptic_act.T, post_synaptic)
        C = self.C*np.repeat(pre_synaptic_act, axis=0, repeats=self.output_dim).T
        D = self.D*np.repeat(post_synaptic, axis=0, repeats=self.input_dim)
        if self.modulatory:
            self.weight_save = self.weight_save + self.modulatory_signal*self.eta * (self.A * h + self.B + C + D)
        else:
            self.weight_save = self.weight_save + (1-self.learning_rate)*self.eta*(self.A*h + self.B + C + D)
        if self.recurrent:
            recur_h = np.matmul(self.recur_trace.T, post_synaptic)
            recur_C = self.recur_C * np.repeat(self.recur_trace, axis=0, repeats=self.output_dim).T
            recur_D = self.recur_D * np.repeat(post_synaptic, axis=0, repeats=self.output_dim)
            if self.modulatory:
                self.recur_weight_save = self.recur_weight_save + \
                    self.recur_modulatory_signal*self.recur_eta * \
                    (self.recur_A * recur_h + self.recur_B + recur_C + recur_D)
            else:
                self.recur_weight_save = self.recur_weight_save + \
                    (1-self.learning_rate)*self.recur_eta * \
                    (self.recur_A*recur_h + self.recur_B + recur_C + recur_D)
            self.recur_trace = post_synaptic

        return post_synaptic if not action_layer else post_activity

    def reset(self):
        if self.modulatory:
            self.modulatory_signal = np.zeros((1, 1))
            if self.recurrent:
                self.recur_modulatory_signal = np.zeros((1, 1))
        if self.recurrent:
            self.recur_trace = np.zeros((1, self.output_dim))
            self.recur_weight_save = deepcopy(self.recur_weights)
        self.weight_save = deepcopy(self.weights)

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
        #self.learning_rate_actual = np.e**(1/(-20+10*self.learning_rate.squeeze()))



def compute_returns(seed, network, num_eps_samples, num_env_rollouts=1, evo_itr=0):
    """

    :param seed:
    :param network:
    :param num_eps_samples:
    :param num_env_rollouts:
    :param evo_itr:
    :return:
    """
    avg_stand = 0
    returns = list()
    total_env_interacts = 0
    max_env_interacts = 500
    network_cpy = deepcopy(network)
    local_env = gym.make("Ant-v2")
    eps_samples = network.generate_eps_samples(seed, num_eps_samples)

    avg_action = 0.0
    # iterate through the noise samples
    for _sample in range(eps_samples.shape[0]):
        return_avg = 0.0
        network = deepcopy(network_cpy)
        network.update_params(eps_samples[_sample])

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
        returns.append(return_avg / num_env_rollouts)
    return np.array(returns), total_env_interacts / num_env_rollouts, 0



class ParamSampler:
    def __init__(self, sample_type, num_eps_samples, noise_std=0.01):
        """
        Evolutionary Strategies Optimizer
        :param sample_type: (str) type of noise sampling
        :param num_eps_samples: (int) number of noise samples to generate
        :param noise_std: (float) noise standard deviation
        """
        self.noise_std = noise_std
        self.sample_type = sample_type
        self.num_eps_samples = num_eps_samples
        if self.sample_type == "antithetic":
            assert (self.num_eps_samples % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.num_eps_samples / 2)

    def sample(self, params, seed, num_eps_samples):
        """
        Sample noise for network parameters
        :param params: (ndarray) network parameters
        :param seed: (int) random seed used to sample
        :param num_eps_samples (int) number of noise samples to evaluate
        :return: (ndarray) sampled noise
        """
        if seed is not None:
            rand_m = np.random.RandomState(seed)
        else:
            rand_m = np.random.RandomState()
        sample = None
        if self.sample_type == "antithetic":
            epsilon_half = rand_m.randn(num_eps_samples//2, params.size)
            sample = np.concatenate([epsilon_half, - epsilon_half]) * self.noise_std
        elif self.sample_type == "normal":
            sample = rand_m.randn(num_eps_samples, params.size) * self.noise_std
        return sample


class ESNetwork:
    def __init__(self, params, noise_std=0.01,
            num_eps_samples=64, sample_type="normal"):
        """
        :param params:
        :param num_eps_samples:
        :param sample_type:
        """
        self.params = params
        self.es_optim = ParamSampler(noise_std=noise_std,
            sample_type=sample_type, num_eps_samples=num_eps_samples)

    def parameters(self):
        """
        Return list of network parameters
        :return: (ndarray) network parameters
        """
        params = list()
        for _param in range(len(self.params)):
            params.append(self.params[_param].params())
        return np.concatenate(params, axis=0)

    def generate_eps_samples(self, seed, num_eps_samples):
        """
        Generate noise samples for list of parameters
        :param seed: (int) random number seed
        :param num_eps_samples (int) number of noise samples to evaluate
        :return: (ndarray) parameter noise
        """
        params = self.parameters()
        sample = self.es_optim.sample(params, seed, num_eps_samples)
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


class LinearNetwork(ESNetwork):
    def __init__(self, input_size, output_size, noise_std=0.01,
            action_noise_std=None, num_eps_samples=64, sample_type="normal", neuron_type="LIF"):
        self.params = list()
        self.recurrent = True
        self.neuromodulated = True
        self.neuromodulation_type = "sigmoid" #"sigmoid"

        self.hidden_base_1 = 32
        self.hidden_base_2 = 32
        self.input_dim = input_size
        self.output_dim = output_size
        if self.neuromodulated:
            if self.recurrent:
                self.modulators = 5 # 2*1 + 2*1 + 1*1
            else:
                self.modulators = 3 # 1*1 + 1*1 + 1*1
            self.output_dim += self.modulators

        self.neuron_module_1 = PlasticWeightModule(
            self.input_dim, self.hidden_base_1,
            recurrent=self.recurrent, modulatory=self.neuromodulated
        )
        self.neuron_module_2 = PlasticWeightModule(
            self.hidden_base_1, self.hidden_base_2,
            recurrent=self.recurrent, modulatory=self.neuromodulated
        )
        self.action_module = PlasticWeightModule(
            self.hidden_base_2, self.output_dim,
            recurrent=self.recurrent, modulatory=self.neuromodulated
        )

        self.params.append(self.neuron_module_1)
        self.params.append(self.neuron_module_2)
        self.params.append(self.action_module)

        super(LinearNetwork, self).__init__(noise_std=noise_std,
                params=self.params, num_eps_samples=num_eps_samples, sample_type=sample_type)

    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def forward(self, fp_input):
        neural_activity = self.neuron_module_1.forward(
            func = np.tanh,
            spikes = fp_input + np.random.normal(loc=0.0, scale=0.01, size=(self.input_dim,)),
            activity = None,
            action_layer = False,
        )
        neural_activity = self.neuron_module_2.forward(
            func = np.tanh,
            spikes = neural_activity + np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_base_1,)),
            activity = None,
            action_layer = False,
        )
        action = self.action_module.forward(
            func = np.tanh,
            spikes = neural_activity + np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_base_2,)),
            activity = None,
            action_layer = self.neuromodulated,
        )

        if self.neuromodulated:
            action_full = action.squeeze()
            action = np.tanh(action_full[:-self.modulators]) + \
                np.random.normal(loc=0.0, scale=0.01, size=(self.output_dim-self.modulators,))
            if self.neuromodulation_type == "sigmoid":
                modulatory_signal = self.action_module.zero_centered_sigmoid(action_full[-self.modulators:], gain=1)
            elif self.neuromodulation_type == "tanh":
                modulatory_signal = self.action_module.hard_tanh(action_full[-self.modulators:], gain=1)
            else:
                raise TypeError("No Modulation Type {}".format(self.neuromodulation_type))
            self.neuron_module_1.modulatory_signal = modulatory_signal[0]
            self.neuron_module_2.modulatory_signal = modulatory_signal[1]
            self.action_module.modulatory_signal = modulatory_signal[2]
            if self.recurrent:
                self.neuron_module_1.recur_modulatory_signal = modulatory_signal[3]
                self.neuron_module_2.recur_modulatory_signal = modulatory_signal[4]
        else:
            action = (action + np.random.normal(loc=0.0, scale=0.01, size=(self.output_dim,))).squeeze()

        return action

    def reset(self):
        self.neuron_module_1.reset()
        self.neuron_module_2.reset()
        self.action_module.reset()


class EvolutionaryOptimizer:
    def __init__(self, network, num_workers=1, epsilon_samples=48, noise_std_limit=None,
                 learning_rate_limit=None, learning_rate=0.01, weight_decay=0.01, max_iterations=2000,
                 noise_std_decay=None, learning_rate_decay=None):
        """

        :param network:
        :param num_workers:
        :param epsilon_samples:
        :param learning_rate:
        """
        assert (epsilon_samples % num_workers == 0), "Epsilon sample size not divis num workers"
        self.adam = False
        self.network = network
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon_samples = epsilon_samples
        self.init_learning_rate = learning_rate
        self.init_noise_std = self.network.es_optim.noise_std
        self.optimizer = Adam(network.parameters(), learning_rate)

        self.noise_std_limit = noise_std_limit
        self.learning_rate_limit = learning_rate_limit

        self.noise_std_decay = noise_std_decay
        self.learning_rate_decay = learning_rate_decay
        #self.init_noise_std if noise_std_limit is None else noise_std_limit

    def parallel_returns(self, x):
        """
        Function call for collecting parallelized rewards
        :param x: (tuple(int, int)) worker id and seed
        :return: (list) collected returns
        """
        worker_id, seed, iteration = x
        return compute_returns(seed=seed, evo_itr=iteration,
            network=self.network, num_eps_samples=self.epsilon_samples//self.num_workers)

    def compute_weight_decay(self, weight_decay, model_param_list):
        """
        Compute weight decay penalty
        :param weight_decay: (float) weight decay coefficient
        :param model_param_list: (ndarray) weight parameters
        :return: (float) weight decay penalty
        """
        model_param_grid = np.array(model_param_list + self.network.parameters())
        return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

    def normalized_rank(self, rewards):
        """
        Rank the rewards and normalize them.
        """
        ranked = ss.rankdata(rewards)
        norm = (ranked - 1) / (len(ranked) - 1)
        norm -= 0.5
        return norm

    def update(self, iteration, meta_itr, offset):
        """
        Update weights of given model using OpenAI-ES
        :param iteration: (int) iteration number used for random seed sampling
        :return: None
        """
        samples = list()
        timestep_list = list()
        sample_returns = list()
        random_seeds = [
            (_, (iteration*self.num_workers*7 + _ + meta_itr*55 + offset) % MAX_SEED, iteration)
                for _ in range(self.num_workers)]
        with Pool(self.num_workers) as p:
            values = p.map(func=self.parallel_returns, iterable=random_seeds)

        num_corr_tot = 0
        total_timesteps = 0
        for _worker in range(self.num_workers):
            seed = random_seeds[_worker]
            returns, timesteps, num_corr = values[_worker]

            num_corr_tot += num_corr
            timestep_list.append(timesteps)
            total_timesteps += timesteps
            sample_returns += returns.tolist()
            samples += [self.network.generate_eps_samples(
                seed[1], self.epsilon_samples//self.num_workers)]

        num_corr_tot /= self.num_workers
        eps = np.concatenate(samples)
        returns = np.array(sample_returns)

        if self.weight_decay > 0:
            l2_decay = self.compute_weight_decay(self.weight_decay, eps)
            returns += l2_decay

        returns = self.normalized_rank(returns)

        returns = (returns - np.mean(returns)) / (returns.std() + 1e-5)
        scale_mu = (self.learning_rate / (self.epsilon_samples * self.network.es_optim.noise_std))
        change_mu = scale_mu * np.dot(eps.T, returns)

        if self.adam:
            change_mu = -change_mu
            self.optimizer.t += 1
            a = self.optimizer.stepsize * \
                np.sqrt(1.0 - self.optimizer.beta2 ** self.optimizer.t)\
                / (1.0 - self.optimizer.beta1 ** self.optimizer.t)
            m = self.optimizer.beta1 * self.optimizer.m + \
                (1.0 - self.optimizer.beta1) * change_mu
            v = self.optimizer.beta2 * self.optimizer.v + \
                (1.0 - self.optimizer.beta2) * (change_mu * change_mu)
            step = -a * m / (np.sqrt(v) + self.optimizer.epsilon)

        self.network.update_params(change_mu, add_eps=True)

        if self.learning_rate_limit is not None:
            self.learning_rate = (1-(iteration/self.max_iterations))\
                *self.learning_rate + (iteration/self.max_iterations)*self.learning_rate_limit
        elif self.learning_rate_decay is not None:
            self.learning_rate = self.learning_rate*self.learning_rate_decay

        if self.noise_std_limit is not None:
            self.network.es_optim.noise_std = (1-(iteration/self.max_iterations))\
                *self.init_noise_std + (iteration/self.max_iterations)*self.noise_std_limit
        elif self.noise_std_decay is not None:
            self.network.es_optim.noise_std = self.network.es_optim.noise_std*self.noise_std_decay

        avg_return_rec = sum(sample_returns) / len(sample_returns)
        return avg_return_rec, total_timesteps, num_corr_tot



envrn = gym.make("Ant-v2")
envrn.reset()

if __name__ == "__main__":
    t_time = 0.0

    workers = 8
    max_itr = 3000

    # tinker high LR low etc and low etc high LR

    eps_samples            = 64
    learning_rate_         = 0.04 #0.2
    learning_rate_decay_   = 0.999 #0.995

    param_noise_std        = 0.04 #0.1
    param_noise_std_decay_ = 0.999 #0.995


    n_type = "linear"
    spinal_net = LinearNetwork(
        input_size        = envrn.observation_space.shape[0],
        output_size       = envrn.action_space.shape[0],#.n,
        action_noise_std  = None,
        num_eps_samples   = eps_samples,
        noise_std         = param_noise_std,
        neuron_type       = n_type
    )

    es_optim = EvolutionaryOptimizer(
        spinal_net,
        weight_decay        = 0.01,
        num_workers         = workers,
        max_iterations      = max_itr,
        epsilon_samples     = eps_samples,
        learning_rate       = learning_rate_,
        noise_std_decay     = param_noise_std_decay_,
        learning_rate_decay = learning_rate_decay_,
    )

    import pickle
    top_reward = -100000.0
    reward_list = list()
    for _i in range(es_optim.max_iterations):
        r, t, pr = es_optim.update(_i, 1, 1)
        t_time += t
        reward_list.append((r, _i, t_time))

        with open("saves/save_{}_net_rew_{}.pkl".format(n_type, 0), "wb") as f:
            pickle.dump(reward_list, f)

        if r >= top_reward:
            print("New Best Performance!", round(r, 5), _i, round(t/eps_samples, 5),
                  learning_rate_, param_noise_std, eps_samples)
            with open("saves/save_{}_net_{}.pkl".format(n_type, 0), "wb") as f:
                pickle.dump(spinal_net, f)
            top_reward = r
        else:
            print(round(r, 5), _i, round(t/eps_samples, 5),
                  learning_rate_, param_noise_std, eps_samples)
    print("~~~~~~~~~~~~~~~~~~~~")



"""
Graph displaying the peak reward for each plasticity/neuromod type,
also showing the average sum of rewards for each leg being disabled
as a fraction over the avg reward, and also for each max-timestep train

Show the average weight delta for plasticity and neuromod for 
disabled legs and w/o leg disablement

Also, show the effects of plasticity on task using reward feedback
e.g. [Incorporate reward into neuromod and solve task where Ant
  has to walk forward && backward, show difference in neuromod and plasticity etc]
  -> It must interact with the environment to learn what task to solve...
  
Show generalization abilities for both of these tasks for each max-timestep train (250, 500, maybe 1K)

Claims about embodiment?

"""




"""
Alpha annealing for neuromod and plast
"""
"""
* Long- and Short-term plasticity differences
* Generalization between SNN v ANN
    (potential-based, spike-timing based, hebbian?)
* Plasticity under STRONG weight penalty STDP v hebbian
"""









