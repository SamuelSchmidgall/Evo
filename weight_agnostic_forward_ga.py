import gym
import numpy as np
from multiprocessing import Pool
from copy import deepcopy

MAX_SEED = 2**16 - 1

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    return (np.tanh(x/2.0) + 1.0)/2.0

def relu(x):
    return np.maximum(0, x)

def identity(x):
    return x

def gaussian(x):
    return np.exp(-np.multiply(x, x) / 2.0)

def gaussian2(x):
    return np.exp(-np.multiply(x, x) * 2.0)


def step(x):
    return 1.0*(x>0.0)

ACTIVATIONS = { 
    "tanh" : np.tanh,
    "id"   : identity,
    "sin"  : lambda x: np.sin(np.pi*x),
    "cos"  : lambda x: np.cos(np.pi*x),
    "sig"  : sigmoid,
    "step" : step,
    "clip" : lambda x:np.clip(x, -1, 1)
}

# only using activations that work with plasticity
ACTIVATION_CHOICES = [
    "tanh",
    "sin" ,
    "cos" ,
    "sig" ,
    "clip",
    "step",
]


class WANN:
    def __init__(self, graph):
        self.graph = graph

    def forward(self, x, weight_value):
        # fifo queue for DFS
        x = x.squeeze()
        prior_nodes = set()
        for _inp in self.graph['input']:
            self.graph['nodes'][_inp]['val'] += x[_inp]
        queue = deepcopy(self.graph['input'])
        while len(queue) > 0:
            node = queue[0]
            if len(self.graph['nodes'][node]['outgoing']) > 0:
                w_val = ACTIVATIONS[self.graph['nodes'][node]['activation']](self.graph['nodes'][node]['val'])
                # for self connections
                self.graph['nodes'][node]['val'] = 0
                for outgoing_node in self.graph['nodes'][node]['outgoing']:
                    if outgoing_node not in prior_nodes:
                        queue.append(outgoing_node)
                    prior_nodes.add(outgoing_node)
                    self.graph['nodes'][outgoing_node]['val'] += w_val*weight_value
            #else:
            #    self.graph['nodes'][node]['val'] = 0
            queue.pop(0)
        output = deepcopy([ACTIVATIONS[self.graph['nodes'][_]['activation']](self.graph['nodes'][_]['val']) for _ in self.graph['output']])
        for _outp in self.graph['output']:
            self.graph['nodes'][_outp]['val'] = 0
        return np.tanh(np.array(output))

    def reset(self):
        for node in self.graph['nodes']:
            self.graph['nodes'][node]['val'] = 0


def mutate(sub_elite, num_modifications):
    random_operations_prob = [0.5, 0.5, 0.0]
    random_operations = ["addweight", "node", "activation"]
    for _mod in range(num_modifications):
        operation = np.random.choice(random_operations, p=random_operations_prob)
        if operation == "activation":
            node = np.random.choice(list(sub_elite["nodes"].keys()))
            sub_elite["nodes"][node]["activation"] \
                = "tanh"#np.random.choice(ACTIVATION_CHOICES)
        if operation == "node":
            nodes = sub_elite["nodes"]
            _conns = [(nodes[_node]["outgoing"], _node)
                      for _node in nodes if len(nodes[_node]["outgoing"]) > 0]
            _connections = list()
            for _connection in _conns:
                for _node in _connection[0]:
                    _connections.append((_connection[1], _node))
            _conn = _connections[np.random.choice(list(range(len(_connections))))]
            _outgoing, _receiving = _conn[0], _conn[1]
            _new_node_id = sub_elite["next_node_id"]
            sub_elite["next_node_id"] += 1
            sub_elite["nodes"][_new_node_id] = {
                "val": 0,
                "activation": "tanh",#np.random.choice(ACTIVATION_CHOICES),
                "outgoing": {_receiving},
            }
            sub_elite["nodes"][_outgoing]["outgoing"].remove(_receiving)
            sub_elite["nodes"][_outgoing]["outgoing"].add(_new_node_id)
        elif operation == "addweight":
            for _try in range(10):
                node1 = np.random.choice(list(sub_elite["nodes"].keys()))
                node2 = np.random.choice(list(sub_elite["nodes"].keys()))
                if node2 in sub_elite["input"] or \
                        (node1 == node1 and node1 in sub_elite["output"]):
                    continue
                elif node2 not in sub_elite["output"] and \
                        len(sub_elite["nodes"][node2]['outgoing']) == 0:
                    continue
                sub_elite["nodes"][node1]['outgoing'].add(node2)
                break
        return sub_elite


def compute_returns(seed, graph, num_env_rollouts, mutate_prob=0.01):
    """
    :param seed:
    :return:
    """
    avg_stand = 0
    returns = list()
    max_env_interacts = 500
    total_env_interacts = 0
    local_env = gym.make("Ant-v2")

    w_samples = 6
    intra_mutate_prob = mutate_prob
    for _sample in range(len(graph)):
        return_avg = 0.0
        weight_samples = [-2, -1, -0.5, 0.5, 1, 2] #[np.random.uniform(-2, 2) for _ in range(w_samples)]
        for _w_sample in weight_samples:
            for _roll in range(num_env_rollouts):
                network = WANN(deepcopy(graph[_sample]))
                network.reset()
                state = local_env.reset()
                for _inter in range(max_env_interacts):
                    if np.random.uniform(0.0, 1.0) <= intra_mutate_prob:
                        network.graph = mutate(network.graph, 1)
                    state = state.reshape((1, state.size))
                    # appending bias activity
                    state = np.concatenate((state, np.ones((1, 1))), axis=1)
                    action1 = network.forward(state, _w_sample).squeeze()
                    state, reward, game_over, _info = local_env.step(action1)
                    return_avg += reward
                    avg_stand += 1
                    total_env_interacts += 1
                    if game_over:
                        break
            #rp = np.random.uniform(0, 1)
            #if rp < 0.2:
            # todo: penalize connections
        returns.append(return_avg / (w_samples*num_env_rollouts))
    return np.array(returns), total_env_interacts / (w_samples*num_env_rollouts), 0



class GAParamSampler:
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

    def sample(self, params, seed):
        """
        Sample noise for network parameters
        :param params: (ndarray) network parameters
        :param seed: (int) random seed used to sample
        :return: (ndarray) sampled noise
        """
        p_size = params.size
        noise_vectors = list()
        """ Regenerate original param """
        for _seed in seed:
            noise_vector = np.zeros((1, p_size))
            for _sub_seed in _seed:
                rand_m = np.random.RandomState(seed=_sub_seed)
                noise_vector += \
                    rand_m.randn(1, p_size) * self.noise_std
            noise_vectors.append(noise_vector)
        noise_vector = np.concatenate(noise_vectors, axis=0)
        return noise_vector




"""
Todo:
 We can find the best architecture with weight sharing
 -> We can find the best architecture with WEIGHT AND NODE GROWTH ONLINE RANDOMLY anneal
"""


class GAOptimizer:
    def __init__(self, input_dim, output_dim, num_workers=1, epsilon_samples=48, max_iterations=2000, num_env_rollouts=1, optimization="elite_selection"):
        assert (epsilon_samples % num_workers == 0), "Epsilon sample size not divis num workers"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_workers = num_workers
        self.max_iterations = max_iterations
        self.epsilon_samples = epsilon_samples
        self.num_env_rollouts = num_env_rollouts

        self.optimization = optimization

        if self.optimization == "elite_selection":
            self.elites = 30
            self.unchanged_elites = 10
            self.total_elites = self.elites + self.unchanged_elites

        elif self.optimization == "tournament":
            self.tournament_size = 8
            self.cull_percentage = 0.25
            self.unchanged_percentage = 0.10
            self.cull_population = int(epsilon_samples*self.cull_percentage)
            self.unchanged_elites = int(epsilon_samples*self.unchanged_percentage)
            self.population_size = epsilon_samples - self.unchanged_elites
            self.total_elites = self.population_size + self.unchanged_elites

        init_elite_graph = list()
        self.input_nodes = list(range(self.input_dim))
        self.output_nodes = [_ + self.input_dim for _ in range(self.output_dim)]
        for _ in range(self.total_elites):
            input_nodes = np.random.choice(
                self.input_nodes, replace=False, size=(np.random.randint(1, self.input_dim),))
            random_perm = set(input_nodes)
            graph = {
                "input": self.input_nodes,
                "output": self.output_nodes,
                "nodes":{
                    _: {"outgoing":
                        set(np.random.choice(
                            self.output_nodes,
                            replace=False,
                            size=(1,))) if _ in random_perm else set(), #np.random.randint(1, self.output_dim)
                        "activation": "id",
                        "val": 0,
                        }
                    for _ in self.input_nodes
                },
                "next_node_id": self.input_dim + self.output_dim,
            }
            graph["nodes"].update({
                    _:{"outgoing": set(),
                       "activation": "tanh",
                       "val": 0
                       }
                for _ in self.output_nodes
                })
            init_elite_graph.append(graph)

        if self.optimization == "elite_selection":
            self.previous_elites = [
                (-1000, [0, ], init_elite_graph[_+self.elites]) for _ in range(self.unchanged_elites)]
            self.elite_candidates = [
                (-1000, [0,], init_elite_graph[_]) for _ in range(self.elites)] + self.previous_elites
        elif self.optimization == "tournament":
            self.previous_elites = [
                (-1000, [0, ], init_elite_graph[_+self.population_size]) for _ in range(self.unchanged_elites)]
            self.elite_candidates = [
                (-1000, [0,], init_elite_graph[_]) for _ in range(self.population_size)] + self.previous_elites

        self.best = self.elite_candidates[-1]
        self.elite_list = list(range(self.total_elites))

    def parallel_returns(self, x):
        """
        Function call for collecting parallelized rewards
        :param x: (tuple(int, int)) worker id and seed
        :return: (list) collected returns
        """
        worker_id, seed, graph = x
        return compute_returns(seed=seed, graph=graph, num_env_rollouts=self.num_env_rollouts)

    def update(self, iteration):
        """
        To insert a node, we split an existing connection into two connections that
        pass through this new hidden node. The activation function of this new node is randomly assigned.
        New connections are added between previously unconnected nodes, respecting the feed-forward
        property of the network. When activation functions of hidden nodes are changed, they are assigned at
        random.
        """
        net_seeds = list()
        graph_list = list()
        timestep_list = list()
        sample_returns = list()
        #np.random.seed(iteration*7**2)

        mutation_seeds = list()
        for _w in range(self.num_workers):
            graph = list()
            seed_samples = list()
            for _k in range(self.epsilon_samples // self.num_workers):
                elite = np.random.choice(self.elite_list)
                seed_samples.append([0])#[self.elite_candidates[elite][1] +
                      #[7*(iteration*self.epsilon_samples) +_k + _w*self.num_workers]])
                graph.append(self.elite_candidates[elite][2])
            mutation_seeds.append((_w, seed_samples, graph))
        with Pool(self.num_workers) as p:
            values = p.map(func=self.parallel_returns, iterable=mutation_seeds)

        # todo: we dont need pool if we return seed or net id
        num_corr_tot = 0
        total_timesteps = 0
        for _worker in range(self.num_workers):
            _, net_seed, graph_l = mutation_seeds[_worker]
            returns, timesteps, num_corr = values[_worker]
            num_corr_tot += num_corr
            timestep_list.append(timesteps)
            total_timesteps += timesteps

            sample_returns.append(returns)
            net_seeds += net_seed
            graph_list += graph_l
        sample_returns = np.concatenate(sample_returns)

        net_rew = [(sample_returns[_k], net_seeds[_k], graph_list[_k]) for _k in range(len(net_seeds))]
        net_rew.sort(key=lambda x: x[0], reverse=True)

        if net_rew[0][0] > self.best[0]:
            self.best = net_rew[0]

        self.previous_elites = net_rew[:self.unchanged_elites]
        self.elite_list = list(range(self.total_elites))

        if self.optimization == "elite_selection":
            self.elite_candidates = net_rew[:self.elites]

        elif self.optimization == "tournament":
            self.elite_candidates = list()
            net_rew = net_rew[:-self.cull_population]
            len_net_rew = range(len(net_rew))
            for _ in range(self.total_elites):
                tournament = [net_rew[np.random.choice(len_net_rew)] for _ in range(self.tournament_size)]
                tournament.sort(key=lambda x: x[0], reverse=True)
                self.elite_candidates.append(tournament[0])
            self.elite_list = list(range(self.total_elites))

        # todo: learn probabilities of each, higher prob of remove to encourage sparseness?
        for _elite in range(len(self.elite_candidates)):
            _sub_elite = deepcopy(self.elite_candidates[_elite][2])
            num_modifications = np.random.randint(low=1, high=4)
            _sub_elite = mutate(_sub_elite, num_modifications)
            self.elite_candidates[_elite] = \
                (self.elite_candidates[_elite][0], self.elite_candidates[_elite][1], _sub_elite)

        if self.optimization == "elite_selection" or self.optimization == "tournament":
            self.elite_candidates += self.previous_elites

        avg_return_rec = sum(sample_returns) / len(sample_returns)
        return avg_return_rec, total_timesteps, num_corr_tot, max([_[0] for _ in self.elite_candidates]), self.best[0]


envrn = gym.make("Ant-v2")
envrn.reset()

if __name__ == "__main__":
    t_time = 0.0

    workers = 8
    max_itr = 1000

    eps_samples = 128
    num_net_env_rollouts = 3

    n_type = "linear"
    optimization = "tournament"

    es_optim = GAOptimizer(
        input_dim        = envrn.observation_space.shape[0],
        output_dim       = envrn.action_space.shape[0],
        num_workers      = workers,
        max_iterations   = max_itr,
        epsilon_samples  = eps_samples,
        num_env_rollouts = num_net_env_rollouts,
        optimization     = optimization
    )

    import pickle
    top_reward = -100000.0
    reward_list = list()
    for _i in range(es_optim.max_iterations):
        r, t, pr, max_elite, max_best = es_optim.update(_i)
        t_time += t
        reward_list.append((r, _i, t_time))

        with open("save_{}_net_rew_{}.pkl".format(n_type, 0), "wb") as f:
            pickle.dump(reward_list, f)

        with open("save_{}_net_{}.pkl".format(n_type, 0), "wb") as f:
            pickle.dump(es_optim.elite_candidates, f)

        if r >= top_reward:
            print("New Best Performance!", round(r, 5), _i, round(t/eps_samples, 5), eps_samples, max_elite, max_best)
            top_reward = r
        else:
            print(round(r, 5), _i, round(t/eps_samples, 5), eps_samples, max_elite, max_best)
    print("~~~~~~~~~~~~~~~~~~~~")



