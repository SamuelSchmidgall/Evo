import gym
import numpy as np
from multiprocessing import Pool
from copy import deepcopy

MAX_SEED = 2**16 - 1


#def sigmoid(v):
#    return 1 / (1 + np.exp(-v))

#def relu(x):
#    x[x < 0] = 0
#    return x

def identity(x):
    return x


ACTIVATIONS = {"tan":np.tanh, "id":identity, "sin":np.sin, "cos":np.cos, "abs":np.abs, } # "sig":sigmoid


class WANN:
    def __init__(self, graph):
        self.graph = graph
        self.layers = list()
        for layer in range(self.graph["depth"]):
            self.layers.append(np.zeros((self.graph["layer_len"][layer],)))

    def forward(self, x, weight_value):
        self.layers[0] = x.squeeze()
        for layer in range(1, len(self.layers)):
            for node in self.graph["layer"][layer]:
                g_node = self.graph["nodes"][node]
                node_conns = g_node['incoming']
                for conn in node_conns:
                    g_conn = self.graph["nodes"][conn]
                    self.layers[layer][g_node["pos"]] += \
                        self.layers[g_conn["depth"]][g_conn["pos"]]*weight_value
                self.layers[layer][g_node["pos"]] = ACTIVATIONS[
                    g_node["activation"]](self.layers[layer][g_node["pos"]])
        output = np.zeros((1,))
        for node in self.graph["layer"][999]:
            g_node = self.graph["nodes"][node]
            node_conns = g_node['incoming']
            for conn in node_conns:
                g_conn = self.graph["nodes"][conn]
                output += self.layers[g_conn["depth"]][g_conn["pos"]] * weight_value
        return output

    def reset(self):
        pass


def compute_returns(seed, graph):
    """
    :param seed:
    :return:
    """
    avg_stand = 0
    returns = list()
    max_env_interacts = 500
    total_env_interacts = 0
    local_env = gym.make("CartPoleSwingup-v1")

    avg_action = 0.0
    num_env_rollouts = 1
    weight_samples = [np.random.uniform(-2, 2)]#[-2, -1, -0.5, 0.5, 1, 2]
    for _sample in range(len(graph)):
        return_avg = 0.0
        network = WANN(graph[_sample])
        for _w_sample in weight_samples:
            for _roll in range(num_env_rollouts):
                network.reset()
                state = local_env.reset()
                for _inter in range(max_env_interacts):
                    state = state.reshape((1, state.size))
                    action1 = network.forward(state, _w_sample)
                    avg_action += action1
                    state, reward, game_over, _info = local_env.step(action1)
                    return_avg += reward
                    avg_stand += 1
                    total_env_interacts += 1
                    if game_over:
                        break
            rp = np.random.uniform(0, 1)
            if rp < 0.2:
                conns = list()
                nodes = network.graph["nodes"]
                _conns = [(nodes[_node]["incoming"], _node)
                    for _node in nodes if nodes[_node]["incoming"] is not None]
                for _conn in _conns:
                    for _sub_conn in _conn[0]:
                        conns.append((_sub_conn, _conn[1]))
                num_conns = len(conns)
                return_avg -= num_conns*0.03
            returns.append(return_avg / num_env_rollouts)
    return np.array(returns), total_env_interacts / num_env_rollouts, 0



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
    def __init__(self, num_workers=1, epsilon_samples=48, weight_decay=0.01, max_iterations=2000):
        assert (epsilon_samples % num_workers == 0), "Epsilon sample size not divis num workers"
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.max_iterations = max_iterations
        self.epsilon_samples = epsilon_samples

        self.elites = 100
        self.elite_saves = 5

        init_elite_graph = list()
        self.output_nodes = [999]
        self.input_nodes = [0, 1, 2, 3]
        for _ in range(self.elites):
            graph = {
                "layer":{
                    0: [0, 1, 2, 3],
                    1: [4,],
                    999: [5,],
                },
                "nodes":{
                    0: {"incoming": None, "depth": 0, "activation": "id", "pos": 0},
                    1: {"incoming": None, "depth": 0, "activation": "id", "pos": 1},
                    2: {"incoming": None, "depth": 0, "activation": "id", "pos": 2},
                    3: {"incoming": None, "depth": 0, "activation": "id", "pos": 3},
                    4: {"incoming": [3], "depth": 1, "activation": "id", "pos": 0},
                    5: {"incoming": np.random.permutation(self.input_nodes).tolist(),
                        "depth": 999, "activation": "id", "pos": 0},

                },
                "depth": 2,
                "layer_len":{
                    0: 4,
                    1: 1,
                    999: 1,
                },
                "next_node_id": 6,
            }
            init_elite_graph.append(graph)

        self.previous_elites = [
            (-1000, [0,], init_elite_graph[_]) for _ in range(self.elites)]
        self.elite_candidates = self.previous_elites
        self.elite_list = np.array(list(range(len(self.elite_candidates))))

    def parallel_returns(self, x):
        """
        Function call for collecting parallelized rewards
        :param x: (tuple(int, int)) worker id and seed
        :return: (list) collected returns
        """
        worker_id, seed, graph = x
        return compute_returns(seed=seed, graph=graph)

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
        np.random.seed(iteration*13*7**2)

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
        if iteration == 0:
            self.elite_candidates = net_rew[:self.elites]
        else:
            self.elite_candidates = net_rew[:self.elites-self.elite_saves] + self.previous_elites

        #self.elite_candidates = [_ for _ in self.elite_candidates if len(_[2]) > 0]
        self.elite_list = list(range(len(self.elite_candidates)))

        # todo: learn probabilities of each, higher prob of remove to encourage sparseness?
        random_operations = ["addweight", "activation", "node", "none"]
        random_operations_prob = [0.15, 0.15, 0.15, 0.55]
        for _elite in range(len(self.elite_candidates)):
            _sub_elite = deepcopy(self.elite_candidates[_elite][2])
            operation = np.random.choice(random_operations, p=random_operations_prob)
            if operation == "activation":
                node = np.random.choice(list(self.elite_candidates[_elite][2]["nodes"].keys()))
                self.elite_candidates[_elite][2]["nodes"][node]["activation"]\
                    = np.random.choice(list(ACTIVATIONS.keys()))
            elif operation == "node":
                conns = list()
                nodes = _sub_elite["nodes"]
                _conns = [(nodes[_node]["incoming"], _node)
                          for _node in nodes if nodes[_node]["incoming"] is not None]
                for _conn in _conns:
                    for _sub_conn in _conn[0]:
                        conns.append((_sub_conn, _conn[1]))
                random_conn = conns[np.random.choice(list(range(len(conns))))]
                new_node_id = _sub_elite["next_node_id"]
                incoming_node, outgoing_node = random_conn
                new_node_layer = _sub_elite["nodes"][incoming_node]["depth"] + 1
                if _sub_elite["nodes"][incoming_node]["depth"] + 1 \
                    == _sub_elite["nodes"][outgoing_node]["depth"]:
                    _sub_elite["nodes"][outgoing_node]["depth"] += 1
                    if _sub_elite["nodes"][outgoing_node]["depth"] not in _sub_elite["layer"]:
                        _sub_elite["layer"][_sub_elite["nodes"][outgoing_node]["depth"]] = list()
                        _sub_elite["layer_len"][_sub_elite["nodes"][outgoing_node]["depth"]] = 0
                        _sub_elite["depth"] += 1
                    _sub_elite["layer"][_sub_elite["nodes"][outgoing_node]["depth"]].append(outgoing_node)
                    _sub_elite["layer"][_sub_elite["nodes"][outgoing_node]["depth"]-1].remove(outgoing_node)
                    _sub_elite["layer_len"][_sub_elite["nodes"][outgoing_node]["depth"]] += 1
                    _sub_elite["layer_len"][_sub_elite["nodes"][outgoing_node]["depth"]-1] -= 1
                    _sub_elite["nodes"][outgoing_node]["pos"] = _sub_elite["layer_len"][
                        _sub_elite["nodes"][outgoing_node]["depth"]] - 1
                if new_node_layer not in _sub_elite["layer"]:
                    _sub_elite["layer"][new_node_layer] = list()
                    _sub_elite["layer_len"][new_node_layer] = 0
                    _sub_elite["depth"] += 1
                _sub_elite["nodes"][outgoing_node]["incoming"].remove(incoming_node)
                _sub_elite["nodes"][outgoing_node]["incoming"].append(new_node_id)
                new_node_position = _sub_elite["layer_len"][new_node_layer]
                _sub_elite["layer_len"][new_node_layer] += 1
                _sub_elite["nodes"][new_node_id] = {
                    "depth": new_node_layer,
                    "pos": new_node_position,
                    "incoming": [incoming_node],
                    "activation": np.random.choice(list(ACTIVATIONS.keys())),
                }
                _sub_elite["layer"][new_node_layer].append(new_node_id)
                _sub_elite["next_node_id"] += 1
            elif operation == "addweight":
                node1 = np.random.choice(list(_sub_elite["nodes"].keys()))
                node2 = np.random.choice(list(_sub_elite["nodes"].keys()))
                if node1 == node2: continue
                node1_d = (node1, _sub_elite["nodes"][node1])
                node2_d = (node2, _sub_elite["nodes"][node2])
                nodes = sorted([node1_d, node2_d], key=lambda x: x[1]["depth"])
                node1_d, node2_d = nodes[0], nodes[1]
                if node1_d[1]['depth'] >= node2_d[1]['depth'] or \
                        _sub_elite["nodes"][node2]["incoming"] is None or\
                        node1 in _sub_elite["nodes"][node2]["incoming"]: continue
                _sub_elite["nodes"][node2_d[0]]["incoming"].append(node1_d[0])
            self.elite_candidates[_elite] = \
                (self.elite_candidates[_elite][0], self.elite_candidates[_elite][1], _sub_elite)

        # todo: evaluate each elite over 30 ts and set new prev elite
        self.previous_elites = sorted(
            self.elite_candidates, key=lambda x: x[0], reverse=True)[:self.elite_saves]

        avg_return_rec = sum(sample_returns) / len(sample_returns)
        return avg_return_rec, total_timesteps, num_corr_tot, max([_[0] for _ in self.elite_candidates])



# todo: rebuild using node indices


envrn = gym.make("CartPoleSwingup-v1")
envrn.reset()

if __name__ == "__main__":
    t_time = 0.0

    workers = 8
    max_itr = 3000

    eps_samples = 200

    n_type = "linear"

    es_optim = GAOptimizer(
        weight_decay        = 0.01,
        num_workers         = workers,
        max_iterations      = max_itr,
        epsilon_samples     = eps_samples,
    )

    import pickle
    top_reward = -100000.0
    reward_list = list()
    for _i in range(es_optim.max_iterations):
        r, t, pr, max_elite = es_optim.update(_i)
        t_time += t
        reward_list.append((r, _i, t_time))

        with open("save_{}_net_rew_{}.pkl".format(n_type, 0), "wb") as f:
            pickle.dump(reward_list, f)

        if r >= top_reward:
            print("New Best Performance!", round(r, 5), _i, round(t/eps_samples, 5), eps_samples, max_elite)
            top_reward = r
        else:
            print(round(r, 5), _i, round(t/eps_samples, 5), eps_samples, max_elite)
    print("~~~~~~~~~~~~~~~~~~~~")



