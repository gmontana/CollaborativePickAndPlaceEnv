import gym
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import (
    DIR_TO_VEC,
)
import torch
from torch_geometric.data import Data as GeometricData

class FlatObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.env = env
        self.observation_space = None
        dim = self.n_objects * 2 * 2 + self.n_agents * 4
        self.single_space = gym.spaces.Box(np.ones(dim)*-1, np.ones(dim)*100, dtype=np.float32)
        self.observation_space = [self.single_space for _ in range(self.n_agents)]

        self.shared_space = gym.spaces.Box(
            np.ones(dim * self.n_agents) * -1,
            np.ones(dim * self.n_agents) * 100,
            dtype=np.float32
            )
        self.share_observation_space = [self.shared_space for _ in range(self.n_agents)]

    def observation(self, obs):
        """
        Flattens the observation dictionary into a 1D numpy array

        Args:
        obs (dict): The observation dictionary.

        Returns:
        list: List of flattened observation array as np.ndarray
        """
        flattened_obs_all = []
        num_agents = len(obs)

        # Information for each agent
        for i in range(num_agents):
            flattened_obs = []
            agent_key = f'agent_{i}'
            agent_obs = obs[agent_key]

            # Self information
            flattened_obs.extend(agent_obs['self']['position'])
            flattened_obs.append(int(agent_obs['self']['picker']))
            flattened_obs.append(agent_obs['self']['carrying_object']
                                 if agent_obs['self']['carrying_object'] is not None else -1)

            # Other agents' information
            for other_agent in agent_obs['agents']:
                flattened_obs.extend(other_agent['position'])
                flattened_obs.append(int(other_agent['picker']))
                flattened_obs.append(
                    other_agent['carrying_object'] if other_agent['carrying_object'] is not None else -1)

            # Objects' information
            for obj in obs[agent_key]['objects']:
                flattened_obs.extend(obj['position'])

            # Goals' information
            for goal in obs[agent_key]['goals']:
                flattened_obs.extend(goal)
            flattened_obs = np.array(flattened_obs)
            flattened_obs_all.append(flattened_obs)

        return flattened_obs_all




class GridObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = None
        single_observation_space = {}

        dim = self.n_objects * 2 * 2 + self.n_agents * 4
        single_observation_space['actor'] = gym.spaces.Box(np.ones(dim) * -1, np.ones(dim) * 100, dtype=np.float32)

        single_observation_space['image'] = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(*env.grid_size, 6),
            dtype=np.float32)
        self.observation_space = [single_observation_space for _ in range(self.n_agents)]
    def observation(self, obs):
        '''
        Converts the observation dictionary into a 3D grid representation.

        The grid is represented as a 3D NumPy array with dimensions (grid_width, grid_length, 3),
        where the last dimension corresponds to different channels for agents, objects, and goals.
        Each cell in the grid can be either 0 or 1, indicating the absence or presence of an entity.

        Args:
        obs (dict): The observation dictionary containing information about agents, objects, and goals.
        grid_size (tuple): A tuple representing the size of the grid as (grid_width, grid_length).

        Returns:
        np.ndarray: A 3D NumPy array representing the grid.
        '''


        obs_all = []

        for _, agent_data in obs.items():
            single_obs = {}

            grid = np.zeros((*self.grid_size, 6))
            x, y = agent_data['self']['position']
            grid[x, y, 3] = 1  # ID layer
            grid[x, y, 0] = 1
            grid[x, y, 4] = 1 if agent_data['self']['carrying_object'] is not None else -1
            grid[x, y, 5] = agent_data['self']['picker']

            for other_agent in agent_data['agents']:
                x, y = other_agent['position']
                grid[x, y, 0] = 1
                grid[x, y, 4] = 1 if other_agent['carrying_object'] is not None else -1
                grid[x, y, 5] = other_agent['picker']

            for obj in agent_data['objects']:
                x, y = obj['position']
                grid[x, y, 1] = 1

            for goal in agent_data['goals']:
                x, y = goal
                grid[x, y, 2] = 1

            single_obs['image'] = grid
            obs_all.append(single_obs)

        return (obs_all)



class RelGraphWrapper(gym.ObservationWrapper):
    """
    Observation Wrapper to create heterogeneous spatial graph from 2-dim observation.
    """

    def __init__(
        self, env, attr_mapping, background_id="b0", abs_id="None"
    ):
        """
        Inputs:
            env (gym.Env): the Gym environment
            attr_mapping (dict): dictionary of entity features
            background_id (str): code for the set of spatial relations used
            abs_id (str): any additional, non-spatial relations
        """
        super().__init__(env)

        self.dim = self.n_objects * 2 * 2 + self.n_agents * 4
        self.node_dim = self.n_objects * 2 + self.n_agents
        single_observation_space = {}
        single_observation_space['actor'] = gym.spaces.Box(np.ones(self.dim) * -1, np.ones(self.dim) * 100, dtype=np.float32)
        single_observation_space['image'] = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(*env.grid_size, 6),
            dtype=np.float32)
        self.observation_space = [single_observation_space for _ in range(self.n_agents)]
        self.shared_space = gym.spaces.Box(np.ones(self.dim * self.n_agents) * -1, np.ones(self.dim * self.n_agents) * 100,
                                           dtype=np.float32)
        self.share_observation_space = [self.shared_space for _ in range(self.n_agents)]


        a, b, c = self.observation_space[0]["image"].shape
        if b == c:
            self.n_attr, self.field_size = a, c
            print("Image dimensions are reversed")
        else:
            self.field_size, self.n_attr = a, c
        print(f"field_size:{self.field_size} and n_attributes:{self.n_attr}")

        self.attr_mapping = attr_mapping
        assert (
            len(self.attr_mapping) == self.n_attr
        ), f"Attribute mapping ({len(self.attr_mapping)}) needs to have a key for each attribute ({self.n_attr})"
        self.background_id = background_id
        self.rel_deter_func = self.id_to_rule_list(self.background_id)
        self.n_rel_rules = len(self.rel_deter_func)

        print("Number of relational rules", self.n_rel_rules)
        self.adj_shape = (self.node_dim,self.node_dim, self.n_rel_rules)
        self.node_obs_shape = (self.node_dim, self.n_attr)
        self.set_graph_obs_space()

    def set_graph_obs_space(self):
        self.node_observation_space = []
        self.adj_observation_space = []
        self.edge_observation_space = []
        self.agent_id_observation_space = []
        self.share_agent_id_observation_space = []
        for _ in range(self.n_agents):
            edge_dim = 1  # NOTE hardcoding edge dimension
            agent_id_dim = 1  # NOTE hardcoding agent id dimension
            self.node_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=self.node_obs_shape, dtype=np.float32
                )
            )
            self.adj_observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=self.adj_shape, dtype=np.float32)
            )
            self.edge_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(edge_dim,), dtype=np.float32
                )
            )
            self.agent_id_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(agent_id_dim,), dtype=np.float32
                )
            )
            self.share_agent_id_observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.n_agents * agent_id_dim,),
                    dtype=np.float32,
                )
            )

    def extract_dense_attributes(self, data):
        # Compute the sum of attributes along the last dimension
        attribute_sum = np.sum(abs(data), axis=2)

        non_zero_count = np.count_nonzero(attribute_sum)
        non_zero_indices = np.nonzero(attribute_sum)

        # Filter out elements whose attributes sum to zero
        filtered_data = data[non_zero_indices]

        attribute_vectors = []

        for i in range(self.n_attr):
            attribute_vectors.append(np.reshape(filtered_data[:, i], non_zero_count))

        return attribute_vectors

    def img2adj(self, img):
        """
        Takes an img/grid with multiple layers and returns vectorized attributes
        Parameters
        ----------
        img : image of the environment
        Returns
        -------
        unary_t (attribute per entity), binary_t (relation between entities)

        """
        img = img.astype(np.int32)
        objs = []
        for y, row in enumerate(img):
            for x, pixel in enumerate(row):
                # print(pixel)
                if np.sum(abs(pixel)) == 0:
                    continue
                obj = GridObject(x, y, attr=pixel)
                objs.append(obj)


        # create spatial tensors that gives for every rel. det rule a binary indicator between the entities
        self.spatial_tensors = [
            np.zeros([self.node_dim, self.node_dim])
            for _ in range(len(self.rel_deter_func))
        ]  # 14  81x81 vectors for each relation
        for obj_idx1, obj1 in enumerate(objs):
            for obj_idx2, obj2 in enumerate(objs):
                direction_vec = DIR_TO_VEC[1]
                for rel_idx, func in enumerate(self.rel_deter_func):
                    if func(obj1, obj2, direction_vec):
                        self.spatial_tensors[rel_idx][obj_idx1, obj_idx2] = 1.0

        all_binaries = self.spatial_tensors  # + self.abstract_tensors
        adj = np.stack(all_binaries, axis=2) # TODO this could potentially stack the adj wrong
        return adj

    def agent_id(self):
        return [[np.array(i)] for i in range(self.n_agents)]

    def actor_observations(self, agent_obs):
        flattened_obs = []

        # Self information
        flattened_obs.extend(agent_obs['self']['position'])
        flattened_obs.append(int(agent_obs['self']['picker']))
        flattened_obs.append(agent_obs['self']['carrying_object']
                             if agent_obs['self']['carrying_object'] is not None else -1)

        # Other agents' information
        for other_agent in agent_obs['agents']:
            flattened_obs.extend(other_agent['position'])
            flattened_obs.append(int(other_agent['picker']))
            flattened_obs.append(
                other_agent['carrying_object'] if other_agent['carrying_object'] is not None else -1)

        # Objects' information
        for obj in agent_obs['objects']:
            flattened_obs.extend(obj['position'])

        # Goals' information
        for goal in agent_obs['goals']:
            flattened_obs.extend(goal)

        return np.array(flattened_obs)

    def grid_observation(self, agent_data):
        single_obs = {}
        grid = np.zeros((*self.grid_size, 6))
        x, y = agent_data['self']['position']
        grid[x, y, 3] = 1  # ID layer
        grid[x, y, 0] = 1
        grid[x, y, 4] = 1 if agent_data['self']['carrying_object'] is not None else -1
        grid[x, y, 5] = agent_data['self']['picker']

        for other_agent in agent_data['agents']:
            x, y = other_agent['position']
            grid[x, y, 0] = 1
            grid[x, y, 4] = 1 if other_agent['carrying_object'] is not None else -1
            grid[x, y, 5] = other_agent['picker']

        for obj in agent_data['objects']:
            x, y = obj['position']
            grid[x, y, 1] = 1

        for goal in agent_data['goals']:
            x, y = goal
            grid[x, y, 2] = 1

        return grid


    def observation(self, observation):
        adj_n = []
        node_obs_n = []
        actor_obs_n = []
        for ob in observation.values():
            actor_obs_n.append(self.actor_observations(ob))
            grid = self.grid_observation(ob)

            adj_n.append(self.img2adj(grid))
            node_obs_n.append(np.stack(self.extract_dense_attributes(grid.astype(np.int32)), axis=-1))


        agent_id_n = self.agent_id()
        return actor_obs_n, agent_id_n, node_obs_n, adj_n


    def step(self, action):
        observation, reward_n, done_n, info = self.env.step(action)
        adj_n = []
        node_obs_n = []
        actor_obs_n = []
        for ob in observation.values():
            actor_obs_n.append(self.actor_observations(ob))
            grid = self.grid_observation(ob)

            adj_n.append(self.img2adj(grid))
            node_obs = np.stack(self.extract_dense_attributes(grid.astype(np.int32)), axis=-1)
            if node_obs.shape[0]<self.node_dim:
                node_obs = np.concatenate([node_obs, np.zeros((self.node_dim-node_obs.shape[0],self.n_rel_rules))])
            node_obs_n.append(node_obs)

        agent_id_n = self.agent_id()

        return actor_obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info

    def id_to_rule_list(self, background_id):
        if background_id in ["b0", "nolocal"]:
            rel_deter_func = [
                is_left,
                is_right,
                is_front,
                is_back,
                is_aligned,
                is_close,
            ]
        elif background_id in ["b2", "local"]:
            rel_deter_func = [
                is_top_adj,
                is_left_adj,
                is_top_right_adj,
                is_top_left_adj,
                is_down_adj,
                is_right_adj,
                is_down_left_adj,
                is_down_right_adj,
            ]
        elif background_id in ["b3", "all"]:
            rel_deter_func = [
                is_top_adj,
                is_left_adj,
                is_top_right_adj,
                is_top_left_adj,
                is_down_adj,
                is_right_adj,
                is_down_left_adj,
                is_down_right_adj,
                is_left,
                is_right,
                is_front,
                is_back,
                is_aligned,
                is_close,
            ]
        else:
            rel_deter_func = None
        return rel_deter_func


def is_front(obj1, obj2, direction_vec) -> bool:
    diff = obj2.pos - obj1.pos
    return diff @ direction_vec > 0.1


def is_back(obj1, obj2, direction_vec) -> bool:
    diff = obj2.pos - obj1.pos
    return diff @ direction_vec < -0.1


def is_left(obj1, obj2, direction_vec) -> bool:
    left_vec = rotate_vec2d(direction_vec, -90)
    diff = obj2.pos - obj1.pos
    return diff @ left_vec > 0.1


def is_right(obj1, obj2, direction_vec) -> bool:
    left_vec = rotate_vec2d(direction_vec, 90)
    diff = obj2.pos - obj1.pos
    return diff @ left_vec > 0.1


#### auxilliary rules


def is_close(obj1, obj2, direction_vec=None) -> bool:
    distance = np.abs(obj1.pos - obj2.pos)
    return np.sum(distance) == 1


def is_aligned(obj1, obj2, direction_vec=None) -> bool:
    diff = obj2.pos - obj1.pos
    return np.any(diff == 0)


#### convolutional rules
def is_top_adj(obj1, obj2, direction_vec=None) -> bool:
    return obj1.x == obj2.x and obj1.y == obj2.y + 1


def is_left_adj(obj1, obj2, direction_vec=None) -> bool:
    return obj1.y == obj2.y and obj1.x == obj2.x - 1


def is_top_left_adj(obj1, obj2, direction_vec=None) -> bool:
    return obj1.y == obj2.y + 1 and obj1.x == obj2.x - 1


def is_top_right_adj(obj1, obj2, direction_vec=None) -> bool:
    return obj1.y == obj2.y + 1 and obj1.x == obj2.x + 1


def is_down_adj(obj1, obj2, direction_vec=None) -> bool:
    return is_top_adj(obj2, obj1)


def is_right_adj(obj1, obj2, direction_vec=None) -> bool:
    return is_left_adj(obj2, obj1)


def is_down_right_adj(obj1, obj2, direction_vec=None) -> bool:
    return is_top_left_adj(obj2, obj1)


def is_down_left_adj(obj1, obj2, direction_vec=None) -> bool:
    return is_top_right_adj(obj2, obj1)


#### Another set of rules which quarters the grid into areas
def top_left(obj1, obj2, direction_vec) -> bool:
    return (obj1.x - obj2.x) <= (obj1.y - obj2.y)


def top_right(obj1, obj2, direction_vec) -> bool:
    return -(obj1.x - obj2.x) <= (obj1.y - obj2.y)


def down_left(obj1, obj2, direction_vec) -> bool:
    return top_right(obj2, obj1, direction_vec)


def down_right(obj1, obj2, direction_vec) -> bool:
    return top_left(obj2, obj1, direction_vec)


def is_food(obj1, obj2, mapping) -> bool:
    """
    All to all rule between agents and foods
    """
    return obj1.attr[mapping["agent"]] != 0 and obj2.attr[mapping["food"]] != 0


def is_agent(obj1, obj2, mapping) -> bool:
    """
    This rule goes from all agents to agents, including self-loops
    """
    return obj2.attr[mapping["agent"]] != 0


def is_other_player(obj1, obj2, mapping) -> bool:
    """
    This rule only goes from the observing agent to the other agents
    """
    return obj1[mapping["id"]] != 0 and obj2.attr[mapping["agent"]] != 0


class GridObject:
    "object is specified by its location"

    def __init__(self, x, y, attr=None):
        self.x = x
        self.y = y
        self.attr = attr

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def name(self):
        return "_" + str(self.x) + str(self.y)


def rotate_vec2d(vec, degrees):
    """
    rotate a vector anti-clockwise
    :param vec:
    :param degrees:
    :return:
    """
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R @ vec

def to_gd(data: torch.Tensor, unary_t, GAT=False) -> GeometricData:
    """
    takes batch of adjacency geometric data and transforms it to a GeometricData object for torch.geometric

    Parameters
    --------
    data : heterogeneous adjacency matrix (nb_relations, nb_objects, nb_objects)
    unary_t: node features
    GAT : full graph without heterogenous relations for GAT
    """
    unary_t = torch.tensor(unary_t, dtype=torch.float32)
    data = torch.tensor(data)
    nz = torch.nonzero(data)

    # list of node to node indicating an edge
    edge_index = nz[:, 1:].T

    if GAT:
        return GeometricData(x=unary_t, edge_index=edge_index)
    else:
        # list of all edges and what relationtype they have
        edge_attr = nz[:, 0]
        return GeometricData(x=unary_t, edge_index=edge_index, edge_attr=edge_attr)
