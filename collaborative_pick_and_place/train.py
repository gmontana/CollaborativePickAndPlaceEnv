from macpp import MultiAgentPickAndPlace
from q_learning import QLearning
from sacred import Experiment
import yaml

# Define constants for configuration file and experiment name
CONFIG_FILE = 'config.yaml'
EXPERIMENT_NAME = 'q_learning_experiment'

ex = Experiment(EXPERIMENT_NAME)

def initialize_environment(config):
    """Initialize environment from a configuration dictionary."""
    return MultiAgentPickAndPlace(
        width=config['env_width'],
        length=config['env_length'],
        n_agents=config['env_n_agents'],
        n_pickers=config['env_n_pickers'],
        n_objects=config['env_n_objects'],
        enable_rendering=config['env_enable_rendering']
    )

@ex.config
def cfg():

    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # Unpack configuration directly
    q_table_filename = config['q_table_filename']
    env_width = config['env_width']
    env_length = config['env_length']
    env_n_agents = config['env_n_agents']
    env_n_pickers = config['env_n_pickers']
    env_n_objects = config['env_n_objects']
    env_enable_rendering = config['env_enable_rendering']
    episodes = config['episodes']
    max_steps_per_episode = config['max_steps_per_episode']
    discount_factor = config['discount_factor']
    min_exploration = config['min_exploration']
    exploration_rate = config['exploration_rate']
    exploration_decay = config['exploration_decay']
    min_learning_rate = config['min_learning_rate']
    learning_rate = config['learning_rate']
    learning_rate_decay = config['learning_rate_decay']

@ex.main
def run_experiment(episodes, max_steps_per_episode, learning_rate, discount_factor, exploration_rate, exploration_decay, min_exploration, env_width, env_length, env_n_agents, env_n_pickers, env_n_objects, env_enable_rendering, q_table_filename):

    # initialise env
    env = initialize_environment({
        'env_width': env_width,
        'env_length': env_length,
        'env_n_agents': env_n_agents,
        'env_n_pickers': env_n_pickers,
        'env_n_objects': env_n_objects,
        'env_enable_rendering': env_enable_rendering
    })

    # train algo
    q_learning = QLearning(env, learning_rate, discount_factor, exploration_rate, exploration_decay, min_exploration)
    q_table, rewards_all_episodes, _ = q_learning.train(episodes, max_steps_per_episode)

    # save final policy 
    q_table.save_q_table(q_table_filename)

if __name__ == "__main__":
    ex.run_commandline()

