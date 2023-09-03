from macpp import MultiAgentPickAndPlace
from q_learning import QTable, QLearning
from train import initialize_environment
import yaml
import os

def load_and_play_policy(config_file, num_play_episodes, num_max_steps):
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found!")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    config['env_enable_rendering'] = True

    env = initialize_environment(config)

    loaded_q_table = QTable(n_agents=env.n_agents, action_space=env.get_action_space())
    loaded_q_table.load_q_table(config['q_table_filename'])  

    q_learning = QLearning(env)  

    for _ in range(num_play_episodes):
        success, total_return, total_steps = q_learning.execute(num_max_steps)
        print(f"Success: {success} | Total steps: {total_steps} | Total return: {total_return}.")

if __name__ == "__main__":

    config_file = "config.yaml"
    num_play_episodes = 10
    num_max_steps = 50
    load_and_play_policy(config_file, num_play_episodes, num_max_steps)

