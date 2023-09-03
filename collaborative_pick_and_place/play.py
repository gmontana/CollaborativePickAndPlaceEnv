import hydra
from omegaconf import DictConfig
from macpp import MultiAgentPickAndPlace
from q_learning import QTable, QLearning
import imageio

@hydra.main(config_name="config")
def load_and_play_policy(cfg: DictConfig):
    # Set rendering to True for playing
    cfg.env_enable_rendering = True

    # Initialize environment
    env = MultiAgentPickAndPlace(
        cell_size=cfg.cell_size,
        width=cfg.env_width,
        length=cfg.env_length,
        n_agents=cfg.env_n_agents,
        n_pickers=cfg.env_n_pickers,
        n_objects=cfg.env_n_objects,
        enable_rendering=cfg.env_enable_rendering
    )

    # Load the Q-table
    loaded_q_table = QTable(n_agents=env.n_agents, action_space=env.get_action_space())
    loaded_q_table.load_q_table(cfg.q_table_filename)

    # Initialize Q-learning agent
    q_learning = QLearning(env)

    # Play the game
    for _ in range(cfg.num_play_episodes):
        success, total_return, total_steps = q_learning.execute(cfg.num_max_steps)
        print(f"Success: {success} | Total steps: {total_steps} | Total return: {total_return}.")

if __name__ == "__main__":
    load_and_play_policy()

