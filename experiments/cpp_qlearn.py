import os
import sys
import wandb
import numpy as np
from environments.collaborative_pick_and_place.macpp import MultiAgentPickAndPlace
from algos.q_learning import QLearning

sys.path.append('/home/gm13/Dropbox/mycode/envs/multi_agent_rl')


# Parameters from config.yaml
q_table_filename = "q_table.npy"
cell_size = 300
env_width = 4
env_length = 4
env_n_agents = 2
env_n_pickers = 1
env_n_objects = 2
env_enable_rendering = False
episodes = 500000
max_steps_per_episode = 50
discount_factor = 0.90
min_exploration = 0.02
exploration_rate = 1.0
exploration_decay = 0.99
min_learning_rate = 0.01
learning_rate = 0.1
learning_rate_decay = 0.995

# Initialize WandB
wandb.init(project="cpp_qlearn", config={
    "q_table_filename": q_table_filename,
    "cell_size": cell_size,
    "env_width": env_width,
    "env_length": env_length,
    "env_n_agents": env_n_agents,
    "env_n_pickers": env_n_pickers,
    "env_n_objects": env_n_objects,
    "env_enable_rendering": env_enable_rendering,
    "episodes": episodes,
    "max_steps_per_episode": max_steps_per_episode,
    "discount_factor": discount_factor,
    "min_exploration": min_exploration,
    "exploration_rate": exploration_rate,
    "exploration_decay": exploration_decay,
    "min_learning_rate": min_learning_rate,
    "learning_rate": learning_rate,
    "learning_rate_decay": learning_rate_decay
})

def run_experiment():
    # Access parameters using WandB's API
    cfg = wandb.config

    # Initialise env
    env = MultiAgentPickAndPlace(
        cell_size=cfg.cell_size,
        width=cfg.env_width,
        length=cfg.env_length,
        n_agents=cfg.env_n_agents,
        n_pickers=cfg.env_n_pickers,
        n_objects=cfg.env_n_objects,
        enable_rendering=cfg.env_enable_rendering,
    )

    # Train algo
    q_learning = QLearning(
        env, 
        cfg.learning_rate, 
        cfg.discount_factor, 
        cfg.exploration_rate, 
        cfg.exploration_decay, 
        cfg.min_exploration
    )

    _, rewards_all_episodes, _, _ = q_learning.train(cfg.episodes, cfg.max_steps_per_episode)
    wandb.log({"Rewards for All Episodes": rewards_all_episodes})

    # Save final policy 
    # q_table.save_q_table(cfg.q_table_filename)

    wandb.finish()

if __name__ == "__main__":
    run_experiment()

