import sys
import wandb
import numpy as np
import matplotlib.pyplot as plt
import yaml

sys.path.append('/home/gm13/Dropbox/mycode/envs/multi_agent_rl')

from environments.collaborative_pick_and_place.macpp import MultiAgentPickAndPlace
from algos.q_learning import QLearning

# Load parameters from config.yaml
with open("config.yaml", 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Initialize WandB
wandb.init(project="cpp_qlean", config=cfg)

def run_experiment():
    # Access parameters using WandB's API
    cfg = wandb.config

    # initialise env
    env = MultiAgentPickAndPlace(
        cell_size=cfg.cell_size,
        width=cfg.env_width,
        length=cfg.env_length,
        n_agents=cfg.env_n_agents,
        n_pickers=cfg.env_n_pickers,
        n_objects=cfg.env_n_objects,
        enable_rendering=cfg.env_enable_rendering,
    )

if __name__ == "__main__":
    run_experiment()

