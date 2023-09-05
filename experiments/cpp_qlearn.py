# import os
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

def run_training(self, episodes, max_steps_per_episode):
    rewards_all_episodes = []
    successful_episodes = 0
    steps_per_episode = []

    for episode in range(episodes):
        state = self.env.reset()
        state_hash = self.env.get_hashed_state()
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            total_reward, done, next_state_hash = self.train(state_hash)
            rewards_current_episode += total_reward
            state_hash = next_state_hash

            if done:
                break

        rewards_all_episodes.append(rewards_current_episode)
        steps_per_episode.append(step)

        # Check for a successful episode
        if step <= max_steps_per_episode:
            successful_episodes += 1

        # Logging or any other operations you want to perform at the end of each episode

    return rewards_all_episodes, steps_per_episode

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

    # Log rewards and other metrics
    wandb.log({
        "Rewards for All Episodes": rewards_all_episodes,
        "Average Reward": np.mean(rewards_all_episodes),
    })

    # Save and log the final policy as an artifact (if needed)
    # artifact = wandb.Artifact('model', type='model')
    # artifact.add_file(cfg.q_table_filename)
    # wandb.log_artifact(artifact)

    # Save final policy 
    # q_table.save_q_table(cfg.q_table_filename)

    wandb.finish()

    # Create some summary stats and videos after training
    num_episodes = 30
    successes = []
    total_returns = []
    total_steps_list = []

    print(f"\nExecuting learned policy over {num_episodes} episodes for statistics:")
    for _ in range(num_episodes):
        success, total_return, total_steps = q_learning.execute(cfg.num_max_steps, save_video=True)
        successes.append(success)
        total_returns.append(total_return)
        total_steps_list.append(total_steps)

if __name__ == "__main__":
    run_experiment()

