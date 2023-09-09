import sys
import wandb
import numpy as np
import os 

PATH = os.path.expanduser("~/Dropbox/mycode/envs/multi_agent_rl")
print(PATH)
sys.path.append(PATH)

from environments.collaborative_pick_and_place.macpp import MultiAgentPickAndPlace
from algorithms.qlearn import QLearning


# Configuration parameters
config = {
    "q_table_filename": "q_table.npy",
    "cell_size": 300,
    "env_width": 3,
    "env_length": 3,
    "env_n_agents": 2,
    "env_n_pickers": 1,
    "env_n_objects": 1,
    "env_enable_rendering": False,
    "train_episodes": 200000,
    "max_steps_per_episode": 50,
    "discount_factor": 0.90,
    "min_exploration": 0.02,
    "exploration_rate": 1.0,
    "exploration_decay": 0.99,
    "min_learning_rate": 0.01,
    "learning_rate": 0.1,
    "learning_rate_decay": 0.995,
}

# Initialize WandB
wandb.init(project="cpp_qlearn", name="tabular_q", config=config)

def run_training(q_learning, train_episodes, max_steps_per_episode):
    rewards_all_episodes = []
    successful_episodes = 0
    steps_all_episodes = []

    for episode in range(train_episodes):  
        state_hash = q_learning.env.reset()
        rewards_current_episode = 0

        step = -1  
        for step in range(max_steps_per_episode):
            total_reward, done, next_state_hash = q_learning.train(state_hash)
            rewards_current_episode += total_reward
            state_hash = next_state_hash
            if done:
                break

        rewards_all_episodes.append(rewards_current_episode)
        steps_all_episodes.append(step + 1) 
        print(f"Episode: {episode} | Steps: {step+1} | Rewards: {rewards_current_episode}")

        # Check for a successful episode
        if step + 1 <= max_steps_per_episode:
            successful_episodes += 1

        # Log rewards and average rewards every N episodes (e.g., every 100 episodes)
        if (episode + 1) % 100 == 0:
            wandb.log({
                "Average Reward": np.mean(rewards_all_episodes[-100:]),
                "Average Steps": np.mean(steps_all_episodes[-100:])
            })

    return rewards_all_episodes, steps_all_episodes

def run_experiment():

    cfg = wandb.config

    # Initialise environment 
    env = MultiAgentPickAndPlace(
        cell_size=cfg.cell_size,
        width=cfg.env_width,
        length=cfg.env_length,
        n_agents=cfg.env_n_agents,
        n_pickers=cfg.env_n_pickers,
        n_objects=cfg.env_n_objects,
        enable_rendering=cfg.env_enable_rendering,
    )

    # Set up Q learning algorithm 
    q_learning = QLearning(
        env, 
        cfg.learning_rate, 
        cfg.discount_factor, 
        cfg.exploration_rate, 
        cfg.exploration_decay, 
        cfg.min_exploration
    )

    # Perform training
    rewards_all_episodes, _ = run_training(q_learning, cfg.train_episodes, cfg.max_steps_per_episode)

    # Log rewards and other metrics
    print(f"Average Reward: {np.mean(rewards_all_episodes)}")

    # Save and log the final policy as an artifact (if needed)
    # artifact = wandb.Artifact('model', type='model')
    # artifact.add_file(cfg.q_table_filename)
    # wandb.log_artifact(artifact)

    # Save final policy 
    # q_table.save_q_table(cfg.q_table_filename)

    wandb.save("model.h5")
    wandb.finish()

    # Create some summary stats and videos after training
    num_episodes = 30
    successes = []
    total_returns = []
    total_steps_list = []

    print(f"\nExecuting learned policy over {num_episodes} episodes for statistics:")
    for _ in range(num_episodes):
        success, total_return, total_steps = q_learning.execute(cfg.max_steps_per_episode)
        successes.append(success)
        total_returns.append(total_return)
        total_steps_list.append(total_steps)

if __name__ == "__main__":
    run_experiment()
