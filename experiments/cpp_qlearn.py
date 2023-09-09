import sys
import wandb
import numpy as np
import os 
import datetime

PATH = os.path.expanduser("~/Dropbox/mycode/envs/multi_agent_rl")
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

def run_execution():

    # Re-initialise environment 
    env = MultiAgentPickAndPlace(
        cell_size=cfg.cell_size,
        width=cfg.env_width,
        length=cfg.env_length,
        n_agents=cfg.env_n_agents,
        n_pickers=cfg.env_n_pickers,
        n_objects=cfg.env_n_objects,
        enable_rendering=True,
        create_video=True
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

    # Load the policy
    q_learning.q_table.load_q_table(cfg.q_table_filename)

    # Create some summary stats and videos after training
    num_episodes = 10
    successes = []
    total_returns = []
    total_steps_list = []

    print(f"\nExecuting learned policy over {num_episodes} episodes for statistics:")
    for i in range(num_episodes):
        success, total_return, total_steps = q_learning.execute(cfg.max_steps_per_episode)
        successes.append(success)
        total_returns.append(total_return)
        total_steps_list.append(total_steps)

        # save the video
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f"policy_video_{i}_{timestamp}.mp4"
        env.save_video(video_filename)


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

    # Save final policy 
    q_learning.q_table.save_q_table(cfg.q_table_filename)

    # wandb.save("model.h5")
    wandb.finish()

if __name__ == "__main__":
    # run_experiment()
    run_execution()
