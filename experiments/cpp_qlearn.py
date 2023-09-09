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

def run_episodes(q_learning, train_episodes, max_steps_per_episode, training):
    rewards_all_episodes = []
    successful_episodes = 0
    steps_all_episodes = []

    if not training:
        train_episodes=1

    print(f"Running {train_episodes} episode(s)...")

    for episode in range(train_episodes):  
        state_hash = q_learning.env.reset()
        rewards_current_episode = 0

        step = -1  
        for step in range(max_steps_per_episode):
            if training:
                total_reward, done, next_state_hash = q_learning.train(state_hash)
            else:
                total_reward, done, next_state_hash = q_learning.execute(state_hash)
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

        # Save the video when non trainng 
        if not training:
            print("Saving the video")
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            video_filename = f"policy_video_{i}_{timestamp}.mp4"
            env.save_video(video_filename)

    return rewards_all_episodes, steps_all_episodes

def run_experiment(training):

    cfg = wandb.config

    enable_rendering=False if training else True
    create_video=False if training else True

    # Initialise environment 
    env = MultiAgentPickAndPlace(
        cell_size=cfg.cell_size,
        width=cfg.env_width,
        length=cfg.env_length,
        n_agents=cfg.env_n_agents,
        n_pickers=cfg.env_n_pickers,
        n_objects=cfg.env_n_objects,
        enable_rendering=enable_rendering,
        create_video=create_video
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

    # load the policy before execution
    if not training:
        q_learning.q_table.load_q_table(cfg.q_table_filename)

    # run episodes
    rewards_all_episodes, _ = run_episodes(q_learning, cfg.train_episodes, cfg.max_steps_per_episode, training)

    # save the policy after training 
    if training:
        q_learning.q_table.save_q_table(cfg.q_table_filename)

    wandb.finish()

if __name__ == "__main__":
    training=False
    run_experiment(training)
