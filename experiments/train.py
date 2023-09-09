import hydra
from omegaconf import DictConfig
from macpp import MultiAgentPickAndPlace
from q_learning import QLearning
import numpy as np
import matplotlib.pyplot as plt

@hydra.main(config_name="config", version_base="1.1")
def run_experiment(cfg: DictConfig):
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

    # train algo
    q_learning = QLearning(
        env, 
        cfg.learning_rate, 
        cfg.discount_factor, 
        cfg.exploration_rate, 
        cfg.exploration_decay, 
        cfg.min_exploration
    )

    # print("Action Space:", env.get_action_space())

    q_table, rewards_all_episodes, _, avg_rewards_all_episodes = q_learning.train(cfg.episodes, cfg.max_steps_per_episode)

    # save final policy 
    q_table.save_q_table(cfg.q_table_filename)

    # Initialise env again to create a video 
    env_with_video = MultiAgentPickAndPlace(
        cell_size=cfg.cell_size,
        width=cfg.env_width,
        length=cfg.env_length,
        n_agents=cfg.env_n_agents,
        n_pickers=cfg.env_n_pickers,
        n_objects=cfg.env_n_objects,
        enable_rendering=False,
        create_video=True
    )

    q_learning.env = env_with_video
    q_learning.q_table = q_table

    # Create a training curve
    plt.plot(avg_rewards_all_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    config_details = f"Grid Size: {cfg.env_width}x{cfg.env_length}\nNumber of Agents: {cfg.env_n_agents}\nNumber of Pickers: {cfg.env_n_pickers}\nNumber of Objects: {cfg.env_n_objects}"
    plt.title(f"Training Curve\n{config_details}")
    plot_filename = "training_curve.png" 
    plt.savefig(plot_filename)
    print(f"Training curve saved to: {plot_filename}")
    plt.savefig("training_curve.png")

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

    average_return = sum(total_returns) / num_episodes
    average_steps = sum(total_steps_list) / num_episodes
    success_rate = sum(successes) / num_episodes

    std_dev_return = np.std(total_returns)
    std_dev_steps = np.std(total_steps_list)
    std_dev_success = np.std(successes)

    print(f"\nSummary Statistics over {num_episodes} episodes:")
    print(f"Average Return: {average_return:.2f} (±{std_dev_return:.2f})")
    print(f"Average Steps: {average_steps:.2f} (±{std_dev_steps:.2f})")
    print(f"Success Rate: {success_rate:.2f} (±{std_dev_success:.2f})")

if __name__ == "__main__":
    run_experiment()
