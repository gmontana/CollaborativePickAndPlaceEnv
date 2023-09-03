import hydra
from omegaconf import DictConfig
from macpp import MultiAgentPickAndPlace
from q_learning import QLearning

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
    q_table, rewards_all_episodes, _ = q_learning.train(cfg.episodes, cfg.max_steps_per_episode)

    # save final policy 
    q_table.save_q_table(cfg.q_table_filename)

if __name__ == "__main__":
    run_experiment()

