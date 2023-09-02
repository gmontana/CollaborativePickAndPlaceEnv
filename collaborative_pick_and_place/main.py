import random
from macpp import MultiAgentPickAndPlace

# Initialize the environment
env = MultiAgentPickAndPlace(width=5, length=5, n_agents=3, n_pickers=2, enable_rendering=False)

num_episodes = 100
max_steps_per_episode = 300
moving_avg_lenght = 10
episode_returns = []

for episode in range(num_episodes):
    state = env.reset()
    episode_return = 0

    for step in range(max_steps_per_episode):

        # print(env.print_state())
        actions = [random.choice(env.action_space) for _ in range(env.n_agents)]

        _, rewards, done = env.step(actions)
        episode_return += sum(rewards)

        # print(env.print_state())
        if done:
            break

    episode_returns.append(episode_return)

    if len(episode_returns) > moving_avg_lenght:
        episode_returns.pop(0)  
    moving_avg = sum(episode_returns) / len(episode_returns)

    print(f"Episode {episode + 1:4d} | Return: {episode_return:6.2f} | Moving Avg: {moving_avg:6.2f} ")
