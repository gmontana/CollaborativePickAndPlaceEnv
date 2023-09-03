import random
from macpp import MultiAgentPickAndPlace

# Q leraning params
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 1000
MAX_STEPS = 100
ROLLING_AVG = 10

env = MultiAgentPickAndPlace(width=5, length=5, n_agents=3, n_pickers=2, enable_rendering=False)

Q_table = []

def epsilon_greedy_action_selection(state):
    state_repr = env.get_hashed_state(state)
    if random.uniform(0, 1) < EPSILON:
        return [random.choice(env.get_action_space) for _ in range(env.n_agents)]
    else:
        return [max(env.action_space, key=lambda action: Q_table.get((state_repr, action), 0)) for _ in range(env.n_agents)]

# Initialize the environment

episode_returns = []

for episode in range(EPISODES):

    state = env.reset()
    episode_return = 0

    for step in range(MAX_STEPS):

        # print(env.print_state())
        actions = [random.choice(env.action_space) for _ in range(env.n_agents)]

        _, rewards, done = env.step(actions)
        episode_return += sum(rewards)

        # print(env.print_state())
        if done:
            break

    episode_returns.append(episode_return)

    if len(episode_returns) > ROLLING_AVG:
        episode_returns.pop(0)  
    moving_avg = sum(episode_returns) / len(episode_returns)

    print(f"Episode {episode + 1:4d} | Return: {episode_return:6.2f} | Moving Avg: {moving_avg:6.2f} ")
