import random
from macpp import MultiAgentPickAndPlace

# Initialize the environment
env = MultiAgentPickAndPlace(width=10, length=10, n_agents=4, n_pickers=2, enable_rendering=True)

num_episodes = 100
max_steps_per_episode = 300

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):

        # print(env.print_state())
        actions = [random.choice(env.action_space) for _ in range(env.n_agents)]

        next_state, rewards, done = env.step(actions)

        # print(env.print_state())

        if done:
            break

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

