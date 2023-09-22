import gym
import macpp
import time

env = gym.make('macpp-3x3-2-1-1-v0', debug_mode=True)

n_episodes = 10
for episode in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        actions = {agent: env.action_space.sample() for agent in env.agents}
        next_obs, rewards, done, _ = env.step(actions)
        total_reward += sum(rewards.values())
        obs = next_obs
        time.sleep(0.1)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()

