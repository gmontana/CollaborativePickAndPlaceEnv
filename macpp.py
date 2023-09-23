import argparse
import gym
import macpp
import time

def game_loop(env, render=False):
    """
    Run a single game loop.
    """
    env.reset()
    done = False

    while not done:
        actions = env.action_space.sample()
        _, _, done, _ = env.step(actions)

        if render:
            env.render()

        time.sleep(0.5)

def main(game_count=1, render=False):
    env = gym.make('macpp-3x3-2-1-1-v0', debug_mode=True)

    for episode in range(game_count):
        game_loop(env, render)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Play the MACPP environment.")
    parser.add_argument("--render", action="store_true", help="Render the environment.")
    parser.add_argument("--times", type=int, default=1, help="How many times to run the game.")
    
    args = parser.parse_args()
    main(args.times, args.render)

