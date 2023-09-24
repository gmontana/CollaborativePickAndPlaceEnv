<p align="center">
  <strong>Collaborative Pick and Place</strong>
</p>

<p align="center">
  <img width="150px" src="logo.png" alt="Collaborative Pick and Place Environment" />
</p>

<p align="center">
  A multi-agent reinforcement learning environment
</p>



<!-- TABLE OF CONTENTS -->
<h1> Table of Contents </h1>

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Installation](#installation)
- [Usage](#usage)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Rewards](#rewards)
- [Please Cite](#please-cite)
- [Contributing](#contributing)
- [Contact](#contact)

<!-- OVERVIEW -->
# Overview

In the Collaborative Pick and Place environment, multiple agents collaborate in a grid-based world to achieve a common objective. Their mission is to efficiently pick up and place objects (green circles) into designated goal positions (gray rectangles). Agents are divided into two categories:

- **Pickers**: These agents can automatically collect objects while traversing the grid but cannot place them in goal positions.

- **Non-pickers**: In contrast, non-picker agents can deposit objects into goal positions upon reaching them but cannot pick up objects from the grid.

Success in this task relies on the agents' ability to develop a collaborative strategy. Agents can perform movements in four cardinal directions (up, down, left, right) and have the option to wait in their current position. They can also engage in collaboration by using the 'PASS' action to transfer objects between adjacent agents. Successful object placement in goal positions results in rewards for the agents, and the task is considered complete when all objects are in their designated goal positions.

<!-- GETTING STARTED -->
# Getting Started

## Installation

You can install the Collaborative Pick and Place environment using pip:

```sh
pip install collaborative-pick-place

```
Or to ensure that you have the latest version:
```sh
git clone https://github.com/gmontana/collaborative_pick_and_place
cd collaborative_pick_and_place
pip install -e .
```

<!-- USAGE EXAMPLES -->
# Usage

Create environments with the gym framework.
First import
```python
import macpp
```

Then create an environment:
```python
env = gym.make("macpp-3x3-2-1-2-v0")
```

We offer a variety of environments using this template:
```
macpp-{grid_size[0]}x{grid_size[1]}-{n_agents}-{n_pickers}-{n_objects}-v0
```

However you can register your own variation using different parameters:
```python
from gym.envs.registration register

env_name = f"macpp-{grid_size[0]}x{grid_size[1]}-{n_agents}-{n_pickers}-{n_objects}-v0"
    register(
        id=env_name,
        entry_point='macpp.core.environment:MACPPEnv',
        kwargs={
            'grid_size': grid_size,
            'n_agents': n_agents,
            'n_pickers': n_pickers,
            'n_objects': n_objects
        }
    )

```

Similarly to Gym, but adapted to multi-agent settings step() function is defined as
```python
nobs, nreward, ndone, ninfo = env.step(actions)
```

Where n-obs, n-rewards, n-done and n-info are LISTS of N items (where N is the number of agents). The i'th element of each list should be assigned to the i'th agent.

## Observation Space

## Action space

actions is a LIST of N INTEGERS (one of each agent) that should be executed in that step. The integers should correspond to the Enum below:

```python
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PASS = 4
    WAIT = 5
```
Valid actions can always be sampled like in a gym environment, using:
```python
env.action_space.sample() # [2, 3, 0, 1]
```
Also, ALL actions are valid. If an agent cannot move to a location or load, his action will be replaced with `NONE` automatically.

## Rewards

The rewards are calculated as follows. When one or more agents load a food, the food level is rewarded to the agents weighted with the level of each agent. Then the reward is normalised so that at the end, the sum of the rewards (if all foods have been picked-up) is one. 
If you prefer code:

```python
for a in adj_players: # the players that participated in loading the food
    a.reward = float(a.level * food) # higher-leveled agents contribute more and are rewarded more. 
    if self._normalize_reward:
        a.reward = a.reward / float(
            adj_player_level * self._food_spawned
        )  # normalize reward so that the final sum of rewards is one.
```

<!-- CITATION -->
# Please Cite
The paper that first uses this implementation of Level-based Foraging (LBF) and achieves state-of-the-art results:
```
paper_here
```

<!-- CONTRIBUTING -->
# Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- CONTACT -->
# Contact

Giovanni Montana - g.montana@warwick.ac.uk

Project Link: [https://github.com/gmontana/collaborative_pick_and_place](https://github.com/gmontana/collaborative_pick_and_place)
