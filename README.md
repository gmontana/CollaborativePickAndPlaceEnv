<p align="center">
  <strong>Collaborative Pick and Place</strong>
</p>

<p align="center">
  <img width="300px" src="logo.png" alt="Collaborative Pick and Place Environment" />
</p>

<p align="center">
  A multi-agent reinforcement learning environment
</p>



<!-- TABLE OF CONTENTS -->
<h1> Table of Contents </h1>

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Interactive](#interactive)
- [Usage](#usage)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Rewards](#rewards)
- [Contributing](#contributing)
- [Contact](#contact)

<!-- OVERVIEW -->

# Overview

This Gym environment is designed for multi-agent reinforcement learning, where the primary goal is for agents to collaboratively pick up boxes and place them at designated targets within a grid-world. Agents are categorized into "Pickers," who can only pick up boxes, and "Droppers," who can only place boxes into goal positions. To succeed, agents must learn to efficiently pass boxes between Pickers and Droppers. The environment supports movements in four cardinal directions, a wait action for strategic positioning, and a pass action for box transfer, emphasizing teamwork for achieving the objective of placing all boxes into their respective goals. Unfilled goal positions are marked as red bounding boxes, while filled ones are indicated in green.

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

<!-- INTERACTIVE -->

## Interactive

An interactive mode is available to manually control a two-agent game.

<!-- USAGE EXAMPLES -->
# Usage

Create environments with the gym framework.
First import
```python
import macpp
```

Then create an environment:
```python
env = gym.make("macpp-3x3-2a-1p-2o-v0")
```

We offer a variety of environments using this template:
```
macpp-{grid_size[0]}x{grid_size[1]}-{n_agents}a-{n_pickers}p-{n_objects}o-v0
```

However you can register your own variation using different parameters:
```python
from gym.envs.registration register

env_name = f"macpp-{grid_size[0]}x{grid_size[1]}-{n_agents}a-{n_pickers}p-{n_objects}o-v0"
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

The observation space for the environment is structured as a dictionary where each agent in the environment has its own observation. Each agent's observation is itself a dictionary with the following keys:

- `self`: This key maps to a dictionary that describes the observing agent's own state:
  - `position`: A tuple representing the 2D position on the grid.
  - `picker`: A boolean value indicating if the agent is a picker (can pick up objects) or not.
  - `carrying_object`: Either an integer ID representing the object the agent is currently carrying or `None` if the agent isn't carrying any object.

- `agents`: This key maps to a list of dictionaries. Each dictionary in this list represents the state of another agent in the environment (excluding the observing agent itself). Each dictionary contains the same fields, i.e., `position`, `picker`, and `carrying_object`.

- `objects`: This key maps to a list of dictionaries. Each dictionary in this list represents an object in the environment. Each dictionary contains:
  - `id`: An integer representing the unique ID of the object.
  - `position`: A tuple representing the object's position on the grid.

- `goals`: This key maps to a list of tuples. Each tuple in this list represents the 2D position of a goal on the grid.

Here is a representation of this structure:

```python
{
    'agent_0': {
        'self': {
            'position': (x, y),
            'picker': True/False,
            'carrying_object': ID/None
        },
        'agents': [
            {
                'position': (x, y),
                'picker': True/False,
                'carrying_object': ID/None
            },
            ... (other agents)
        ],
        'objects': [
            {
                'id': ID,
                'position': (x, y)
            },
            ... (other objects)
        ],
        'goals': [
            (x, y),
            ... (other goals)
        ]
    },
    'agent_1': { ... },
    ... (other agents)
}
```

## Action space

The action space within the environment is represented by a list of integers, with each integer corresponding to a specific action that an agent can perform during a step. Each entry in the list corresponds to one agent, meaning the length of the list is equal to the number of agents in the environment.

Actions that agents can perform are enumerated as follows:

- **UP (0)**: Moves the agent up in the grid.
- **DOWN (1)**: Moves the agent down in the grid.
- **LEFT (2)**: Moves the agent left in the grid.
- **RIGHT (3)**: Moves the agent right in the grid.
- **PASS (4)**: Enables the agent to pass an object to another agent.
- **WAIT (5)**: The agent waits or takes no action during the step.

For example, if there are three agents in the environment, and you want the first agent to move up, the second to pass an object, and the third to wait, you would represent this set of actions as `[0, 4, 5]`.

Valid actions for all agents can be sampled using the environment's action space, similar to other gym environments. This can be particularly useful for random action selection or testing purposes:

```python
env.action_space.sample() # Example output: [2, 3, 0, 1]
```

## Rewards

The following rewards can be assigned to the agents based on their actions within the environment:

- **REWARD_STEP**: Penalizes the agent for each action taken, encouraging efficiency and goal-directed behavior.
- **REWARD_GOOD_PASS**: Awarded when an agent successfully passes an object to another agent, promoting collaboration.
- **REWARD_BAD_PASS**: Penalizes an agent for an unsuccessful pass attempt, encouraging accurate and thoughtful passing of objects.
- **REWARD_DROP**: Awarded when an agent successfully places an object in its designated goal location, reinforcing the objective of accurate placement.
- **REWARD_PICKUP**: Awarded when an agent picks up an object, incentivizing the collection of objects for task completion.
- **REWARD_COMPLETION**: A significant reward given upon the successful completion of the task, motivating agents to achieve the collective goal efficiently.


<!-- CONTRIBUTING -->
# Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, feel free to fork the repository, make your changes, and submit a pull request.

<!-- CONTACT -->
# Contact

For questions, suggestions, or collaborations, please contact Giovanni Montana at g.montana@warwick.ac.uk.

