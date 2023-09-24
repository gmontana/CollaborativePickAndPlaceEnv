# Collaborative Pick and Place Environment

## Overview 

In the Collaborative Pick and Place environment, a cohort of agents collaborates within a grid-based realm to achieve a shared objective. Their mission entails the efficient collection and depositing of objects, represented as green circles, into designated goal positions, depicted as gray rectangles. The agents within this environment fall into two distinct categories:

- **Pickers**: These agents possess the capability to autonomously collect objects as they traverse the grid. However, they lack the ability to deposit these items into goal positions.

- **Non-pickers**: In contrast, non-picker agents can effortlessly deposit objects into goal positions when they reach them, but they do not possess the capacity to pick up objects from the grid.

The successful execution of this task hinges on the agents' acquisition of a collaborative strategy. Adjacent agents within the grid can collaborate by transferring objects amongst themselves. The task is considered complete when all objects find their designated places on the goal positions.

## Action space

The action space in this environment can be categorized into three main types:

- **Movement actions**: each agent can be UP, DOWN, LEFT and RIGHT. Collisions amongst agents are avoided, but the agents can move over objects, regardless of whether they are already carrying an object. 

- **WAIT**: this action results in the agent not moving

- **PASS**: this action facilitates the transfer of objects between agents, enabling them to work collaboratively to achieve the task at hand. 

## Reward structure 

## Observation space

