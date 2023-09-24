# Collaborative Pick and Place Environment

## Overview 

In the Collaborative Pick and Place environment, agents work together in a grid-based world to achieve a common goal. Their mission is to efficiently collect green circle objects and place them into gray rectangle goal positions. There are two types of agents in this environment:

- **Pickers**: These agents can pick up objects as they move around the grid but can't place them in goal positions.

- **Non-pickers**: These agents can place objects in goal positions but can't pick them up from the grid.

To succeed, agents must collaborate. When two agents are next to each other on the grid, they can transfer objects between them. The task is completed when all objects are in their designated goal positions.

## Action space

The action space in this environment can be categorized into three main types:

- **Movement actions**: each agent can be UP, DOWN, LEFT and RIGHT. Collisions amongst agents are avoided, but the agents can move over objects, regardless of whether they are already carrying an object. 

- **WAIT**: this action results in the agent not moving

- **PASS**: this action facilitates the transfer of objects between agents, enabling them to work collaboratively to achieve the task at hand. 

## Reward structure 

## Observation space

