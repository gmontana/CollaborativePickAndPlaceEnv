# Collaborative Pick and Place Environment

## Overview
Welcome to the Collaborative Pick and Place environment, where multiple agents work together in a grid-based world to accomplish a common task. The goal is to successfully pick up and place objects into designated goal positions, emphasizing teamwork and specialization among agents.

- Agents: Represented by colored squares, each agent type has a distinct color.
- Objects: Shown as green circles.
- Goal Positions: Indicated by gray rectangles.

## Agents
There are two types of agents in this environment:

- Pickers: they automatically pick up objects when moving over them.
- Non-pickers: they automatically drop off objects when moving over goal positions.

### Actions
Agents can move in four cardinal directions—up, down, left, and right—and can also wait in their current position. To transfer an object to an adjacent agent, both agents must perform the 'pass' action simultaneously. 
- When a Picker passes an object to Non-Picker, they both receive a positive reward.
- When a Non-Picker passes an object to Picker, they both receive a negative reward.

