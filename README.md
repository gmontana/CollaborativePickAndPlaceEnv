# Collaborative Pick and Place Environment

## Overview
In the Collaborative Pick and Place environment, multiple agents work together in a grid-based world to accomplish a common task. The goal is to successfully pick up and place objects (green circles) into designated goal positions (gray rectangles). The agents populating this environment can be of two types:

- **Pickers**: can automatically pick up objects when moving over them.
- **Non-pickers**: can automatically drop off objects when moving over goal positions.

In order to successfully move and drop objects on goal positions, the agents need to learn a collaborative strategy.

Agents can move in four cardinal directions—up, down, left, and right—and can also wait in their current position. To transfer an object to an adjacent agent, both agents must perform a 'pass' action simultaneously. 
- When a Picker passes an object to Non-Picker, they both receive a positive reward.
- When a Non-Picker passes an object to Picker, they both receive a negative reward.

For each object correctly dropped, the agents receive a reward. Upon completing the entire task, all agents are rewarded.
