# Collaborative Pick and Place Environment

## Overview
In the Collaborative Pick and Place environment, multiple agents collaborate within a grid-based world to perform a task involving the pickup and placement of objects into designated goal positions. This cooperative game emphasizes teamwork and specialization among agents, each with distinct roles and responsibilities. The primary objective is to successfully place all objects at their respective goal locations, with agents receiving rewards based on their collaborative efforts.

## Agents
There are two types of agents in this environment:

### 1. Pickers
- Role: Pickers are specialized in picking up objects.
- Actions: They can move in four cardinal directions—up, down, left, and right—and can also wait in their current position.
- Object Interaction: Pickers automatically pick up objects when they move over a cell containing an object, provided they are not already carrying an object.

### 2. Non-pickers
- Role: Non-pickers are specialized in dropping off objects at goal positions.
- Actions: Similar to Pickers, they can move in four directions and wait.
- Object Interaction: Non-pickers automatically drop off objects when they move over a goal position, earning a positive reward for doing so.

## Object Handling
- **Pickers** automatically pick up objects when they move over a cell containing one, provided they aren't already carrying an object.
- **Non-pickers** automatically drop off objects when they move over a goal position, earning a positive reward for doing so.

## Object Transfer
Agents can transfer objects to one another under certain conditions:
- To pass an object from one agent to another, both agents must perform the `pass` action simultaneously.
- When a Picker passes an object to a Non-picker, both agents receive a positive reward.
- When a Non-picker passes an object to a Picker, both agents receive a negative reward.

## Object Representation
- Objects are represented as green circles on the grid.
- Goal positions are indicated by gray rectangles.
- When an agent is carrying an object, a green bounding box appears around the agent, indicating the carried object.

## Task Completion
The task is considered successfully completed when all objects have been placed on their respective goal positions. Upon successful completion, all agents receive a final reward based on their contributions.

