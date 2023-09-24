# Collaborative Pick and Place Environment

## Overview 

In the Collaborative Pick and Place environment, a cohort of agents collaborates within a grid-based realm to achieve a shared objective. Their mission entails the efficient collection and depositing of objects, represented as green circles, into designated goal positions, depicted as gray rectangles. The agents within this environment fall into two distinct categories:

- **Pickers**: These agents possess the capability to autonomously collect objects as they traverse the grid. However, they lack the ability to deposit these items into goal positions.

- **Non-pickers**: In contrast, non-picker agents can effortlessly deposit objects into goal positions when they reach them, but they do not possess the capacity to pick up objects from the grid.

The successful execution of this task hinges on the agents' acquisition of a collaborative strategy. Adjacent agents within the grid can collaborate by transferring objects amongst themselves. The task is considered complete when all objects find their designated places on the goal positions.

## Action space
The environment has defined an Action enum with the following actions: UP, DOWN, LEFT, RIGHT, PASS, and WAIT.

The PASS action allows two agents - one carrying an object and one not carrying anything - to pass the obect between them. The action has to be taken simultanouysly, and requires the agent to be adjecent to each other on the grid. 
The action space for the environment is a tuple of discrete spaces, each with 6 possible actions (one for each agent).

## Reward structure 

## Observation space

