# Collaborative Pick and Place Environment

## Overview

In the Collaborative Pick and Place environment, multiple agents work together in a grid-based world to accomplish a common task. The goal is to 
successfully pick up and place objects (green circles) into designated goal positions (gray rectangles). The agents populating this environment 
can be of two types: **Pickers** have the ability to automatically pick up objects when moving over them, but are not able to drop them off;
**Non-pickers** have the ability to automatically drop off objects they are carrying simply by reaching the goal positions, but they are not able to pick up objects from the grid. 
In order to successfully move and drop objects on goal positions, the agents must learn a collaborative strategy. Any two ageent that are adejcent to each other in the grid can collaborate by transfering an object from one agent to each other. The task is completed when all the objects are placed on the goal locations.


## Action space
The environment has defined an Action enum with the following actions: UP, DOWN, LEFT, RIGHT, PASS, and WAIT.

The PASS action allows two agents - one carrying an object and one not carrying anything - to pass the obect between them. The action has to be taken simultanouysly, and requires the agent to be adjecent to each other on the grid. 
The action space for the environment is a tuple of discrete spaces, each with 6 possible actions (one for each agent).

## Reward structure 

## Observation space

