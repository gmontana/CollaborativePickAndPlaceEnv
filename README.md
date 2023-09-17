
# **Collaborative Pick and Place**

In "Collaborative Pick and Place," agents work together in a grid environment to pick up and place objects in designated goal positions. The game emphasizes teamwork, with agents having distinct roles and responsibilities. The game concludes when all objects are correctly placed, rewarding agents based on their collaborative efforts.

# **Environment Description**

The environment consists of a grid where agents can move, interact with objects, and achieve specific goals. Agents are either:
- **Pickers**: Can pick up objects
- **Non-pickers**: Can drop off objects

## **Rules**

### Movement:**
- Agents can move one cell in directions: `up`, `down`, `left`, or `right`.
- They cannot move into cells occupied by other agents.
- They can, however, occupy cells containing objects.

### Object Interaction:**
- **Pickers** automatically pick up objects when they move over a cell containing one, provided they aren't already carrying an object.
- **Non-pickers** automatically drop off objects when they move over a goal position, earning a positive reward for doing so.

## Passing Objects:**
- Agents can pass objects to adjacent agents.
- Both agents must use the "pass" action at the same time.
  - **Picker to Non-picker**: Both receive a positive reward.
  - **Non-picker to Picker**: Both receive a negative reward.

### Termination:**
- The game ends when all objects are placed on goal positions.
- Upon conclusion, all agents receive a completion reward.

