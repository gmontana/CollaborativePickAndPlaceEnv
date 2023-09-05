Agent Actions:

    "move_up": Move one cell upwards.
    "move_down": Move one cell downwards.
    "move_left": Move one cell to the left.
    "move_right": Move one cell to the right.
    "pass": Pass an object to a neighboring agent.

Allowed Actions:

Agents can move to adjacent cells on the grid if the cell is unoccupied by another agent.

Agents can only pass an object to a neighboring agent if:
    The neighboring agent is in an adjacent cell.
    The neighboring agent is not already carrying an object.
    The current agent is carrying an object.

Picker Status:

    Agents can be classified as "pickers" or "non-pickers."
    Picker agents can pick up objects from the grid.
    Non-picker agents cannot pick up objects, but they can receive objects from picker agents through passing.

Object Interaction:

    Agents can pick up objects from the grid if they are designated as "pickers" and they are not currently carrying an object.
    When an agent picks up an object, the object is removed from the grid.

Passing Objects:

    Picker agents can pass objects to neighboring agents through the "pass" action.
    Passing an object requires both the current agent and the adjacent agent to take the "pass" action simultaneously.
    If the adjacent agent is a picker and not carrying an object, it will receive the object.

Goals:

    The environment contains goal positions where objects need to be delivered.
    The goal is considered fulfilled when all objects are placed in their corresponding goal positions.

Rewards and Termination:

    Agents receive rewards based on their actions and achievements:
    Moving incurs a negative reward (REWARD_STEP).
    Successfully passing an object to a non-picker incurs a positive reward (REWARD_GOOD_PASS).
    Passing an object to a picker incurs a negative reward (REWARD_BAD_PASS).
    Dropping an object on a goal incurs a positive reward (REWARD_DROP).
    Fulfilling all goals results in a termination reward (REWARD_COMPLETION).
    The environment terminates when all goals are fulfilled.

Grid Constraints:

    Agents cannot move out of the grid boundaries.
    The grid size must be large enough to accommodate all agents and objects.
