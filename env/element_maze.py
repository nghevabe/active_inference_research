position_states = []

base_actions = ["LEFT", "UP", "RIGHT", "DOWN"]  # Base action
agent_actions = base_actions + []  # Agent action

position_observes = []


def init_element(size):
    for x in range(size):
        for y in range(size):
            position_states.append("S" + str(x) + str(y))
            position_observes.append("P" + str(x) + str(y))


init_element(5)
print(position_observes)
print(position_states)

model_description = {
    "observations": {
        "position_obs": {
            "elements": position_observes,
            "depends_on": ["position_state"],
        },
    },

    "controls": {
        "agent_action": {
            "elements": agent_actions,
        },
    },

    "states": {
        "position_state": {
            "elements": position_states,
            "depends_on": ["position_state"],
            "controlled_by": ["agent_action"],
        },
    },
}


def append_action(action_id):
    agent_actions.append(action_id)
    print("agent_actions: " + str(agent_actions))


for x in range(6):
    for y in range(6):
        print("P" + str(x) + str(y), end=" ")
    print("")
