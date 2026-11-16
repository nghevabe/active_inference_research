position_states = []

base_actions = ["LEFT", "UP", "RIGHT", "DOWN"]  # Base action
agent_actions = base_actions + []  # Agent action

position_observes = []

maze_size = 4

lst_slice_b = []
lst_action_extended = []


def init_element(size):
    for x in range(size):
        for y in range(size):
            position_states.append("S" + str(x) + str(y))
            position_observes.append("P" + str(x) + str(y))


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


def update_model_description():
    return {
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




# init_element(maze_size)
# print("position_observes: ")
# print(position_observes)
# print("position_states: ")
# print(position_states)
#
# test_model = extend_action_space("SWIM_UP")
#
# print("XXX_current_model: " + str(test_model.B))


