from pymdp.distribution import compile_model

position_states = []

base_actions = ["LEFT", "UP", "RIGHT", "DOWN"]  # Base action
agent_actions = base_actions + []  # Agent action

position_observes = []


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


def build_matrix_a(model, lst_position_observe, lst_position_state):

    for item_state in lst_position_state:
        for item_observe in lst_position_observe:
            if item_observe[1:3] == item_state[1:3]:
                model.A["position_obs"][item_observe, item_state] = 1
            else:
                model.A["position_obs"][item_observe, item_state] = 0


def build_matrix_b(model, lst_position_state, lst_base_actions):

    for action in lst_base_actions:
        for i in lst_position_state:
            for j in lst_position_state:
                model.B["position_state"][j, i, action] = 0

    # S00 S01 S02 S03 S04 S05
    # S10 S11 S12 S13 S14 S15
    # S20 S21 S22 S23 S24 S25
    # S30 S31 S32 S33 S34 S35
    # S40 S41 S42 S43 S44 S45
    # S50 S51 S52 S53 S54 S55

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "FIT"] = 0.10
    current_model.B["comfort"]["LittleCool", "Warm", "FIT"] = 0.85
    current_model.B["comfort"]["Cool", "Warm", "FIT"] = 0.05
    current_model.B["comfort"]["LittleCold", "Warm", "FIT"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "FIT"] = 0.00


init_element(6)
print("position_observes: ")
print(position_observes)
print("position_states: ")
print(position_states)


current_model = compile_model(update_model_description())
build_matrix_a(current_model, position_observes, position_states)

print("XXX_current_model: "+str(current_model.A))

