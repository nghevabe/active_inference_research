comforts = ["Warm", "LittleCool", "Cool", "LittleCold", "Cold"]  # hidden state

base_actions = ["FIT", "FDT", "NA"]  # Base action
base_actions_test = ["NA"]  # Base action
agent_actions = base_actions + []  # Agent action

temperatures = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]  # Observe Temperature

lst_pairing_state = [
    "Warm_Warm", "Warm_LittleCool", "Warm_Cool", "Warm_LittleCold", "Warm_Cold",
    "LittleCool_Warm", "LittleCool_LittleCool", "LittleCool_Cool", "LittleCool_LittleCold", "LittleCool_Cold",
    "Cool_Warm", "Cool_LittleCool", "Cool_Cool", "Cool_LittleCold", "Cool_Cold",
    "LittleCold_Warm", "LittleCold_LittleCool", "LittleCold_Cool", "LittleCold_LittleCold", "LittleCold_Cold",
    "Cold_Warm", "Cold_LittleCool", "Cold_Cool", "Cold_LittleCold", "Cold_Cold"
]


lst_slice_b = []
lst_action_extended = []

model_description = {
    "observations": {
        "temperature_obs": {
            "elements": temperatures,
            "depends_on": ["comfort"],
        },
    },

    "controls": {
        "agent_action": {
            "elements": agent_actions,
        },
    },

    "states": {
        "comfort": {
            "elements": comforts,
            "depends_on": ["comfort"],
            "controlled_by": ["agent_action"],
        },
    },
}


def update_model_description():
    return {
        "observations": {
            "temperature_obs": {
                "elements": temperatures,
                "depends_on": ["comfort"],
            },
        },

        "controls": {
            "agent_action": {
                "elements": agent_actions,
            },
        },

        "states": {
            "comfort": {
                "elements": comforts,
                "depends_on": ["comfort"],
                "controlled_by": ["agent_action"],
            },
        },
    }


def append_action(action_id):
    agent_actions.append(action_id)
    print("agent_actions: " + str(agent_actions))
