comforts = ["UncomfortableLeft", "NeutralLeft", "Comfortable", "NeutralRight", "UncomfortableRight"]  # hidden state

base_actions = ["IT", "DT", "NA"]  # Base action
agent_actions = base_actions + []  # Agent action

temperatures = ["T0", "T1", "T2", "T3", "T4", "T5"]  # Observe Temperature
lights = ["L0", "L1", "L2", "L3", "L4", "L5"]  # Observe Light

lst_pairing_state = [
    "UncomfortableLeft_UncomfortableLeft", "UncomfortableLeft_NeutralLeft", "UncomfortableLeft_Comfortable", "UncomfortableLeft_NeutralRight", "UncomfortableLeft_UncomfortableRight",
    "NeutralLeft_UncomfortableLeft", "NeutralLeft_NeutralLeft", "NeutralLeft_Comfortable", "NeutralLeft_NeutralRight", "NeutralLeft_UncomfortableRight",
    "Comfortable_UncomfortableLeft", "Comfortable_NeutralLeft", "Comfortable_Comfortable", "Comfortable_NeutralRight", "Comfortable_UncomfortableRight",
    "NeutralRight_UncomfortableLeft", "NeutralRight_NeutralLeft", "NeutralRight_Comfortable", "NeutralRight_NeutralRight", "NeutralRight_UncomfortableRight",
    "UncomfortableRight_UncomfortableLeft", "UncomfortableRight_NeutralLeft", "UncomfortableRight_Comfortable", "UncomfortableRight_NeutralRight", "UncomfortableRight_UncomfortableRight"
]


lst_slice_b = []
lst_action_extended = []

model_description = {
    "observations": {
        "temperature_obs": {
            "elements": temperatures,
            "depends_on": ["comfort"],
        },
        "light_obs": {
            "elements": lights,
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
            "light_obs": {
                "elements": lights,
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
