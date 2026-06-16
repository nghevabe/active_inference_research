comforts = ["Uncomfortable", "Neutral", "Comfortable"]  # hidden state

agent_actions = ["IL", "DL", "IT", "DT", "NA"]  # Agent action
temperatures = ["T0", "T1", "T2", "T3", "T4", "T5"]  # Observe Temperature
lights = ["L0", "L1", "L2", "L3", "L4", "L5"]  # Observe Light
humidity = ["H0", "H1", "H2", "H3", "H4", "H5"]  # Observe Humidity

lst_pairing_state = [
    "Uncomfortable_Uncomfortable", "Uncomfortable_Neutral", "Uncomfortable_Comfortable",
    "Neutral_Uncomfortable", "Neutral_Neutral", "Neutral_Comfortable",
    "Comfortable_Uncomfortable", "Comfortable_Neutral", "Comfortable_Comfortable"
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
        "humidity_obs": {
            "elements": humidity,
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
            "humidity_obs": {
                "elements": humidity,
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
