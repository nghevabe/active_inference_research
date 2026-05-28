from jax import numpy as jnp
from pymdp.distribution import compile_model

comforts = ["Uncomfortable", "Neutral", "Comfortable"]  # hidden state

agent_actions = ["IL", "DL", "IT", "DT", "NA"]  # Agent action
agent_actions_2 = ["IL", "DL", "IT", "DT", "NA", "ACN2"]  # Agent action
user_actions = ["uIL", "uDL", "uIT", "uDT", "uNA"]  # User feedback action

temperatures = ["T1", "T2", "T3", "T4"]  # Observe Temperature
lights = ["L0", "L1", "L2", "L3"]  # Observe Light
times = ["Morning", "Afternoon", "Evening", "Night"]  # Observe Time

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
        "time_obs": {
            "elements": times,
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
            "time_obs": {
                "elements": times,
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


def update_matrix(current_model):
    build_matrix_a(current_model)
    build_matrix_b(current_model)
    build_matrix_c("Comfortable", current_model)
    build_matrix_d(current_model)


def update_model(current_model_description):
    return compile_model(current_model_description)


def build_matrix_a(current_model):
    # temperature_obs define
    current_model.A["temperature_obs"]["T1", "Uncomfortable"] = 0.35
    current_model.A["temperature_obs"]["T2", "Uncomfortable"] = 0.20
    current_model.A["temperature_obs"]["T3", "Uncomfortable"] = 0.30
    current_model.A["temperature_obs"]["T4", "Uncomfortable"] = 0.15

    current_model.A["temperature_obs"]["T1", "Neutral"] = 0.10
    current_model.A["temperature_obs"]["T2", "Neutral"] = 0.20
    current_model.A["temperature_obs"]["T3", "Neutral"] = 0.40
    current_model.A["temperature_obs"]["T4", "Neutral"] = 0.30

    current_model.A["temperature_obs"]["T1", "Comfortable"] = 0.05
    current_model.A["temperature_obs"]["T2", "Comfortable"] = 0.10
    current_model.A["temperature_obs"]["T3", "Comfortable"] = 0.35
    current_model.A["temperature_obs"]["T4", "Comfortable"] = 0.50

    # light_obs define
    # Assumption:
    # - L0: too dark
    # - L1: dim / low light
    # - L2: moderate light
    # - L3: bright light
    #
    # For Uncomfortable, extreme lighting conditions are more likely.
    # For Neutral, middle lighting levels are more likely.
    # For Comfortable, moderate / suitable lighting is more likely.
    current_model.A["light_obs"]["L0", "Uncomfortable"] = 0.35
    current_model.A["light_obs"]["L1", "Uncomfortable"] = 0.20
    current_model.A["light_obs"]["L2", "Uncomfortable"] = 0.20
    current_model.A["light_obs"]["L3", "Uncomfortable"] = 0.25

    current_model.A["light_obs"]["L0", "Neutral"] = 0.15
    current_model.A["light_obs"]["L1", "Neutral"] = 0.25
    current_model.A["light_obs"]["L2", "Neutral"] = 0.35
    current_model.A["light_obs"]["L3", "Neutral"] = 0.25

    current_model.A["light_obs"]["L0", "Comfortable"] = 0.10
    current_model.A["light_obs"]["L1", "Comfortable"] = 0.25
    current_model.A["light_obs"]["L2", "Comfortable"] = 0.45
    current_model.A["light_obs"]["L3", "Comfortable"] = 0.20

    # time_obs define
    # Assumption:
    # Time of day alone does not strongly determine comfort state.
    # Therefore, use uniform likelihood across all time observations.

    current_model.A["time_obs"]["Morning", "Uncomfortable"] = 0.25
    current_model.A["time_obs"]["Afternoon", "Uncomfortable"] = 0.25
    current_model.A["time_obs"]["Evening", "Uncomfortable"] = 0.25
    current_model.A["time_obs"]["Night", "Uncomfortable"] = 0.25

    current_model.A["time_obs"]["Morning", "Neutral"] = 0.25
    current_model.A["time_obs"]["Afternoon", "Neutral"] = 0.25
    current_model.A["time_obs"]["Evening", "Neutral"] = 0.25
    current_model.A["time_obs"]["Night", "Neutral"] = 0.25

    current_model.A["time_obs"]["Morning", "Comfortable"] = 0.25
    current_model.A["time_obs"]["Afternoon", "Comfortable"] = 0.25
    current_model.A["time_obs"]["Evening", "Comfortable"] = 0.25
    current_model.A["time_obs"]["Night", "Comfortable"] = 0.25


def build_matrix_b(current_model):
    # fill in the transition model (B) tensor
    # note that it's specified as ["to", "from", "action"]
    # Hidden states: Uncomfortable, Neutral, Comfortable
    # Agent actions: IL, DL, IT, DT, NA
    #
    # Assumption:
    # Each action has a different random transition distribution.
    # For each [from_state, action], probabilities over to_state sum to 1.0.

    # =========================================================
    # IL: Increase Light
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "IL"] = 0.25
    current_model.B["comfort"]["Neutral", "Uncomfortable", "IL"] = 0.40
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "IL"] = 0.35

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "IL"] = 0.20
    current_model.B["comfort"]["Neutral", "Neutral", "IL"] = 0.45
    current_model.B["comfort"]["Comfortable", "Neutral", "IL"] = 0.35

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "IL"] = 0.30
    current_model.B["comfort"]["Neutral", "Comfortable", "IL"] = 0.25
    current_model.B["comfort"]["Comfortable", "Comfortable", "IL"] = 0.45

    # =========================================================
    # DL: Decrease Light
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "DL"] = 0.30
    current_model.B["comfort"]["Neutral", "Uncomfortable", "DL"] = 0.25
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "DL"] = 0.45

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "DL"] = 0.25
    current_model.B["comfort"]["Neutral", "Neutral", "DL"] = 0.50
    current_model.B["comfort"]["Comfortable", "Neutral", "DL"] = 0.25

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "DL"] = 0.20
    current_model.B["comfort"]["Neutral", "Comfortable", "DL"] = 0.35
    current_model.B["comfort"]["Comfortable", "Comfortable", "DL"] = 0.45

    # =========================================================
    # IT: Increase Temperature
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "IT"] = 0.35
    current_model.B["comfort"]["Neutral", "Uncomfortable", "IT"] = 0.40
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "IT"] = 0.25

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "IT"] = 0.30
    current_model.B["comfort"]["Neutral", "Neutral", "IT"] = 0.40
    current_model.B["comfort"]["Comfortable", "Neutral", "IT"] = 0.30

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "IT"] = 0.25
    current_model.B["comfort"]["Neutral", "Comfortable", "IT"] = 0.30
    current_model.B["comfort"]["Comfortable", "Comfortable", "IT"] = 0.45

    # =========================================================
    # DT: Decrease Temperature
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "DT"] = 0.40
    current_model.B["comfort"]["Neutral", "Uncomfortable", "DT"] = 0.35
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "DT"] = 0.25

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "DT"] = 0.25
    current_model.B["comfort"]["Neutral", "Neutral", "DT"] = 0.35
    current_model.B["comfort"]["Comfortable", "Neutral", "DT"] = 0.40

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "DT"] = 0.35
    current_model.B["comfort"]["Neutral", "Comfortable", "DT"] = 0.25
    current_model.B["comfort"]["Comfortable", "Comfortable", "DT"] = 0.40

    # =========================================================
    # NA: No Action
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "NA"] = 0.45
    current_model.B["comfort"]["Neutral", "Uncomfortable", "NA"] = 0.35
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "NA"] = 0.20

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "NA"] = 0.20
    current_model.B["comfort"]["Neutral", "Neutral", "NA"] = 0.55
    current_model.B["comfort"]["Comfortable", "Neutral", "NA"] = 0.25

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "NA"] = 0.15
    current_model.B["comfort"]["Neutral", "Comfortable", "NA"] = 0.25
    current_model.B["comfort"]["Comfortable", "Comfortable", "NA"] = 0.60


def build_matrix_c(preferred_state, current_model):
    eps = 1e-16

    for obs in temperatures:
        current_model.C["temperature_obs"][obs] = jnp.log(
            current_model.A["temperature_obs"][obs, preferred_state] + eps
        )

    for obs in lights:
        current_model.C["light_obs"][obs] = jnp.log(
            current_model.A["light_obs"][obs, preferred_state] + eps
        )

    for obs in times:
        current_model.C["time_obs"][obs] = jnp.log(
            current_model.A["time_obs"][obs, preferred_state] + eps
        )


def build_matrix_d(current_model):
    # =========================================================
    # D matrix: Initial prior belief over hidden states
    # =========================================================
    # Hidden state factor: comfort
    #
    # The agent initially believes that the user/environment is
    # most likely in the Uncomfortable state.
    current_model.D["comfort"]["Uncomfortable"] = 0.30
    current_model.D["comfort"]["Neutral"] = 0.65
    current_model.D["comfort"]["Comfortable"] = 0.05
