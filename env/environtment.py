import jax
from jax import numpy as jnp
from env.elements import temperatures, agent_actions


def build_matrix_a(current_model):
    # temperature_obs define
    current_model.A["temperature_obs"]["T0", "Warm"] = 0.40
    current_model.A["temperature_obs"]["T1", "Warm"] = 0.25
    current_model.A["temperature_obs"]["T2", "Warm"] = 0.15
    current_model.A["temperature_obs"]["T3", "Warm"] = 0.10
    current_model.A["temperature_obs"]["T4", "Warm"] = 0.05
    current_model.A["temperature_obs"]["T5", "Warm"] = 0.05

    current_model.A["temperature_obs"]["T0", "LittleCool"] = 0.05
    current_model.A["temperature_obs"]["T1", "LittleCool"] = 0.25
    current_model.A["temperature_obs"]["T2", "LittleCool"] = 0.35
    current_model.A["temperature_obs"]["T3", "LittleCool"] = 0.20
    current_model.A["temperature_obs"]["T4", "LittleCool"] = 0.10
    current_model.A["temperature_obs"]["T5", "LittleCool"] = 0.05

    current_model.A["temperature_obs"]["T0", "Cool"] = 0.05
    current_model.A["temperature_obs"]["T1", "Cool"] = 0.10
    current_model.A["temperature_obs"]["T2", "Cool"] = 0.20
    current_model.A["temperature_obs"]["T3", "Cool"] = 0.35
    current_model.A["temperature_obs"]["T4", "Cool"] = 0.20
    current_model.A["temperature_obs"]["T5", "Cool"] = 0.10

    current_model.A["temperature_obs"]["T0", "LittleCold"] = 0.05
    current_model.A["temperature_obs"]["T1", "LittleCold"] = 0.10
    current_model.A["temperature_obs"]["T2", "LittleCold"] = 0.20
    current_model.A["temperature_obs"]["T3", "LittleCold"] = 0.25
    current_model.A["temperature_obs"]["T4", "LittleCold"] = 0.35
    current_model.A["temperature_obs"]["T5", "LittleCold"] = 0.05

    current_model.A["temperature_obs"]["T0", "Cold"] = 0.05
    current_model.A["temperature_obs"]["T1", "Cold"] = 0.05
    current_model.A["temperature_obs"]["T2", "Cold"] = 0.10
    current_model.A["temperature_obs"]["T3", "Cold"] = 0.15
    current_model.A["temperature_obs"]["T4", "Cold"] = 0.25
    current_model.A["temperature_obs"]["T5", "Cold"] = 0.40


def build_matrix_b(current_model):
    # fill in the transition model (B) tensor
    # note that it's specified as ["to", "from", "action"]
    # Hidden states: "Warm", "LittleCool", "Cool", "LittleCold", "Cold"
    # Agent actions: FIT, FDT, NA
    #
    # Assumption:
    # Each action has a different random transition distribution.
    # For each [from_state, action], probabilities over to_state sum to 1.0.

    # =========================================================
    # FIT: Fan Increase Temperature
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Warm", "FIT"] = 1.00
    current_model.B["comfort"]["Cool", "Warm", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Warm", "FIT"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "FIT"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCool", "FIT"] = 0.50
    current_model.B["comfort"]["Cool", "LittleCool", "FIT"] = 0.50
    current_model.B["comfort"]["LittleCold", "LittleCool", "FIT"] = 0.00
    current_model.B["comfort"]["Cold", "LittleCool", "FIT"] = 0.00

    # From Cool
    current_model.B["comfort"]["Warm", "Cool", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cool", "FIT"] = 0.00
    current_model.B["comfort"]["Cool", "Cool", "FIT"] = 1.00
    current_model.B["comfort"]["LittleCold", "Cool", "FIT"] = 0.00
    current_model.B["comfort"]["Cold", "Cool", "FIT"] = 0.00

    # From LittleCold
    current_model.B["comfort"]["Warm", "LittleCold", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "FIT"] = 0.00
    current_model.B["comfort"]["Cool", "LittleCold", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCold", "FIT"] = 1.00
    current_model.B["comfort"]["Cold", "LittleCold", "FIT"] = 0.00

    # From Cold
    current_model.B["comfort"]["Warm", "Cold", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "FIT"] = 0.00
    current_model.B["comfort"]["Cool", "Cold", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cold", "FIT"] = 0.00
    current_model.B["comfort"]["Cold", "Cold", "FIT"] = 1.00

    # =========================================================
    # FDT: Fan Decrease Temperature
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "FDT"] = 1.00
    current_model.B["comfort"]["LittleCool", "Warm", "FDT"] = 0.00
    current_model.B["comfort"]["Cool", "Warm", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Warm", "FDT"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "FDT"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "FDT"] = 0.05
    current_model.B["comfort"]["LittleCool", "LittleCool", "FDT"] = 1.00
    current_model.B["comfort"]["Cool", "LittleCool", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCool", "FDT"] = 0.00
    current_model.B["comfort"]["Cold", "LittleCool", "FDT"] = 0.00

    # From Cool
    current_model.B["comfort"]["Warm", "Cool", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cool", "FDT"] = 1.00
    current_model.B["comfort"]["Cool", "Cool", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cool", "FDT"] = 0.00
    current_model.B["comfort"]["Cold", "Cool", "FDT"] = 0.00

    # From LittleCold
    current_model.B["comfort"]["Warm", "LittleCold", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "FDT"] = 0.00
    current_model.B["comfort"]["Cool", "LittleCold", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCold", "FDT"] = 1.00
    current_model.B["comfort"]["Cold", "LittleCold", "FDT"] = 0.00

    # From Cold
    current_model.B["comfort"]["Warm", "Cold", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "FDT"] = 0.00
    current_model.B["comfort"]["Cool", "Cold", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cold", "FDT"] = 0.00
    current_model.B["comfort"]["Cold", "Cold", "FDT"] = 1.00

    # =========================================================
    # NA: No Action
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "NA"] = 1.00
    current_model.B["comfort"]["LittleCool", "Warm", "NA"] = 0.00
    current_model.B["comfort"]["Cool", "Warm", "NA"] = 0.00
    current_model.B["comfort"]["LittleCold", "Warm", "NA"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "NA"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "NA"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCool", "NA"] = 1.00
    current_model.B["comfort"]["Cool", "LittleCool", "NA"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCool", "NA"] = 0.00
    current_model.B["comfort"]["Cold", "LittleCool", "NA"] = 0.00

    # From Cool
    current_model.B["comfort"]["Warm", "Cool", "NA"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cool", "NA"] = 0.00
    current_model.B["comfort"]["Cool", "Cool", "NA"] = 1.00
    current_model.B["comfort"]["LittleCold", "Cool", "NA"] = 0.00
    current_model.B["comfort"]["Cold", "Cool", "NA"] = 0.00

    # From LittleCold
    current_model.B["comfort"]["Warm", "LittleCold", "NA"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "NA"] = 0.00
    current_model.B["comfort"]["Cool", "LittleCold", "NA"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCold", "NA"] = 1.00
    current_model.B["comfort"]["Cold", "LittleCold", "NA"] = 0.00

    # From Cold
    current_model.B["comfort"]["Warm", "Cold", "NA"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "NA"] = 0.00
    current_model.B["comfort"]["Cool", "Cold", "NA"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cold", "NA"] = 0.00
    current_model.B["comfort"]["Cold", "Cold", "NA"] = 1.00

    # =========================================================
    # AIT: AirCondition Increase Temperature
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "AIT"] = 1.00
    current_model.B["comfort"]["LittleCool", "Warm", "AIT"] = 0.00
    current_model.B["comfort"]["Cool", "Warm", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Warm", "AIT"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "AIT"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCool", "AIT"] = 1.00
    current_model.B["comfort"]["Cool", "LittleCool", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCool", "AIT"] = 0.00
    current_model.B["comfort"]["Cold", "LittleCool", "AIT"] = 0.00

    # From Cool
    current_model.B["comfort"]["Warm", "Cool", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cool", "AIT"] = 0.00
    current_model.B["comfort"]["Cool", "Cool", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cool", "AIT"] = 1.00
    current_model.B["comfort"]["Cold", "Cool", "AIT"] = 0.00

    # From LittleCold
    current_model.B["comfort"]["Warm", "LittleCold", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "AIT"] = 0.00
    current_model.B["comfort"]["Cool", "LittleCold", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCold", "AIT"] = 0.50
    current_model.B["comfort"]["Cold", "LittleCold", "AIT"] = 0.50

    # From Cold
    current_model.B["comfort"]["Warm", "Cold", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "AIT"] = 0.00
    current_model.B["comfort"]["Cool", "Cold", "AIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cold", "AIT"] = 0.00
    current_model.B["comfort"]["Cold", "Cold", "AIT"] = 1.00


    # =========================================================
    # ADT: AirCondition Decrease Temperature
    # =========================================================

    # =========================================================
    #               T0       T1, T2       T3       T4, T5       T6
    # comforts = ["Warm", "LittleCool", "Cool", "LittleCold", "Cold"]  # hidden state

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "ADT"] = 1.00
    current_model.B["comfort"]["LittleCool", "Warm", "ADT"] = 0.00
    current_model.B["comfort"]["Cool", "Warm", "ADT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Warm", "ADT"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "ADT"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "ADT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCool", "ADT"] = 1.00
    current_model.B["comfort"]["Cool", "LittleCool", "ADT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCool", "ADT"] = 0.00
    current_model.B["comfort"]["Cold", "LittleCool", "ADT"] = 0.00

    # From Cool
    current_model.B["comfort"]["Warm", "Cool", "ADT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cool", "ADT"] = 0.00
    current_model.B["comfort"]["Cool", "Cool", "ADT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cool", "ADT"] = 1.00
    current_model.B["comfort"]["Cold", "Cool", "ADT"] = 0.00

    # From LittleCold
    current_model.B["comfort"]["Warm", "LittleCold", "ADT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "ADT"] = 0.00
    current_model.B["comfort"]["Cool", "LittleCold", "ADT"] = 0.50
    current_model.B["comfort"]["LittleCold", "LittleCold", "ADT"] = 0.50
    current_model.B["comfort"]["Cold", "LittleCold", "ADT"] = 0.00

    # From Cold
    current_model.B["comfort"]["Warm", "Cold", "ADT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "ADT"] = 0.00
    current_model.B["comfort"]["Cool", "Cold", "ADT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cold", "ADT"] = 1.00
    current_model.B["comfort"]["Cold", "Cold", "ADT"] = 0.00


    # =========================================================
    # XDT: X Decrease Temperature
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "XDT"] = 1.00
    current_model.B["comfort"]["LittleCool", "Warm", "XDT"] = 0.00
    current_model.B["comfort"]["Cool", "Warm", "XDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Warm", "XDT"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "XDT"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "XDT"] = 0.50
    current_model.B["comfort"]["LittleCool", "LittleCool", "XDT"] = 0.50
    current_model.B["comfort"]["Cool", "LittleCool", "XDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCool", "XDT"] = 0.00
    current_model.B["comfort"]["Cold", "LittleCool", "XDT"] = 0.00

    # From Cool
    current_model.B["comfort"]["Warm", "Cool", "XDT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cool", "XDT"] = 1.00
    current_model.B["comfort"]["Cool", "Cool", "XDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cool", "XDT"] = 0.00
    current_model.B["comfort"]["Cold", "Cool", "XDT"] = 0.00

    # From LittleCold
    current_model.B["comfort"]["Warm", "LittleCold", "XDT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "XDT"] = 0.00
    current_model.B["comfort"]["Cool", "LittleCold", "XDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCold", "XDT"] = 1.00
    current_model.B["comfort"]["Cold", "LittleCold", "XDT"] = 0.00

    # From Cold
    current_model.B["comfort"]["Warm", "Cold", "XDT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "XDT"] = 0.00
    current_model.B["comfort"]["Cool", "Cold", "XDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cold", "XDT"] = 0.00
    current_model.B["comfort"]["Cold", "Cold", "XDT"] = 1.00

def build_matrix_c(preferred_state, current_model):
    eps = 1e-16

    for obs in temperatures:
        current_model.C["temperature_obs"][obs] = jnp.log(
            current_model.A["temperature_obs"][obs, preferred_state] + eps
        )


def build_matrix_d(current_model):
    # =========================================================
    # D matrix: Initial prior belief over hidden states
    # =========================================================
    # Hidden state factor: comfort
    #
    # The agent initially believes that the user/environment is
    # most likely in the Uncomfortable state.
    current_model.D["comfort"]["Warm"] = 0.05
    current_model.D["comfort"]["LittleCool"] = 0.10
    current_model.D["comfort"]["Cool"] = 0.65
    current_model.D["comfort"]["LittleCold"] = 0.15
    current_model.D["comfort"]["Cold"] = 0.05
    # ======


def get_B_action_matrix(model, action_input, comforts_input):
    """
    Get B action slice.

    Expected B shape:
        B[to_state, from_state, action]

    This version supports model as dict:
        model["B"][0]

    where:
        model["B"][0] is the B matrix for the comfort hidden-state factor.
    """

    action_idx = agent_actions.index(action_input)

    # ---------------------------------------------------------
    # Case 1: model is dict, used by Agent(**model_agent)
    # ---------------------------------------------------------
    if isinstance(model, dict):
        B_dist = model["B"][0]
        B_array = extract_distribution_array(B_dist)

        return B_array[:, :, action_idx]

    # ---------------------------------------------------------
    # Case 2: fallback for old object-style model
    # ---------------------------------------------------------
    return jnp.array([
        [
            model.B["comfort"][to_state, from_state, action_input]
            for from_state in comforts_input
        ]
        for to_state in comforts_input
    ])


def predict_next_state_belief(
    model,
    qs_current,
    action_input,
    comforts_input,
):
    """
    Predict next prior belief using B and selected action.

    qs_current shape:
        (num_states,)

    B_a shape:
        (num_states_to, num_states_from)

    Output:
        q(s_{t+1}) = B_a @ q(s_t)
    """

    B_a = get_B_action_matrix(
        model=model,
        action_input=action_input,
        comforts_input=comforts_input,
    )

    qs_next = B_a @ qs_current

    # Safety normalization
    qs_next = qs_next / jnp.sum(qs_next)

    return qs_next


def environment_step(action_input, current_temperatures):
    """
    Simulate environment transition and generate new observations.

    current_true_state: string, e.g. "Uncomfortable"
    action: string, e.g. "IL"
    """

    #               T0       T1, T2       T3       T4, T5       T6
    # comforts = ["Warm", "LittleCool", "Cool", "LittleCold", "Cold"]  # hidden state
    # temperatures = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]  # Observe Temperature

    next_temperatures_index = temperatures.index(current_temperatures)

    if action_input == "FIT" and next_temperatures_index <= 2:
        next_temperatures_index = next_temperatures_index + 1
    if action_input == "FDT" and 0 < next_temperatures_index <= 3:
        next_temperatures_index = next_temperatures_index - 1

    if action_input == "NA":
        next_temperatures_index = next_temperatures_index

    # New Action
    if action_input == "AIT" and 2 < next_temperatures_index <= 5:
        next_temperatures_index = next_temperatures_index + 1
    if action_input == "ADT" and next_temperatures_index > 3:
        next_temperatures_index = next_temperatures_index - 1

    if action_input == "XDT" and 0 < next_temperatures_index <= 3:
        next_temperatures_index = next_temperatures_index - 1

    label_next_temperature = temperatures[next_temperatures_index]

    return label_next_temperature, next_temperatures_index
# ======


def extract_distribution_array(dist):
    """
    Extract raw probability array from either:
    - raw JAX / NumPy array
    - Distribution-like object
    """

    candidate_attrs = [
        "values",
        "array",
        "data",
        "tensor",
        "params",
        "parameters",
        "probabilities",
        "probs",
        "_values",
        "_array",
        "_data",
        "_tensor",
    ]

    for attr in candidate_attrs:
        if hasattr(dist, attr):
            value = getattr(dist, attr)

            if callable(value):
                value = value()

            try:
                return jnp.asarray(value)
            except Exception:
                pass

    try:
        return jnp.asarray(dist)
    except Exception as e:
        raise TypeError(
            "Cannot extract raw array from B distribution object."
        ) from e