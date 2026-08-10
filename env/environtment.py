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
    # Agent actions: IT, DT, NA
    #
    # Assumption:
    # Each action has a different random transition distribution.
    # For each [from_state, action], probabilities over to_state sum to 1.0.

    # =========================================================
    # IT: Increase Temperature
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "IT"] = 0.05
    current_model.B["comfort"]["LittleCool", "Warm", "IT"] = 0.55
    current_model.B["comfort"]["Cool", "Warm", "IT"] = 0.20
    current_model.B["comfort"]["LittleCold", "Warm", "IT"] = 0.15
    current_model.B["comfort"]["Cold", "Warm", "IT"] = 0.05

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "IT"] = 0.10
    current_model.B["comfort"]["LittleCool", "LittleCool", "IT"] = 0.25
    current_model.B["comfort"]["Cool", "LittleCool", "IT"] = 0.45
    current_model.B["comfort"]["LittleCold", "LittleCool", "IT"] = 0.15
    current_model.B["comfort"]["Cold", "LittleCool", "IT"] = 0.05

    # From Comfortable
    current_model.B["comfort"]["Warm", "Cool", "IT"] = 0.10
    current_model.B["comfort"]["LittleCool", "Cool", "IT"] = 0.10
    current_model.B["comfort"]["Cool", "Cool", "IT"] = 0.20
    current_model.B["comfort"]["LittleCold", "Cool", "IT"] = 0.50
    current_model.B["comfort"]["Cold", "Cool", "IT"] = 0.10

    # From NeutralRight
    current_model.B["comfort"]["Warm", "LittleCold", "IT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "IT"] = 0.05
    current_model.B["comfort"]["Cool", "LittleCold", "IT"] = 0.10
    current_model.B["comfort"]["LittleCold", "LittleCold", "IT"] = 0.25
    current_model.B["comfort"]["Cold", "LittleCold", "IT"] = 0.60

    # From UncomfortableRight
    current_model.B["comfort"]["Warm", "Cold", "IT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "IT"] = 0.05
    current_model.B["comfort"]["Cool", "Cold", "IT"] = 0.10
    current_model.B["comfort"]["LittleCold", "Cold", "IT"] = 0.15
    current_model.B["comfort"]["Cold", "Cold", "IT"] = 0.70

    # =========================================================
    # DT: Decrease Temperature
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "DT"] = 0.70
    current_model.B["comfort"]["LittleCool", "Warm", "DT"] = 0.15
    current_model.B["comfort"]["Cool", "Warm", "DT"] = 0.10
    current_model.B["comfort"]["LittleCold", "Warm", "DT"] = 0.05
    current_model.B["comfort"]["Cold", "Warm", "DT"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "DT"] = 0.50
    current_model.B["comfort"]["LittleCool", "LittleCool", "DT"] = 0.20
    current_model.B["comfort"]["Cool", "LittleCool", "DT"] = 0.15
    current_model.B["comfort"]["LittleCold", "LittleCool", "DT"] = 0.10
    current_model.B["comfort"]["Cold", "LittleCool", "DT"] = 0.05

    # From Comfortable
    current_model.B["comfort"]["Warm", "Cool", "DT"] = 0.10
    current_model.B["comfort"]["LittleCool", "Cool", "DT"] = 0.45
    current_model.B["comfort"]["Cool", "Cool", "DT"] = 0.20
    current_model.B["comfort"]["LittleCold", "Cool", "DT"] = 0.15
    current_model.B["comfort"]["Cold", "Cool", "DT"] = 0.10

    # From NeutralRight
    current_model.B["comfort"]["Warm", "LittleCold", "DT"] = 0.10
    current_model.B["comfort"]["LittleCool", "LittleCold", "DT"] = 0.15
    current_model.B["comfort"]["Cool", "LittleCold", "DT"] = 0.45
    current_model.B["comfort"]["LittleCold", "LittleCold", "DT"] = 0.25
    current_model.B["comfort"]["Cold", "LittleCold", "DT"] = 0.05

    # From UncomfortableRight
    current_model.B["comfort"]["Warm", "Cold", "DT"] = 0.10
    current_model.B["comfort"]["LittleCool", "Cold", "DT"] = 0.10
    current_model.B["comfort"]["Cool", "Cold", "DT"] = 0.10
    current_model.B["comfort"]["LittleCold", "Cold", "DT"] = 0.45
    current_model.B["comfort"]["Cold", "Cold", "DT"] = 0.25

    # =========================================================
    # NA: No Action
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "NA"] = 0.90
    current_model.B["comfort"]["LittleCool", "Warm", "NA"] = 0.04
    current_model.B["comfort"]["Cool", "Warm", "NA"] = 0.02
    current_model.B["comfort"]["LittleCold", "Warm", "NA"] = 0.02
    current_model.B["comfort"]["Cold", "Warm", "NA"] = 0.02

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "NA"] = 0.04
    current_model.B["comfort"]["LittleCool", "LittleCool", "NA"] = 0.90
    current_model.B["comfort"]["Cool", "LittleCool", "NA"] = 0.04
    current_model.B["comfort"]["LittleCold", "LittleCool", "NA"] = 0.01
    current_model.B["comfort"]["Cold", "LittleCool", "NA"] = 0.01

    # From Comfortable
    current_model.B["comfort"]["Warm", "Cool", "NA"] = 0.01
    current_model.B["comfort"]["LittleCool", "Cool", "NA"] = 0.04
    current_model.B["comfort"]["Cool", "Cool", "NA"] = 0.90
    current_model.B["comfort"]["LittleCold", "Cool", "NA"] = 0.04
    current_model.B["comfort"]["Cold", "Cool", "NA"] = 0.01

    # From NeutralRight
    current_model.B["comfort"]["Warm", "LittleCold", "NA"] = 0.01
    current_model.B["comfort"]["LittleCool", "LittleCold", "NA"] = 0.01
    current_model.B["comfort"]["Cool", "LittleCold", "NA"] = 0.04
    current_model.B["comfort"]["LittleCold", "LittleCold", "NA"] = 0.90
    current_model.B["comfort"]["Cold", "LittleCold", "NA"] = 0.04

    # From UncomfortableRight
    current_model.B["comfort"]["Warm", "Cold", "NA"] = 0.02
    current_model.B["comfort"]["LittleCool", "Cold", "NA"] = 0.02
    current_model.B["comfort"]["Cool", "Cold", "NA"] = 0.02
    current_model.B["comfort"]["LittleCold", "Cold", "NA"] = 0.04
    current_model.B["comfort"]["Cold", "Cold", "NA"] = 0.90


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

    next_temperatures_index = temperatures.index(current_temperatures)

    if action_input == "IT":
        next_temperatures_index = next_temperatures_index + 1
    if action_input == "DT":
        next_temperatures_index = next_temperatures_index - 1
    if action_input == "NA":
        next_temperatures_index = next_temperatures_index

    if next_temperatures_index < 0:
        next_temperatures_index = 0

    if next_temperatures_index > len(temperatures)-1:
        next_temperatures_index = len(temperatures)-1

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