import jax
from jax import numpy as jnp


from env.elements import temperatures, lights, humidity, agent_actions

# ["UncomfortableLeft", "NeutralLeft", "Comfortable", "NeutralRight", "UncomfortableRight"]

def build_matrix_a(current_model):
    # temperature_obs define
    current_model.A["temperature_obs"]["T0", "UncomfortableLeft"] = 0.40
    current_model.A["temperature_obs"]["T1", "UncomfortableLeft"] = 0.25
    current_model.A["temperature_obs"]["T2", "UncomfortableLeft"] = 0.15
    current_model.A["temperature_obs"]["T3", "UncomfortableLeft"] = 0.10
    current_model.A["temperature_obs"]["T4", "UncomfortableLeft"] = 0.05
    current_model.A["temperature_obs"]["T5", "UncomfortableLeft"] = 0.05

    current_model.A["temperature_obs"]["T0", "NeutralLeft"] = 0.05
    current_model.A["temperature_obs"]["T1", "NeutralLeft"] = 0.25
    current_model.A["temperature_obs"]["T2", "NeutralLeft"] = 0.35
    current_model.A["temperature_obs"]["T3", "NeutralLeft"] = 0.20
    current_model.A["temperature_obs"]["T4", "NeutralLeft"] = 0.10
    current_model.A["temperature_obs"]["T5", "NeutralLeft"] = 0.05

    current_model.A["temperature_obs"]["T0", "Comfortable"] = 0.05
    current_model.A["temperature_obs"]["T1", "Comfortable"] = 0.10
    current_model.A["temperature_obs"]["T2", "Comfortable"] = 0.20
    current_model.A["temperature_obs"]["T3", "Comfortable"] = 0.35
    current_model.A["temperature_obs"]["T4", "Comfortable"] = 0.20
    current_model.A["temperature_obs"]["T5", "Comfortable"] = 0.10

    current_model.A["temperature_obs"]["T0", "NeutralRight"] = 0.05
    current_model.A["temperature_obs"]["T1", "NeutralRight"] = 0.10
    current_model.A["temperature_obs"]["T2", "NeutralRight"] = 0.20
    current_model.A["temperature_obs"]["T3", "NeutralRight"] = 0.35
    current_model.A["temperature_obs"]["T4", "NeutralRight"] = 0.25
    current_model.A["temperature_obs"]["T5", "NeutralRight"] = 0.05

    current_model.A["temperature_obs"]["T0", "UncomfortableRight"] = 0.05
    current_model.A["temperature_obs"]["T1", "UncomfortableRight"] = 0.05
    current_model.A["temperature_obs"]["T2", "UncomfortableRight"] = 0.10
    current_model.A["temperature_obs"]["T3", "UncomfortableRight"] = 0.15
    current_model.A["temperature_obs"]["T4", "UncomfortableRight"] = 0.25
    current_model.A["temperature_obs"]["T5", "UncomfortableRight"] = 0.40

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
    current_model.A["light_obs"]["L0", "UncomfortableLeft"] = 0.25
    current_model.A["light_obs"]["L1", "UncomfortableLeft"] = 0.20
    current_model.A["light_obs"]["L2", "UncomfortableLeft"] = 0.15
    current_model.A["light_obs"]["L3", "UncomfortableLeft"] = 0.15
    current_model.A["light_obs"]["L4", "UncomfortableLeft"] = 0.15
    current_model.A["light_obs"]["L5", "UncomfortableLeft"] = 0.10

    current_model.A["light_obs"]["L0", "NeutralLeft"] = 0.10
    current_model.A["light_obs"]["L1", "NeutralLeft"] = 0.15
    current_model.A["light_obs"]["L2", "NeutralLeft"] = 0.25
    current_model.A["light_obs"]["L3", "NeutralLeft"] = 0.20
    current_model.A["light_obs"]["L4", "NeutralLeft"] = 0.20
    current_model.A["light_obs"]["L5", "NeutralLeft"] = 0.10

    current_model.A["light_obs"]["L0", "Comfortable"] = 0.10
    current_model.A["light_obs"]["L1", "Comfortable"] = 0.15
    current_model.A["light_obs"]["L2", "Comfortable"] = 0.20
    current_model.A["light_obs"]["L3", "Comfortable"] = 0.25
    current_model.A["light_obs"]["L4", "Comfortable"] = 0.20
    current_model.A["light_obs"]["L5", "Comfortable"] = 0.10

    current_model.A["light_obs"]["L0", "NeutralRight"] = 0.10
    current_model.A["light_obs"]["L1", "NeutralRight"] = 0.20
    current_model.A["light_obs"]["L2", "NeutralRight"] = 0.20
    current_model.A["light_obs"]["L3", "NeutralRight"] = 0.25
    current_model.A["light_obs"]["L4", "NeutralRight"] = 0.15
    current_model.A["light_obs"]["L5", "NeutralRight"] = 0.10

    current_model.A["light_obs"]["L0", "UncomfortableRight"] = 0.10
    current_model.A["light_obs"]["L1", "UncomfortableRight"] = 0.15
    current_model.A["light_obs"]["L2", "UncomfortableRight"] = 0.15
    current_model.A["light_obs"]["L3", "UncomfortableRight"] = 0.15
    current_model.A["light_obs"]["L4", "UncomfortableRight"] = 0.20
    current_model.A["light_obs"]["L5", "UncomfortableRight"] = 0.25

    # humidity_obs define
    # Assumption:
    # Todo Maybe remove humidity_obs

    # current_model.A["humidity_obs"]["H0", "UncomfortableLeft"] = 0.40
    # current_model.A["humidity_obs"]["H1", "UncomfortableLeft"] = 0.25
    # current_model.A["humidity_obs"]["H2", "UncomfortableLeft"] = 0.15
    # current_model.A["humidity_obs"]["H3", "UncomfortableLeft"] = 0.10
    # current_model.A["humidity_obs"]["H4", "UncomfortableLeft"] = 0.05
    # current_model.A["humidity_obs"]["H5", "UncomfortableLeft"] = 0.05
    #
    # current_model.A["humidity_obs"]["H0", "NeutralLeft"] = 0.05
    # current_model.A["humidity_obs"]["H1", "NeutralLeft"] = 0.25
    # current_model.A["humidity_obs"]["H2", "NeutralLeft"] = 0.35
    # current_model.A["humidity_obs"]["H3", "NeutralLeft"] = 0.20
    # current_model.A["humidity_obs"]["H4", "NeutralLeft"] = 0.10
    # current_model.A["humidity_obs"]["H5", "NeutralLeft"] = 0.05
    #
    # current_model.A["humidity_obs"]["H0", "Comfortable"] = 0.05
    # current_model.A["humidity_obs"]["H1", "Comfortable"] = 0.10
    # current_model.A["humidity_obs"]["H2", "Comfortable"] = 0.20
    # current_model.A["humidity_obs"]["H3", "Comfortable"] = 0.35
    # current_model.A["humidity_obs"]["H4", "Comfortable"] = 0.20
    # current_model.A["humidity_obs"]["H5", "Comfortable"] = 0.10
    #
    # current_model.A["humidity_obs"]["H0", "NeutralRight"] = 0.05
    # current_model.A["humidity_obs"]["H1", "NeutralRight"] = 0.10
    # current_model.A["humidity_obs"]["H2", "NeutralRight"] = 0.20
    # current_model.A["humidity_obs"]["H3", "NeutralRight"] = 0.35
    # current_model.A["humidity_obs"]["H4", "NeutralRight"] = 0.25
    # current_model.A["humidity_obs"]["H5", "NeutralRight"] = 0.05
    #
    # current_model.A["humidity_obs"]["H0", "UncomfortableRight"] = 0.05
    # current_model.A["humidity_obs"]["H1", "UncomfortableRight"] = 0.05
    # current_model.A["humidity_obs"]["H2", "UncomfortableRight"] = 0.10
    # current_model.A["humidity_obs"]["H3", "UncomfortableRight"] = 0.15
    # current_model.A["humidity_obs"]["H4", "UncomfortableRight"] = 0.25
    # current_model.A["humidity_obs"]["H5", "UncomfortableRight"] = 0.40


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
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "IL"] = 0.50
    current_model.B["comfort"]["Neutral", "Uncomfortable", "IL"] = 0.35
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "IL"] = 0.15

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "IL"] = 0.35
    current_model.B["comfort"]["Neutral", "Neutral", "IL"] = 0.50
    current_model.B["comfort"]["Comfortable", "Neutral", "IL"] = 0.15

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "IL"] = 0.30
    current_model.B["comfort"]["Neutral", "Comfortable", "IL"] = 0.25
    current_model.B["comfort"]["Comfortable", "Comfortable", "IL"] = 0.45

    # =========================================================
    # DL: Decrease Light
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "DL"] = 0.40
    current_model.B["comfort"]["Neutral", "Uncomfortable", "DL"] = 0.30
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "DL"] = 0.30

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "DL"] = 0.25
    current_model.B["comfort"]["Neutral", "Neutral", "DL"] = 0.50
    current_model.B["comfort"]["Comfortable", "Neutral", "DL"] = 0.25

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "DL"] = 0.15
    current_model.B["comfort"]["Neutral", "Comfortable", "DL"] = 0.35
    current_model.B["comfort"]["Comfortable", "Comfortable", "DL"] = 0.50

    # =========================================================
    # IT: Increase Temperature
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "IT"] = 0.55
    current_model.B["comfort"]["Neutral", "Uncomfortable", "IT"] = 0.30
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "IT"] = 0.15

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "IT"] = 0.45
    current_model.B["comfort"]["Neutral", "Neutral", "IT"] = 0.45
    current_model.B["comfort"]["Comfortable", "Neutral", "IT"] = 0.10

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "IT"] = 0.35
    current_model.B["comfort"]["Neutral", "Comfortable", "IT"] = 0.30
    current_model.B["comfort"]["Comfortable", "Comfortable", "IT"] = 0.35

    # =========================================================
    # DT: Decrease Temperature
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "DT"] = 0.15
    current_model.B["comfort"]["Neutral", "Uncomfortable", "DT"] = 0.35
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "DT"] = 0.50

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "DT"] = 0.15
    current_model.B["comfort"]["Neutral", "Neutral", "DT"] = 0.40
    current_model.B["comfort"]["Comfortable", "Neutral", "DT"] = 0.45

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "DT"] = 0.10
    current_model.B["comfort"]["Neutral", "Comfortable", "DT"] = 0.40
    current_model.B["comfort"]["Comfortable", "Comfortable", "DT"] = 0.50

    # =========================================================
    # NA: No Action
    # =========================================================

    # From Uncomfortable
    current_model.B["comfort"]["Uncomfortable", "Uncomfortable", "NA"] = 0.90
    current_model.B["comfort"]["Neutral", "Uncomfortable", "NA"] = 0.05
    current_model.B["comfort"]["Comfortable", "Uncomfortable", "NA"] = 0.05

    # From Neutral
    current_model.B["comfort"]["Uncomfortable", "Neutral", "NA"] = 0.05
    current_model.B["comfort"]["Neutral", "Neutral", "NA"] = 0.90
    current_model.B["comfort"]["Comfortable", "Neutral", "NA"] = 0.05

    # From Comfortable
    current_model.B["comfort"]["Uncomfortable", "Comfortable", "NA"] = 0.05
    current_model.B["comfort"]["Neutral", "Comfortable", "NA"] = 0.05
    current_model.B["comfort"]["Comfortable", "Comfortable", "NA"] = 0.90


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

    for obs in humidity:
        current_model.C["humidity"][obs] = jnp.log(
            current_model.A["humidity"][obs, preferred_state] + eps
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


def environment_step(action_input, current_temperatures,
                     current_lights, current_humidity):
    """
    Simulate environment transition and generate new observations.

    current_true_state: string, e.g. "Uncomfortable"
    action: string, e.g. "IL"
    """


    next_temperatures_index = temperatures.index(current_temperatures)
    next_lights_index = lights.index(current_lights)
    next_humidity_index = humidity.index(current_humidity)

    if action_input == "IL":
        next_lights_index = next_lights_index + 1
    if action_input == "DL":
        next_lights_index = next_lights_index - 1
    if action_input == "IT":
        next_temperatures_index = next_temperatures_index + 1
    if action_input == "DT":
        next_temperatures_index = next_temperatures_index - 1
    if action_input == "ACN1":
        next_humidity_index = next_humidity_index + 1
    if action_input == "ACN2":
        next_humidity_index = next_humidity_index - 1

    if next_temperatures_index < 0:
        next_temperatures_index = 0
    if next_lights_index < 0:
        next_lights_index = 0
    if next_humidity_index < 0:
        next_humidity_index = 0

    if next_temperatures_index > len(temperatures)-1:
        next_temperatures_index = len(temperatures)-1
    if next_lights_index > len(lights)-1:
        next_lights_index = len(lights)-1
    if next_humidity_index > len(humidity)-1:
        next_humidity_index = len(humidity)-1

    label_next_temperature = temperatures[next_temperatures_index]
    label_next_light = lights[next_lights_index]
    label_next_humidity = humidity[next_humidity_index]

    return (label_next_temperature, label_next_light, label_next_humidity, next_temperatures_index,
            next_lights_index, next_humidity_index)
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