from jax import numpy as jnp
from pymdp.distribution import compile_model
from env.elements import temperatures, lights, update_model_description, lst_action_extended, lst_slice_b, \
    lst_pairing_state, humidity, append_action


def update_matrix_agent():
    current_model = compile_model(update_model_description())
    build_matrix_a(current_model)
    build_matrix_b(current_model)
    build_matrix_c("Comfortable", current_model)
    build_matrix_d(current_model)
    return current_model


def update_model_agent(current_model_description):
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

    # humidity_obs define
    # Assumption:

    current_model.A["humidity_obs"]["H1", "Uncomfortable"] = 0.25
    current_model.A["humidity_obs"]["H2", "Uncomfortable"] = 0.35
    current_model.A["humidity_obs"]["H3", "Uncomfortable"] = 0.20
    current_model.A["humidity_obs"]["H4", "Uncomfortable"] = 0.20

    current_model.A["humidity_obs"]["H1", "Neutral"] = 0.35
    current_model.A["humidity_obs"]["H2", "Neutral"] = 0.25
    current_model.A["humidity_obs"]["H3", "Neutral"] = 0.30
    current_model.A["humidity_obs"]["H4", "Neutral"] = 0.10

    current_model.A["humidity_obs"]["H1", "Comfortable"] = 0.30
    current_model.A["humidity_obs"]["H2", "Comfortable"] = 0.20
    current_model.A["humidity_obs"]["H3", "Comfortable"] = 0.15
    current_model.A["humidity_obs"]["H4", "Comfortable"] = 0.35


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

    for obs in humidity:
        current_model.C["humidity_obs"][obs] = jnp.log(
            current_model.A["humidity_obs"][obs, preferred_state] + eps
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


def add_slice_b_by_action(action_id):
    # current_model = update_model_agent(update_model_description())
    current_model = update_matrix_agent()
    for item_str in lst_pairing_state:
        state_str = item_str.split("_")
        state_to = state_str[0]
        state_from = state_str[1]
        average_distribution = calculate_average_distribution(current_model, state_to, state_from)
        current_model.B["comfort"][state_to, state_from, action_id] = average_distribution

    lst_action_extended.append(action_id)
    lst_slice_b.append(current_model.B["comfort"][:, :, action_id])
    return current_model


def calculate_average_distribution(current_model, state_to, state_from):
    lst_item = current_model.B["comfort"][state_to, state_from, :]
    filtered_list = [value for value in lst_item if value > 0]
    total_item = 0.0
    for item in filtered_list:
        print(item)
        total_item = total_item + item
    average = total_item / len(filtered_list)
    return round(average, 2)


# def extend_action_space(action_id):
#     append_action(action_id)
#     add_slice_b_by_action(action_id)


def extend_action_space(action_id):
    append_action(action_id)
    add_slice_b_by_action(action_id)
    agent_model = update_matrix_agent()
    slice_index = 0
    for action in lst_action_extended:
        agent_model.B["comfort"][:, :, action] = lst_slice_b[slice_index]
        slice_index = slice_index + 1

    return agent_model


# def ates_update_matrix_agent(ates_action_id, current_comfort):
#     if ates_action_id in lst_action_extended:

