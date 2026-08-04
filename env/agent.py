
from pymdp.distribution import compile_model
from env.elements import update_model_description, lst_action_extended, lst_slice_b, \
    lst_pairing_state, append_action


import jax
from pymdp.agent import Agent
from jax import numpy as jnp
import jax.tree_util as jtu
from env.elements import comforts, agent_actions
from env.environtment import \
    temperatures, lights, environment_step, \
    predict_next_state_belief
from utils.util import sample_top_k_with_temperature


def update_matrix_agent():
    current_model = compile_model(update_model_description())
    build_matrix_a(current_model)
    build_matrix_b(current_model)
    # build_matrix_c("Comfortable", current_model)
    build_matrix_c(current_model)
    build_matrix_d(current_model)
    return current_model


def update_model_agent(current_model_description):
    return compile_model(current_model_description)


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
    current_model.A["temperature_obs"]["T3", "NeutralRight"] = 0.25
    current_model.A["temperature_obs"]["T4", "NeutralRight"] = 0.35
    current_model.A["temperature_obs"]["T5", "NeutralRight"] = 0.05

    current_model.A["temperature_obs"]["T0", "UncomfortableRight"] = 0.05
    current_model.A["temperature_obs"]["T1", "UncomfortableRight"] = 0.05
    current_model.A["temperature_obs"]["T2", "UncomfortableRight"] = 0.10
    current_model.A["temperature_obs"]["T3", "UncomfortableRight"] = 0.15
    current_model.A["temperature_obs"]["T4", "UncomfortableRight"] = 0.25
    current_model.A["temperature_obs"]["T5", "UncomfortableRight"] = 0.40

    # light_obs define

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
    current_model.A["light_obs"]["L3", "NeutralRight"] = 0.15
    current_model.A["light_obs"]["L4", "NeutralRight"] = 0.25
    current_model.A["light_obs"]["L5", "NeutralRight"] = 0.10

    current_model.A["light_obs"]["L0", "UncomfortableRight"] = 0.10
    current_model.A["light_obs"]["L1", "UncomfortableRight"] = 0.15
    current_model.A["light_obs"]["L2", "UncomfortableRight"] = 0.15
    current_model.A["light_obs"]["L3", "UncomfortableRight"] = 0.15
    current_model.A["light_obs"]["L4", "UncomfortableRight"] = 0.20
    current_model.A["light_obs"]["L5", "UncomfortableRight"] = 0.25


def build_matrix_b(current_model):
    # fill in the transition model (B) tensor
    # note that it's specified as ["to", "from", "action"]
    # Hidden states: "UncomfortableLeft", "NeutralLeft", "Comfortable", "NeutralRight", "UncomfortableRight"
    # Agent actions: IT, DT, NA
    #
    # Assumption:
    # Each action has a different random transition distribution.
    # For each [from_state, action], probabilities over to_state sum to 1.0.

    # =========================================================
    # IT: Increase Temperature
    # =========================================================

    # From UncomfortableLeft
    current_model.B["comfort"]["UncomfortableLeft", "UncomfortableLeft", "IT"] = 0.05
    current_model.B["comfort"]["NeutralLeft", "UncomfortableLeft", "IT"] = 0.55
    current_model.B["comfort"]["Comfortable", "UncomfortableLeft", "IT"] = 0.20
    current_model.B["comfort"]["NeutralRight", "UncomfortableLeft", "IT"] = 0.15
    current_model.B["comfort"]["UncomfortableRight", "UncomfortableLeft", "IT"] = 0.05

    # From NeutralLeft
    current_model.B["comfort"]["UncomfortableLeft", "NeutralLeft", "IT"] = 0.10
    current_model.B["comfort"]["NeutralLeft", "NeutralLeft", "IT"] = 0.25
    current_model.B["comfort"]["Comfortable", "NeutralLeft", "IT"] = 0.45
    current_model.B["comfort"]["NeutralRight", "NeutralLeft", "IT"] = 0.15
    current_model.B["comfort"]["UncomfortableRight", "NeutralLeft", "IT"] = 0.05

    # From Comfortable
    current_model.B["comfort"]["UncomfortableLeft", "Comfortable", "IT"] = 0.10
    current_model.B["comfort"]["NeutralLeft", "Comfortable", "IT"] = 0.10
    current_model.B["comfort"]["Comfortable", "Comfortable", "IT"] = 0.20
    current_model.B["comfort"]["NeutralRight", "Comfortable", "IT"] = 0.50
    current_model.B["comfort"]["UncomfortableRight", "Comfortable", "IT"] = 0.10

    # From NeutralRight
    current_model.B["comfort"]["UncomfortableLeft", "NeutralRight", "IT"] = 0.00
    current_model.B["comfort"]["NeutralLeft", "NeutralRight", "IT"] = 0.05
    current_model.B["comfort"]["Comfortable", "NeutralRight", "IT"] = 0.10
    current_model.B["comfort"]["NeutralRight", "NeutralRight", "IT"] = 0.25
    current_model.B["comfort"]["UncomfortableRight", "NeutralRight", "IT"] = 0.60

    # From UncomfortableRight
    current_model.B["comfort"]["UncomfortableLeft", "UncomfortableRight", "IT"] = 0.00
    current_model.B["comfort"]["NeutralLeft", "UncomfortableRight", "IT"] = 0.05
    current_model.B["comfort"]["Comfortable", "UncomfortableRight", "IT"] = 0.10
    current_model.B["comfort"]["NeutralRight", "UncomfortableRight", "IT"] = 0.15
    current_model.B["comfort"]["UncomfortableRight", "UncomfortableRight", "IT"] = 0.70

    # =========================================================
    # DT: Decrease Temperature
    # =========================================================

    # From UncomfortableLeft
    current_model.B["comfort"]["UncomfortableLeft", "UncomfortableLeft", "DT"] = 0.70
    current_model.B["comfort"]["NeutralLeft", "UncomfortableLeft", "DT"] = 0.15
    current_model.B["comfort"]["Comfortable", "UncomfortableLeft", "DT"] = 0.10
    current_model.B["comfort"]["NeutralRight", "UncomfortableLeft", "DT"] = 0.05
    current_model.B["comfort"]["UncomfortableRight", "UncomfortableLeft", "DT"] = 0.00

    # From NeutralLeft
    current_model.B["comfort"]["UncomfortableLeft", "NeutralLeft", "DT"] = 0.50
    current_model.B["comfort"]["NeutralLeft", "NeutralLeft", "DT"] = 0.20
    current_model.B["comfort"]["Comfortable", "NeutralLeft", "DT"] = 0.15
    current_model.B["comfort"]["NeutralRight", "NeutralLeft", "DT"] = 0.10
    current_model.B["comfort"]["UncomfortableRight", "NeutralLeft", "DT"] = 0.05

    # From Comfortable
    current_model.B["comfort"]["UncomfortableLeft", "Comfortable", "DT"] = 0.10
    current_model.B["comfort"]["NeutralLeft", "Comfortable", "DT"] = 0.45
    current_model.B["comfort"]["Comfortable", "Comfortable", "DT"] = 0.20
    current_model.B["comfort"]["NeutralRight", "Comfortable", "DT"] = 0.15
    current_model.B["comfort"]["UncomfortableRight", "Comfortable", "DT"] = 0.10

    # From NeutralRight
    current_model.B["comfort"]["UncomfortableLeft", "NeutralRight", "DT"] = 0.10
    current_model.B["comfort"]["NeutralLeft", "NeutralRight", "DT"] = 0.15
    current_model.B["comfort"]["Comfortable", "NeutralRight", "DT"] = 0.45
    current_model.B["comfort"]["NeutralRight", "NeutralRight", "DT"] = 0.25
    current_model.B["comfort"]["UncomfortableRight", "NeutralRight", "DT"] = 0.05

    # From UncomfortableRight
    current_model.B["comfort"]["UncomfortableLeft", "UncomfortableRight", "DT"] = 0.10
    current_model.B["comfort"]["NeutralLeft", "UncomfortableRight", "DT"] = 0.10
    current_model.B["comfort"]["Comfortable", "UncomfortableRight", "DT"] = 0.10
    current_model.B["comfort"]["NeutralRight", "UncomfortableRight", "DT"] = 0.45
    current_model.B["comfort"]["UncomfortableRight", "UncomfortableRight", "DT"] = 0.25

    # =========================================================
    # NA: No Action
    # =========================================================

    # From UncomfortableLeft
    current_model.B["comfort"]["UncomfortableLeft", "UncomfortableLeft", "NA"] = 0.90
    current_model.B["comfort"]["NeutralLeft", "UncomfortableLeft", "NA"] = 0.04
    current_model.B["comfort"]["Comfortable", "UncomfortableLeft", "NA"] = 0.02
    current_model.B["comfort"]["NeutralRight", "UncomfortableLeft", "NA"] = 0.02
    current_model.B["comfort"]["UncomfortableRight", "UncomfortableLeft", "NA"] = 0.02

    # From NeutralLeft
    current_model.B["comfort"]["UncomfortableLeft", "NeutralLeft", "NA"] = 0.04
    current_model.B["comfort"]["NeutralLeft", "NeutralLeft", "NA"] = 0.90
    current_model.B["comfort"]["Comfortable", "NeutralLeft", "NA"] = 0.04
    current_model.B["comfort"]["NeutralRight", "NeutralLeft", "NA"] = 0.01
    current_model.B["comfort"]["UncomfortableRight", "NeutralLeft", "NA"] = 0.01

    # From Comfortable
    current_model.B["comfort"]["UncomfortableLeft", "Comfortable", "NA"] = 0.01
    current_model.B["comfort"]["NeutralLeft", "Comfortable", "NA"] = 0.04
    current_model.B["comfort"]["Comfortable", "Comfortable", "NA"] = 0.90
    current_model.B["comfort"]["NeutralRight", "Comfortable", "NA"] = 0.04
    current_model.B["comfort"]["UncomfortableRight", "Comfortable", "NA"] = 0.01

    # From NeutralRight
    current_model.B["comfort"]["UncomfortableLeft", "NeutralRight", "NA"] = 0.01
    current_model.B["comfort"]["NeutralLeft", "NeutralRight", "NA"] = 0.01
    current_model.B["comfort"]["Comfortable", "NeutralRight", "NA"] = 0.04
    current_model.B["comfort"]["NeutralRight", "NeutralRight", "NA"] = 0.90
    current_model.B["comfort"]["UncomfortableRight", "NeutralRight", "NA"] = 0.04

    # From UncomfortableRight
    current_model.B["comfort"]["UncomfortableLeft", "UncomfortableRight", "NA"] = 0.02
    current_model.B["comfort"]["NeutralLeft", "UncomfortableRight", "NA"] = 0.02
    current_model.B["comfort"]["Comfortable", "UncomfortableRight", "NA"] = 0.02
    current_model.B["comfort"]["NeutralRight", "UncomfortableRight", "NA"] = 0.04
    current_model.B["comfort"]["UncomfortableRight", "UncomfortableRight", "NA"] = 0.90


def build_matrix_c(current_model):
    # Temperature preferences
    current_model.C["temperature_obs"]["T0"] = -4.0
    current_model.C["temperature_obs"]["T1"] = 2.0
    current_model.C["temperature_obs"]["T2"] = 2.0
    current_model.C["temperature_obs"]["T3"] = 4.0
    current_model.C["temperature_obs"]["T4"] = 1.0
    current_model.C["temperature_obs"]["T5"] = -4.0

    # Light preferences
    current_model.C["light_obs"]["L0"] = -3.0
    current_model.C["light_obs"]["L1"] = 1.0
    current_model.C["light_obs"]["L2"] = 2.0
    current_model.C["light_obs"]["L3"] = 4.0
    current_model.C["light_obs"]["L4"] = 2.0
    current_model.C["light_obs"]["L5"] = -2.0

    return current_model


def build_matrix_d(current_model):
    # =========================================================
    # D matrix: Initial prior belief over hidden states
    # =========================================================
    # Hidden state factor: comfort
    #
    # The agent initially believes that the user/environment is
    # most likely in the Uncomfortable state.
    current_model.D["comfort"]["UncomfortableLeft"] = 0.05
    current_model.D["comfort"]["NeutralLeft"] = 0.10
    current_model.D["comfort"]["Comfortable"] = 0.65
    current_model.D["comfort"]["NeutralRight"] = 0.15
    current_model.D["comfort"]["UncomfortableRight"] = 0.05
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


def calculate_average_distribution(
    current_model,
    state_to,
    state_from,
):
    source_actions = ["IT", "DT", "NA"]

    return sum(
        float(
            current_model.B["comfort"][
                state_to,
                state_from,
                action,
            ]
        )
        for action in source_actions
    ) / len(source_actions)


def extend_action_space(action_id):
    append_action(action_id)
    add_slice_b_by_action(action_id)
    print("Done_1")
    agent_model = update_matrix_agent()
    print("Done_2")
    slice_index = 0
    for action in lst_action_extended:
        agent_model.B["comfort"][:, :, action] = lst_slice_b[slice_index]
        slice_index = slice_index + 1

    return agent_model


def get_ates_positive_point(ates_action_id, current_state, expect_state, current_model, belief_prob, upd_point_ratio):
    current_transition_prob = current_model.B["comfort"][expect_state, current_state, ates_action_id]
    positive_point = (1 - current_transition_prob) * upd_point_ratio / 100 * belief_prob / 100
    return round(positive_point, 2)


def get_ates_negative_point(ates_action_id, current_state, expect_state, current_model, belief_prob, upd_point_ratio):
    current_transition_prob = current_model.B["comfort"][expect_state, current_state, ates_action_id]
    negative_point = (1 - current_transition_prob) * upd_point_ratio / 100 * belief_prob / 100
    return round(negative_point, 2)


def ates_positive_update(ates_action_id, current_state, expect_state, current_model, belief_prob, upd_point_ratio):
    lst_transition_prob = current_model.B["comfort"][:, current_state, ates_action_id]
    remainder_prob_num = len(lst_transition_prob) - 1
    ates_positive_point = get_ates_positive_point(ates_action_id, current_state, expect_state, current_model,
                                                  belief_prob, upd_point_ratio)
    ates_positive_remain_prob = ates_positive_point / remainder_prob_num
    return round(ates_positive_remain_prob, 3)


def ates_negative_update(ates_action_id, current_state, expect_state, current_model, belief_prob, upd_point_ratio):
    lst_transition_prob = current_model.B["comfort"][:, current_state, ates_action_id]
    remainder_prob_num = len(lst_transition_prob) - 1
    ates_negative_point = get_ates_negative_point(ates_action_id, current_state, expect_state, current_model,
                                                  belief_prob, upd_point_ratio)
    ates_negative_remain_prob = ates_negative_point / remainder_prob_num
    return round(ates_negative_remain_prob, 3)


def ates_update_matrix_positive(ates_action_id, current_state, expect_state, current_model, ates_diff, ates_point):
    current_model.B["comfort"][expect_state, current_state, ates_action_id] += ates_point
    for item_str in lst_pairing_state:
        state_str = item_str.split("_")
        state_to = state_str[0]
        state_from = state_str[1]
        if state_from == current_state and state_to != expect_state:
            current_model.B["comfort"][state_to, state_from, ates_action_id] -= ates_diff
    return current_model
    # return current_model.B["comfort"][:, current_state, ates_action_id]


def ates_update_matrix_negative(chosen_action_id, current_state, expect_state, current_model, ates_diff, ates_point):
    current_model.B["comfort"][expect_state, current_state, chosen_action_id] -= ates_point
    for item_str in lst_pairing_state:
        state_str = item_str.split("_")
        state_to = state_str[0]
        state_from = state_str[1]
        if state_from == current_state and state_to != expect_state:
            current_model.B["comfort"][state_to, state_from, chosen_action_id] += ates_diff
    return current_model
    # return current_model.B["comfort"][:, current_state, chosen_action_id]


def infer_belief_once(
        model_agent,
        temperature_observed,
        light_observed,
        qs_prior_input=None,
):
    gamma = 1

    agent = Agent(**model_agent, gamma=gamma, policy_len=1)

    temperature_idx = temperatures.index(temperature_observed)
    light_idx = lights.index(light_observed)

    observations = [
        jnp.full((agent.batch_size, 1), temperature_idx),
        jnp.full((agent.batch_size, 1), light_idx),
    ]

    if qs_prior_input is None:
        # First run: use D as prior
        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 1),
            agent.D
        )
    else:
        # Later runs: use previous posterior as prior
        qs_init = [
            jnp.expand_dims(
                jnp.expand_dims(qs_prior_input, axis=0),
                axis=1
            )
        ]

    qs = agent.infer_states(observations, qs_init)

    comfort_belief = jnp.squeeze(qs[0], axis=(0, 1, 2))

    print("\n===== BELIEF INFERENCE =====")
    print(f"Observed temperature: {temperature_observed}")
    print(f"Observed light:       {light_observed}")

    print("\nPosterior belief over comfort:")
    for i, state in enumerate(comforts):
        print(f"{state}: {float(comfort_belief[i]):.4f}")

    current_comfort_idx = int(jnp.argmax(comfort_belief))
    print(f"Most likely comfort state: {comforts[current_comfort_idx]}")

    return comfort_belief


def run_agent(
        model_agent,
        temperature_observed,
        light_observed,
        qs_prior_input=None,
        rng_key=None,
):
    # =========================================================
    # Run Active Inference agent
    # =========================================================

    gamma = 1  # deterministic behavior; smaller gamma -> more stochastic behavior

    # Create agent
    agent = Agent(**model_agent, gamma=gamma, policy_len=1)

    temperature_idx = temperatures.index(temperature_observed)
    light_idx = lights.index(light_observed)

    # Each observation must have shape: (batch_size, time_dim)
    # agent.batch_size defaults to 1.
    temperature_observation = jnp.full(
        (agent.batch_size, 1),
        temperature_idx
    )

    light_observation = jnp.full(
        (agent.batch_size, 1),
        light_idx
    )

    # Multi-modality observation list
    observations = [
        temperature_observation,
        light_observation,
    ]

    if qs_prior_input is None:
        # First run: use agent.D as initial prior
        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 1),
            agent.D
        )
    else:
        # Later runs: use previous posterior belief as current prior
        # qs_prior_input shape: (num_states,)
        # required shape: (batch_size, time_dim, num_states)
        qs_init = [
            jnp.expand_dims(
                jnp.expand_dims(qs_prior_input, axis=0),
                axis=1
            )
        ]

    # ---------------------------------------------------------
    # Infer hidden states
    # ---------------------------------------------------------

    qs = agent.infer_states(observations, qs_init)

    print("\n===== INITIAL OBSERVATION =====")
    print(f"Observed temperature: {temperature_observed}")
    print(f"Observed light:       {light_observed}")

    print("\n===== DEBUG SHAPE =====")
    print("qs[0].shape:", qs[0].shape)

    # qs[0] currently has shape: (batch_size, 1, 1, num_states)
    # Example: (1, 1, 1, 3)
    # For printing, convert it to a clean vector: (3,)
    comfort_belief = jnp.squeeze(qs[0], axis=(0, 1, 2))

    print("comfort_belief.shape:", comfort_belief.shape)

    print("\n===== POSTERIOR BELIEF OVER COMFORT =====")
    for i, state in enumerate(comforts):
        print(f"{state}: {float(comfort_belief[i]):.4f}")

    current_comfort_idx = int(jnp.argmax(comfort_belief))
    print(f"\nMost likely comfort state: {comforts[current_comfort_idx]}")

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Prepare posterior belief for policy inference
    # ---------------------------------------------------------
    # agent.infer_policies expects qs with shape:
    # (batch_size, time_dim, num_states)
    #
    # Current qs[0] shape: (1, 1, 1, 3)
    # We remove only the extra singleton axis at axis=2:
    # Result shape: (1, 1, 3)

    qs_for_policy = [
        jnp.squeeze(q, axis=2)
        for q in qs
    ]

    print("\n===== DEBUG POLICY INPUT SHAPE =====")
    print("qs_for_policy[0].shape:", qs_for_policy[0].shape)

    # ---------------------------------------------------------
    # Infer policies and sample action
    # ---------------------------------------------------------

    q_pi, G = agent.infer_policies(qs_for_policy)

    print("\n===== POLICY INFERENCE =====")
    print("q_pi:", q_pi)
    print("q_pi.shape:", q_pi.shape)
    print("G:", G)

    # rng_key must match batch dimension.
    # q_pi.shape = (batch_size, num_policies)
    # Therefore rng_key.shape should be (batch_size, 2)
    # ---------------------------------------------------------
    # Custom stochastic sampling directly from q_pi
    # ---------------------------------------------------------

    # Initialize rng_key once before the interaction loop.
    # If this code is not inside a loop yet, placing it here is still fine.
    if rng_key is None:
        rng_key = jax.random.PRNGKey(25)

    probs = q_pi[0]  # shape: (num_actions,)

    # Split the key before sampling so each random draw uses a fresh subkey.
    rng_key, action_sample_key = jax.random.split(rng_key)

    # ======

    chosen_action_idx, chosen_action, rng_key, top_indices, top_probs_temp = sample_top_k_with_temperature(
        q_pi=q_pi,
        rng_key=rng_key,
        agent_actions=agent_actions,
        k=3,
        temperature=0.1,
    )

    # ======

    # chosen_action_idx = int(jnp.argmax(q_pi[0]))
    # chosen_action = agent_actions[chosen_action_idx]

    print("\n===== CUSTOM STOCHASTIC ACTION SELECTED =====")
    print("probs:", probs)
    print(f"Action chosen: {chosen_action}")

    print("\n===== ACTION PROBABILITIES =====")
    for i, action in enumerate(agent_actions):
        print(f"{i}: {action:6s} | q_pi={float(q_pi[0][i]):.4f} | G={float(G[0][i]):.4f}")

    # =========================================================
    # Execute action a_t in the environment
    # =========================================================

    # In a real simulator, the environment should maintain its own true hidden state.
    # For this first version, we approximate the current true state using the most
    # likely inferred comfort state.
    current_infer_state = comforts[current_comfort_idx]

    print("\n===== EXECUTE ACTION =====")
    print(f"Current infer state used by simulator: {current_infer_state}")
    print(f"Executed action a_t: {chosen_action}")

    (label_next_temperature, label_next_light, next_temperatures_index, next_lights_index) = environment_step(
        action_input=chosen_action,
        current_temperatures=temperature_observed,
        current_lights=light_observed,
    )

    print("\n===== ENVIRONMENT RESULT =====")
    # print(f"Next true hidden state s_t+1: {next_true_state}")
    print(f"New temperature observation: {label_next_temperature}")
    print(f"New light observation:       {label_next_light}")

    # =========================================================
    # Predict next prior q(s_{t+1}) using B and selected action
    # =========================================================

    qs_prior_next = predict_next_state_belief(
        model=model_agent,
        qs_current=comfort_belief,
        action_input=chosen_action,
        comforts_input=comforts,
    )

    print("\n===== PREDICTED PRIOR AFTER ACTION =====")
    for i, state in enumerate(comforts):
        print(
            f"Prior q(s_t+1={state}) before new observation: "
            f"{float(qs_prior_next[i]):.4f}"
        )

    # =========================================================
    # Do not infer the next observation here.
    #
    # qs_prior_next is the predicted prior for the next timestep:
    #
    # q^-(s_t+1) = B[a_t] @ q(s_t)
    #
    # It will be passed into the next run_agent() call. At that
    # point, the new observation will be processed exactly once:
    #
    # q(s_t+1) ∝ P(o_t+1 | s_t+1) * q^-(s_t+1)
    # =========================================================

    return {
        "current_temperature": temperature_observed,
        "current_light": light_observed,
        "current_belief": comfort_belief,
        "next_temperature": label_next_temperature,
        "next_light": label_next_light,
        "predicted_prior_next": qs_prior_next,
        "chosen_action": chosen_action,
        "q_pi": q_pi,
        "G": G,
        "rng_key": rng_key,
    }


# Baseline
