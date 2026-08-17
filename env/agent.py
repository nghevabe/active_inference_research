
from pymdp.distribution import compile_model
from env.elements import update_model_description, lst_action_extended, lst_slice_b, \
    lst_pairing_state, append_action, base_actions

import jax
from pymdp.agent import Agent
from jax import numpy as jnp
import jax.tree_util as jtu
from env.elements import comforts, agent_actions
from env.environtment import \
    temperatures, environment_step, \
    predict_next_state_belief
from utils.util import sample_top_k_with_temperature


def update_matrix_agent():
    current_model = compile_model(update_model_description())
    build_matrix_a(current_model)
    build_matrix_b(current_model)
    # build_matrix_c("Cool", current_model)
    build_matrix_c(current_model)
    build_matrix_d(current_model)
    return current_model


def update_model_agent(current_model_description):
    return compile_model(current_model_description)


def build_matrix_a(current_model):
    # temperature_obs define
    current_model.A["temperature_obs"]["T0", "Warm"] = 0.85
    current_model.A["temperature_obs"]["T1", "Warm"] = 0.10
    current_model.A["temperature_obs"]["T2", "Warm"] = 0.05
    current_model.A["temperature_obs"]["T3", "Warm"] = 0.00
    current_model.A["temperature_obs"]["T4", "Warm"] = 0.00
    current_model.A["temperature_obs"]["T5", "Warm"] = 0.00
    current_model.A["temperature_obs"]["T6", "Warm"] = 0.00

    current_model.A["temperature_obs"]["T0", "LittleCool"] = 0.10
    current_model.A["temperature_obs"]["T1", "LittleCool"] = 0.40
    current_model.A["temperature_obs"]["T2", "LittleCool"] = 0.40
    current_model.A["temperature_obs"]["T3", "LittleCool"] = 0.10
    current_model.A["temperature_obs"]["T4", "LittleCool"] = 0.00
    current_model.A["temperature_obs"]["T5", "LittleCool"] = 0.00
    current_model.A["temperature_obs"]["T6", "LittleCool"] = 0.00

    current_model.A["temperature_obs"]["T0", "Cool"] = 0.00
    current_model.A["temperature_obs"]["T1", "Cool"] = 0.05
    current_model.A["temperature_obs"]["T2", "Cool"] = 0.05
    current_model.A["temperature_obs"]["T3", "Cool"] = 0.80
    current_model.A["temperature_obs"]["T4", "Cool"] = 0.05
    current_model.A["temperature_obs"]["T5", "Cool"] = 0.05
    current_model.A["temperature_obs"]["T6", "Cool"] = 0.00

    current_model.A["temperature_obs"]["T0", "LittleCold"] = 0.00
    current_model.A["temperature_obs"]["T1", "LittleCold"] = 0.00
    current_model.A["temperature_obs"]["T2", "LittleCold"] = 0.00
    current_model.A["temperature_obs"]["T3", "LittleCold"] = 0.10
    current_model.A["temperature_obs"]["T4", "LittleCold"] = 0.40
    current_model.A["temperature_obs"]["T5", "LittleCold"] = 0.40
    current_model.A["temperature_obs"]["T6", "LittleCold"] = 0.10

    current_model.A["temperature_obs"]["T0", "Cold"] = 0.00
    current_model.A["temperature_obs"]["T1", "Cold"] = 0.00
    current_model.A["temperature_obs"]["T2", "Cold"] = 0.00
    current_model.A["temperature_obs"]["T3", "Cold"] = 0.00
    current_model.A["temperature_obs"]["T4", "Cold"] = 0.05
    current_model.A["temperature_obs"]["T5", "Cold"] = 0.10
    current_model.A["temperature_obs"]["T6", "Cold"] = 0.85


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
    current_model.B["comfort"]["Warm", "Warm", "FIT"] = 0.10
    current_model.B["comfort"]["LittleCool", "Warm", "FIT"] = 0.85
    current_model.B["comfort"]["Cool", "Warm", "FIT"] = 0.05
    current_model.B["comfort"]["LittleCold", "Warm", "FIT"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "FIT"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "FIT"] = 0.05
    current_model.B["comfort"]["LittleCool", "LittleCool", "FIT"] = 0.45
    current_model.B["comfort"]["Cool", "LittleCool", "FIT"] = 0.45
    current_model.B["comfort"]["LittleCold", "LittleCool", "FIT"] = 0.05
    current_model.B["comfort"]["Cold", "LittleCool", "FIT"] = 0.00

    # From Cool
    current_model.B["comfort"]["Warm", "Cool", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cool", "FIT"] = 0.00
    current_model.B["comfort"]["Cool", "Cool", "FIT"] = 0.95
    current_model.B["comfort"]["LittleCold", "Cool", "FIT"] = 0.05
    current_model.B["comfort"]["Cold", "Cool", "FIT"] = 0.00

    # From LittleCold
    current_model.B["comfort"]["Warm", "LittleCold", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "FIT"] = 0.00
    current_model.B["comfort"]["Cool", "LittleCold", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCold", "FIT"] = 0.95
    current_model.B["comfort"]["Cold", "LittleCold", "FIT"] = 0.05

    # From Cold
    current_model.B["comfort"]["Warm", "Cold", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "FIT"] = 0.00
    current_model.B["comfort"]["Cool", "Cold", "FIT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cold", "FIT"] = 0.05
    current_model.B["comfort"]["Cold", "Cold", "FIT"] = 0.95

    # =========================================================
    # FDT: Fan Decrease Temperature
    # =========================================================

    # From Warm
    current_model.B["comfort"]["Warm", "Warm", "FDT"] = 0.90
    current_model.B["comfort"]["LittleCool", "Warm", "FDT"] = 0.05
    current_model.B["comfort"]["Cool", "Warm", "FDT"] = 0.05
    current_model.B["comfort"]["LittleCold", "Warm", "FDT"] = 0.00
    current_model.B["comfort"]["Cold", "Warm", "FDT"] = 0.00

    # From LittleCool
    current_model.B["comfort"]["Warm", "LittleCool", "FDT"] = 0.05
    current_model.B["comfort"]["LittleCool", "LittleCool", "FDT"] = 0.95
    current_model.B["comfort"]["Cool", "LittleCool", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "LittleCool", "FDT"] = 0.00
    current_model.B["comfort"]["Cold", "LittleCool", "FDT"] = 0.00

    # From Cool
    current_model.B["comfort"]["Warm", "Cool", "FDT"] = 0.05
    current_model.B["comfort"]["LittleCool", "Cool", "FDT"] = 0.95
    current_model.B["comfort"]["Cool", "Cool", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cool", "FDT"] = 0.00
    current_model.B["comfort"]["Cold", "Cool", "FDT"] = 0.00

    # From LittleCold
    current_model.B["comfort"]["Warm", "LittleCold", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCool", "LittleCold", "FDT"] = 0.00
    current_model.B["comfort"]["Cool", "LittleCold", "FDT"] = 0.05
    current_model.B["comfort"]["LittleCold", "LittleCold", "FDT"] = 0.95
    current_model.B["comfort"]["Cold", "LittleCold", "FDT"] = 0.00

    # From Cold
    current_model.B["comfort"]["Warm", "Cold", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCool", "Cold", "FDT"] = 0.00
    current_model.B["comfort"]["Cool", "Cold", "FDT"] = 0.00
    current_model.B["comfort"]["LittleCold", "Cold", "FDT"] = 0.05
    current_model.B["comfort"]["Cold", "Cold", "FDT"] = 0.95

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


def build_matrix_c(current_model):
    # Temperature preferences T3
    # current_model.C["temperature_obs"]["T0"] = -4.0
    # current_model.C["temperature_obs"]["T1"] = 2.0
    # current_model.C["temperature_obs"]["T2"] = 2.0
    # current_model.C["temperature_obs"]["T3"] = 4.0
    # current_model.C["temperature_obs"]["T4"] = 1.0
    # current_model.C["temperature_obs"]["T5"] = -4.0
    # current_model.C["temperature_obs"]["T6"] = -4.0


    # # Temperature preferences for T6
    # current_model.C["temperature_obs"]["T0"] = -4.0
    # current_model.C["temperature_obs"]["T1"] = -2.0
    # current_model.C["temperature_obs"]["T2"] = 0.0
    # current_model.C["temperature_obs"]["T3"] = 1.0
    # current_model.C["temperature_obs"]["T4"] = 2.0
    # current_model.C["temperature_obs"]["T5"] = 4.0
    # current_model.C["temperature_obs"]["T6"] = 6.0


    # # Temperature preferences for T0
    # current_model.C["temperature_obs"]["T0"] = 6.0
    # current_model.C["temperature_obs"]["T1"] = 4.0
    # current_model.C["temperature_obs"]["T2"] = 2.0
    # current_model.C["temperature_obs"]["T3"] = 1.0
    # current_model.C["temperature_obs"]["T4"] = 0.0
    # current_model.C["temperature_obs"]["T5"] = -2.0
    # current_model.C["temperature_obs"]["T6"] = -4.0


    # Temperature preferences for T6
    current_model.C["temperature_obs"]["T0"] = 0.0
    current_model.C["temperature_obs"]["T1"] = 0.0
    current_model.C["temperature_obs"]["T2"] = 0.0
    current_model.C["temperature_obs"]["T3"] = 0.0
    current_model.C["temperature_obs"]["T4"] = 0.0
    current_model.C["temperature_obs"]["T5"] = 0.0
    current_model.C["temperature_obs"]["T6"] = 6.0

    return current_model


def build_matrix_d(current_model):
    # =========================================================
    # D matrix: Initial prior belief over hidden states
    # =========================================================
    # Hidden state factor: comfort
    #
    # The agent initially believes that the user/environment is
    # most likely in the Uncomfortable state.
    current_model.D["comfort"]["Warm"] = 0.05
    current_model.D["comfort"]["LittleCool"] = 0.05
    current_model.D["comfort"]["Cool"] = 0.75
    current_model.D["comfort"]["LittleCold"] = 0.10
    current_model.D["comfort"]["Cold"] = 0.05
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
    filtered_list = lst_item[0:len(base_actions)]
    total_item = 0.0
    for item in filtered_list:
        total_item = total_item + item
    average = total_item / len(filtered_list)
    return average


# def calculate_average_distribution(
#     current_model,
#     state_to,
#     state_from,
# ):
#     source_actions = ["IT", "DT", "NA"]
#
#     return sum(
#         float(
#             current_model.B["comfort"][
#                 state_to,
#                 state_from,
#                 action,
#             ]
#         )
#         for action in source_actions
#     ) / len(source_actions)


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
        qs_prior_input=None,
):
    gamma = 1

    agent = Agent(**model_agent, gamma=gamma, policy_len=1)

    temperature_idx = temperatures.index(temperature_observed)

    observations = [
        jnp.full((agent.batch_size, 1), temperature_idx)
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

    print("\nPosterior belief over comfort:")
    for i, state in enumerate(comforts):
        print(f"{state}: {float(comfort_belief[i]):.4f}")

    current_comfort_idx = int(jnp.argmax(comfort_belief))
    print(f"Most likely comfort state: {comforts[current_comfort_idx]}")

    return comfort_belief


def run_agent(
        model_agent,
        temperature_observed,
        qs_prior_input=None,
        rng_key=None,
):
    # =========================================================
    # Run Active Inference agent
    # =========================================================

    gamma = 1

    # ---------------------------------------------------------
    # Create agent
    # ---------------------------------------------------------
    agent = Agent(
        **model_agent,
        gamma=gamma,
        policy_len=1,
    )

    temperature_idx = temperatures.index(temperature_observed)

    # Each observation must have shape:
    # (batch_size, time_dim)
    #
    # agent.batch_size defaults to 1.
    temperature_observation = jnp.full(
        (agent.batch_size, 1),
        temperature_idx,
    )

    # Multi-modality observation list
    observations = [
        temperature_observation
    ]

    # =========================================================
    # Prepare prior for hidden-state inference
    # =========================================================

    if qs_prior_input is None:
        # -----------------------------------------------------
        # First interaction step:
        # use agent.D as the initial prior
        # -----------------------------------------------------
        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 1),
            agent.D,
        )
    else:
        # -----------------------------------------------------
        # Later interaction steps:
        #
        # qs_prior_input is the predicted prior:
        #
        # q^-(s_t) = B[a_{t-1}] @ q(s_{t-1})
        #
        # Input shape:
        # (num_states,)
        #
        # Required shape:
        # (batch_size, time_dim, num_states)
        # -----------------------------------------------------
        qs_init = [
            jnp.expand_dims(
                jnp.expand_dims(
                    qs_prior_input,
                    axis=0,
                ),
                axis=1,
            )
        ]

    # =========================================================
    # Infer hidden states
    # =========================================================

    qs = agent.infer_states(
        observations,
        qs_init,
    )

    print("\n===== INITIAL OBSERVATION =====")
    print(
        f"Observed temperature: "
        f"{temperature_observed}"
    )

    print("\n===== DEBUG SHAPE =====")
    print(
        "qs[0].shape:",
        qs[0].shape,
    )

    # ---------------------------------------------------------
    # qs[0] currently has shape:
    #
    # (batch_size, 1, 1, num_states)
    #
    # Example:
    # (1, 1, 1, 5)
    #
    # Convert it to the full posterior belief vector:
    #
    # (num_states,)
    #
    # IMPORTANT:
    # comfort_belief remains a SOFT posterior distribution.
    # We do not convert it to one-hot.
    # ---------------------------------------------------------
    comfort_belief = jnp.squeeze(
        qs[0],
        axis=(0, 1, 2),
    )

    print(
        "comfort_belief.shape:",
        comfort_belief.shape,
    )

    print(
        "\n===== POSTERIOR BELIEF OVER COMFORT ====="
    )

    for i, state in enumerate(comforts):
        print(
            f"{state}: "
            f"{float(comfort_belief[i]):.4f}"
        )

    # ---------------------------------------------------------
    # MAP estimate of the current hidden comfort state.
    #
    # This is only the most probable state under the posterior:
    #
    # s_MAP = argmax_s q(s)
    #
    # IMPORTANT:
    # This is NOT the true hidden state and does NOT replace
    # the full soft posterior distribution.
    # ---------------------------------------------------------
    current_comfort_idx = int(
        jnp.argmax(comfort_belief)
    )

    current_comfort_map = comforts[
        current_comfort_idx
    ]

    print(
        f"\nMost likely comfort state "
        f"(MAP estimate): "
        f"{current_comfort_map}"
    )

    # =========================================================
    # Prepare posterior belief for policy inference
    # =========================================================
    #
    # agent.infer_policies expects:
    #
    # (batch_size, time_dim, num_states)
    #
    # Current qs[0] shape:
    #
    # (1, 1, 1, num_states)
    #
    # Remove only the extra singleton axis at axis=2:
    #
    # (1, 1, num_states)
    # =========================================================

    qs_for_policy = [
        jnp.squeeze(
            q,
            axis=2,
        )
        for q in qs
    ]

    print(
        "\n===== DEBUG POLICY INPUT SHAPE ====="
    )

    print(
        "qs_for_policy[0].shape:",
        qs_for_policy[0].shape,
    )

    # =========================================================
    # Infer policies
    # =========================================================

    q_pi, G = agent.infer_policies(
        qs_for_policy
    )

    print("\n===== POLICY INFERENCE =====")
    print(
        "q_pi:",
        q_pi,
    )
    print(
        "q_pi.shape:",
        q_pi.shape,
    )
    print(
        "G:",
        G,
    )

    # =========================================================
    # Direct stochastic sampling from q_pi
    # =========================================================
    #
    # IMPORTANT:
    #
    # We do NOT:
    # - use top-k
    # - apply an additional sampling temperature
    # - modify q_pi before action selection
    #
    # Therefore:
    #
    # P(action = a) = q_pi[a]
    #
    # =========================================================

    # ---------------------------------------------------------
    # Initialize RNG only if no RNG key was supplied.
    #
    # For formal experiments, it is better to initialize
    # rng_key once outside run_agent(), for example:
    #
    # rng_key = jax.random.PRNGKey(RANDOM_SEED)
    #
    # and then pass the returned rng_key into the next step.
    # ---------------------------------------------------------
    if rng_key is None:
        rng_key = jax.random.PRNGKey(25)

    # q_pi shape:
    #
    # (batch_size, num_policies)
    #
    # policy_len = 1, therefore each policy corresponds
    # directly to one action in this experiment.
    probs = jnp.asarray(
        q_pi[0],
        dtype=jnp.float32,
    )

    # Numerical safety
    probs = jnp.clip(
        probs,
        min=0.0,
    )

    probs_sum = jnp.sum(probs)

    probs = jnp.where(
        probs_sum > 0.0,
        probs / probs_sum,
        jnp.ones_like(probs) / probs.shape[0],
    )

    # ---------------------------------------------------------
    # Split the RNG key exactly once for this action sample.
    #
    # rng_key:
    # carried forward to the next interaction step
    #
    # action_sample_key:
    # used only for the current random draw
    # ---------------------------------------------------------
    rng_key, action_sample_key = jax.random.split(
        rng_key
    )

    chosen_action_idx = int(
        jax.random.choice(
            action_sample_key,
            probs.shape[0],
            p=probs,
        )
    )

    chosen_action = agent_actions[
        chosen_action_idx
    ]

    print(
        "\n===== STOCHASTIC ACTION SELECTED "
        "DIRECTLY FROM q_pi ====="
    )

    print(
        "Sampling probabilities:",
        probs,
    )

    print(
        f"Action chosen: "
        f"{chosen_action}"
    )

    print(
        "\n===== ACTION PROBABILITIES ====="
    )

    for i, action in enumerate(agent_actions):
        print(
            f"{i}: {action:6s} "
            f"| q_pi={float(q_pi[0][i]):.4f} "
            f"| sample_p={float(probs[i]):.4f} "
            f"| G={float(G[0][i]):.4f}"
        )

    # =========================================================
    # Execute action a_t in the environment
    # =========================================================
    #
    # The environment transition is driven by:
    #
    # temperature_observed + chosen_action
    #
    # The MAP comfort state is diagnostic only.
    # It is NOT treated as the true hidden state.
    # =========================================================

    print("\n===== EXECUTE ACTION =====")

    print(
        f"Current inferred comfort "
        f"(MAP estimate only): "
        f"{current_comfort_map}"
    )

    print(
        f"Executed action a_t: "
        f"{chosen_action}"
    )

    (
        label_next_temperature,
        next_temperatures_index,
    ) = environment_step(
        action_input=chosen_action,
        current_temperatures=temperature_observed,
    )

    print("\n===== ENVIRONMENT RESULT =====")

    print(
        f"New temperature observation: "
        f"{label_next_temperature}"
    )

    # =========================================================
    # Predict next prior q^-(s_{t+1})
    # using B and selected action
    # =========================================================
    #
    # q^-(s_{t+1})
    #     =
    # B[a_t] @ q(s_t)
    #
    # IMPORTANT:
    # We use the FULL SOFT posterior comfort_belief here,
    # not its argmax/MAP state.
    # =========================================================

    qs_prior_next = predict_next_state_belief(
        model=model_agent,
        qs_current=comfort_belief,
        action_input=chosen_action,
        comforts_input=comforts,
    )

    print(
        "\n===== PREDICTED PRIOR AFTER ACTION ====="
    )

    for i, state in enumerate(comforts):
        print(
            f"Prior q(s_t+1={state}) "
            f"before new observation: "
            f"{float(qs_prior_next[i]):.4f}"
        )

    # =========================================================
    # Do not infer the next observation here.
    #
    # qs_prior_next is the predicted prior for the next
    # timestep:
    #
    # q^-(s_t+1)
    #     =
    # B[a_t] @ q(s_t)
    #
    # It will be passed into the next run_agent() call.
    #
    # At that point, the new observation is incorporated
    # exactly once:
    #
    # q(s_t+1)
    #     ∝
    # P(o_t+1 | s_t+1)
    # *
    # q^-(s_t+1)
    #
    # =========================================================

    return {
        "current_temperature": temperature_observed,

        # Full soft posterior:
        "current_belief": comfort_belief,

        # MAP estimate for diagnostics only:
        "current_comfort_map_idx": current_comfort_idx,
        "current_comfort_map": current_comfort_map,

        # Environment output:
        "next_temperature": label_next_temperature,
        "next_temperature_idx": next_temperatures_index,

        # Predicted prior for next inference step:
        "predicted_prior_next": qs_prior_next,

        # Selected action:
        "chosen_action_idx": chosen_action_idx,
        "chosen_action": chosen_action,

        # Policy inference:
        "q_pi": q_pi,
        "G": G,

        # Actual sampling distribution:
        "sampling_probs": probs,

        # RNG state to carry to next step:
        "rng_key": rng_key,
    }

# Baseline
