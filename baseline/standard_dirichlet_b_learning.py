import time
import copy
import jax
from pymdp.agent import Agent
from jax import numpy as jnp
import jax.tree_util as jtu

from env.agent import extend_action_space
from env.elements import comforts, agent_actions
from env.environtment import \
    temperatures, lights, humidity, environment_step, \
    predict_next_state_belief
from utils.util import extract_distribution_array_and_attr, sample_top_k_with_temperature, build_noisy_agent_b_from_env


def run_agent_standard_dirichlet_b_learning(
        model_agent,
        temperature_observed,
        light_observed,
        humidity_observed,
        qs_prior_input=None,
        alpha_B_input=None,
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
    humidity_idx = humidity.index(humidity_observed)

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

    humidity_observation = jnp.full(
        (agent.batch_size, 1),
        humidity_idx
    )

    # Multi-modality observation list
    observations = [
        temperature_observation,
        light_observation,
        humidity_observation,
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
    print(f"Observed humidity:        {humidity_observed}")

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

    # chosen_action_idx = int(
    #     jax.random.choice(
    #         action_sample_key,
    #         a=jnp.arange(len(agent_actions)),
    #         p=probs
    #     )
    # )
    #
    # chosen_action = agent_actions[chosen_action_idx]

    # ======

    chosen_action_idx, chosen_action, rng_key, top_indices, top_probs_temp = sample_top_k_with_temperature(
        q_pi=q_pi,
        rng_key=rng_key,
        agent_actions=agent_actions,
        k=4,
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

    (label_next_temperature, label_next_light, label_next_humidity, next_temperatures_index, next_lights_index,
     next_humidity_index) = environment_step(
        action_input=chosen_action,
        current_temperatures=temperature_observed,
        current_lights=light_observed,
        current_humidity=humidity_observed,
    )

    print("\n===== ENVIRONMENT RESULT =====")
    # print(f"Next true hidden state s_t+1: {next_true_state}")
    print(f"New temperature observation: {label_next_temperature}")
    print(f"New light observation:       {label_next_light}")
    print(f"New humidity observation:        {label_next_humidity}")

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
        print(f"Prior q(s_t+1={state}) before new observation: {float(qs_prior_next[i]):.4f}")

    # =========================================================
    # Update q(s) via VFE using new observation
    # =========================================================

    next_temperature_observation = jnp.full(
        (agent.batch_size, 1),
        next_temperatures_index
    )

    next_light_observation = jnp.full(
        (agent.batch_size, 1),
        next_lights_index
    )

    next_humidity_observation = jnp.full(
        (agent.batch_size, 1),
        next_humidity_index
    )

    next_observations = [
        next_temperature_observation,
        next_light_observation,
        next_humidity_observation,
    ]

    # infer_states expects qs_init with batch and time dimensions.
    # qs_prior_next has shape: (num_states,)
    # Convert to: (batch_size, time_dim, num_states) = (1, 1, 3)
    qs_init_next = [
        jnp.expand_dims(
            jnp.expand_dims(qs_prior_next, axis=0),
            axis=1
        )
    ]

    qs_next = agent.infer_states(next_observations, qs_init_next)

    next_comfort_belief = jnp.squeeze(qs_next[0], axis=(0, 1, 2))

    # =========================================================
    # Standard Dirichlet B-learning
    # =========================================================
    # We update B using the expected transition:
    # q(s_t) --a_t--> q(s_{t+1})
    #
    # This is a soft-count update:
    # alpha_B[s_next, s_current, action] += q(s_current) * q(s_next)

    updated_model_agent, alpha_B_updated = standard_dirichlet_b_learning_update(
        model_agent=model_agent,
        qs_current=comfort_belief,
        qs_next=next_comfort_belief,
        action_input=chosen_action,
        alpha_B_input=alpha_B_input,
        learning_rate=1.0,
        prior_strength=16.0,
    )

    print("\n===== STANDARD DIRICHLET B-LEARNING =====")
    print(f"Updated B slice for action: {chosen_action}")

    action_idx = agent_actions.index(chosen_action)

    for from_idx, from_state in enumerate(comforts):
        print(f"\nFrom state: {from_state}")
        for to_idx, to_state in enumerate(comforts):
            prob = updated_model_agent["B"][0][to_idx, from_idx, action_idx]
            print(f"  To {to_state}: {float(prob):.4f}")

    return {
        "current_temperature": temperature_observed,
        "current_light": light_observed,
        "current_humidity": humidity_observed,
        "current_belief": comfort_belief,
        "next_temperature": label_next_temperature,
        "next_light": label_next_light,
        "next_humidity": label_next_humidity,
        "predicted_prior_next": qs_prior_next,
        "next_belief": next_comfort_belief,
        "chosen_action": chosen_action,
        "q_pi": q_pi,
        "G": G,
        "rng_key": rng_key,
        # New outputs for B-learning
        "updated_model_agent": updated_model_agent,
        "alpha_B": alpha_B_updated,
    }

    print("\n===== UPDATED POSTERIOR q(s_t+1) VIA VFE =====")
    for i, state in enumerate(comforts):
        print(f"{state}: {float(next_comfort_belief[i]):.4f}")

    next_comfort_idx = int(jnp.argmax(next_comfort_belief))
    print(f"\nMost likely next comfort state: {comforts[next_comfort_idx]}")


def standard_dirichlet_b_learning_update(
        model_agent,
        qs_current,
        qs_next,
        action_input,
        alpha_B_input=None,
        learning_rate=1.0,
        prior_strength=16.0,
):
    """
    Standard Dirichlet B-learning.

    model_agent["B"] is expected to be a list:
        model_agent["B"][0] = B distribution for comfort factor

    B shape:
        (num_states_to, num_states_from, num_actions)
    """

    action_idx = agent_actions.index(action_input)

    # ---------------------------------------------------------
    # Get B for the first hidden-state factor: comfort
    # ---------------------------------------------------------

    B_dist = model_agent["B"][0]

    B_current, b_attr = extract_distribution_array_and_attr(B_dist)

    print("\n===== DEBUG B-LEARNING =====")
    print("type(model_agent['B']):", type(model_agent["B"]))
    print("type(B_dist):", type(B_dist))
    print("B attr used:", b_attr)
    print("B_current.shape:", B_current.shape)

    # ---------------------------------------------------------
    # Initialize or reuse Dirichlet concentration
    # ---------------------------------------------------------

    if alpha_B_input is None:
        alpha_B = B_current * prior_strength
    else:
        alpha_B = alpha_B_input

    # ---------------------------------------------------------
    # Soft transition count:
    # q(s_t=s) * q(s_t+1=s')
    # shape: (num_states_to, num_states_from)
    # ---------------------------------------------------------

    transition_count = jnp.outer(qs_next, qs_current)

    print("transition_count.shape:", transition_count.shape)
    print("action_input:", action_input)
    print("action_idx:", action_idx)

    # ---------------------------------------------------------
    # Update selected action slice
    # ---------------------------------------------------------

    alpha_B = alpha_B.at[:, :, action_idx].add(
        learning_rate * transition_count
    )

    # ---------------------------------------------------------
    # Normalize Dirichlet counts into B probabilities
    # Sum over next state dimension.
    # ---------------------------------------------------------

    B_updated = alpha_B / jnp.sum(
        alpha_B,
        axis=0,
        keepdims=True
    )

    # ---------------------------------------------------------
    # Put updated B back into model_agent["B"][0]
    # ---------------------------------------------------------

    updated_model_agent = dict(model_agent)

    B_list = list(updated_model_agent["B"])

    B_list[0] = assign_distribution_array(
        dist=B_dist,
        new_array=B_updated,
        attr_name=b_attr,
    )

    updated_model_agent["B"] = B_list

    return updated_model_agent, alpha_B


def assign_distribution_array(dist, new_array, attr_name):
    """
    Assign updated array back to the same Distribution object format.
    """

    updated_dist = copy.deepcopy(dist)

    if attr_name is not None:
        try:
            setattr(updated_dist, attr_name, new_array)
            return updated_dist
        except Exception:
            pass

    # Fallback: try common writable attributes
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
        if hasattr(updated_dist, attr):
            try:
                setattr(updated_dist, attr, new_array)
                return updated_dist
            except Exception:
                pass

    # Last fallback:
    # If Agent can accept raw JAX array, use raw array directly.
    return new_array


agent_model = extend_action_space("ACN1")
agent_model = extend_action_space("ACN2")

# =========================================================
# Initial observation
# =========================================================

temperature_observed = "T4"
light_observed = "L3"
humidity_observed = "H4"


# =========================================================
# Initial recurrent variables
# =========================================================

qs_prior = None
alpha_B = None

rng_key = jax.random.PRNGKey(int(time.time()))

history = []

# =========================================================
# Agent-environment interaction loop
# =========================================================

for t in range(10):
    print(f"\n================ AGENT LOOP STEP {t + 1} ================")

    result = run_agent_standard_dirichlet_b_learning(
        model_agent=agent_model,
        temperature_observed=temperature_observed,
        light_observed=light_observed,
        humidity_observed=humidity_observed,
        qs_prior_input=qs_prior,
        alpha_B_input=alpha_B,
        rng_key=rng_key,
    )

    history.append(result)

    print("\n===== STEP SUMMARY =====")
    print(
        "Current observation:",
        result["current_temperature"],
        result["current_light"],
        result["current_humidity"],
    )

    print("Chosen action:", result["chosen_action"])

    print(
        "Next observation:",
        result["next_temperature"],
        result["next_light"],
        result["next_humidity"],
    )

    print("Current belief:", result["current_belief"])
    print("Predicted prior next:", result["predicted_prior_next"])
    print("Next belief:", result["next_belief"])

    # =====================================================
    # Important:
    # Updated B model becomes the model for next step
    # =====================================================

    agent_model = result["updated_model_agent"]

    # =====================================================
    # Important:
    # Updated Dirichlet concentration parameters are reused
    # in the next step
    # =====================================================

    alpha_B = result["alpha_B"]

    # =====================================================
    # Important:
    # Next posterior becomes prior for next step
    # =====================================================

    qs_prior = result["next_belief"]

    # =====================================================
    # Important:
    # Next observation becomes current observation
    # for next step
    # =====================================================

    temperature_observed = result["next_temperature"]
    light_observed = result["next_light"]
    humidity_observed = result["next_humidity"]

    # =====================================================
    # Important:
    # Keep random key evolving
    # =====================================================

    rng_key = result["rng_key"]
