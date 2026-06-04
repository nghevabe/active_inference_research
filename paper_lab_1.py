import jax
from pymdp.agent import Agent
from jax import numpy as jnp
import jax.tree_util as jtu
from env.agent import extend_action_space
from env.elements import comforts, agent_actions
from env.environtment import \
    temperatures, lights, humidity, environment_step, \
    predict_next_state_belief


def run_agent(model_agent):
    # =========================================================
    # Run Active Inference agent
    # =========================================================

    gamma = 1  # deterministic behavior; smaller gamma -> more stochastic behavior

    # Create agent
    agent = Agent(**model_agent, gamma=gamma, policy_len=1)

    temperature_observed = "T2"
    light_observed = "L1"
    humidity_observed = "H2"

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

    qs_init = jtu.tree_map(
        lambda x: jnp.expand_dims(x, 1),
        agent.D
    )

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
    rng_key = jax.random.PRNGKey(10)

    probs = q_pi[0]  # shape: (num_actions,)

    # Split the key before sampling so each random draw uses a fresh subkey.
    rng_key, action_sample_key = jax.random.split(rng_key)

    chosen_action_idx = int(
        jax.random.choice(
            action_sample_key,
            a=jnp.arange(len(agent_actions)),
            p=probs
        )
    )

    chosen_action = agent_actions[chosen_action_idx]

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

    rng_key = jax.random.PRNGKey(100)

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

    print("\n===== B transition from Uncomfortable =====")
    for action in agent_actions:
        print(f"\nAction: {action}")
        for next_state in comforts:
            value = model_agent.B["comfort"][next_state, current_infer_state, action]
            print(f"  P({next_state} | {current_infer_state}, {action}) = {float(value):.4f}")

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

    print("\n===== UPDATED POSTERIOR q(s_t+1) VIA VFE =====")
    for i, state in enumerate(comforts):
        print(f"{state}: {float(next_comfort_belief[i]):.4f}")

    next_comfort_idx = int(jnp.argmax(next_comfort_belief))
    print(f"\nMost likely next comfort state: {comforts[next_comfort_idx]}")



agent_model = extend_action_space("ACN1")
agent_model = extend_action_space("ACN2")

run_agent(agent_model)