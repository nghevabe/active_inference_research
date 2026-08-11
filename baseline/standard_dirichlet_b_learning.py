import time
import copy
import jax
from pymdp.agent import Agent
from jax import numpy as jnp
import jax.tree_util as jtu

from env.agent import extend_action_space
from env.elements import comforts, agent_actions
from env.environtment import \
    temperatures, environment_step, \
    predict_next_state_belief
from utils.util import extract_distribution_array_and_attr, sample_top_k_with_temperature, build_noisy_agent_b_from_env


# =========================================================
# Configuration
# =========================================================

# Existing known actions have correct / ground-truth-like B slices.
# Only newly introduced actions are learned in this baseline.
LEARNABLE_ACTIONS = {"AIT", "ADT"}

DIRICHLET_PRIOR_STRENGTH = 16.0
DIRICHLET_LEARNING_RATE = 1.0
DIRICHLET_EPSILON = 1e-6


# =========================================================
# Standard Dirichlet B-learning Agent
# =========================================================

def run_agent_standard_dirichlet_b_learning(
    model_agent,
    temperature_observed,
    qs_prior_input=None,
    alpha_B_input=None,
    previous_belief=None,
    previous_action=None,
    rng_key=None,
):
    """
    Run one interaction step using Standard Dirichlet B-learning.

    New interaction architecture:

        predicted prior q^-(s_t)
                    +
            observation o_t
                    ↓
              posterior q(s_t)
                    ↓
        learn previous transition:
        q(s_{t-1}), a_{t-1}, q(s_t)
                    ↓
              infer policy
                    ↓
               choose a_t
                    ↓
             environment step
                    ↓
          next observation o_{t+1}
                    ↓
         predict q^-(s_{t+1})
                    ↓
                  return

    IMPORTANT:
    - o_t is inferred exactly once.
    - q(s_t) is NOT reused as the next prior.
    - q^-(s_{t+1}) = B[a_t] @ q(s_t) is the next prior.
    - Standard B-learning is applied only to AIT / ADT.
    """

    gamma = 1  # policy precision parameter

    # =========================================================
    # 1. Create inference agent using current B
    # =========================================================

    inference_agent = Agent(
        **model_agent,
        gamma=gamma,
        policy_len=1,
    )

    # =========================================================
    # 2. Build current observation
    # =========================================================

    temperature_idx = temperatures.index(temperature_observed)

    temperature_observation = jnp.full(
        (inference_agent.batch_size, 1),
        temperature_idx,
    )

    observations = [
        temperature_observation,
    ]

    # =========================================================
    # 3. Prepare prior q^-(s_t)
    # =========================================================

    if qs_prior_input is None:
        # First interaction:
        # use D as initial prior.
        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 1),
            inference_agent.D,
        )

    else:
        # Later interactions:
        # qs_prior_input is predicted prior q^-(s_t)
        #
        # Input shape:
        #   (num_states,)
        #
        # Required shape:
        #   (batch_size, time_dim, num_states)
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
    # 4. Infer CURRENT hidden state exactly once
    # =========================================================

    qs = inference_agent.infer_states(
        observations,
        qs_init,
    )

    # Expected shape now:
    # (batch_size, time_dim, extra_dim, num_states)
    #
    # Current environment:
    # num_states = 5
    #
    # Example:
    # (1, 1, 1, 5)
    comfort_belief = jnp.squeeze(
        qs[0],
        axis=(0, 1, 2),
    )

    print("\n===== CURRENT OBSERVATION =====")
    print(f"Observed temperature: {temperature_observed}")

    print("\n===== POSTERIOR BELIEF OVER COMFORT =====")

    for i, state in enumerate(comforts):
        print(
            f"{state}: "
            f"{float(comfort_belief[i]):.4f}"
        )

    current_comfort_idx = int(
        jnp.argmax(comfort_belief)
    )

    print(
        "\nMost likely comfort state:",
        comforts[current_comfort_idx],
    )

    # =========================================================
    # 5. Learn PREVIOUS transition
    # =========================================================
    #
    # We can only learn transition t-1 -> t now because
    # current observation o_t has just been received and
    # q(s_t) has just been inferred.
    #
    # previous_belief:
    #   q(s_{t-1})
    #
    # previous_action:
    #   a_{t-1}
    #
    # comfort_belief:
    #   q(s_t)
    # =========================================================

    updated_model_agent = model_agent
    alpha_B_updated = alpha_B_input

    if (
        previous_belief is not None
        and previous_action is not None
    ):
        if previous_action in LEARNABLE_ACTIONS:

            print(
                "\n===== STANDARD DIRICHLET "
                "B-LEARNING ====="
            )

            print(
                "Learning previous transition:"
            )

            print(
                f"Previous action: {previous_action}"
            )

            (
                updated_model_agent,
                alpha_B_updated,
            ) = standard_dirichlet_b_learning_update(
                model_agent=model_agent,
                qs_current=previous_belief,
                qs_next=comfort_belief,
                action_input=previous_action,
                alpha_B_input=alpha_B_input,
                learning_rate=DIRICHLET_LEARNING_RATE,
                prior_strength=DIRICHLET_PRIOR_STRENGTH,
                epsilon=DIRICHLET_EPSILON,
            )

            print_updated_b_slice(
                model_agent=updated_model_agent,
                action_input=previous_action,
            )

        else:
            print(
                "\n===== STANDARD DIRICHLET "
                "B-LEARNING ====="
            )
            print(
                f"Skip action {previous_action}: "
                "known action, B slice is fixed."
            )

    else:
        print(
            "\n===== STANDARD DIRICHLET "
            "B-LEARNING ====="
        )
        print(
            "No previous transition available "
            "(first interaction step)."
        )

    # =========================================================
    # 6. IMPORTANT:
    # Recreate Agent with UPDATED B
    # =========================================================
    #
    # If previous transition updated AIT/ADT,
    # policy inference in this same interaction should use
    # the newly learned B.
    # =========================================================

    policy_agent = Agent(
        **updated_model_agent,
        gamma=gamma,
        policy_len=1,
    )

    # =========================================================
    # 7. Prepare posterior for policy inference
    # =========================================================

    # qs[0] shape:
    #   (1, 1, 1, 5)
    #
    # infer_policies expects:
    #   (1, 1, 5)
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
    # 8. Policy inference
    # =========================================================

    q_pi, G = policy_agent.infer_policies(
        qs_for_policy
    )

    print("\n===== POLICY INFERENCE =====")

    print("q_pi:", q_pi)
    print("q_pi.shape:", q_pi.shape)
    print("G:", G)

    # =========================================================
    # 9. Random action selection
    # =========================================================

    if rng_key is None:
        rng_key = jax.random.PRNGKey(25)

    probs = q_pi[0]

    # Avoid k > number of actions.
    top_k = min(
        4,
        len(agent_actions),
    )

    (
        chosen_action_idx,
        chosen_action,
        rng_key,
        top_indices,
        top_probs_temp,
    ) = sample_top_k_with_temperature(
        q_pi=q_pi,
        rng_key=rng_key,
        agent_actions=agent_actions,
        k=top_k,
        temperature=0.1,
    )

    print(
        "\n===== CUSTOM STOCHASTIC "
        "ACTION SELECTED ====="
    )

    print("probs:", probs)
    print(
        f"Action chosen: {chosen_action}"
    )

    print(
        "\n===== ACTION PROBABILITIES ====="
    )

    for i, action in enumerate(agent_actions):
        print(
            f"{i}: {action:6s} "
            f"| q_pi={float(q_pi[0][i]):.4f} "
            f"| G={float(G[0][i]):.4f}"
        )

    # =========================================================
    # 10. Execute CURRENT action in environment
    # =========================================================

    current_infer_state = comforts[
        current_comfort_idx
    ]

    print("\n===== EXECUTE ACTION =====")

    print(
        "Current inferred state:",
        current_infer_state,
    )

    print(
        f"Executed action a_t: "
        f"{chosen_action}"
    )

    (
        label_next_temperature,
        next_temperature_index,
    ) = environment_step(
        action_input=chosen_action,
        current_temperatures=temperature_observed,
    )

    print("\n===== ENVIRONMENT RESULT =====")

    print(
        "New temperature observation:",
        label_next_temperature,
    )

    # =========================================================
    # 11. Predict next PRIOR only
    # =========================================================
    #
    # Do NOT infer label_next_temperature here.
    #
    # That observation belongs to the next interaction.
    #
    # q^-(s_{t+1}) =
    #       B[a_t] @ q(s_t)
    # =========================================================

    qs_prior_next = predict_next_state_belief(
        model=updated_model_agent,
        qs_current=comfort_belief,
        action_input=chosen_action,
        comforts_input=comforts,
    )

    print(
        "\n===== PREDICTED PRIOR "
        "AFTER ACTION ====="
    )

    for i, state in enumerate(comforts):
        print(
            f"Prior q^-(s_t+1={state}): "
            f"{float(qs_prior_next[i]):.4f}"
        )

    # =========================================================
    # 12. Return
    # =========================================================

    return {
        "current_temperature":
            temperature_observed,

        "current_belief":
            comfort_belief,

        "current_inferred_state":
            current_infer_state,

        "next_temperature":
            label_next_temperature,

        "predicted_prior_next":
            qs_prior_next,

        "chosen_action":
            chosen_action,

        "chosen_action_idx":
            chosen_action_idx,

        "q_pi":
            q_pi,

        "G":
            G,

        "rng_key":
            rng_key,

        "updated_model_agent":
            updated_model_agent,

        "alpha_B":
            alpha_B_updated,
    }


# =========================================================
# Standard Dirichlet B-learning update
# =========================================================

def standard_dirichlet_b_learning_update(
    model_agent,
    qs_current,
    qs_next,
    action_input,
    alpha_B_input=None,
    learning_rate=1.0,
    prior_strength=16.0,
    epsilon=1e-6,
):
    """
    Standard Dirichlet B-learning.

    B indexing:

        B[to_state, from_state, action]

    Soft transition count:

        N(s_t, s_t-1 | a)
            =
        q(s_t) outer q(s_t-1)

    Therefore:

        transition_count[to, from]
            =
        q_next[to] * q_current[from]

    Standard learning adds evidence for the
    transition that is inferred to have occurred.

    It does NOT perform an explicit negative update.
    """

    # =========================================================
    # Only new actions are learnable in this experiment
    # =========================================================

    if action_input not in LEARNABLE_ACTIONS:
        return model_agent, alpha_B_input

    action_idx = agent_actions.index(
        action_input
    )

    # =========================================================
    # Extract B distribution
    # =========================================================

    B_dist = model_agent["B"][0]

    (
        B_current,
        b_attr,
    ) = extract_distribution_array_and_attr(
        B_dist
    )

    print("\n===== DEBUG B-LEARNING =====")

    print(
        "type(model_agent['B']):",
        type(model_agent["B"]),
    )

    print(
        "type(B_dist):",
        type(B_dist),
    )

    print(
        "B attr used:",
        b_attr,
    )

    print(
        "B_current.shape:",
        B_current.shape,
    )

    # =========================================================
    # Initialize Dirichlet concentration
    # =========================================================
    #
    # Avoid exact zero concentration parameters.
    #
    # Dirichlet requires:
    #
    #     alpha > 0
    #
    # B itself can still contain values that are effectively
    # zero after normalization.
    # =========================================================

    if alpha_B_input is None:

        B_safe = jnp.clip(
            B_current,
            epsilon,
            None,
        )

        # Re-normalize each:
        #
        #   [from_state, action]
        #
        # distribution over to_state.
        B_safe = B_safe / jnp.sum(
            B_safe,
            axis=0,
            keepdims=True,
        )

        alpha_B = (
            B_safe
            * prior_strength
        )

    else:
        alpha_B = alpha_B_input

        # Safety check in case an earlier array
        # contains exact zeros.
        alpha_B = jnp.maximum(
            alpha_B,
            epsilon,
        )

    # =========================================================
    # Soft transition evidence
    # =========================================================
    #
    # qs_current:
    #   q(s_{t-1})
    #
    # qs_next:
    #   q(s_t)
    #
    # shape:
    #   (num_states_to, num_states_from)
    # =========================================================

    transition_count = jnp.outer(
        qs_next,
        qs_current,
    )

    print(
        "transition_count.shape:",
        transition_count.shape,
    )

    print(
        "action_input:",
        action_input,
    )

    print(
        "action_idx:",
        action_idx,
    )

    # =========================================================
    # Add evidence to selected action slice
    # =========================================================

    alpha_B = alpha_B.at[
        :,
        :,
        action_idx,
    ].add(
        learning_rate
        * transition_count
    )

    # =========================================================
    # Convert concentration -> B probability
    # =========================================================
    #
    # Normalize over "to_state".
    #
    # B[to, from, action]
    # =========================================================

    denominator = jnp.sum(
        alpha_B,
        axis=0,
        keepdims=True,
    )

    B_updated = (
        alpha_B
        / denominator
    )

    # =========================================================
    # Put updated B back into model
    # =========================================================

    updated_model_agent = dict(
        model_agent
    )

    B_list = list(
        updated_model_agent["B"]
    )

    B_list[0] = assign_distribution_array(
        dist=B_dist,
        new_array=B_updated,
        attr_name=b_attr,
    )

    updated_model_agent["B"] = B_list

    return (
        updated_model_agent,
        alpha_B,
    )


# =========================================================
# Print B slice
# =========================================================

def print_updated_b_slice(
    model_agent,
    action_input,
):
    action_idx = agent_actions.index(
        action_input
    )

    B_dist = model_agent["B"][0]

    B_array, _ = (
        extract_distribution_array_and_attr(
            B_dist
        )
    )

    print(
        f"\n===== UPDATED B SLICE: "
        f"{action_input} ====="
    )

    for from_idx, from_state in enumerate(
        comforts
    ):
        print(
            f"\nFrom state: {from_state}"
        )

        for to_idx, to_state in enumerate(
            comforts
        ):
            prob = B_array[
                to_idx,
                from_idx,
                action_idx,
            ]

            print(
                f"  To {to_state}: "
                f"{float(prob):.4f}"
            )


# =========================================================
# Assign updated array into Distribution object
# =========================================================

def assign_distribution_array(
    dist,
    new_array,
    attr_name,
):
    """
    Assign updated array back to the same
    Distribution object format.
    """

    updated_dist = copy.deepcopy(dist)

    if attr_name is not None:
        try:
            setattr(
                updated_dist,
                attr_name,
                new_array,
            )
            return updated_dist

        except Exception:
            pass

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
        if hasattr(
            updated_dist,
            attr,
        ):
            try:
                setattr(
                    updated_dist,
                    attr,
                    new_array,
                )
                return updated_dist

            except Exception:
                pass

    # Last fallback:
    # return raw JAX array if Agent accepts it.
    return new_array


# =========================================================
# Finalize the LAST pending transition
# =========================================================

def finalize_standard_dirichlet_b_learning(
    model_agent,
    temperature_observed,
    qs_prior_input,
    previous_belief,
    previous_action,
    alpha_B_input,
):
    """
    Learn the final transition after the interaction loop.

    Reason:

    B-learning is delayed until o_{t+1} is observed.

    After the final environment action there is still one
    pending transition:

        q(s_{T-1})
            |
          a_{T-1}
            |
            v
        observation o_T

    We infer o_T once, obtain q(s_T), and then update B.

    No policy/action is selected here.
    """

    if (
        previous_belief is None
        or previous_action is None
    ):
        return (
            model_agent,
            alpha_B_input,
            None,
        )

    agent = Agent(
        **model_agent,
        gamma=1,
        policy_len=1,
    )

    temperature_idx = temperatures.index(
        temperature_observed
    )

    temperature_observation = jnp.full(
        (agent.batch_size, 1),
        temperature_idx,
    )

    observations = [
        temperature_observation,
    ]

    if qs_prior_input is None:
        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 1),
            agent.D,
        )

    else:
        qs_init = [
            jnp.expand_dims(
                jnp.expand_dims(
                    qs_prior_input,
                    axis=0,
                ),
                axis=1,
            )
        ]

    qs = agent.infer_states(
        observations,
        qs_init,
    )

    final_belief = jnp.squeeze(
        qs[0],
        axis=(0, 1, 2),
    )

    print(
        "\n===== FINAL OBSERVATION ====="
    )

    print(
        "Observed temperature:",
        temperature_observed,
    )

    print(
        "\n===== FINAL POSTERIOR ====="
    )

    for i, state in enumerate(comforts):
        print(
            f"{state}: "
            f"{float(final_belief[i]):.4f}"
        )

    # Known actions stay fixed.
    if previous_action not in LEARNABLE_ACTIONS:

        print(
            "\nFinal transition uses known action "
            f"{previous_action}; skip B-learning."
        )

        return (
            model_agent,
            alpha_B_input,
            final_belief,
        )

    (
        updated_model_agent,
        alpha_B_updated,
    ) = standard_dirichlet_b_learning_update(
        model_agent=model_agent,
        qs_current=previous_belief,
        qs_next=final_belief,
        action_input=previous_action,
        alpha_B_input=alpha_B_input,
        learning_rate=DIRICHLET_LEARNING_RATE,
        prior_strength=DIRICHLET_PRIOR_STRENGTH,
        epsilon=DIRICHLET_EPSILON,
    )

    print_updated_b_slice(
        model_agent=updated_model_agent,
        action_input=previous_action,
    )

    return (
        updated_model_agent,
        alpha_B_updated,
        final_belief,
    )


# =========================================================
# Build model with NEW actions
# =========================================================

agent_model = extend_action_space(
    "AIT"
)

agent_model = extend_action_space(
    "ADT"
)


# =========================================================
# Initial observation
# =========================================================

temperature_observed = "T1"


# =========================================================
# Recurrent variables
# =========================================================

# Predicted prior q^-(s_t)
qs_prior = None

# Dirichlet concentration tensor
alpha_B = None

# Needed because B-learning is performed when the
# NEXT posterior becomes available.
previous_belief = None
previous_action = None

rng_key = jax.random.PRNGKey(
    int(time.time())
)

history = []


# =========================================================
# Agent-environment interaction loop
# =========================================================

NUM_STEPS = 20

for t in range(NUM_STEPS):

    print(
        "\n========================================"
    )

    print(
        f"AGENT LOOP STEP {t + 1}"
    )

    print(
        "========================================"
    )

    result = (
        run_agent_standard_dirichlet_b_learning(
            model_agent=agent_model,
            temperature_observed=temperature_observed,
            qs_prior_input=qs_prior,
            alpha_B_input=alpha_B,
            previous_belief=previous_belief,
            previous_action=previous_action,
            rng_key=rng_key,
        )
    )

    history.append(result)

    # =====================================================
    # Summary
    # =====================================================

    print("\n===== STEP SUMMARY =====")

    print(
        "Current observation:",
        result["current_temperature"],
    )

    print(
        "Current belief:",
        result["current_belief"],
    )

    print(
        "Chosen action:",
        result["chosen_action"],
    )

    print(
        "Next observation:",
        result["next_temperature"],
    )

    print(
        "Predicted prior next:",
        result["predicted_prior_next"],
    )

    # =====================================================
    # Updated B becomes model for next step
    # =====================================================

    agent_model = result[
        "updated_model_agent"
    ]

    # =====================================================
    # Keep Dirichlet concentration
    # =====================================================

    alpha_B = result[
        "alpha_B"
    ]

    # =====================================================
    # Save CURRENT posterior/action.
    #
    # They will become the previous transition evidence
    # when o_{t+1} is inferred in the next loop.
    # =====================================================

    previous_belief = result[
        "current_belief"
    ]

    previous_action = result[
        "chosen_action"
    ]

    # =====================================================
    # IMPORTANT:
    #
    # Predicted prior, NOT posterior, becomes next prior.
    # =====================================================

    qs_prior = result[
        "predicted_prior_next"
    ]

    # =====================================================
    # Environment observation for next interaction
    # =====================================================

    temperature_observed = result[
        "next_temperature"
    ]

    # =====================================================
    # Keep RNG state
    # =====================================================

    rng_key = result[
        "rng_key"
    ]


# =========================================================
# Learn the final pending transition
# =========================================================
#
# Without this block:
#
# N environment actions produce only N-1 B updates,
# because the result of the final action is not inferred
# until another interaction starts.
# =========================================================

(
    agent_model,
    alpha_B,
    final_belief,
) = finalize_standard_dirichlet_b_learning(
    model_agent=agent_model,
    temperature_observed=temperature_observed,
    qs_prior_input=qs_prior,
    previous_belief=previous_belief,
    previous_action=previous_action,
    alpha_B_input=alpha_B,
)


print(
    "\n========================================"
)

print(
    "STANDARD DIRICHLET B-LEARNING FINISHED"
)

print(
    "========================================"
)