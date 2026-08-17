import copy

import jax
import jax.tree_util as jtu
from jax import numpy as jnp
from pymdp.agent import Agent

from env.agent import extend_action_space
from env.elements import comforts, agent_actions
from env.environtment import (
    temperatures,
    environment_step,
    predict_next_state_belief,
)
from utils.util import extract_distribution_array_and_attr


# =========================================================
# Configuration
# =========================================================

# Existing known actions have fixed / ground-truth-like B.
# Only newly introduced actions are learned.
LEARNABLE_ACTIONS = {
    "AIT",
    "ADT",
    "XDT",
}

# Dirichlet hyperparameters.
DIRICHLET_PRIOR_STRENGTH = 16.0
DIRICHLET_LEARNING_RATE = 1.0
DIRICHLET_EPSILON = 1e-6

# Active Inference configuration.
POLICY_LENGTH = 1
GAMMA = 1.0

# Experiment configuration.
INITIAL_TEMPERATURE = "T0"
NUM_STEPS = 30

# IMPORTANT:
# Use deterministic seeds for reproducible experiments.
#
# When comparing:
#   Average-B
#   Dirichlet
#   ATES
#
# use the SAME seed for each corresponding run.
RANDOM_SEED = 50


# =========================================================
# Direct stochastic action sampling from q_pi
# =========================================================

def sample_action_from_q_pi(
    q_pi,
    rng_key,
):
    """
    Sample directly from the policy posterior.

    No:
        - top-k filtering
        - additional temperature scaling
        - greedy transformation

    Therefore:

        a_t ~ Q(pi)

    For policy_len = 1, each policy corresponds to one action,
    so q_pi[0] can directly be treated as the action
    probability distribution.
    """

    probs = jnp.asarray(
        q_pi[0],
        dtype=jnp.float32,
    )

    # Numerical safety.
    probs = jnp.clip(
        probs,
        a_min=0.0,
    )

    prob_sum = jnp.sum(probs)

    # Defensive fallback.
    # Normally q_pi already sums to 1.
    probs = jnp.where(
        prob_sum > 0.0,
        probs / prob_sum,
        jnp.ones_like(probs) / probs.shape[0],
    )

    # JAX PRNG keys are immutable.
    # Always split before using one for stochastic sampling.
    rng_key, action_key = jax.random.split(
        rng_key
    )

    chosen_action_idx = int(
        jax.random.choice(
            action_key,
            probs.shape[0],
            p=probs,
        )
    )

    chosen_action = agent_actions[
        chosen_action_idx
    ]

    return (
        chosen_action_idx,
        chosen_action,
        rng_key,
        probs,
    )


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
    Run one interaction step using Standard Dirichlet
    B-learning.

    Interaction architecture:

        predicted prior q^-(s_t)
                +
          observation o_t
                |
                v
         posterior q(s_t)
                |
                v
       learn previous transition

       q(s_{t-1}),
       a_{t-1},
       q(s_t)

                |
                v
          infer policies
                |
                v
         sample action a_t
                |
                v
          environment step
                |
                v
       observation o_{t+1}
                |
                v
     predict prior q^-(s_{t+1})
                |
                v
              return

    Important:

    1. o_t is inferred exactly once.

    2. q(s_t) is the current posterior.

    3. q(s_t) is NOT blindly reused as the next prior.

    4. The next prior is:

           q^-(s_{t+1})
               =
           B[a_t] @ q(s_t)

    5. Standard Dirichlet B-learning is applied only to
       newly introduced actions:

           AIT
           ADT
           XDT

    6. Known actions keep their original B slices fixed.

    7. Action selection is sampled DIRECTLY from q_pi.
       No additional top-k or temperature transformation.
    """

    if rng_key is None:
        rng_key = jax.random.PRNGKey(
            RANDOM_SEED
        )

    # =====================================================
    # 1. Create inference agent using CURRENT B
    # =====================================================

    inference_agent = Agent(
        **model_agent,
        gamma=GAMMA,
        policy_len=POLICY_LENGTH,
    )

    # =====================================================
    # 2. Build current observation
    # =====================================================

    temperature_idx = temperatures.index(
        temperature_observed
    )

    temperature_observation = jnp.full(
        (
            inference_agent.batch_size,
            1,
        ),
        temperature_idx,
    )

    observations = [
        temperature_observation,
    ]

    # =====================================================
    # 3. Prepare prior q^-(s_t)
    # =====================================================

    if qs_prior_input is None:
        # First interaction step:
        #
        # use model D as initial hidden-state prior.

        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(
                x,
                axis=1,
            ),
            inference_agent.D,
        )

    else:
        # Later interaction steps:
        #
        # qs_prior_input:
        #
        #     q^-(s_t)
        #
        # Expected input:
        #
        #     (num_states,)
        #
        # Agent expects:
        #
        #     (
        #       batch_size,
        #       time_dim,
        #       num_states
        #     )

        qs_init = [
            jnp.expand_dims(
                jnp.expand_dims(
                    qs_prior_input,
                    axis=0,
                ),
                axis=1,
            )
        ]

    # =====================================================
    # 4. Infer CURRENT hidden state exactly once
    # =====================================================

    qs = inference_agent.infer_states(
        observations,
        qs_init,
    )

    # Current shape:
    #
    #     qs[0]:
    #       (
    #         batch_size,
    #         time_dim,
    #         extra_dim,
    #         num_states
    #       )
    #
    # In this environment:
    #
    #     (1, 1, 1, 5)

    comfort_belief = jnp.squeeze(
        qs[0],
        axis=(0, 1, 2),
    )

    print(
        "\n===== CURRENT OBSERVATION ====="
    )

    print(
        f"Observed temperature: "
        f"{temperature_observed}"
    )

    print(
        "\n===== POSTERIOR BELIEF "
        "OVER COMFORT ====="
    )

    for i, state in enumerate(
        comforts
    ):
        print(
            f"{state}: "
            f"{float(comfort_belief[i]):.4f}"
        )

    current_comfort_idx = int(
        jnp.argmax(
            comfort_belief
        )
    )

    current_infer_state = comforts[
        current_comfort_idx
    ]

    print(
        "\nMost likely comfort state:",
        current_infer_state,
    )

    # =====================================================
    # 5. Learn PREVIOUS transition
    # =====================================================
    #
    # At interaction t we now have:
    #
    #     previous_belief
    #         =
    #     q(s_{t-1})
    #
    #     previous_action
    #         =
    #     a_{t-1}
    #
    #     comfort_belief
    #         =
    #     q(s_t)
    #
    # Thus transition t-1 -> t can now be learned.
    # =====================================================

    updated_model_agent = model_agent
    alpha_B_updated = alpha_B_input

    if (
        previous_belief is not None
        and previous_action is not None
    ):

        print(
            "\n===== STANDARD DIRICHLET "
            "B-LEARNING ====="
        )

        if previous_action in LEARNABLE_ACTIONS:

            print(
                "Learning previous transition:"
            )

            print(
                f"Previous action: "
                f"{previous_action}"
            )

            (
                updated_model_agent,
                alpha_B_updated,
            ) = (
                standard_dirichlet_b_learning_update(
                    model_agent=model_agent,
                    qs_current=previous_belief,
                    qs_next=comfort_belief,
                    action_input=previous_action,
                    alpha_B_input=alpha_B_input,
                    learning_rate=(
                        DIRICHLET_LEARNING_RATE
                    ),
                    prior_strength=(
                        DIRICHLET_PRIOR_STRENGTH
                    ),
                    epsilon=(
                        DIRICHLET_EPSILON
                    ),
                )
            )

            print_updated_b_slice(
                model_agent=updated_model_agent,
                action_input=previous_action,
            )

        else:
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

    # =====================================================
    # 6. Recreate Agent using UPDATED B
    # =====================================================
    #
    # If the previous transition updated B,
    # the current policy inference should immediately
    # use that new B.
    # =====================================================

    policy_agent = Agent(
        **updated_model_agent,
        gamma=GAMMA,
        policy_len=POLICY_LENGTH,
    )

    # =====================================================
    # 7. Prepare CURRENT posterior for policy inference
    # =====================================================

    qs_for_policy = [
        jnp.squeeze(
            q,
            axis=2,
        )
        for q in qs
    ]

    print(
        "\n===== DEBUG POLICY INPUT "
        "SHAPE ====="
    )

    print(
        "qs_for_policy[0].shape:",
        qs_for_policy[0].shape,
    )

    # =====================================================
    # 8. Infer policies
    # =====================================================

    q_pi, G = policy_agent.infer_policies(
        qs_for_policy
    )

    print(
        "\n===== POLICY INFERENCE ====="
    )

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

    # =====================================================
    # 9. Sample action DIRECTLY from q_pi
    # =====================================================

    (
        chosen_action_idx,
        chosen_action,
        rng_key,
        action_probs,
    ) = sample_action_from_q_pi(
        q_pi=q_pi,
        rng_key=rng_key,
    )

    print(
        "\n===== STOCHASTIC ACTION "
        "SELECTED DIRECTLY FROM q_pi ====="
    )

    print(
        "Sampling probabilities:",
        action_probs,
    )

    print(
        f"Action chosen: "
        f"{chosen_action}"
    )

    print(
        "\n===== ACTION PROBABILITIES ====="
    )

    for i, action in enumerate(
        agent_actions
    ):
        print(
            f"{i}: "
            f"{action:6s} "
            f"| q_pi="
            f"{float(q_pi[0][i]):.4f} "
            f"| sample_p="
            f"{float(action_probs[i]):.4f} "
            f"| G="
            f"{float(G[0][i]):.4f}"
        )

    # =====================================================
    # 10. Execute CURRENT action in environment
    # =====================================================

    print(
        "\n===== EXECUTE ACTION ====="
    )

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

    print(
        "\n===== ENVIRONMENT RESULT ====="
    )

    print(
        "New temperature observation:",
        label_next_temperature,
    )

    # =====================================================
    # 11. Predict NEXT PRIOR only
    # =====================================================
    #
    # IMPORTANT:
    #
    # Do NOT infer label_next_temperature here.
    #
    # That observation belongs to the NEXT interaction.
    #
    #
    #     q^-(s_{t+1})
    #
    #         =
    #
    #     B[a_t] @ q(s_t)
    #
    # =====================================================

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

    for i, state in enumerate(
        comforts
    ):
        print(
            f"Prior q^-(s_t+1={state}): "
            f"{float(qs_prior_next[i]):.4f}"
        )

    # =====================================================
    # 12. Return interaction result
    # =====================================================

    return {
        "current_temperature":
            temperature_observed,

        "current_belief":
            comfort_belief,

        "current_inferred_state":
            current_infer_state,

        "next_temperature":
            label_next_temperature,

        "next_temperature_index":
            next_temperature_index,

        "predicted_prior_next":
            qs_prior_next,

        "chosen_action":
            chosen_action,

        "chosen_action_idx":
            chosen_action_idx,

        "q_pi":
            q_pi,

        "action_probs":
            action_probs,

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
    Standard Dirichlet learning for transition model B.

    B indexing convention:

        B[
            to_state,
            from_state,
            action
        ]

    The soft transition evidence is:

        N(to, from | action)

            =

        q_next[to]
            *
        q_current[from]

    Therefore:

        transition_count

            =

        outer(
            q_next,
            q_current,
        )

    Standard Dirichlet learning only ADDS observed
    transition evidence.

    There is no explicit negative update.

    This function updates only actions included in:

        LEARNABLE_ACTIONS
    """

    # =====================================================
    # Only new actions are learnable
    # =====================================================

    if action_input not in LEARNABLE_ACTIONS:
        return (
            model_agent,
            alpha_B_input,
        )

    action_idx = agent_actions.index(
        action_input
    )

    # =====================================================
    # Extract B tensor
    # =====================================================

    B_dist = model_agent["B"][0]

    (
        B_current,
        b_attr,
    ) = extract_distribution_array_and_attr(
        B_dist
    )

    B_current = jnp.asarray(
        B_current,
        dtype=jnp.float32,
    )

    print(
        "\n===== DEBUG B-LEARNING ====="
    )

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

    # =====================================================
    # Initialize Dirichlet concentration tensor
    # =====================================================
    #
    # We use current B as the prior mean:
    #
    #     alpha_0
    #
    #         =
    #
    #     B_initial * prior_strength
    #
    #
    # Dirichlet requires alpha > 0,
    # therefore exact zero entries are replaced with epsilon.
    # =====================================================

    if alpha_B_input is None:

        B_safe = jnp.clip(
            B_current,
            min=epsilon,
        )

        # Normalize over TO state.
        #
        # B:
        #
        #     [to, from, action]
        #
        # For every:
        #
        #     (from, action)
        #
        # sum_to B[to, from, action] = 1

        B_safe = (
            B_safe
            / jnp.sum(
                B_safe,
                axis=0,
                keepdims=True,
            )
        )

        alpha_B = (
            B_safe
            * prior_strength
        )

    else:
        alpha_B = jnp.asarray(
            alpha_B_input,
            dtype=jnp.float32,
        )

        # Defensive numerical safety.
        alpha_B = jnp.maximum(
            alpha_B,
            epsilon,
        )

    # =====================================================
    # Soft transition evidence
    # =====================================================
    #
    # qs_current:
    #
    #     q(s_{t-1})
    #
    # qs_next:
    #
    #     q(s_t)
    #
    #
    # transition_count:
    #
    #     [
    #       to_state,
    #       from_state
    #     ]
    #
    # =====================================================

    qs_current = jnp.asarray(
        qs_current,
        dtype=jnp.float32,
    )

    qs_next = jnp.asarray(
        qs_next,
        dtype=jnp.float32,
    )

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

    # =====================================================
    # Add evidence only to the executed action slice
    # =====================================================

    alpha_B = alpha_B.at[
        :,
        :,
        action_idx,
    ].add(
        learning_rate
        * transition_count
    )

    # =====================================================
    # Convert concentration parameters back to B
    # =====================================================
    #
    # Dirichlet mean:
    #
    #         alpha_i
    #     ----------------
    #       sum_j alpha_j
    #
    #
    # Normalize across TO states:
    #
    #     axis = 0
    #
    # because:
    #
    #     B[to, from, action]
    # =====================================================

    denominator = jnp.sum(
        alpha_B,
        axis=0,
        keepdims=True,
    )

    denominator = jnp.maximum(
        denominator,
        epsilon,
    )

    B_updated = (
        alpha_B
        / denominator
    )

    # =====================================================
    # Put UPDATED B back into model
    # =====================================================

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
# Print updated B slice
# =========================================================

def print_updated_b_slice(
    model_agent,
    action_input,
):
    """
    Print:

        B[
            to_state,
            from_state,
            action_input
        ]

    for debugging Dirichlet learning.
    """

    action_idx = agent_actions.index(
        action_input
    )

    B_dist = model_agent["B"][0]

    (
        B_array,
        _,
    ) = extract_distribution_array_and_attr(
        B_dist
    )

    print(
        f"\n===== UPDATED B SLICE: "
        f"{action_input} ====="
    )

    for (
        from_idx,
        from_state,
    ) in enumerate(
        comforts
    ):

        print(
            f"\nFrom state: "
            f"{from_state}"
        )

        for (
            to_idx,
            to_state,
        ) in enumerate(
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
# Assign updated array into pymdp Distribution
# =========================================================

def assign_distribution_array(
    dist,
    new_array,
    attr_name,
):
    """
    Assign updated array back into the same
    pymdp Distribution object representation.

    A deepcopy is used so the original distribution is not
    directly mutated.
    """

    updated_dist = copy.deepcopy(
        dist
    )

    # First try the attribute discovered by:
    #
    # extract_distribution_array_and_attr()

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

    # Defensive fallbacks for possible Distribution
    # implementations.

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
    #
    # Return raw JAX array if the Agent implementation
    # accepts it.

    return new_array


# =========================================================
# Finalize LAST pending transition
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
    Learn the final pending transition after the interaction
    loop has finished.

    Why this is needed:

    B-learning for action a_t occurs only after observation
    o_{t+1} becomes available.

    Therefore after N environment actions, the final action
    still has one pending transition.

    We infer the final observation once:

        q(s_T)

    and use:

        q(s_{T-1})
        a_{T-1}
        q(s_T)

    to perform the last B update.

    No policy inference or environment action is performed
    here.
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
        gamma=GAMMA,
        policy_len=POLICY_LENGTH,
    )

    temperature_idx = temperatures.index(
        temperature_observed
    )

    temperature_observation = jnp.full(
        (
            agent.batch_size,
            1,
        ),
        temperature_idx,
    )

    observations = [
        temperature_observation,
    ]

    # =====================================================
    # Prepare final prior
    # =====================================================

    if qs_prior_input is None:

        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(
                x,
                axis=1,
            ),
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

    # =====================================================
    # Infer final posterior
    # =====================================================

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

    for i, state in enumerate(
        comforts
    ):
        print(
            f"{state}: "
            f"{float(final_belief[i]):.4f}"
        )

    # =====================================================
    # Known action -> no B learning
    # =====================================================

    if (
        previous_action
        not in LEARNABLE_ACTIONS
    ):

        print(
            "\nFinal transition uses known action "
            f"{previous_action}; "
            "skip B-learning."
        )

        return (
            model_agent,
            alpha_B_input,
            final_belief,
        )

    # =====================================================
    # Learn final transition
    # =====================================================

    print(
        "\n===== FINAL STANDARD "
        "DIRICHLET B-LEARNING ====="
    )

    print(
        "Learning final transition:"
    )

    print(
        f"Previous action: "
        f"{previous_action}"
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
        learning_rate=(
            DIRICHLET_LEARNING_RATE
        ),
        prior_strength=(
            DIRICHLET_PRIOR_STRENGTH
        ),
        epsilon=(
            DIRICHLET_EPSILON
        ),
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

agent_model = extend_action_space(
    "XDT"
)


# =========================================================
# Initial observation
# =========================================================

temperature_observed = (
    INITIAL_TEMPERATURE
)


# =========================================================
# Recurrent variables
# =========================================================

# Predicted prior:
#
#     q^-(s_t)

qs_prior = None


# Dirichlet concentration tensor:
#
#     alpha_B[
#         to,
#         from,
#         action
#     ]

alpha_B = None


# Needed because B-learning is delayed until the NEXT
# posterior becomes available.

previous_belief = None
previous_action = None


# Reproducible random number generator.

rng_key = jax.random.PRNGKey(
    RANDOM_SEED
)


# Save interaction history.

history = []


# =========================================================
# Experiment information
# =========================================================

print(
    "\n========================================"
)

print(
    "STANDARD DIRICHLET B-LEARNING"
)

print(
    "========================================"
)

print(
    f"Initial temperature: "
    f"{INITIAL_TEMPERATURE}"
)

print(
    f"Number of steps: "
    f"{NUM_STEPS}"
)

print(
    f"Random seed: "
    f"{RANDOM_SEED}"
)

print(
    f"Policy length: "
    f"{POLICY_LENGTH}"
)

print(
    f"Gamma: "
    f"{GAMMA}"
)

print(
    f"Dirichlet prior strength: "
    f"{DIRICHLET_PRIOR_STRENGTH}"
)

print(
    f"Dirichlet learning rate: "
    f"{DIRICHLET_LEARNING_RATE}"
)

print(
    "Action selection: "
    "DIRECT SAMPLE FROM q_pi"
)

print(
    "Learnable actions:",
    LEARNABLE_ACTIONS,
)


# =========================================================
# Agent-environment interaction loop
# =========================================================

for t in range(
    NUM_STEPS
):

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
            temperature_observed=(
                temperature_observed
            ),
            qs_prior_input=qs_prior,
            alpha_B_input=alpha_B,
            previous_belief=(
                previous_belief
            ),
            previous_action=(
                previous_action
            ),
            rng_key=rng_key,
        )
    )

    history.append(
        result
    )

    # =====================================================
    # Step summary
    # =====================================================

    print(
        "\n===== STEP SUMMARY ====="
    )

    print(
        "Current observation:",
        result[
            "current_temperature"
        ],
    )

    print(
        "Current belief:",
        result[
            "current_belief"
        ],
    )

    print(
        "Chosen action:",
        result[
            "chosen_action"
        ],
    )

    print(
        "Next observation:",
        result[
            "next_temperature"
        ],
    )

    print(
        "Predicted prior next:",
        result[
            "predicted_prior_next"
        ],
    )

    # =====================================================
    # UPDATED B becomes model for next interaction
    # =====================================================

    agent_model = result[
        "updated_model_agent"
    ]

    # =====================================================
    # Preserve Dirichlet concentration tensor
    # =====================================================

    alpha_B = result[
        "alpha_B"
    ]

    # =====================================================
    # Save CURRENT posterior/action
    #
    # At the NEXT interaction:
    #
    #     current_belief
    #
    # becomes:
    #
    #     q(s_{t-1})
    #
    # and chosen_action becomes:
    #
    #     a_{t-1}
    #
    # =====================================================

    previous_belief = result[
        "current_belief"
    ]

    previous_action = result[
        "chosen_action"
    ]

    # =====================================================
    # Predicted PRIOR becomes next prior
    # =====================================================

    qs_prior = result[
        "predicted_prior_next"
    ]

    # =====================================================
    # Environment observation becomes next observation
    # =====================================================

    temperature_observed = result[
        "next_temperature"
    ]

    # =====================================================
    # Preserve updated JAX PRNG key
    # =====================================================

    rng_key = result[
        "rng_key"
    ]


# =========================================================
# Learn final pending transition
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


# =========================================================
# Final experiment summary
# =========================================================

print(
    "\n========================================"
)

print(
    "STANDARD DIRICHLET "
    "B-LEARNING FINISHED"
)

print(
    "========================================"
)

print(
    "Final environment observation:",
    temperature_observed,
)

print(
    "Total interaction steps:",
    len(history),
)

print(
    "Random seed:",
    RANDOM_SEED,
)

print(
    "\n===== ACTION HISTORY ====="
)

for step_idx, item in enumerate(
    history,
    start=1,
):
    print(
        f"Step {step_idx:02d}: "
        f"{item['current_temperature']} "
        f"--{item['chosen_action']}--> "
        f"{item['next_temperature']}"
    )


# =========================================================
# Target-reaching statistics
# =========================================================

TARGET_TEMPERATURE = "T6"

first_target_step = None

for step_idx, item in enumerate(
    history,
    start=1,
):
    if (
        item["next_temperature"]
        == TARGET_TEMPERATURE
    ):
        first_target_step = step_idx
        break


print(
    "\n===== TARGET SUMMARY ====="
)

print(
    "Target temperature:",
    TARGET_TEMPERATURE,
)

if first_target_step is None:

    print(
        "Target reached: NO"
    )

else:

    print(
        "Target reached: YES"
    )

    print(
        "First target transition step:",
        first_target_step,
    )