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


# ============================================================
# CONFIGURATION
# ============================================================

# Only newly introduced actions are learnable.
#
# Known actions keep their original / ground-truth-like
# transition model fixed.
LEARNABLE_ACTIONS = (
    "AIT",
    "ADT",
    "XDT",
)


# ------------------------------------------------------------
# Dirichlet B-learning
# ------------------------------------------------------------

DIRICHLET_PRIOR_STRENGTH = 16.0
DIRICHLET_LEARNING_RATE = 10.0
DIRICHLET_EPSILON = 1e-6


# ------------------------------------------------------------
# Active Inference
# ------------------------------------------------------------

POLICY_LENGTH = 1
GAMMA = 1.0


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

INITIAL_TEMPERATURE = "T0"
NUM_STEPS = 30

# IMPORTANT:
#
# Use exactly the same seed when comparing:
#
#   Average-B
#   Dirichlet
#   ATES
#
RANDOM_SEED = 50


# ============================================================
# LOGGING UTILITIES
# ============================================================

def print_section(title):
    print(
        "\n"
        "============================================================"
    )
    print(title)
    print(
        "============================================================"
    )


def print_subsection(title):
    print(
        "\n"
        "------------------------------------------------------------"
    )
    print(title)
    print(
        "------------------------------------------------------------"
    )


def print_vector(
    title,
    vector,
    labels,
    precision=6,
):
    """
    Print a probability / belief vector.
    """

    print(f"\n{title}")

    for idx, label in enumerate(labels):
        print(
            f"  {label:12s}: "
            f"{float(vector[idx]):.{precision}f}"
        )


def print_transition_evidence(
    transition_count,
):
    """
    Print soft transition evidence:

        q_next[to]
        *
        q_current[from]

    Matrix orientation:

        rows    = TO state
        columns = FROM state
    """

    print("\nSOFT TRANSITION EVIDENCE")
    print(
        "rows = TO state | columns = FROM state"
    )

    header = (
        f"{'TO \\ FROM':12s}"
        + "".join(
            f"{state:>13s}"
            for state in comforts
        )
    )

    print(header)

    for to_idx, to_state in enumerate(comforts):

        row = f"{to_state:12s}"

        for from_idx in range(len(comforts)):

            value = float(
                transition_count[
                    to_idx,
                    from_idx,
                ]
            )

            row += f"{value:13.6f}"

        print(row)


def extract_b_array(
    model_agent,
):
    """
    Extract B tensor from pymdp distribution.

    Returns
    -------
    B_array:
        B[
            to_state,
            from_state,
            action
        ]

    B_dist:
        Original pymdp Distribution object.

    b_attr:
        Internal array attribute used by Distribution.
    """

    B_dist = model_agent["B"][0]

    (
        B_array,
        b_attr,
    ) = extract_distribution_array_and_attr(
        B_dist
    )

    B_array = jnp.asarray(
        B_array,
        dtype=jnp.float32,
    )

    return (
        B_array,
        B_dist,
        b_attr,
    )


def print_b_slice(
    B_array,
    action_input,
    title=None,
    precision=6,
):
    """
    Print one action slice:

        B[
            to_state,
            from_state,
            action
        ]

    Matrix orientation:

        rows    = TO state
        columns = FROM state
    """

    action_idx = agent_actions.index(
        action_input
    )

    if title is None:
        title = f"B SLICE | ACTION = {action_input}"

    print(f"\n{title}")

    print(
        "rows = TO state | columns = FROM state"
    )

    header = (
        f"{'TO \\ FROM':12s}"
        + "".join(
            f"{state:>13s}"
            for state in comforts
        )
    )

    print(header)

    for to_idx, to_state in enumerate(comforts):

        row = f"{to_state:12s}"

        for from_idx in range(len(comforts)):

            value = float(
                B_array[
                    to_idx,
                    from_idx,
                    action_idx,
                ]
            )

            row += (
                f"{value:13.{precision}f}"
            )

        print(row)


def print_b_delta_slice(
    B_before,
    B_after,
    action_input,
    precision=6,
):
    """
    Print:

        Delta B = B_after - B_before

    for a single action.
    """

    action_idx = agent_actions.index(
        action_input
    )

    print(
        f"\nDELTA B | ACTION = {action_input}"
    )

    print(
        "Delta B = B_after - B_before"
    )

    print(
        "rows = TO state | columns = FROM state"
    )

    header = (
        f"{'TO \\ FROM':12s}"
        + "".join(
            f"{state:>13s}"
            for state in comforts
        )
    )

    print(header)

    for to_idx, to_state in enumerate(comforts):

        row = f"{to_state:12s}"

        for from_idx in range(len(comforts)):

            delta = float(
                B_after[
                    to_idx,
                    from_idx,
                    action_idx,
                ]
                -
                B_before[
                    to_idx,
                    from_idx,
                    action_idx,
                ]
            )

            row += (
                f"{delta:+13.{precision}f}"
            )

        print(row)


def print_b_update_comparison(
    B_before,
    B_after,
    action_input,
):
    """
    Standard learning log:

        B BEFORE
        B AFTER
        DELTA B

    This same format should later also be used
    for ATES.
    """

    print_subsection(
        f"B UPDATE COMPARISON | {action_input}"
    )

    print_b_slice(
        B_array=B_before,
        action_input=action_input,
        title="B BEFORE LEARNING",
    )

    print_b_slice(
        B_array=B_after,
        action_input=action_input,
        title="B AFTER LEARNING",
    )

    print_b_delta_slice(
        B_before=B_before,
        B_after=B_after,
        action_input=action_input,
    )


def print_all_learnable_b_slices(
    model_agent,
    title,
):
    """
    Print B slices for all learnable actions.
    """

    B_array, _, _ = extract_b_array(
        model_agent
    )

    print_section(title)

    for action in LEARNABLE_ACTIONS:

        print_b_slice(
            B_array=B_array,
            action_input=action,
        )


def print_initial_final_b_comparison(
    initial_model,
    final_model,
):
    """
    Compare B at experiment start and experiment end.

    Useful for Dirichlet-vs-ATES evaluation.
    """

    (
        B_initial,
        _,
        _,
    ) = extract_b_array(
        initial_model
    )

    (
        B_final,
        _,
        _,
    ) = extract_b_array(
        final_model
    )

    print_section(
        "GLOBAL INITIAL vs FINAL B COMPARISON"
    )

    for action in LEARNABLE_ACTIONS:

        print_subsection(
            f"ACTION = {action}"
        )

        print_b_slice(
            B_array=B_initial,
            action_input=action,
            title="INITIAL B",
        )

        print_b_slice(
            B_array=B_final,
            action_input=action,
            title="FINAL B",
        )

        print_b_delta_slice(
            B_before=B_initial,
            B_after=B_final,
            action_input=action,
        )


# ============================================================
# PYMDP DISTRIBUTION ASSIGNMENT
# ============================================================

def assign_distribution_array(
    dist,
    new_array,
    attr_name,
):
    """
    Assign updated array back into pymdp Distribution.

    deepcopy is used to avoid mutating the original
    Distribution object directly.
    """

    updated_dist = copy.deepcopy(
        dist
    )

    # --------------------------------------------------------
    # First try attribute detected by utility function
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Defensive fallbacks
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Last fallback
    # --------------------------------------------------------

    return new_array


# ============================================================
# DIRECT STOCHASTIC ACTION SAMPLING FROM q_pi
# ============================================================

def sample_action_from_q_pi(
    q_pi,
    rng_key,
):
    """
    Sample directly from policy posterior.

    No:
        - greedy argmax
        - top-k
        - extra temperature scaling
        - manual probability transformation

    For policy_len = 1:

        each policy corresponds to one action

    therefore:

        a_t ~ Q(pi)
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

    prob_sum = jnp.sum(
        probs
    )

    probs = jnp.where(
        prob_sum > 0.0,
        probs / prob_sum,
        jnp.ones_like(probs)
        / probs.shape[0],
    )

    # JAX PRNG keys are immutable.
    rng_key, action_key = (
        jax.random.split(
            rng_key
        )
    )

    chosen_action_idx = int(
        jax.random.choice(
            action_key,
            probs.shape[0],
            p=probs,
        )
    )

    chosen_action = (
        agent_actions[
            chosen_action_idx
        ]
    )

    return (
        chosen_action_idx,
        chosen_action,
        rng_key,
        probs,
    )


# ============================================================
# STANDARD DIRICHLET B-LEARNING
# ============================================================

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

    ----------------------------------------------------------
    B indexing
    ----------------------------------------------------------

        B[
            to_state,
            from_state,
            action
        ]

    ----------------------------------------------------------
    Soft transition evidence
    ----------------------------------------------------------

        N(
            s_t = to,
            s_{t-1} = from
            |
            action
        )

        =

        q(s_t = to)
        *
        q(s_{t-1} = from)

    Therefore:

        transition_count

        =

        outer(
            q_next,
            q_current
        )

    ----------------------------------------------------------
    Dirichlet update
    ----------------------------------------------------------

        alpha_new

        =

        alpha_old
        +
        learning_rate
        *
        transition_count

    only for the executed action slice.

    ----------------------------------------------------------
    Posterior mean
    ----------------------------------------------------------

        B[to, from, action]

        =

        alpha[to, from, action]
        /
        sum_to alpha[to, from, action]

    ----------------------------------------------------------
    Important
    ----------------------------------------------------------

    Standard Dirichlet learning adds positive evidence.

    It does NOT explicitly apply a negative transition
    correction.

    Some B probabilities may nevertheless decrease after
    normalization because another TO-state received more
    concentration mass.
    """

    # ========================================================
    # Only new actions are learnable
    # ========================================================

    if action_input not in LEARNABLE_ACTIONS:

        return (
            model_agent,
            alpha_B_input,
        )

    action_idx = agent_actions.index(
        action_input
    )

    # ========================================================
    # Extract CURRENT B
    # ========================================================

    (
        B_current,
        B_dist,
        b_attr,
    ) = extract_b_array(
        model_agent
    )

    # Keep exact snapshot before learning.
    B_before_update = jnp.array(
        B_current,
        copy=True,
    )

    # ========================================================
    # Initialize / restore Dirichlet concentration
    # ========================================================

    if alpha_B_input is None:

        # ----------------------------------------------------
        # Current B becomes prior mean
        #
        # alpha_0 = B_initial * prior_strength
        # ----------------------------------------------------

        B_safe = jnp.clip(
            B_current,
            min=epsilon,
        )

        # Normalize across TO states:
        #
        # sum_to B[to, from, action] = 1
        B_safe = (
            B_safe
            /
            jnp.sum(
                B_safe,
                axis=0,
                keepdims=True,
            )
        )

        alpha_B = (
            B_safe
            *
            prior_strength
        )

    else:

        alpha_B = jnp.asarray(
            alpha_B_input,
            dtype=jnp.float32,
        )

        alpha_B = jnp.maximum(
            alpha_B,
            epsilon,
        )

    # ========================================================
    # Prepare beliefs
    # ========================================================

    qs_current = jnp.asarray(
        qs_current,
        dtype=jnp.float32,
    )

    qs_next = jnp.asarray(
        qs_next,
        dtype=jnp.float32,
    )

    # ========================================================
    # Soft transition evidence
    # ========================================================

    transition_count = jnp.outer(
        qs_next,
        qs_current,
    )

    # ========================================================
    # Learning-event log
    # ========================================================

    print_section(
        "STANDARD DIRICHLET B-LEARNING EVENT"
    )

    print(
        f"Executed previous action : "
        f"{action_input}"
    )

    print(
        f"Action index             : "
        f"{action_idx}"
    )

    print(
        f"Prior strength           : "
        f"{prior_strength}"
    )

    print(
        f"Learning rate            : "
        f"{learning_rate}"
    )

    print_vector(
        title="q(s_t-1) | PREVIOUS BELIEF",
        vector=qs_current,
        labels=comforts,
    )

    print_vector(
        title="q(s_t) | CURRENT BELIEF",
        vector=qs_next,
        labels=comforts,
    )

    print_transition_evidence(
        transition_count
    )

    # ========================================================
    # Dirichlet concentration update
    # ========================================================

    alpha_B = alpha_B.at[
        :,
        :,
        action_idx,
    ].add(
        learning_rate
        *
        transition_count
    )

    # ========================================================
    # Convert alpha -> posterior mean B
    # ========================================================

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
        /
        denominator
    )

    # ========================================================
    # Standardized B log
    # ========================================================

    print_b_update_comparison(
        B_before=B_before_update,
        B_after=B_updated,
        action_input=action_input,
    )

    # ========================================================
    # Put UPDATED B back into model
    # ========================================================

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

    updated_model_agent["B"] = (
        B_list
    )

    return (
        updated_model_agent,
        alpha_B,
    )


# ============================================================
# ONE AGENT-ENVIRONMENT INTERACTION STEP
# ============================================================

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
    One Active Inference interaction step with
    Standard Dirichlet B-learning.

    Sequence:

        predicted prior q^-(s_t)
                +
           observation o_t
                |
                v
          posterior q(s_t)
                |
                v
        learn transition
        from previous step
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
        observation o_(t+1)
                |
                v
        predict q^-(s_(t+1))

    Learning is delayed by one interaction because transition:

        s_(t-1)
           --a_(t-1)-->
        s_t

    can only be learned after observation o_t allows inference
    of q(s_t).
    """

    if rng_key is None:

        rng_key = (
            jax.random.PRNGKey(
                RANDOM_SEED
            )
        )

    # ========================================================
    # 1. Agent using CURRENT B
    # ========================================================

    inference_agent = Agent(
        **model_agent,
        gamma=GAMMA,
        policy_len=POLICY_LENGTH,
    )

    # ========================================================
    # 2. Observation
    # ========================================================

    temperature_idx = (
        temperatures.index(
            temperature_observed
        )
    )

    temperature_observation = (
        jnp.full(
            (
                inference_agent.batch_size,
                1,
            ),
            temperature_idx,
        )
    )

    observations = [
        temperature_observation,
    ]

    # ========================================================
    # 3. Prior q^-(s_t)
    # ========================================================

    if qs_prior_input is None:

        # First interaction uses D.
        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(
                x,
                axis=1,
            ),
            inference_agent.D,
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

    # ========================================================
    # 4. Infer CURRENT state
    # ========================================================

    qs = inference_agent.infer_states(
        observations,
        qs_init,
    )

    comfort_belief = jnp.squeeze(
        qs[0],
        axis=(0, 1, 2),
    )

    current_comfort_idx = int(
        jnp.argmax(
            comfort_belief
        )
    )

    current_infer_state = (
        comforts[
            current_comfort_idx
        ]
    )

    print_subsection(
        "CURRENT STATE INFERENCE"
    )

    print(
        "Observed temperature:",
        temperature_observed,
    )

    print_vector(
        title="Posterior q(s_t)",
        vector=comfort_belief,
        labels=comforts,
    )

    print(
        "\nMost likely state:",
        current_infer_state,
    )

    # ========================================================
    # 5. Learn PREVIOUS transition
    # ========================================================

    updated_model_agent = (
        model_agent
    )

    alpha_B_updated = (
        alpha_B_input
    )

    if (
        previous_belief is not None
        and previous_action is not None
    ):

        if (
            previous_action
            in LEARNABLE_ACTIONS
        ):

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

        else:

            print_subsection(
                "B-LEARNING SKIPPED"
            )

            print(
                f"Previous action = "
                f"{previous_action}"
            )

            print(
                "Reason: known action; "
                "B slice is fixed."
            )

    else:

        print_subsection(
            "B-LEARNING SKIPPED"
        )

        print(
            "Reason: first interaction; "
            "no previous transition exists."
        )

    # ========================================================
    # 6. Recreate agent using UPDATED B
    # ========================================================

    policy_agent = Agent(
        **updated_model_agent,
        gamma=GAMMA,
        policy_len=POLICY_LENGTH,
    )

    # ========================================================
    # 7. Current posterior for policy inference
    # ========================================================

    qs_for_policy = [
        jnp.squeeze(
            q,
            axis=2,
        )
        for q in qs
    ]

    # ========================================================
    # 8. Infer policies
    # ========================================================

    q_pi, G = (
        policy_agent.infer_policies(
            qs_for_policy
        )
    )

    print_subsection(
        "POLICY INFERENCE"
    )

    print(
        "q_pi:",
        q_pi,
    )

    print(
        "G:",
        G,
    )

    # ========================================================
    # 9. Sample directly from q_pi
    # ========================================================

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
        "\nACTION PROBABILITIES"
    )

    for i, action in enumerate(
        agent_actions
    ):

        print(
            f"  {i:2d} | "
            f"{action:6s} | "
            f"q_pi="
            f"{float(q_pi[0][i]):.6f} | "
            f"sample_p="
            f"{float(action_probs[i]):.6f} | "
            f"G="
            f"{float(G[0][i]):.6f}"
        )

    print(
        "\nChosen action:",
        chosen_action,
    )

    # ========================================================
    # 10. Execute CURRENT action
    # ========================================================

    (
        label_next_temperature,
        next_temperature_index,
    ) = environment_step(
        action_input=chosen_action,
        current_temperatures=(
            temperature_observed
        ),
    )

    print_subsection(
        "ENVIRONMENT TRANSITION"
    )

    print(
        "Current observation :",
        temperature_observed,
    )

    print(
        "Current state       :",
        current_infer_state,
    )

    print(
        "Executed action     :",
        chosen_action,
    )

    print(
        "Next observation    :",
        label_next_temperature,
    )

    # ========================================================
    # 11. Predict NEXT prior
    # ========================================================
    #
    # Do NOT infer next observation here.
    #
    # q^-(s_(t+1))
    #
    # =
    #
    # B[a_t] @ q(s_t)
    # ========================================================

    qs_prior_next = (
        predict_next_state_belief(
            model=updated_model_agent,
            qs_current=comfort_belief,
            action_input=chosen_action,
            comforts_input=comforts,
        )
    )

    print_vector(
        title="Predicted prior q^-(s_t+1)",
        vector=qs_prior_next,
        labels=comforts,
    )

    # ========================================================
    # 12. Return
    # ========================================================

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


# ============================================================
# FINALIZE LAST PENDING TRANSITION
# ============================================================

def finalize_standard_dirichlet_b_learning(
    model_agent,
    temperature_observed,
    qs_prior_input,
    previous_belief,
    previous_action,
    alpha_B_input,
):
    """
    Learn final pending transition.

    With delayed transition learning:

        action a_t

    can only be learned after:

        observation o_(t+1)

    is available.

    Therefore, after N environment actions, one transition
    remains pending.
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

    temperature_idx = (
        temperatures.index(
            temperature_observed
        )
    )

    temperature_observation = (
        jnp.full(
            (
                agent.batch_size,
                1,
            ),
            temperature_idx,
        )
    )

    observations = [
        temperature_observation,
    ]

    # ========================================================
    # Prepare final prior
    # ========================================================

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

    # ========================================================
    # Final state inference
    # ========================================================

    qs = agent.infer_states(
        observations,
        qs_init,
    )

    final_belief = jnp.squeeze(
        qs[0],
        axis=(0, 1, 2),
    )

    print_section(
        "FINAL PENDING TRANSITION"
    )

    print(
        "Final observation:",
        temperature_observed,
    )

    print_vector(
        title="Final posterior q(s_T)",
        vector=final_belief,
        labels=comforts,
    )

    # ========================================================
    # Known action: no learning
    # ========================================================

    if (
        previous_action
        not in LEARNABLE_ACTIONS
    ):

        print(
            "\nFinal action:",
            previous_action,
        )

        print(
            "Known action -> skip B-learning."
        )

        return (
            model_agent,
            alpha_B_input,
            final_belief,
        )

    # ========================================================
    # Learn final transition
    # ========================================================

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

    return (
        updated_model_agent,
        alpha_B_updated,
        final_belief,
    )


# ============================================================
# BUILD MODEL WITH NEW ACTIONS
# ============================================================

agent_model = extend_action_space(
    "AIT"
)

agent_model = extend_action_space(
    "ADT"
)

agent_model = extend_action_space(
    "XDT"
)


# ============================================================
# SAVE INITIAL MODEL
# ============================================================
#
# Important for experiment-level:
#
#     Initial B
#         vs
#     Final B
#
# comparison.
# ============================================================

initial_agent_model = copy.deepcopy(
    agent_model
)


# ============================================================
# INITIAL OBSERVATION
# ============================================================

temperature_observed = (
    INITIAL_TEMPERATURE
)


# ============================================================
# RECURRENT VARIABLES
# ============================================================

# Predicted hidden-state prior:
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


# Previous posterior/action needed for delayed learning.
previous_belief = None
previous_action = None


# Reproducible stochastic action sampling.
rng_key = jax.random.PRNGKey(
    RANDOM_SEED
)


# Interaction history.
history = []


# ============================================================
# EXPERIMENT HEADER
# ============================================================

print_section(
    "STANDARD DIRICHLET B-LEARNING EXPERIMENT"
)

print(
    f"Initial temperature        : "
    f"{INITIAL_TEMPERATURE}"
)

print(
    f"Number of steps            : "
    f"{NUM_STEPS}"
)

print(
    f"Random seed                : "
    f"{RANDOM_SEED}"
)

print(
    f"Policy length              : "
    f"{POLICY_LENGTH}"
)

print(
    f"Gamma                      : "
    f"{GAMMA}"
)

print(
    f"Dirichlet prior strength   : "
    f"{DIRICHLET_PRIOR_STRENGTH}"
)

print(
    f"Dirichlet learning rate    : "
    f"{DIRICHLET_LEARNING_RATE}"
)

print(
    "Action selection           : "
    "DIRECT SAMPLE FROM q_pi"
)

print(
    "Learnable actions          : "
    f"{LEARNABLE_ACTIONS}"
)


# ============================================================
# INITIAL B
# ============================================================

print_all_learnable_b_slices(
    model_agent=agent_model,
    title="INITIAL B SLICES",
)


# ============================================================
# AGENT-ENVIRONMENT LOOP
# ============================================================

for t in range(
    NUM_STEPS
):

    print_section(
        f"AGENT LOOP STEP {t + 1:02d}"
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

    # ========================================================
    # STEP SUMMARY
    # ========================================================

    print_subsection(
        "STEP SUMMARY"
    )

    print(
        f"Step                : "
        f"{t + 1}"
    )

    print(
        "Current observation :",
        result[
            "current_temperature"
        ],
    )

    print(
        "Inferred state      :",
        result[
            "current_inferred_state"
        ],
    )

    print(
        "Chosen action       :",
        result[
            "chosen_action"
        ],
    )

    print(
        "Next observation    :",
        result[
            "next_temperature"
        ],
    )

    # ========================================================
    # Updated B becomes next model
    # ========================================================

    agent_model = result[
        "updated_model_agent"
    ]

    # ========================================================
    # Preserve alpha
    # ========================================================

    alpha_B = result[
        "alpha_B"
    ]

    # ========================================================
    # Current posterior/action become previous values
    # at next interaction
    # ========================================================

    previous_belief = result[
        "current_belief"
    ]

    previous_action = result[
        "chosen_action"
    ]

    # ========================================================
    # Predicted next prior
    # ========================================================

    qs_prior = result[
        "predicted_prior_next"
    ]

    # ========================================================
    # Environment observation
    # ========================================================

    temperature_observed = result[
        "next_temperature"
    ]

    # ========================================================
    # Updated PRNG key
    # ========================================================

    rng_key = result[
        "rng_key"
    ]


# ============================================================
# FINAL PENDING TRANSITION
# ============================================================

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


# ============================================================
# FINAL EXPERIMENT SUMMARY
# ============================================================

print_section(
    "STANDARD DIRICHLET B-LEARNING FINISHED"
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


# ============================================================
# ACTION HISTORY
# ============================================================

print_subsection(
    "ACTION HISTORY"
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


# ============================================================
# TARGET-REACHING STATISTICS
# ============================================================

TARGET_TEMPERATURE = "T6"

first_target_step = None


for step_idx, item in enumerate(
    history,
    start=1,
):

    if (
        item["next_temperature"]
        ==
        TARGET_TEMPERATURE
    ):

        first_target_step = (
            step_idx
        )

        break


print_subsection(
    "TARGET SUMMARY"
)

print(
    "Target temperature:",
    TARGET_TEMPERATURE,
)

if first_target_step is None:

    print(
        "Target reached:",
        "NO",
    )

else:

    print(
        "Target reached:",
        "YES",
    )

    print(
        "First target transition step:",
        first_target_step,
    )


# ============================================================
# FINAL B SLICES
# ============================================================

print_all_learnable_b_slices(
    model_agent=agent_model,
    title="FINAL B SLICES",
)


# ============================================================
# INITIAL vs FINAL B
# ============================================================

print_initial_final_b_comparison(
    initial_model=initial_agent_model,
    final_model=agent_model,
)