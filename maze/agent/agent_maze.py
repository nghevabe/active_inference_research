from pymdp.agent import Agent
from pymdp.distribution import compile_model
from jax import numpy as jnp
import jax.tree_util as jtu
import jax

from maze.env.element_maze import agent_actions, position_observes, base_actions, update_model_description, position_states
from maze.env.environtment_maze import build_matrix_a, build_matrix_b, build_matrix_c, environment_maze_step


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


def get_B_action_matrix(model, action_input, position_input):
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
            model.B["position_state"][to_state, from_state, action_input]
            for from_state in position_input
        ]
        for to_state in position_input
    ])


def predict_next_state_belief(
    model,
    qs_current,
    action_input,
    position_input,
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
        position_input=position_input,
    )

    qs_next = B_a @ qs_current

    # Safety normalization
    qs_next = qs_next / jnp.sum(qs_next)

    return qs_next


def update_matrix_agent():
    model = compile_model(update_model_description())
    build_matrix_a(model, position_observes, position_states)
    build_matrix_b(model, position_states, base_actions)
    build_matrix_c(model, position_observes, "P10")
    return model


def run_agent(
        model_agent,
        position_observed,
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

    position_idx = position_observes.index(position_observed)

    # Each observation must have shape:
    # (batch_size, time_dim)
    #
    # agent.batch_size defaults to 1.
    position_observation = jnp.full(
        (agent.batch_size, 1),
        position_idx,
    )

    # Multi-modality observation list
    observations = [
        position_observation
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
        f"Observed position: "
        f"{position_observed}"
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
    position_belief = jnp.squeeze(
        qs[0],
        axis=(0, 1, 2),
    )

    print(
        "position_belief.shape:",
        position_belief.shape,
    )

    print(
        "\n===== POSTERIOR BELIEF OVER position ====="
    )

    lst_obj_belief = []
    for i, state in enumerate(position_states):
        str_object_belief = f"{state}: "+f"{float(position_belief[i]):.4f}"
        lst_obj_belief.append(str_object_belief)
        print(
            f"{state}: "
            f"{float(position_belief[i]):.4f}"
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
    current_position_idx = int(
        jnp.argmax(position_belief)
    )

    current_position_map = position_states[
        current_position_idx
    ]

    print(
        f"\nMost likely position state "
        f"(MAP estimate): "
        f"{current_position_map}"
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

    print("\n===== EXECUTE ACTION =====")

    print(
        f"Current inferred comfort "
        f"(MAP estimate only): "
        f"{current_position_map}"
    )

    print(
        f"Executed action a_t: "
        f"{chosen_action}"
    )

    (
        label_next_position,
        next_position_index,
    ) = environment_maze_step(
        action_input=chosen_action,
        current_position=position_observed,
    )

    print("\n===== ENVIRONMENT RESULT =====")

    print(
        f"New position observation: "
        f"{label_next_position}"
    )

    qs_prior_next = predict_next_state_belief(
        model=model_agent,
        qs_current=position_belief,
        action_input=chosen_action,
        position_input=position_states,
    )

    print(
        "\n===== PREDICTED PRIOR AFTER ACTION ====="
    )

    for i, state in enumerate(position_states):
        print(
            f"Prior q(s_t+1={state}) "
            f"before new observation: "
            f"{float(qs_prior_next[i]):.4f}"
        )

    return {
        "current_position": position_observed,

        # Full soft posterior:
        "current_belief": position_belief,

        # MAP estimate for diagnostics only:
        "current_position_map_idx": current_position_idx,
        "current_position_map": current_position_map,

        # Environment output:
        "next_position": label_next_position,
        "next_position_idx": next_position_index,

        # Predicted prior for next inference step:
        "predicted_prior_next": qs_prior_next,

        # Selected action:
        "chosen_action_idx": chosen_action_idx,
        "chosen_action": chosen_action,
        "lst_obj_belief": lst_obj_belief,

        # Policy inference:
        "q_pi": q_pi,
        "G": G,

        # Actual sampling distribution:
        "sampling_probs": probs,

        # RNG state to carry to next step:
        "rng_key": rng_key,
    }

