import numpy as np
from pymdp.distribution import compile_model
from env.elements import update_model_description, lst_action_extended, lst_slice_b, \
    lst_pairing_state, append_action

import copy
import jax
from pymdp.agent import Agent
from jax import numpy as jnp
import jax.tree_util as jtu
from env.elements import comforts, agent_actions
from env.environtment import \
    temperatures, lights, environment_step, \
    predict_next_state_belief


def extract_distribution_array_and_attr(dist):
    """
    Extract raw probability array from Distribution object
    and remember which attribute was used.
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
                arr = jnp.asarray(value)
                return arr, attr
            except Exception:
                pass

    try:
        arr = jnp.asarray(dist)
        return arr, None
    except Exception as e:
        raise TypeError(
            "Cannot extract raw array from Distribution object. "
            "Please print type(dist) and dir(dist)."
        ) from e


def build_noisy_agent_b_from_env(
        env_model,
        rho=0.10,
        seed=42,
):
    """
    Create a slightly noisy B matrix for the agent from the environment B matrix.

    B_agent = (1 - rho) * B_env + rho * B_noise

    where B_noise is a random valid probability distribution over next states.

    rho = 0.05 -> very small mismatch
    rho = 0.10 -> mild mismatch
    rho = 0.20 -> medium mismatch
    """

    rng = np.random.default_rng(seed)

    agent_model = copy.deepcopy(env_model)

    state_names = [
        "Uncomfortable",
        "Neutral",
        "Comfortable",
    ]

    action_names = [
        "IL",
        "DL",
        "IT",
        "DT",
        "NA",
    ]

    for action in action_names:
        for from_state in state_names:

            # -------------------------------------------------
            # Read original environment transition distribution
            # P(s_next | s_current, action)
            # -------------------------------------------------

            env_probs = np.array([
                env_model.B["comfort"][to_state, from_state, action]
                for to_state in state_names
            ], dtype=float)

            # Safety normalization
            env_probs = env_probs / env_probs.sum()

            # -------------------------------------------------
            # Generate small random transition noise
            # -------------------------------------------------

            noise_probs = rng.dirichlet(
                alpha=np.ones(len(state_names))
            )

            # -------------------------------------------------
            # Mix environment dynamics with noise
            # -------------------------------------------------

            noisy_probs = (1.0 - rho) * env_probs + rho * noise_probs

            # Safety normalization
            noisy_probs = noisy_probs / noisy_probs.sum()

            # -------------------------------------------------
            # Write noisy distribution into agent model
            # -------------------------------------------------

            for to_state, prob in zip(state_names, noisy_probs):
                agent_model.B["comfort"][to_state, from_state, action] = float(prob)

    return agent_model


def sample_top_k_with_temperature(
        q_pi,
        rng_key,
        agent_actions,
        k=3,
        temperature=0.1,
):
    """
    Top-k sampling with temperature.

    Lower temperature -> more deterministic.
    Higher temperature -> more exploratory.
    """

    probs = q_pi[0]

    # Get top-k action indices
    top_indices = jnp.argsort(probs)[-k:]

    # Extract top-k probabilities
    top_probs = probs[top_indices]

    # Apply temperature
    logits = jnp.log(top_probs + 1e-16)
    scaled_logits = logits / temperature

    top_probs_temp = jax.nn.softmax(scaled_logits)

    rng_key, action_sample_key = jax.random.split(rng_key)

    sampled_pos = int(
        jax.random.choice(
            action_sample_key,
            a=jnp.arange(k),
            p=top_probs_temp,
        )
    )

    chosen_action_idx = int(top_indices[sampled_pos])
    chosen_action = agent_actions[chosen_action_idx]

    return chosen_action_idx, chosen_action, rng_key, top_indices, top_probs_temp
