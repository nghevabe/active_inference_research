import copy
import re
from pathlib import Path

import jax.numpy as jnp

from env.elements import comforts, agent_actions
from utils.util import extract_distribution_array_and_attr


def _assign_distribution_array(
    dist,
    new_array,
    attr_name,
):
    """
    Assign an updated array back into a pymdp Distribution object.
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
        if hasattr(updated_dist, attr):
            try:
                setattr(
                    updated_dist,
                    attr,
                    new_array,
                )
                return updated_dist
            except Exception:
                pass

    return new_array


def _normalize_b_from_alpha(
    alpha_B,
    epsilon,
):
    """
    Posterior mean:

        B[to, from, action]
        =
        alpha[to, from, action]
        /
        sum_to alpha[to, from, action]
    """
    alpha_B = jnp.maximum(
        jnp.asarray(
            alpha_B,
            dtype=jnp.float32,
        ),
        epsilon,
    )

    denominator = jnp.sum(
        alpha_B,
        axis=0,
        keepdims=True,
    )

    denominator = jnp.maximum(
        denominator,
        epsilon,
    )

    return alpha_B / denominator


def _initialize_alpha_from_b(
    B_array,
    prior_strength,
    epsilon,
):
    """
    Backward-compatible fallback for an old save file that contains B
    but does not contain alpha_B.
    """
    B_safe = jnp.clip(
        jnp.asarray(
            B_array,
            dtype=jnp.float32,
        ),
        min=epsilon,
    )

    B_safe = (
        B_safe
        /
        jnp.sum(
            B_safe,
            axis=0,
            keepdims=True,
        )
    )

    return B_safe * prior_strength


def load_b_matrix_dirichlet(
    model_agent,
    file_path="matrix_b_dirichlet.txt",
    prior_strength=16.0,
    epsilon=1e-6,
):
    """
    Load both Matrix B and Dirichlet concentration alpha_B.

    Returns
    -------
    updated_model_agent, alpha_B

    New-format behavior
    -------------------
    If [MACHINE_DATA] contains both B and ALPHA entries, both are loaded.
    alpha_B is treated as the Dirichlet state of record, and B is checked
    against normalize(alpha_B). If they differ materially, B is rebuilt
    from alpha_B to keep the posterior mean mathematically consistent.

    Backward compatibility
    ----------------------
    If an older file contains only B transition-list values, B is loaded
    and alpha_B is initialized as:

        alpha_B = normalize(B) * prior_strength
    """

    file_path = Path(file_path)

    # ============================================================
    # Extract current B structure
    # ============================================================
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

    expected_shape = (
        len(comforts),
        len(comforts),
        len(agent_actions),
    )

    if B_current.shape != expected_shape:
        raise ValueError(
            "Unexpected B shape before loading. "
            f"Expected {expected_shape}, got {B_current.shape}."
        )

    # ============================================================
    # No saved file -> use current B as prior mean
    # ============================================================
    if not file_path.exists():

        print(
            "Dirichlet save file not found. "
            "Using current B and initializing alpha_B from it:"
        )
        print(file_path.resolve())

        alpha_B = _initialize_alpha_from_b(
            B_array=B_current,
            prior_strength=prior_strength,
            epsilon=epsilon,
        )

        return (
            model_agent,
            alpha_B,
        )

    text = file_path.read_text(
        encoding="utf-8"
    )

    B_loaded = jnp.array(
        B_current,
        copy=True,
    )

    alpha_loaded = None

    b_entry_count = 0
    alpha_entry_count = 0

    expected_entries = (
        len(comforts)
        *
        len(comforts)
        *
        len(agent_actions)
    )

    # ============================================================
    # Preferred NEW format: [MACHINE_DATA]
    # ============================================================
    machine_match = re.search(
        r"\[MACHINE_DATA\](.*?)\[/MACHINE_DATA\]",
        text,
        flags=re.DOTALL,
    )

    if machine_match is not None:

        alpha_loaded = jnp.zeros(
            expected_shape,
            dtype=jnp.float32,
        )

        machine_text = machine_match.group(1)

        for raw_line in machine_text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            parts = line.split("|")

            if len(parts) != 5:
                continue

            kind, action, from_state, to_state, value_text = (
                part.strip()
                for part in parts
            )

            if action not in agent_actions:
                continue

            if from_state not in comforts:
                continue

            if to_state not in comforts:
                continue

            try:
                value = float(value_text)
            except ValueError:
                continue

            action_idx = agent_actions.index(action)
            from_idx = comforts.index(from_state)
            to_idx = comforts.index(to_state)

            if kind == "B":

                B_loaded = B_loaded.at[
                    to_idx,
                    from_idx,
                    action_idx,
                ].set(value)

                b_entry_count += 1

            elif kind == "ALPHA":

                alpha_loaded = alpha_loaded.at[
                    to_idx,
                    from_idx,
                    action_idx,
                ].set(value)

                alpha_entry_count += 1

        if b_entry_count != expected_entries:
            raise ValueError(
                "Incomplete B data in [MACHINE_DATA]. "
                f"Expected {expected_entries} entries, "
                f"found {b_entry_count}."
            )

        if alpha_entry_count != expected_entries:
            raise ValueError(
                "Incomplete ALPHA data in [MACHINE_DATA]. "
                f"Expected {expected_entries} entries, "
                f"found {alpha_entry_count}."
            )

    # ============================================================
    # OLD format fallback: TRANSITION LIST with B only
    # ============================================================
    else:

        old_transition_pattern = re.compile(
            r"^\s*"
            r"(?P<from_state>\w+)"
            r"\s*->\s*"
            r"(?P<to_state>\w+)"
            r"\s*\|\s*"
            r"(?P<action>\w+)"
            r"\s*\|\s*"
            r"(?:(?:B\s*=\s*)?)"
            r"(?P<b_value>"
            r"[-+]?"
            r"(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][-+]?\d+)?"
            r")"
            r"(?:"
            r"\s*\|\s*"
            r"alpha\s*=\s*"
            r"(?P<alpha_value>"
            r"[-+]?"
            r"(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][-+]?\d+)?"
            r")"
            r")?"
            r"\s*$",
            flags=re.MULTILINE,
        )

        alpha_temp = jnp.zeros(
            expected_shape,
            dtype=jnp.float32,
        )

        for match in old_transition_pattern.finditer(text):

            from_state = match.group("from_state")
            to_state = match.group("to_state")
            action = match.group("action")

            if from_state not in comforts:
                continue

            if to_state not in comforts:
                continue

            if action not in agent_actions:
                continue

            b_value = float(
                match.group("b_value")
            )

            action_idx = agent_actions.index(action)
            from_idx = comforts.index(from_state)
            to_idx = comforts.index(to_state)

            B_loaded = B_loaded.at[
                to_idx,
                from_idx,
                action_idx,
            ].set(b_value)

            b_entry_count += 1

            alpha_value_text = match.group(
                "alpha_value"
            )

            if alpha_value_text is not None:

                alpha_temp = alpha_temp.at[
                    to_idx,
                    from_idx,
                    action_idx,
                ].set(
                    float(alpha_value_text)
                )

                alpha_entry_count += 1

        if b_entry_count != expected_entries:
            raise ValueError(
                "Could not load a complete B matrix from the saved file. "
                f"Expected {expected_entries} transition entries, "
                f"found {b_entry_count}."
            )

        if alpha_entry_count == expected_entries:
            alpha_loaded = alpha_temp

        elif alpha_entry_count == 0:
            alpha_loaded = None

        else:
            raise ValueError(
                "Saved file contains only a partial alpha_B. "
                f"Expected either 0 or {expected_entries} alpha entries, "
                f"found {alpha_entry_count}."
            )

    # ============================================================
    # If alpha was not saved, reconstruct it from B
    # ============================================================
    if alpha_loaded is None:

        alpha_loaded = _initialize_alpha_from_b(
            B_array=B_loaded,
            prior_strength=prior_strength,
            epsilon=epsilon,
        )

        print(
            "Loaded legacy B-only file. "
            "alpha_B was reconstructed from B using "
            f"prior_strength={prior_strength}."
        )

        B_final = (
            alpha_loaded
            /
            jnp.sum(
                alpha_loaded,
                axis=0,
                keepdims=True,
            )
        )

    else:

        alpha_loaded = jnp.maximum(
            jnp.asarray(
                alpha_loaded,
                dtype=jnp.float32,
            ),
            epsilon,
        )

        B_from_alpha = _normalize_b_from_alpha(
            alpha_B=alpha_loaded,
            epsilon=epsilon,
        )

        max_abs_difference = float(
            jnp.max(
                jnp.abs(
                    B_loaded
                    -
                    B_from_alpha
                )
            )
        )

        if max_abs_difference > 1e-5:

            print(
                "WARNING: saved B and normalize(alpha_B) "
                "are not fully consistent."
            )
            print(
                "Max |B - normalize(alpha_B)| = "
                f"{max_abs_difference:.10f}"
            )
            print(
                "Using normalize(alpha_B) as B so the "
                "Dirichlet posterior remains consistent."
            )

            B_final = B_from_alpha

        else:
            B_final = B_loaded

    # ============================================================
    # Put loaded B back into model
    # ============================================================
    updated_model_agent = dict(
        model_agent
    )

    B_list = list(
        updated_model_agent["B"]
    )

    B_list[0] = _assign_distribution_array(
        dist=B_dist,
        new_array=B_final,
        attr_name=b_attr,
    )

    updated_model_agent["B"] = B_list

    print(
        "Dirichlet Matrix B + alpha_B loaded successfully:"
    )
    print(file_path.resolve())
    print(
        "alpha_B total concentration:",
        float(jnp.sum(alpha_loaded)),
    )

    return (
        updated_model_agent,
        alpha_loaded,
    )
