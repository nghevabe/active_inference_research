from datetime import datetime
from pathlib import Path

import jax.numpy as jnp

from env.elements import comforts, agent_actions
from utils.util import extract_distribution_array_and_attr


def save_b_matrix_dirichlet(
    model_agent,
    alpha_B,
    file_path="matrix_b_dirichlet.txt",
):
    """
    Save both posterior-mean Matrix B and Dirichlet concentration alpha_B.

    Orientation:
        B[to_state, from_state, action]
        alpha_B[to_state, from_state, action]

    alpha_B is intentionally kept separate from model_agent because the
    baseline learning code carries the Dirichlet concentration tensor as
    recurrent learning state.
    """

    if alpha_B is None:
        raise ValueError(
            "alpha_B is None. Refusing to save because this would lose "
            "the Dirichlet pseudo-count state."
        )

    B_dist = model_agent["B"][0]
    B_array, _ = extract_distribution_array_and_attr(B_dist)

    B_array = jnp.asarray(B_array, dtype=jnp.float32)
    alpha_B = jnp.asarray(alpha_B, dtype=jnp.float32)

    if B_array.shape != alpha_B.shape:
        raise ValueError(
            "B and alpha_B must have the same shape. "
            f"Got B={B_array.shape}, alpha_B={alpha_B.shape}."
        )

    expected_shape = (
        len(comforts),
        len(comforts),
        len(agent_actions),
    )

    if B_array.shape != expected_shape:
        raise ValueError(
            "Unexpected B shape. "
            f"Expected {expected_shape}, got {B_array.shape}."
        )

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:

        f.write("=" * 100 + "\n")
        f.write("DIRICHLET MODEL - MATRIX B + ALPHA\n")
        f.write("=" * 100 + "\n")
        f.write(
            "Saved at: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(
            "Orientation: "
            "B[TO_STATE, FROM_STATE, ACTION]\n"
        )
        f.write(
            "Alpha orientation: "
            "alpha_B[TO_STATE, FROM_STATE, ACTION]\n\n"
        )

        # ========================================================
        # MACHINE-READABLE SECTION
        # ========================================================
        #
        # This section is the canonical source used by the loader.
        # The human-readable tables below are only for inspection.
        #
        f.write("[MACHINE_DATA]\n")

        for action_idx, action in enumerate(agent_actions):
            for from_idx, from_state in enumerate(comforts):
                for to_idx, to_state in enumerate(comforts):
                    b_value = float(
                        B_array[to_idx, from_idx, action_idx]
                    )
                    alpha_value = float(
                        alpha_B[to_idx, from_idx, action_idx]
                    )

                    f.write(
                        "B|"
                        f"{action}|"
                        f"{from_state}|"
                        f"{to_state}|"
                        f"{b_value:.17g}\n"
                    )

                    f.write(
                        "ALPHA|"
                        f"{action}|"
                        f"{from_state}|"
                        f"{to_state}|"
                        f"{alpha_value:.17g}\n"
                    )

        f.write("[/MACHINE_DATA]\n\n")

        # ========================================================
        # STATES
        # ========================================================
        f.write("STATES\n")
        f.write("-" * 100 + "\n")

        for idx, state in enumerate(comforts):
            f.write(f"{idx}: {state}\n")

        f.write("\n")

        # ========================================================
        # ACTIONS
        # ========================================================
        f.write("ACTIONS\n")
        f.write("-" * 100 + "\n")

        for idx, action in enumerate(agent_actions):
            f.write(f"{idx}: {action}\n")

        f.write("\n")

        # ========================================================
        # B SLICES
        # ========================================================
        f.write("=" * 100 + "\n")
        f.write("B SLICES\n")
        f.write("=" * 100 + "\n\n")

        for action_idx, action in enumerate(agent_actions):

            f.write(f"ACTION = {action}\n")
            f.write(
                "rows = TO state | columns = FROM state\n"
            )
            f.write("-" * 100 + "\n")

            header = f"{'TO \\ FROM':<15}"

            for from_state in comforts:
                header += f"{from_state:>17}"

            f.write(header + "\n")

            for to_idx, to_state in enumerate(comforts):

                row = f"{to_state:<15}"

                for from_idx in range(len(comforts)):
                    value = float(
                        B_array[
                            to_idx,
                            from_idx,
                            action_idx,
                        ]
                    )
                    row += f"{value:>17.10f}"

                f.write(row + "\n")

            f.write("\n")

        # ========================================================
        # ALPHA SLICES
        # ========================================================
        f.write("=" * 100 + "\n")
        f.write("DIRICHLET ALPHA SLICES\n")
        f.write("=" * 100 + "\n\n")

        for action_idx, action in enumerate(agent_actions):

            f.write(f"ACTION = {action}\n")
            f.write(
                "rows = TO state | columns = FROM state\n"
            )
            f.write("-" * 100 + "\n")

            header = f"{'TO \\ FROM':<15}"

            for from_state in comforts:
                header += f"{from_state:>17}"

            f.write(header + "\n")

            for to_idx, to_state in enumerate(comforts):

                row = f"{to_state:<15}"

                for from_idx in range(len(comforts)):
                    value = float(
                        alpha_B[
                            to_idx,
                            from_idx,
                            action_idx,
                        ]
                    )
                    row += f"{value:>17.10f}"

                f.write(row + "\n")

            f.write("\n")

        # ========================================================
        # TRANSITION LIST
        # ========================================================
        f.write("=" * 100 + "\n")
        f.write("TRANSITION LIST - B + ALPHA\n")
        f.write("=" * 100 + "\n\n")

        for action_idx, action in enumerate(agent_actions):

            f.write(f"ACTION = {action}\n")
            f.write("-" * 100 + "\n")

            for from_idx, from_state in enumerate(comforts):
                for to_idx, to_state in enumerate(comforts):

                    b_value = float(
                        B_array[
                            to_idx,
                            from_idx,
                            action_idx,
                        ]
                    )

                    alpha_value = float(
                        alpha_B[
                            to_idx,
                            from_idx,
                            action_idx,
                        ]
                    )

                    f.write(
                        f"{from_state:<12}"
                        f" -> "
                        f"{to_state:<12}"
                        f" | "
                        f"{action:<5}"
                        f" | B = {b_value:.10f}"
                        f" | alpha = {alpha_value:.10f}\n"
                    )

            f.write("\n")

    print(
        "Dirichlet Matrix B + alpha_B saved successfully:"
    )
    print(file_path.resolve())
