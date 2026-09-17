from pathlib import Path
from datetime import datetime


def save_b_matrix_readable(
    agent_model,
    file_path="b_matrix.txt",
):
    """
    Lưu Matrix B của agent_model ra file txt.

    Orientation:
        B[TO_STATE, FROM_STATE, ACTION]

    Không phụ thuộc vào np.asarray(Distribution).
    """

    states = [
        "Warm",
        "LittleCool",
        "Cool",
        "LittleCold",
        "Cold",
    ]

    actions = [
        "FIT",
        "FDT",
        "NA",
        "AIT",
        "ADT",
        "XDT",
    ]

    B = agent_model.B["comfort"]

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("=" * 100 + "\n")
        f.write("AGENT MODEL - MATRIX B\n")
        f.write("=" * 100 + "\n")

        f.write(
            f"Saved at: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        f.write(
            "Orientation: "
            "B[TO_STATE, FROM_STATE, ACTION]\n\n"
        )

        # ======================================================
        # STATE / ACTION INFO
        # ======================================================

        f.write("STATES\n")
        f.write("-" * 100 + "\n")

        for index, state in enumerate(states):
            f.write(
                f"{index}: {state}\n"
            )

        f.write("\n")

        f.write("ACTIONS\n")
        f.write("-" * 100 + "\n")

        for index, action in enumerate(actions):
            f.write(
                f"{index}: {action}\n"
            )

        f.write("\n")

        # ======================================================
        # B SLICES
        # ======================================================

        f.write("=" * 100 + "\n")
        f.write("B SLICES\n")
        f.write("=" * 100 + "\n\n")

        for action in actions:

            f.write(
                f"ACTION = {action}\n"
            )

            f.write(
                "rows = TO state | "
                "columns = FROM state\n"
            )

            f.write("-" * 100 + "\n")

            header = f"{'TO \\ FROM':<15}"

            for from_state in states:
                header += f"{from_state:>17}"

            f.write(
                header + "\n"
            )

            for to_state in states:

                row = f"{to_state:<15}"

                for from_state in states:

                    value = float(
                        B[
                            to_state,
                            from_state,
                            action,
                        ]
                    )

                    row += (
                        f"{value:>17.10f}"
                    )

                f.write(
                    row + "\n"
                )

            f.write("\n")

        # ======================================================
        # TRANSITION LIST
        # ======================================================

        f.write("=" * 100 + "\n")
        f.write("TRANSITION LIST\n")
        f.write("=" * 100 + "\n\n")

        for action in actions:

            f.write(
                f"ACTION = {action}\n"
            )

            f.write("-" * 100 + "\n")

            for from_state in states:

                for to_state in states:

                    value = float(
                        B[
                            to_state,
                            from_state,
                            action,
                        ]
                    )

                    f.write(
                        f"{from_state:<12}"
                        f" -> "
                        f"{to_state:<12}"
                        f" | "
                        f"{action:<5}"
                        f" | "
                        f"{value:.10f}\n"
                    )

            f.write("\n")

    print(
        f"Matrix B saved successfully: {file_path}"
    )