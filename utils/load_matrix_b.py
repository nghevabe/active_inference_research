from pathlib import Path
import re


def load_b_matrix_from_txt(
    agent_model,
    file_path,
):
    """
    Đọc Matrix B từ file txt được tạo bởi
    save_b_matrix_readable()
    rồi gán ngược vào agent_model.B["comfort"].

    Orientation:
        B[TO_STATE, FROM_STATE, ACTION]
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

    file_path = Path(file_path)

    # ==========================================================
    # CHECK FILE
    # ==========================================================
    if not file_path.exists():
        raise FileNotFoundError(
            f"Matrix B file not found: {file_path.resolve()}"
        )

    # ==========================================================
    # READ FILE
    # ==========================================================
    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:
        content = f.read()

    # Chỉ đọc phần TRANSITION LIST
    marker = "TRANSITION LIST"

    if marker not in content:
        raise ValueError(
            "TRANSITION LIST section not found in file."
        )

    transition_content = content.split(
        marker,
        1,
    )[1]

    # ==========================================================
    # REGEX
    #
    # Example:
    # Warm -> LittleCool | AIT | 0.0314893651
    # ==========================================================
    pattern = re.compile(
        r"^\s*"
        r"(Warm|LittleCool|Cool|LittleCold|Cold)"
        r"\s*->\s*"
        r"(Warm|LittleCool|Cool|LittleCold|Cold)"
        r"\s*\|\s*"
        r"(FIT|FDT|NA|AIT|ADT|XDT)"
        r"\s*\|\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*$",
        re.MULTILINE,
    )

    matches = pattern.findall(
        transition_content
    )

    # ==========================================================
    # VALIDATE NUMBER OF VALUES
    # ==========================================================
    expected_count = (
        len(states)
        * len(states)
        * len(actions)
    )

    if len(matches) != expected_count:
        raise ValueError(
            f"Invalid Matrix B file.\n"
            f"Expected {expected_count} transitions, "
            f"but found {len(matches)}."
        )

    # ==========================================================
    # ASSIGN VALUES
    # ==========================================================
    B = agent_model.B["comfort"]

    loaded_count = 0

    for (
        from_state,
        to_state,
        action,
        value_str,
    ) in matches:

        value = float(value_str)

        # Probability validation
        EPS = 1e-9

        if value < -EPS or value > 1.0 + EPS:
            raise ValueError(
                f"Invalid probability:\n"
                f"{from_state} -> {to_state} "
                f"| {action} | {value}"
            )

        B[
            to_state,
            from_state,
            action,
        ] = value

        loaded_count += 1

    print("=" * 70)
    print("MATRIX B LOADED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"File: {file_path.resolve()}"
    )

    print(
        f"Loaded transitions: {loaded_count}"
    )

    return agent_model