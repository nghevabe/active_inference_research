from env.agent import run_agent, extend_action_space
import jax
import time

from utils.util import build_noisy_agent_b_from_env

# =========================================================
# Initialize agent model
# =========================================================

agent_model = extend_action_space("AIT")
agent_model = extend_action_space("ADT")
agent_model = extend_action_space("XDT")


# =========================================================
# Initial observation
# =========================================================

temperature_observed = "T1"

# =========================================================
# Initial recurrent variables
# =========================================================

qs_prior = None
rng_key = jax.random.PRNGKey(int(time.time()))

history = []

# =========================================================
# Agent-environment interaction loop
# =========================================================

for t in range(20):
    print(
        f"\n================ AGENT LOOP STEP {t + 1} ================"
    )

    result = run_agent(
        model_agent=agent_model,
        temperature_observed=temperature_observed,
        qs_prior_input=qs_prior,
        rng_key=rng_key,
    )

    history.append(result)

    print("\n===== STEP SUMMARY =====")
    print(
        "Current observation:",
        result["current_temperature"]
    )

    print("Chosen action:", result["chosen_action"])

    print(
        "Next observation:",
        result["next_temperature"]
    )

    print("Current posterior belief:", result["current_belief"])
    print(
        "Predicted prior for next step:",
        result["predicted_prior_next"],
    )

    qs_prior = result["predicted_prior_next"]

    temperature_observed = result["next_temperature"]

    rng_key = result["rng_key"]