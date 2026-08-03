from env.agent import run_agent, extend_action_space
import jax
import time

from utils.util import build_noisy_agent_b_from_env

# =========================================================
# Initialize agent model
# =========================================================

agent_model = extend_action_space("ACN1")
agent_model = extend_action_space("ACN2")


# =========================================================
# Initial observation
# =========================================================

temperature_observed = "T4"
light_observed = "L3"

# =========================================================
# Initial recurrent variables
# =========================================================

qs_prior = None
rng_key = jax.random.PRNGKey(int(time.time()))

history = []

# =========================================================
# Agent-environment interaction loop
# =========================================================

for t in range(10):
    print(f"\n================ AGENT LOOP STEP {t + 1} ================")

    result = run_agent(
        model_agent=agent_model,
        temperature_observed=temperature_observed,
        light_observed=light_observed,
        qs_prior_input=qs_prior,
        rng_key=rng_key,
    )

    history.append(result)

    print("\n===== STEP SUMMARY =====")
    print(
        "Current observation:",
        result["current_temperature"],
        result["current_light"],
    )

    print("Chosen action:", result["chosen_action"])

    print(
        "Next observation:",
        result["next_temperature"],
        result["next_light"],
    )

    print("Current belief:", result["current_belief"])
    print("Predicted prior next:", result["predicted_prior_next"])
    print("Next belief:", result["next_belief"])

    # =====================================================
    # Important:
    # No Dirichlet B-learning here.
    # The agent model remains fixed across steps.
    # =====================================================

    # agent_model is NOT updated.

    # =====================================================
    # Important:
    # Next posterior becomes prior for next step
    # =====================================================

    qs_prior = result["next_belief"]

    # =====================================================
    # Important:
    # Next observation becomes current observation
    # for next step
    # =====================================================

    temperature_observed = result["next_temperature"]
    light_observed = result["next_light"]

    # =====================================================
    # Important:
    # Keep random key evolving
    # =====================================================

    rng_key = result["rng_key"]