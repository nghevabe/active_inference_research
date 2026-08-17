from env.agent import run_agent, extend_action_space
import jax
import time
from env.elements import base_actions, agent_actions, comforts
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

temperature_observed = "T0"

# =========================================================
# Initial recurrent variables
# =========================================================

qs_prior = None
RANDOM_SEED = 40
rng_key = jax.random.PRNGKey(RANDOM_SEED)

history = []

# =========================================================
# Agent-environment interaction loop
# =========================================================

for t in range(30):
    print(
        f"\n================ AGENT LOOP STEP {t + 1} ================"
    )

    # =========================================================
    # Run one Active Inference interaction step
    # =========================================================
    result = run_agent(
        model_agent=agent_model,
        temperature_observed=temperature_observed,
        qs_prior_input=qs_prior,
        rng_key=rng_key,
    )

    history.append(result)

    # =========================================================
    # Step summary
    # =========================================================
    print("\n===== STEP SUMMARY =====")

    print(
        "Current observation:",
        result["current_temperature"],
    )

    print(
        "Chosen action:",
        result["chosen_action"],
    )

    if result["chosen_action"] not in base_actions:
        print("XXX_New_Action")

    print(
        "Next observation:",
        result["next_temperature"],
    )

    print(
        "Current belief:",
        result["current_belief"],
    )

    print(
        "Current comfort MAP estimate:",
        result["current_comfort_map"],
    )

    print(
        "Predicted prior next:",
        result["predicted_prior_next"],
    )

    # =========================================================
    # Keep the FULL SOFT posterior
    # =========================================================
    #
    # IMPORTANT:
    #
    # Do not convert:
    #
    # q(s_t)
    #
    # into:
    #
    # argmax(q(s_t))
    #
    # here.
    #
    # The complete posterior distribution is preserved for
    # later ATES processing.
    # =========================================================

    current_belief = result["current_belief"]

    # Optional diagnostic only:
    current_comfort_map = result["current_comfort_map"]

    # =========================================================
    # Predicted prior for next timestep
    # =========================================================
    #
    # q^-(s_t+1)
    #     =
    # B[a_t] @ q(s_t)
    #
    # This becomes the prior when the next observation arrives.
    # =========================================================
    qs_prior = result["predicted_prior_next"]

    # =========================================================
    # Move environment observation forward
    # =========================================================
    temperature_observed = result["next_temperature"]

    # =========================================================
    # Carry RNG state forward
    # =========================================================
    rng_key = result["rng_key"]