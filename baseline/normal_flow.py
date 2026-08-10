from env.agent import run_agent, extend_action_space
import jax
import time

from utils.util import build_noisy_agent_b_from_env

# =========================================================
# Initialize agent model
# =========================================================

agent_model = extend_action_space("AIT")
agent_model = extend_action_space("ADT")


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

for t in range(5):
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

    # =====================================================
    # No Dirichlet B-learning in this loop.
    #
    # The transition model remains fixed across timesteps.
    # agent_model is not updated here.
    # =====================================================

    # =====================================================
    # The posterior at the current timestep has already
    # been propagated through B using the selected action:
    #
    # q^-(s_t+1) = B[a_t] @ q(s_t)
    #
    # Therefore predicted_prior_next becomes the prior
    # used with the next observation at timestep t + 1.
    # =====================================================

    qs_prior = result["predicted_prior_next"]

    # =====================================================
    # The environment output becomes the current
    # observation for the next timestep.
    # =====================================================

    temperature_observed = result["next_temperature"]

    # =====================================================
    # Preserve the updated random key so the next action
    # sampling uses a new random state.
    # =====================================================

    rng_key = result["rng_key"]