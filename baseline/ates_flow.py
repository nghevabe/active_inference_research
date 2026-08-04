from env.agent import run_agent, extend_action_space, get_ates_positive_point, ates_positive_update, \
    ates_update_matrix_positive, update_matrix_agent
import jax
import time

# =========================================================
# Initialize agent model
# =========================================================

agent_model = extend_action_space("IL")
agent_model = extend_action_space("DL")


# =========================================================
# Initial observation
# =========================================================

# temperature_observed = "T4"
# light_observed = "L1"

temperature_observed = "T5"
light_observed = "L1"

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

    # Prior predicted by B becomes prior for the next observation.
    qs_prior = result["predicted_prior_next"]

    temperature_observed = result["next_temperature"]
    light_observed = result["next_light"]

    rng_key = result["rng_key"]

    # each loop ask for input ates_action_id_input
    # ates_point = get_ates_positive_point(ates_action_id_input, "Neutral", "Comfortable", agent_model, 70, 30)
    # ates_dif = ates_positive_update(ates_action_id_input, "Neutral", "Comfortable", agent_model, 70, 30)
    # new_model = ates_update_matrix_positive(ates_action_id_input, "Neutral", "Comfortable", agent_model, ates_dif, ates_point)