from env.agent import run_agent, extend_action_space, get_ates_positive_point, ates_positive_update, \
    ates_update_matrix_positive, update_matrix_agent
import jax
import time

from env.elements import base_actions

# =========================================================
# Initialize agent model
# =========================================================

agent_model = extend_action_space("AIT")
agent_model = extend_action_space("ADT")


# =========================================================
# Initial observation
# =========================================================

# temperature_observed = "T4"

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
    print(f"\n================ AGENT LOOP STEP {t + 1} ================")

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

    if result["chosen_action"] not in base_actions:
        print("XXX_New_Action")

    print(
        "Next observation:",
        result["next_temperature"]
    )

    print("Current belief:", result["current_belief"])
    print("Predicted prior next:", result["predicted_prior_next"])

    # Prior predicted by B becomes prior for the next observation.
    qs_prior = result["predicted_prior_next"]

    temperature_observed = result["next_temperature"]

    rng_key = result["rng_key"]

    # each loop ask for input ates_action_id_input
    # ates_point = get_ates_positive_point(ates_action_id_input, "Neutral", "Cool", agent_model, 70, 30)
    # ates_dif = ates_positive_update(ates_action_id_input, "Neutral", "Cool", agent_model, 70, 30)
    # new_model = ates_update_matrix_positive(ates_action_id_input, "Neutral", "Cool", agent_model, ates_dif, ates_point)