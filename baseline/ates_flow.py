from env.agent import run_agent, extend_action_space, get_ates_positive_point, ates_positive_update, \
    ates_update_matrix_positive, update_matrix_agent
import jax
import time

from env.elements import base_actions, agent_actions, comforts
from utils.util import get_max_index

# =========================================================
# Initialize agent model
# =========================================================

agent_model = extend_action_space("AIT")
agent_model = extend_action_space("ADT")
agent_model = extend_action_space("XDT")


# =========================================================
# Initial observation
# =========================================================

# temperature_observed = "T4"

temperature_observed = "T0"

# =========================================================
# Initial recurrent variables
# =========================================================

qs_prior = None
RANDOM_SEED = 50
rng_key = jax.random.PRNGKey(RANDOM_SEED)

history = []
previous_belief = ""
predicted_belief = ""
previous_action = ""

# =========================================================
# Agent-environment interaction loop
# =========================================================

# comforts = ["Warm", "LittleCool", "Cool", "LittleCold", "Cold"]  # hidden state
# print("ZZZ_")
# print(agent_model.B["comfort"]["Warm", "Cool", "AIT"])
# print(agent_model.B["comfort"]["LittleCool", "Cool", "AIT"])
# print(agent_model.B["comfort"]["Cool", "Cool", "AIT"])
# print(agent_model.B["comfort"]["LittleCold", "Cool", "AIT"])
# print(agent_model.B["comfort"]["Cold", "Cool", "AIT"])

#
# for t in range(30):
#     print(f"\n================ AGENT LOOP STEP {t + 1} ================")
#
#     result = run_agent(
#         model_agent=agent_model,
#         temperature_observed=temperature_observed,
#         qs_prior_input=qs_prior,
#         rng_key=rng_key,
#     )
#
#     history.append(result)
#
#     print("\n===== STEP SUMMARY =====")
#     print(
#         "Current observation:",
#         result["current_temperature"]
#     )
#
#     print("Chosen action:", result["chosen_action"])
#
#     if result["chosen_action"] not in base_actions:
#         print("XXX_New_Action")
#
#     print(
#         "Next observation:",
#         result["next_temperature"]
#     )
#
#     print("Current belief:", result["current_belief"])
#     print("Predicted prior next:", result["predicted_prior_next"])
#
#     lst_current_distribution = result["current_belief"].tolist()
#     lst_predicted_distribution = result["predicted_prior_next"].tolist()
#
#     current_belief = comforts[get_max_index(lst_current_distribution)]
#
#     if current_belief == predicted_belief and predicted_belief != "" and previous_action not in base_actions:
#         print(f"predicted_belief: {predicted_belief}")
#         print(f"Positive update for {previous_belief} -> {predicted_belief} with {previous_action}")
#         point = get_ates_positive_point(previous_action, previous_belief, predicted_belief, agent_model, 70, 30)
#         print("XXX HIT positive point = ")
#         print(point)
#
#
#     previous_belief = comforts[get_max_index(lst_current_distribution)]
#     predicted_belief = comforts[get_max_index(lst_predicted_distribution)]
#     previous_action = result["chosen_action"]
#
#     # Prior predicted by B becomes prior for the next observation.
#     qs_prior = result["predicted_prior_next"]
#
#     temperature_observed = result["next_temperature"]
#
#     rng_key = result["rng_key"]
#
#     # each loop ask for input ates_action_id_input
#     # ates_point = get_ates_positive_point(ates_action_id_input, "Neutral", "Cool", agent_model, 70, 30)
#     # ates_dif = ates_positive_update(ates_action_id_input, "Neutral", "Cool", agent_model, 70, 30)
#     # new_model = ates_update_matrix_positive(ates_action_id_input, "Neutral", "Cool", agent_model, ates_dif, ates_point)


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