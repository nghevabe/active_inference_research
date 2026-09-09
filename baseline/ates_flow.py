from env.agent import run_agent, extend_action_space, \
    ates_update_matrix_positive, update_matrix_agent, get_ates_point, ates_update, get_list_positive_and_negative
import jax
import time

from env.elements import base_actions, agent_actions, comforts
from utils.util import get_max_index

# =========================================================
# Initialize agent model
# =========================================================

agent_model_init = extend_action_space("AIT")
agent_model_init = extend_action_space("ADT")
agent_model_init = extend_action_space("XDT")

# print("AIT matrix: ")
# print(agent_model_init.B["comfort"][:, :, "AIT"])
# print("ADT matrix: ")
# print(agent_model_init.B["comfort"][:, :, "ADT"])
# print("XDT matrix: ")
# print(agent_model_init.B["comfort"][:, :, "XDT"])


# =========================================================
# Initial observation
# =========================================================

# temperature_observed = "T4"

print("XXX_Matrix_agent_model Before Learning: ")
print(agent_model_init)

temperature_observed = "T0"

# =========================================================
# Initial recurrent variables
# =========================================================

qs_prior = None
RANDOM_SEED = 50
rng_key = jax.random.PRNGKey(RANDOM_SEED)

history = []
previous_belief = ""
previous_belief_distribution = 0
predicted_belief = ""
previous_action = ""
belief_prob = 0.7

summary_learning_log = []
summary_step_log = []
history_log = []
agent_model = agent_model_init

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

    lst_current_distribution = result["current_belief"].tolist()
    lst_predicted_distribution = result["predicted_prior_next"].tolist()

    current_belief = comforts[get_max_index(lst_current_distribution)]
    current_belief_distribution = max(lst_current_distribution)

    # step_log_str = f"STEP {t + 1} update for {previous_belief} -> {current_belief} with {previous_action} by {point} point"

    if previous_action not in base_actions and previous_action != "" and previous_belief_distribution > belief_prob and current_belief_distribution > belief_prob:
        point = get_ates_point(previous_action, previous_belief, current_belief, agent_model, belief_prob, 30)
        str_log = f"STEP {t + 1} update for {previous_belief} -> {current_belief} with {previous_action} by {point} point"
        summary_learning_log.append(str_log)
        ates_dif = ates_update(previous_action, previous_belief, current_belief, agent_model, belief_prob, 30)
        agent_model = ates_update_matrix_positive(previous_action, previous_belief, current_belief, agent_model,
                                                  ates_dif,
                                                  point)
        print(str_log)

    step_log = {
        "step_index": t,
        "from_observation": result["current_temperature"],
        "from_state_belief": result["current_comfort_map"],
        "from_state_belief_distribution": max(lst_current_distribution),
        "chosen_action": result["chosen_action"],
        "to_observation": result["next_temperature"],
        "to_state_belief": current_belief,
        "to_state_belief_distribution": max(lst_predicted_distribution),
    }

    history_log.append(step_log)

    print("XXX_Matrix_agent_model After Learning: ")
    print(agent_model)

    # each loop ask for input ates_action_id_input
    # ates_point = get_ates_positive_point(ates_action_id_input, "Neutral", "Cool", agent_model, 70, 30)
    # ates_dif = ates_positive_update(ates_action_id_input, "Neutral", "Cool", agent_model, 70, 30)
    # new_model = ates_update_matrix_positive(ates_action_id_input, "Neutral", "Cool", agent_model, ates_dif, ates_point)

    previous_belief = comforts[get_max_index(lst_current_distribution)]
    previous_belief_distribution = max(lst_current_distribution)
    predicted_belief = comforts[get_max_index(lst_predicted_distribution)]
    previous_action = result["chosen_action"]

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

# print("summary_step_log: ")
# counter = 0
# for item in history_log:
#     step_log_str = f"STEP {counter + 1} {item["from_observation"]} ({item["from_state_belief"]} [{item["from_state_belief_distribution"]}]) + {item["chosen_action"]} => {item["to_observation"]} ({item["to_state_belief"]} [{item["to_state_belief_distribution"]}]) "
#     print(item)
#     counter = counter + 1

get_list_positive_and_negative(history_log, "T0")