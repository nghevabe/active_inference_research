from env.agent import run_agent, extend_action_space, \
    ates_update_matrix_positive, get_list_positive_and_negative, \
    get_ates_point, ates_update_matrix_negative, get_ates_point_negative
import jax
import time
from decimal import Decimal
from utils.load_matrix_b import load_b_matrix_from_txt

from env.elements import base_actions, agent_actions, comforts, base_actions_test
from utils.save_matrix_b import save_b_matrix_readable
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

print("XXX_Matrix_agent_model Init: ")
print(agent_model_init)

temperature_observed = "T6"

# =========================================================
# Initial recurrent variables
# =========================================================

qs_prior = None
RANDOM_SEED = 70
rng_key = jax.random.PRNGKey(RANDOM_SEED)

history = []
previous_belief = ""
previous_belief_distribution = 0
predicted_belief = ""
previous_action = ""
belief_prob = 0.5

summary_learning_log = []
summary_step_log = []
history_log = []

agent_model = agent_model_init

# load model 2
load_b_matrix_from_txt(
    agent_model,
    "logs/matrix_b_ates.txt",
)

print("XXX_Matrix_agent_model After Load: ")
print(agent_model)

previous_step = {}

for t in range(80):
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
    # current_belief_distribution = max(lst_current_distribution)
    #
    #
    #
    # if previous_action not in base_actions_test and previous_action != "":
    #     point = get_ates_point(previous_action, previous_belief, current_belief, agent_model, current_belief_distribution, 30)
    #     agent_model = ates_update_matrix_positive(previous_action, previous_belief, current_belief, agent_model,
    #                                               point)
        # print(str_log)

        # str_log = f"STEP {t + 1} update for {previous_belief} -> {current_belief} with {previous_action} by {point} point"
        # summary_learning_log.append(str_log)

    step_log = {
        "step_index": t,
        "from_observation": result["current_temperature"],
        "current_belief_lst": result["lst_obj_belief"],
        "from_state_belief": result["current_comfort_map"],
        "from_state_belief_distribution": max(lst_current_distribution),
        "chosen_action": result["chosen_action"],
        "to_observation": result["next_temperature"],
        "to_state_belief": current_belief,
        "to_state_belief_distribution": max(lst_predicted_distribution),
    }

    history_log.append(step_log)

    if t > 0 and previous_action not in base_actions_test:
        lst_previous = previous_step["current_belief_lst"]
        lst_current = step_log["current_belief_lst"]
        for previous in lst_previous:
            previous_name = previous.split(": ")[0]
            previous_distribute = float(previous.split(": ")[1])
            if previous_distribute > 0.0:
                print("___ZZZ: " + previous_name)
                for current in lst_current:
                    current_name = current.split(": ")[0]
                    current_distribute = float(current.split(": ")[1])
                    if current_distribute > 0.0:
                        current_belief_distribution = previous_distribute * current_distribute
                        print("previous_action: " + str(previous_action))
                        print("previous_name: " + str(previous_name))
                        print("current_name: " + str(current_name))
                        print("current_belief_distribution: " + str(current_belief_distribution))
                        print("positive update for : " + str(t) + " " + str(previous_name) + "("+str(previous_distribute)+")" + " -> " + str(current_name) + "("+str(current_distribute)+")" + " by "+previous_action)
                        point = get_ates_point(previous_action, previous_name, current_name, agent_model,
                                               current_belief_distribution, 40)
                        EPS = 1e-10
                        if point <= EPS:
                            break
                        agent_model = ates_update_matrix_positive(previous_action, previous_name, current_name,
                                                                  agent_model,
                                                                  point)

                                # todo linhth

    previous_step = step_log




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

    previous_belief_lst = lst_current_distribution

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

print("summary_history_log: ")
count_step = 0
for item in history_log:
    step_log_str = f"STEP {count_step + 1} from {item["from_observation"]} to {item["to_observation"]} with {item["chosen_action"]}"
    print(step_log_str)
    count_step = count_step + 1

print("summary_step_log: ")
counter = 0
for item in summary_learning_log:
    # step_log_str = f"STEP {counter + 1} {item["from_observation"]} ({item["from_state_belief"]} [{item["from_state_belief_distribution"]}]) + {item["chosen_action"]} => {item["to_observation"]} ({item["to_state_belief"]} [{item["to_state_belief_distribution"]}]) "
    print(item)
    counter = counter + 1

print("")
print("")
get_list_positive_and_negative(history_log, "T6", agent_model)
print("FINAL MODEL")
print(agent_model)

save_b_matrix_readable(
    agent_model,
    "logs/matrix_b_ates.txt",
)