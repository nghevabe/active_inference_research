from env.agent import extend_action_space, run_agent


# agent_model = extend_action_space("ACN1")
# agent_model = extend_action_space("ACN2")
#
# run_agent(agent_model)


dict_test = [
{
    "state": "Warm",
    "distribute": 0.8279234766960144,
},
{
    "state": "LittleCool",
    "distribute": 0.9613120555877686,
},
{
    "state": "LittleCool",
    "distribute": 0.9613121747970581,
},
{
    "state": "LittleCool",
    "distribute": 0.9913740754127502,
},
]


def get_state_belief_most(
        step_log
):

    dict_state = {}
    for item in step_log:
        dict_state[item["state"]] = 0.0

    for item in step_log:
        dict_state[item["state"]] = dict_state.get(item["state"]) + item["distribute"]

    print(dict_state)

    dict_counter = {}
    for key in dict_state.keys():
        counter = 0
        for item in step_log:
            if item["state"] == key:
                counter = counter + 1
        dict_counter[key] = counter

    print(dict_counter)

    state_max = max(dict_state, key=dict_state.get)
    print(state_max)

    state_max_value = dict_state[state_max]
    print(state_max_value)

    point_average = state_max_value/(dict_counter[state_max])
    print(point_average)
    return state_max, point_average


print(get_state_belief_most(dict_test))

# print("XXX_Matrix B: ")
# print(agent_model.B)
# print("XXX_prob: ")
# print(agent_model.B["comfort"][:, "Neutral", "ACN2"])
# print("XXX_ates_point: ")
# print(get_ates_positive_point("ACN2", "Neutral", "Cool", agent_model, 70, 30))
# print("XXX_ates_dif: ")
# print(ates_positive_update("ACN2", "Neutral", "Cool", agent_model, 70, 30))
# print("XXX_prob_new: ")
# ates_point = get_ates_positive_point("ACN2", "Neutral", "Cool", agent_model, 70, 30)
# ates_dif = ates_positive_update("ACN2", "Neutral", "Cool", agent_model, 70, 30)
# new_model = ates_update_matrix_positive("ACN2", "Neutral", "Cool", agent_model, ates_dif, ates_point)
# print(new_model.B["comfort"][:, "Neutral", "ACN2"])

# [0.24 0.45 0.31] => [0.17 0.38 0.45]
# [0.24 0.45 0.31] => [0.14 0.35 0.50]

#               T0       T1, T2       T3       T4, T5       T6
# comforts = ["Warm", "LittleCool", "Cool", "LittleCold", "Cold"]  # hidden state
# temperatures = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]  # Observe Temperature
