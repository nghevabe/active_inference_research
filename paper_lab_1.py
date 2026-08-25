from env.agent import extend_action_space, run_agent


agent_model = extend_action_space("ACN1")
agent_model = extend_action_space("ACN2")

run_agent(agent_model)

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
