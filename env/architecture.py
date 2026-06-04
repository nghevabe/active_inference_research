from env.environtment import agent_actions, user_actions, \
    lst_pairing_state, update_matrix, lst_slice_b, lst_action_extended, update_model, update_model_description


def append_action(action_id):
    agent_actions.append(action_id)
    user_actions.append("u" + action_id)
    print("agent_actions: " + str(agent_actions))


def add_slice_b_by_action(action_id):
    current_model = update_model(update_model_description())
    update_matrix(current_model)
    for item_str in lst_pairing_state:
        state_str = item_str.split("_")
        state_to = state_str[0]
        state_from = state_str[1]
        average_distribution = calculate_average_distribution(current_model, state_to, state_from)
        current_model.B["comfort"][state_to, state_from, action_id] = average_distribution

    lst_action_extended.append(action_id)
    lst_slice_b.append(current_model.B["comfort"][:, :, action_id])
    return current_model


def calculate_average_distribution(current_model, state_to, state_from):
    lst_item = current_model.B["comfort"][state_to, state_from, :]
    filtered_list = [value for value in lst_item if value > 0]
    total_item = 0.0
    for item in filtered_list:
        print(item)
        total_item = total_item + item
    average = total_item / len(filtered_list)
    return round(average, 2)
