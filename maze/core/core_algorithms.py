from maze.agent.agent_maze import update_matrix_agent
from maze.env.element_maze import lst_action_extended, lst_slice_b, base_actions, append_action, position_states


def add_slice_b_by_action(action_id, lst_position_states):
    current_model = update_matrix_agent()
    for item_to in lst_position_states:
        for item_from in lst_position_states:
            average_distribution = calculate_average_distribution(current_model, item_to, item_from)
            current_model.B["position_state"][item_to, item_from, action_id] = average_distribution

    lst_action_extended.append(action_id)
    lst_slice_b.append(current_model.B["position_state"][:, :, action_id])
    return current_model


def calculate_average_distribution(model, state_to, state_from):
    lst_item = model.B["position_state"][state_to, state_from, :]
    filtered_list = lst_item[0:len(base_actions)]
    total_item = 0.0
    for item in filtered_list:
        total_item = total_item + item
    average = total_item / len(filtered_list)
    return average


def extend_action_space(action_id):
    append_action(action_id)
    model = add_slice_b_by_action(action_id, position_states)
    slice_index = 0
    for action in lst_action_extended:
        model.B["position_state"][:, :, action] = lst_slice_b[slice_index]
        slice_index = slice_index + 1
    return model


def get_ates_point(ates_action_id, current_state, expect_state, current_model, belief_prob, upd_point_ratio):
    current_transition_prob = current_model.B["position_state"][expect_state, current_state, ates_action_id]
    point = (1 - current_transition_prob) * (upd_point_ratio / 100) * belief_prob
    print("XXX_my_point: " + str(point))
    return point


def get_ates_point_negative(ates_action_id, current_state, expect_state, current_model, belief_prob, upd_point_ratio):
    current_transition_prob = current_model.B["position_state"][expect_state, current_state, ates_action_id]
    point = current_transition_prob * upd_point_ratio / 100 * belief_prob
    return point


def ates_update_matrix_positive(ates_action_id, current_state, expect_state, current_model, ates_point):
    current_model.B["position_state"][expect_state, current_state, ates_action_id] += ates_point
    current_model = normalization_matrix_positive(ates_action_id, current_state, expect_state, current_model,
                                                  ates_point)

    return current_model


def ates_update_matrix_negative(ates_action_id, current_state, expect_state, current_model, ates_point):
    current_model.B["position_state"][expect_state, current_state, ates_action_id] -= ates_point
    current_model = normalization_matrix_negative(ates_action_id, current_state, expect_state, current_model,
                                                  ates_point)

    return current_model


def normalization_matrix_positive(ates_action_id, current_state, expect_state, current_model, ates_point):
    print("-----------")
    remain_sum_point = 0
    for item in position_states:
        point_item = current_model.B["position_state"][item, current_state, ates_action_id]

        if item != expect_state:
            print(f"XXX_current_state {current_state}, expect_state {item}, action {ates_action_id} : {point_item}")
            remain_sum_point = remain_sum_point + point_item

    for item in position_states:
        point_item = current_model.B["position_state"][item, current_state, ates_action_id]
        new_sum = abs(remain_sum_point - ates_point)

        if item != expect_state:
            new_point = point_item / remain_sum_point * new_sum
            current_model.B["position_state"][item, current_state, ates_action_id] = new_point

    print("***")

    for item in position_states:
        point_item = current_model.B["position_state"][item, current_state, ates_action_id]
        print(f"XXX_current_state {current_state}, expect_state {item}, action {ates_action_id} : {point_item}")

    print(f"XXX_previous sum = {remain_sum_point} - after sum = {remain_sum_point - ates_point}")

    print("-----------")
    return current_model


def normalization_matrix_negative(ates_action_id, current_state, expect_state, current_model, ates_point):
    print("-----------")
    remain_sum_point = 0
    for item in position_states:
        point_item = current_model.B["position_state"][item, current_state, ates_action_id]

        if item != expect_state:
            print(f"XXX_current_state {current_state}, expect_state {item}, action {ates_action_id} : {point_item}")
            remain_sum_point = remain_sum_point + point_item

    for item in position_states:
        point_item = current_model.B["position_state"][item, current_state, ates_action_id]
        new_sum = remain_sum_point + ates_point

        if item != expect_state:
            new_point = point_item / remain_sum_point * new_sum
            current_model.B["position_state"][item, current_state, ates_action_id] = new_point

    print("***")

    for item in position_states:
        point_item = current_model.B["position_state"][item, current_state, ates_action_id]
        print(f"XXX_current_state {current_state}, expect_state {item}, action {ates_action_id} : {point_item}")

    print(f"XXX_previous sum = {remain_sum_point} - after sum = {remain_sum_point + ates_point}")

    print("-----------")
    return current_model


def get_state_belief_most(
        step_log
):
    state_max = ""
    point_average = 0.0

    for item in step_log:
        state_max = item["state"]
        point_average = item["distribute"]

    return state_max, point_average


def get_list_target_step(step_log, step_index, current_observed):
    lst_target_step = []
    for item in step_log:
        if item["step_index"] == step_index:
            lst_target_step.append(item)
        if item["step_index"] > step_index:
            if item["from_observation"] == current_observed:
                lst_target_step.append(item)
            else:
                break

    return lst_target_step


def negative_update_steps(lst_negative_step, agent_model, current_belief_lst):
    for step in lst_negative_step:
        for item in current_belief_lst:
            target_state_name = item.split(": ")[0]
            target_state_distribute = float(item.split(": ")[1])
            if step["chosen_action"] and str(step["from_state_belief"]) != str(
                    target_state_name) and target_state_distribute > 0:
                print("Update Negative_XXX:")
                print("chosen_action:" + str(step["chosen_action"]))
                print("from_state_belief:" + str(step["from_state_belief"]))
                print("target_state:" + str(target_state_name))
                print("target_state_distribute:" + str(target_state_distribute))
                point = get_ates_point_negative(step["chosen_action"], step["from_state_belief"], target_state_name,
                                                agent_model,
                                                target_state_distribute, 80)
                print("point: " + str(point))
                agent_model = ates_update_matrix_negative(step["chosen_action"], step["from_state_belief"],
                                                          target_state_name,
                                                          agent_model,
                                                          point)

    return agent_model


def get_list_state_target(step_log, target_observed):
    lst_state_target = []
    print("XXX_target_observed: " + str(target_observed))
    for item in step_log:
        if item["from_observation"] == target_observed or item["to_observation"] == target_observed:
            if item["from_observation"] == target_observed:
                state_object = {
                    "state": item["from_state_belief"],
                    "distribute": item["from_state_belief_distribution"],
                }
                lst_state_target.append(state_object)
            if item["to_observation"] == target_observed:
                state_object = {
                    "state": item["to_state_belief"],
                    "distribute": item["to_state_belief_distribution"],
                }
                lst_state_target.append(state_object)
        else:
            break

    return lst_state_target


def get_list_positive_and_negative(
        step_log,
        init_observed,
        agent_model
):
    lst_negative_step = []
    lst_target_step = []
    current_observed = init_observed
    final_model = agent_model

    for item in step_log:
        if item["from_observation"] == current_observed and item["to_observation"] == current_observed:
            lst_negative_step.append(item)
        else:
            lst_target_step.append(item)
            print("XXX_evident: " + str(item))
            current_observed = item["to_observation"]
            lst_target_step = get_list_target_step(step_log, item["step_index"], current_observed)
            most_target_step_belief = lst_target_step[len(lst_target_step) - 1]
            current_belief_lst = most_target_step_belief["current_belief_lst"]
            print("current_belief_lst_ZZZ: " + str(current_belief_lst))
            list_state_target = get_list_state_target(lst_target_step, current_observed)
            print("lst_target_step_XXX: " + str(lst_target_step))
            print("lst_state_target_XXX: " + str(list_state_target))
            target_state, target_state_distribute = get_state_belief_most(list_state_target)
            if len(lst_negative_step) > 0:
                print("Negative Update: " + str(lst_negative_step))
                print("With: " + str(target_state) + " by " + str(target_state_distribute))
                final_model = negative_update_steps(lst_negative_step, agent_model, current_belief_lst)
                lst_negative_step.clear()
                lst_target_step.clear()
                print("=========")

    return final_model
