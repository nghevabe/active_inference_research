from maze.env.element_maze import maze_size, agent_actions, position_observes


def environment_maze_step(action_input, current_position):
    """
    # S00 S01 S02 S03 S04 S05
    # S10 S11 S12 S13 S14 S15
    # S20 S21 S22 S23 S24 S25
    # S30 S31 S32 S33 S34 S35
    # S40 S41 S42 S43 S44 S45
    # S50 S51 S52 S53 S54 S55
    """

    next_pos_x = ""
    next_pos_y = ""

    current_pos_x = current_position[1:2]
    current_pos_y = current_position[2:3]

    if action_input in agent_actions:
        # action LEFT
        if action_input == "LEFT":
            pos_value = int(current_pos_y) - 1
            if pos_value <= 0:
                pos_value = 0

            next_pos_x = current_pos_x
            next_pos_y = str(pos_value)

        # action UP
        if action_input == "UP":
            pos_value = int(current_pos_x) - 1
            if pos_value <= 0:
                pos_value = 0

            next_pos_x = str(pos_value)
            next_pos_y = current_pos_y

        # action RIGHT
        if action_input == "RIGHT":
            pos_value = int(current_pos_y) + 1
            if pos_value >= (maze_size-1):
                pos_value = maze_size-1

            next_pos_x = current_pos_x
            next_pos_y = str(pos_value)

        # action DOWN
        if action_input == "DOWN":
            pos_value = int(current_pos_x) + 1
            if pos_value >= (maze_size-1):
                pos_value = maze_size-1

            next_pos_x = str(pos_value)
            next_pos_y = current_pos_y

        # Extended Action (JUMP_UP)
        if action_input == "JUMP_UP":
            pos_value = int(current_pos_x) - 1
            if pos_value <= 0:
                pos_value = 0

            next_pos_x = str(pos_value)
            next_pos_y = current_pos_y

    print("next_pos_x: "+str(next_pos_x))
    print("next_pos_y: "+str(next_pos_y))
    next_pos_label = str("P" + next_pos_x + next_pos_y)
    next_pos_index = position_observes.index(next_pos_label)

    # next_pos_index = next_pos_label.index(position_observes)

    return next_pos_label, next_pos_index


def build_matrix_a(model, lst_position_observe, lst_position_state):
    for item_state in lst_position_state:
        for item_observe in lst_position_observe:
            if item_observe[1:3] == item_state[1:3]:
                model.A["position_obs"][item_observe, item_state] = 1
            else:
                model.A["position_obs"][item_observe, item_state] = 0


def init_matrix_b(model, lst_position_state, lst_base_actions):
    for action in lst_base_actions:
        for i in lst_position_state:
            for j in lst_position_state:
                model.B["position_state"][j, i, action] = 0


def build_matrix_b(model, lst_position_state, lst_base_actions):
    init_matrix_b(model, lst_position_state, lst_base_actions)
    for action in lst_base_actions:
        for i in lst_position_state:
            for j in lst_position_state:

                pos_from_x = i[1:2]
                pos_from_y = i[2:3]
                pos_to_x = j[1:2]
                pos_to_y = j[2:3]

                # S00 S01 S02 S03 S04 S05
                # S10 S11 S12 S13 S14 S15
                # S20 S21 S22 S23 S24 S25
                # S30 S31 S32 S33 S34 S35
                # S40 S41 S42 S43 S44 S45
                # S50 S51 S52 S53 S54 S55

                # action LEFT
                if action == "LEFT":
                    if int(pos_from_y) - int(pos_to_y) == 1 and int(pos_from_x) == int(pos_to_x):
                        model.B["position_state"][j, i, action] = 1
                    if int(pos_from_y) == 0 and i == j:  # out of boundary left
                        model.B["position_state"][j, i, action] = 1

                # action UP
                if action == "UP":
                    if int(pos_from_x) - int(pos_to_x) == 1 and int(pos_from_y) == int(pos_to_y):
                        model.B["position_state"][j, i, action] = 1
                    if int(pos_from_x) == 0 and i == j:  # check boundary top
                        model.B["position_state"][j, i, action] = 1

                # action RIGHT
                if action == "RIGHT":
                    if int(pos_to_y) - int(pos_from_y) == 1 and int(pos_from_x) == int(pos_to_x):
                        model.B["position_state"][j, i, action] = 1
                    if int(pos_from_y) == (maze_size - 1) and i == j:   # check boundary right
                        model.B["position_state"][j, i, action] = 1

                # action DOWN
                if action == "DOWN":
                    if int(pos_to_x) - int(pos_from_x) == 1 and int(pos_from_y) == int(pos_to_y):
                        model.B["position_state"][j, i, action] = 1
                    if int(pos_from_x) == (maze_size - 1) and i == j:   # check boundary bottom
                        model.B["position_state"][j, i, action] = 1


                # S00 S01 S02 S03
                # S10 S11 S12 S13
                # S20 S21 S22 S23
                # S30 S31 S32 S33


def build_matrix_c(model, lst_position_observe, target_pos):
    for item_obs in lst_position_observe:
        model.C["position_obs"][item_obs] = 0

    model.C["position_obs"][target_pos] = 1