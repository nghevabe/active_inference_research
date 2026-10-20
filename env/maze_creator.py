# from pymdp.distribution import compile_model
#
# from env.element_maze import update_model_description
#
# position_state = []
#
# base_actions = ["LEFT", "UP", "RIGHT", "DOWN"]  # Base action
# agent_actions = base_actions + []  # Agent action
#
# position_observe = []
#
#
# def init_element(size):
#     for x in range(size):
#         for y in range(size):
#             position_state.append("X"+str(x)+str(y))
#             position_observe.append("P"+str(x)+str(y))
#
#
# init_element(6)
# print(position_observe)
# print(position_state)
#
#
# def build_matrix_a(model, lst_position_observe, lst_position_state):
#
#     for item_state in lst_position_state:
#         for item_observe in lst_position_observe:
#             if item_observe[1:3] == item_state[1:3]:
#                 model.A["position_obs"][item_observe, item_state] = 1
#             else:
#                 model.A["position_obs"][item_observe, item_state] = 0
#
#
# current_model = compile_model(update_model_description())
# build_matrix_a(current_model, position_observe, position_state)
#
# print("XXX_current_model: "+current_model.A)


# ['P00', 'P01', 'P02', 'P03', 'P04', 'P05', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35', 'P40', 'P41', 'P42', 'P43', 'P44', 'P45', 'P50', 'P51', 'P52', 'P53', 'P54', 'P55']
# ['S00', 'S01', 'S02', 'S03', 'S04', 'S05', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S50', 'S51', 'S52', 'S53', 'S54', 'S55']
# S00 S01 S02 S03 S04 S05
# S10 S11 S12 S13 S14 S15
# S20 S21 S22 S23 S24 S25
# S30 S31 S32 S33 S34 S35
# S40 S41 S42 S43 S44 S45
# S50 S51 S52 S53 S54 S55