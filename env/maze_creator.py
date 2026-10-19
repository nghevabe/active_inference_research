position_state = []

base_actions = ["LEFT", "UP", "RIGHT", "DOWN"]  # Base action
agent_actions = base_actions + []  # Agent action

position_observe = []


def init_element(size):
    for x in range(size):
        for y in range(size):
            position_state.append("S"+str(x)+str(y))
            position_observe.append("P"+str(x)+str(y))


init_element(6)
print(position_observe)
print(position_state)



for x in range(6):
    for y in range(6):
        print("S" + str(x) + str(y), end=" ")
    print("")




