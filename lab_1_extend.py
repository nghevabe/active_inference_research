import jax.tree_util as jtu
from jax import numpy as jnp
from pymdp.agent import Agent
from pymdp.distribution import compile_model


positions = ["pos_1", "pos_2", "pos_3", "pos_4", "pos_5"]
actions = ["stay", "move_left", "move_right"]

model_description = {
    "observations": {
        "position_obs": {
            "elements": positions,
            "depends_on": ["position"] # we specify that the observation depends on the "position" state factor
        },
    },
    "controls": {
        "movement": {"elements": actions} # we specify the available actions
    },
    "states": {
        "position": {
            "elements": positions,
            "depends_on": ["position"],  # our current position depends on previous position...
            "controlled_by": ["movement"]  # ...and the movement action taken
        },
    },
}

# compile the model structure from the description
model = compile_model(model_description)


# fill in the likelihood (A) tensor
# the observations have an identical mapping to the states (i.e., the agent will perfectly observe its position)
model.A["position_obs"]["pos_1", "pos_1"] = 1.0
model.A["position_obs"]["pos_2", "pos_2"] = 1.0
model.A["position_obs"]["pos_3", "pos_3"] = 1.0
model.A["position_obs"]["pos_4", "pos_4"] = 1.0
model.A["position_obs"]["pos_5", "pos_5"] = 1.0
# model.A["position_obs"].data = jnp.eye(len(positions)) # you could also use the .data attribute to set the identity mapping directly

# fill in the transition model (B) tensor
# note that it's specified as ["to", "from", "action"]

# stay
model.B["position"]["pos_1", "pos_1", "stay"] = 1.0
model.B["position"]["pos_2", "pos_2", "stay"] = 1.0
model.B["position"]["pos_3", "pos_3", "stay"] = 1.0
model.B["position"]["pos_4", "pos_4", "stay"] = 1.0
model.B["position"]["pos_5", "pos_5", "stay"] = 1.0

# moving right
model.B["position"]["pos_2", "pos_1", "move_right"] = 1.0
model.B["position"]["pos_3", "pos_2", "move_right"] = 1.0
model.B["position"]["pos_4", "pos_3", "move_right"] = 1.0
model.B["position"]["pos_5", "pos_4", "move_right"] = 1.0
model.B["position"]["pos_5", "pos_5", "move_right"] = 1.0

# moving left
model.B["position"]["pos_1", "pos_1", "move_left"] = 1.0
model.B["position"]["pos_1", "pos_2", "move_left"] = 1.0
model.B["position"]["pos_2", "pos_3", "move_left"] = 1.0
model.B["position"]["pos_3", "pos_4", "move_left"] = 1.0
model.B["position"]["pos_4", "pos_5", "move_left"] = 1.0

# set preferences (C) tensor - prefer to be at "center_right"
model.C["position_obs"]["pos_5"] = 1.0

gamma = 10 # deterministic behavior; make gamma smaller for stochastic behavior


def get_name_by_index(idx):
    return actions[idx]


def get_lst_name_policy(policies):
    lst_action = []
    for idx in policies:
        lst_action.append(get_name_by_index(idx[0]))

    return lst_action


def get_index_of_best_policies(policies):
    policies_point = list(map(int, policies))
    best_policy = int(policies.max())
    return policies_point.index(best_policy)


# create agent
agent = Agent(**model, gamma=gamma, policy_len=5)

# set up initial observation to be "left"
observation = jnp.full((agent.batch_size, 1), 1) # broadcast to agent's batch size (defaults to 1 agent) and add a time dimension

# get the prior
qs_init = jtu.tree_map(lambda x: jnp.expand_dims(x, 1), agent.D) # qs needs a time dimension too

# print initial beliefs, goal, and action chosen
qs = agent.infer_states([observation], qs_init)
print(f"Current belief about position: {positions[jnp.argmax(qs[0][0])]}")
qs = [jnp.squeeze(q, 1) for q in qs]

print(f"Goal position: {positions[jnp.argmax(agent.C[0])]}")

q_pi, G = agent.infer_policies(qs)
action_idx = agent.sample_action(q_pi)
print(f"Action chosen: {actions[action_idx[0][0]]}")

print("")
print("")
print("=== Policy ===")
best_index = get_index_of_best_policies(G[0])
print(get_lst_name_policy(agent.policies[best_index]))
