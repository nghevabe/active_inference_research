import jax.tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
from pymdp.agent import Agent
from pymdp.distribution import compile_model
from pymdp.envs.env import Env
from pymdp.envs import rollout


positions = ["left", "center_left", "center_right", "right"]
actions = ["move_left", "move_right"]

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
model.A["position_obs"]["left", "left"] = 1.0
model.A["position_obs"]["center_left", "center_left"] = 1.0
model.A["position_obs"]["center_right", "center_right"] = 1.0
model.A["position_obs"]["right", "right"] = 1.0
# model.A["position_obs"].data = jnp.eye(len(positions)) # you could also use the .data attribute to set the identity mapping directly

# fill in the transition model (B) tensor
# note that it's specified as ["to", "from", "action"]
# moving right
model.B["position"]["center_left", "left", "move_right"] = 1.0
model.B["position"]["center_right", "center_left", "move_right"] = 1.0
model.B["position"]["right", "center_right", "move_right"] = 1.0
model.B["position"]["right", "right", "move_right"] = 1.0

# moving left
model.B["position"]["left", "left", "move_left"] = 1.0
model.B["position"]["left", "center_left", "move_left"] = 1.0
model.B["position"]["center_left", "center_right", "move_left"] = 1.0
model.B["position"]["center_right", "right", "move_left"] = 1.0

# set preferences (C) tensor - prefer to be at "center_right"
model.C["position_obs"]["left"] = 1.0
model.C["position_obs"]["center_left"] = 1.0
model.C["position_obs"]["center_right"] = 5.0
model.C["position_obs"]["right"] = 1.0


gamma = 10 # deterministic behavior; make gamma smaller for stochastic behavior

# create agent
agent = Agent(**model, gamma=gamma, policy_len=2)

# set up initial observation to be "left"
observation = jnp.full((agent.batch_size, 1), 0) # broadcast to agent's batch size (defaults to 1 agent) and add a time dimension

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


print("=== POLICIES ===")
for index in range(len(agent.policies)):
    print(agent.policies[index])

print("\n=== Q_PI ===")
print(q_pi)

print("\n=== G ===")
print(G)