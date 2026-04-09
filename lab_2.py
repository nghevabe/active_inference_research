import jax.tree_util as jtu
from jax import numpy as jnp
from jax import random as jr
from pymdp.agent import Agent
from pymdp.distribution import compile_model
from pymdp.envs.env import Env
from pymdp.envs import rollout

num_locations = 3 # these correspond to "left", "center", and "right" and you can just specify ["left", "center", "right"] but we use numbers to show how to use the 'size' key instead of 'elements'
item_list = ["orchard", "apple"]

model_description = {
    "observations": {
        "location_obs": {"size": num_locations, # if you want to use numbers instead of strings, you can use the 'size' key instead of 'elements'
                         "depends_on": ["location_state"],
        },
        "item_obs": {"elements": item_list, # "elements" key for strings
                     "depends_on": ["location_state", "left_state", "center_state", "right_state"],
        },
        "reward_obs": {"elements": ["no_reward", "reward"],
                       "depends_on": ["reward_state"],
        },
    },
    "controls": {
        "move": {"elements": ["stay", "move_left", "move_right"],
        },
        "eat": {"elements": ["noop", "eat"], # noop = no-operation
        # note that if you cannot control a state, you still need to add
        # an action for it (e.g., with elements: ["null"]) for the model to be initialized
        # with the correct dimensions
        },
    },
    "states": {
        "location_state": {"size": num_locations,
                           "depends_on": ["location_state"],
                           "controlled_by": ["move"],
        },
        "reward_state": {"elements": ["no_reward", "reward"],
                            # if you have more than one dependency, the first dependency is its own state factor (at the previous timestep),
                            # then add the other dependencies in the order they are specified (you can skip over some state factors)
                            "depends_on": ["reward_state", "location_state",
                                           "left_state", "center_state", "right_state"],
                            "controlled_by": ["eat"],
        },
        "left_state": {"elements": item_list,
                        "depends_on": ["left_state", "location_state"],
                        "controlled_by": ["eat"],
        },
        "center_state": {"elements": item_list,
                        "depends_on": ["center_state", "location_state"],
                        "controlled_by": ["eat"],
        },
        "right_state": {"elements": item_list,
                        "depends_on": ["right_state", "location_state"],
                        "controlled_by": ["eat"],
        },
    },
}

model = compile_model(model_description)