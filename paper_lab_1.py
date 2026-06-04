import jax
from pymdp.agent import Agent
from jax import numpy as jnp
import jax.tree_util as jtu
from env.environtment import \
    update_model_description, update_model, lst_slice_b, \
    lst_action_extended, update_matrix, temperatures, lights, times, comforts, agent_actions
from env.architecture import append_action, add_slice_b_by_action


def extend_action_space(action_id):
    append_action(action_id)
    add_slice_b_by_action(action_id)


extend_action_space("ACN1")
extend_action_space("ACN2")
current_model = update_model(update_model_description())
update_matrix(current_model)
slice_index = 0

for action in lst_action_extended:
    current_model.B["comfort"][:, :, action] = lst_slice_b[slice_index]
    slice_index = slice_index + 1

print("\n===== B MATRIX / Transition State After =====")
print(current_model.B)


# =========================================================
# Run Active Inference agent
# =========================================================

gamma = 1  # deterministic behavior; smaller gamma -> more stochastic behavior

# Create agent
agent = Agent(**current_model, gamma=gamma, policy_len=1)


temperature_observed = "T1"
light_observed = "L0"
time_observed = "Morning"

temperature_idx = temperatures.index(temperature_observed)
light_idx = lights.index(light_observed)
time_idx = times.index(time_observed)

# Each observation must have shape: (batch_size, time_dim)
# agent.batch_size defaults to 1.
temperature_observation = jnp.full(
    (agent.batch_size, 1),
    temperature_idx
)

light_observation = jnp.full(
    (agent.batch_size, 1),
    light_idx
)

time_observation = jnp.full(
    (agent.batch_size, 1),
    time_idx
)

# Multi-modality observation list
observations = [
    temperature_observation,
    light_observation,
    time_observation,
]


qs_init = jtu.tree_map(
    lambda x: jnp.expand_dims(x, 1),
    agent.D
)

# ---------------------------------------------------------
# Infer hidden states
# ---------------------------------------------------------

qs = agent.infer_states(observations, qs_init)

print("\n===== INITIAL OBSERVATION =====")
print(f"Observed temperature: {temperature_observed}")
print(f"Observed light:       {light_observed}")
print(f"Observed time:        {time_observed}")

print("\n===== DEBUG SHAPE =====")
print("qs[0].shape:", qs[0].shape)

# qs[0] currently has shape: (batch_size, 1, 1, num_states)
# Example: (1, 1, 1, 3)
# For printing, convert it to a clean vector: (3,)
comfort_belief = jnp.squeeze(qs[0], axis=(0, 1, 2))

print("comfort_belief.shape:", comfort_belief.shape)

print("\n===== POSTERIOR BELIEF OVER COMFORT =====")
for i, state in enumerate(comforts):
    print(f"{state}: {float(comfort_belief[i]):.4f}")

current_comfort_idx = int(jnp.argmax(comfort_belief))
print(f"\nMost likely comfort state: {comforts[current_comfort_idx]}")

# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
# Prepare posterior belief for policy inference
# ---------------------------------------------------------
# agent.infer_policies expects qs with shape:
# (batch_size, time_dim, num_states)
#
# Current qs[0] shape: (1, 1, 1, 3)
# We remove only the extra singleton axis at axis=2:
# Result shape: (1, 1, 3)

qs_for_policy = [
    jnp.squeeze(q, axis=2)
    for q in qs
]

print("\n===== DEBUG POLICY INPUT SHAPE =====")
print("qs_for_policy[0].shape:", qs_for_policy[0].shape)

# ---------------------------------------------------------
# Infer policies and sample action
# ---------------------------------------------------------

q_pi, G = agent.infer_policies(qs_for_policy)

print("\n===== POLICY INFERENCE =====")
print("q_pi:", q_pi)
print("q_pi.shape:", q_pi.shape)
print("G:", G)

# rng_key must match batch dimension.
# q_pi.shape = (batch_size, num_policies)
# Therefore rng_key.shape should be (batch_size, 2)
# ---------------------------------------------------------
# Custom stochastic sampling directly from q_pi
# ---------------------------------------------------------

rng_key = jax.random.PRNGKey(10)

probs = q_pi[0]  # shape: (num_actions,)

chosen_action_idx = int(
    jax.random.choice(
        rng_key,
        a=jnp.arange(len(agent_actions)),
        p=probs
    )
)

chosen_action = agent_actions[chosen_action_idx]

print("\n===== CUSTOM STOCHASTIC ACTION SELECTED =====")
print("probs:", probs)
print(f"Action chosen: {chosen_action}")