from pymdp.agent import Agent
from pymdp.distribution import compile_model
from jax import numpy as jnp
import jax.tree_util as jtu
import jax

from env.environtment_maze import predict_next_state_belief, environment_maze_step

position_states = []

base_actions = ["LEFT", "UP", "RIGHT", "DOWN"]  # Base action
agent_actions = base_actions + []  # Agent action

position_observes = []

maze_size = 6

lst_slice_b = []
lst_action_extended = []


def init_element(size):
    for x in range(size):
        for y in range(size):
            position_states.append("S" + str(x) + str(y))
            position_observes.append("P" + str(x) + str(y))


model_description = {
    "observations": {
        "position_obs": {
            "elements": position_observes,
            "depends_on": ["position_state"],
        },
    },

    "controls": {
        "agent_action": {
            "elements": agent_actions,
        },
    },

    "states": {
        "position_state": {
            "elements": position_states,
            "depends_on": ["position_state"],
            "controlled_by": ["agent_action"],
        },
    },
}


def update_model_description():
    return {
        "observations": {
            "position_obs": {
                "elements": position_observes,
                "depends_on": ["position_state"],
            },
        },

        "controls": {
            "agent_action": {
                "elements": agent_actions,
            },
        },

        "states": {
            "position_state": {
                "elements": position_states,
                "depends_on": ["position_state"],
                "controlled_by": ["agent_action"],
            },
        },
    }


def update_matrix_agent():
    model = compile_model(update_model_description())
    build_matrix_a(model, position_observes, position_states)
    build_matrix_b(model, position_states, base_actions)
    build_matrix_c(model, position_observes, "P42")
    return model


def append_action(action_id):
    agent_actions.append(action_id)
    print("agent_actions: " + str(agent_actions))


def build_matrix_a(model, lst_position_observe, lst_position_state):
    for item_state in lst_position_state:
        for item_observe in lst_position_observe:
            if item_observe[1:3] == item_state[1:3]:
                model.A["position_obs"][item_observe, item_state] = 1
            else:
                model.A["position_obs"][item_observe, item_state] = 0


def build_matrix_b(model, lst_position_state, lst_base_actions):
    for action in lst_base_actions:
        for i in lst_position_state:
            for j in lst_position_state:
                model.B["position_state"][j, i, action] = 0

                pos_from_x = i[1:2]
                pos_from_y = i[2:3]
                pos_to_x = j[1:2]
                pos_to_y = j[2:3]

                # action LEFT
                if action == "LEFT":
                    if int(pos_from_y) == 0:  # check in boundary left
                        model.B["position_state"][j, i, action] = 0
                    if int(pos_from_y) - int(pos_to_y) == 1:
                        model.B["position_state"][j, i, action] = 1
                    if int(pos_from_y) == 0 and i == j:  # out of boundary left
                        model.B["position_state"][j, i, action] = 1

                # action UP
                if action == "UP":
                    if int(pos_from_x) == 0:  # check boundary top
                        model.B["position_state"][j, i, action] = 0
                    if int(pos_from_x) - int(pos_to_x) == 1:
                        model.B["position_state"][j, i, action] = 1
                    if int(pos_from_x) == 0 and i == j:  # check boundary top
                        model.B["position_state"][j, i, action] = 1

                # action RIGHT
                if action == "RIGHT":
                    if int(pos_from_y) == (maze_size - 1):   # check boundary right
                        model.B["position_state"][j, i, action] = 0
                    if int(pos_to_y) - int(pos_from_y) == 1:
                        model.B["position_state"][j, i, action] = 1
                    if int(pos_from_y) == (maze_size - 1) and i == j:   # check boundary right
                        model.B["position_state"][j, i, action] = 1

                # action BOTTOM
                if action == "DOWN":
                    if int(pos_from_x) == (maze_size - 1):   # check boundary bottom
                        model.B["position_state"][j, i, action] = 0
                    if int(pos_to_x) - int(pos_from_x) == 1:
                        model.B["position_state"][j, i, action] = 1
                    if int(pos_from_x) == (maze_size - 1) and i == j:   # check boundary bottom
                        model.B["position_state"][j, i, action] = 1

    # S00 S01 S02 S03 S04 S05
    # S10 S11 S12 S13 S14 S15
    # S20 S21 S22 S23 S24 S25
    # S30 S31 S32 S33 S34 S35
    # S40 S41 S42 S43 S44 S45
    # S50 S51 S52 S53 S54 S55

    # From Warm
    # current_model.B["comfort"]["Warm", "Warm", "FIT"] = 0.10
    # current_model.B["comfort"]["LittleCool", "Warm", "FIT"] = 0.85
    # current_model.B["comfort"]["Cool", "Warm", "FIT"] = 0.05
    # current_model.B["comfort"]["LittleCold", "Warm", "FIT"] = 0.00
    # current_model.B["comfort"]["Cold", "Warm", "FIT"] = 0.00


def build_matrix_c(model, lst_position_observe, target_pos):
    for item_obs in lst_position_observe:
        model.C["position_obs"][item_obs] = 0

    model.C["position_obs"][target_pos] = 1


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
    print("Done_1")
    slice_index = 0
    for action in lst_action_extended:
        model.B["position_state"][:, :, action] = lst_slice_b[slice_index]
        slice_index = slice_index + 1
    print("Done_2")
    return model


def get_ates_point(ates_action_id, current_state, expect_state, current_model, belief_prob, upd_point_ratio):
    current_transition_prob = current_model.B["position_state"][expect_state, current_state, ates_action_id]
    point = (1 - current_transition_prob) * (upd_point_ratio / 100) * belief_prob
    print("XXX_my_point: "+str(point))
    return point


def get_ates_point_negative(ates_action_id, current_state, expect_state, current_model, belief_prob, upd_point_ratio):
    current_transition_prob = current_model.B["position_state"][expect_state, current_state, ates_action_id]
    point = current_transition_prob * upd_point_ratio / 100 * belief_prob
    return point


def ates_update_matrix_positive(ates_action_id, current_state, expect_state, current_model, ates_point):
    current_model.B["position_state"][expect_state, current_state, ates_action_id] += ates_point
    current_model = normalization_matrix_positive(ates_action_id, current_state, expect_state, current_model, ates_point)

    return current_model


def ates_update_matrix_negative(ates_action_id, current_state, expect_state, current_model, ates_point):
    current_model.B["position_state"][expect_state, current_state, ates_action_id] -= ates_point
    current_model = normalization_matrix_negative(ates_action_id, current_state, expect_state, current_model, ates_point)

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


def run_agent(
        model_agent,
        position_observed,
        qs_prior_input=None,
        rng_key=None,
):
    # =========================================================
    # Run Active Inference agent
    # =========================================================

    gamma = 1

    # ---------------------------------------------------------
    # Create agent
    # ---------------------------------------------------------
    agent = Agent(
        **model_agent,
        gamma=gamma,
        policy_len=1,
    )

    position_idx = position_observes.index(position_observed)

    # Each observation must have shape:
    # (batch_size, time_dim)
    #
    # agent.batch_size defaults to 1.
    position_observation = jnp.full(
        (agent.batch_size, 1),
        position_idx,
    )

    # Multi-modality observation list
    observations = [
        position_observation
    ]

    # =========================================================
    # Prepare prior for hidden-state inference
    # =========================================================

    if qs_prior_input is None:
        # -----------------------------------------------------
        # First interaction step:
        # use agent.D as the initial prior
        # -----------------------------------------------------
        qs_init = jtu.tree_map(
            lambda x: jnp.expand_dims(x, 1),
            agent.D,
        )
    else:
        # -----------------------------------------------------
        # Later interaction steps:
        #
        # qs_prior_input is the predicted prior:
        #
        # q^-(s_t) = B[a_{t-1}] @ q(s_{t-1})
        #
        # Input shape:
        # (num_states,)
        #
        # Required shape:
        # (batch_size, time_dim, num_states)
        # -----------------------------------------------------
        qs_init = [
            jnp.expand_dims(
                jnp.expand_dims(
                    qs_prior_input,
                    axis=0,
                ),
                axis=1,
            )
        ]

    # =========================================================
    # Infer hidden states
    # =========================================================

    qs = agent.infer_states(
        observations,
        qs_init,
    )

    print("\n===== INITIAL OBSERVATION =====")
    print(
        f"Observed position: "
        f"{position_observed}"
    )

    print("\n===== DEBUG SHAPE =====")
    print(
        "qs[0].shape:",
        qs[0].shape,
    )

    # ---------------------------------------------------------
    # qs[0] currently has shape:
    #
    # (batch_size, 1, 1, num_states)
    #
    # Example:
    # (1, 1, 1, 5)
    #
    # Convert it to the full posterior belief vector:
    #
    # (num_states,)
    #
    # IMPORTANT:
    # comfort_belief remains a SOFT posterior distribution.
    # We do not convert it to one-hot.
    # ---------------------------------------------------------
    position_belief = jnp.squeeze(
        qs[0],
        axis=(0, 1, 2),
    )

    print(
        "position_belief.shape:",
        position_belief.shape,
    )

    print(
        "\n===== POSTERIOR BELIEF OVER position ====="
    )

    lst_obj_belief = []
    for i, state in enumerate(position_states):
        str_object_belief = f"{state}: "+f"{float(position_belief[i]):.4f}"
        lst_obj_belief.append(str_object_belief)
        print(
            f"{state}: "
            f"{float(position_belief[i]):.4f}"
        )

    # ---------------------------------------------------------
    # MAP estimate of the current hidden comfort state.
    #
    # This is only the most probable state under the posterior:
    #
    # s_MAP = argmax_s q(s)
    #
    # IMPORTANT:
    # This is NOT the true hidden state and does NOT replace
    # the full soft posterior distribution.
    # ---------------------------------------------------------
    current_position_idx = int(
        jnp.argmax(position_belief)
    )

    current_position_map = position_states[
        current_position_idx
    ]

    print(
        f"\nMost likely position state "
        f"(MAP estimate): "
        f"{current_position_map}"
    )

    # =========================================================
    # Prepare posterior belief for policy inference
    # =========================================================
    #
    # agent.infer_policies expects:
    #
    # (batch_size, time_dim, num_states)
    #
    # Current qs[0] shape:
    #
    # (1, 1, 1, num_states)
    #
    # Remove only the extra singleton axis at axis=2:
    #
    # (1, 1, num_states)
    # =========================================================

    qs_for_policy = [
        jnp.squeeze(
            q,
            axis=2,
        )
        for q in qs
    ]

    print(
        "\n===== DEBUG POLICY INPUT SHAPE ====="
    )

    print(
        "qs_for_policy[0].shape:",
        qs_for_policy[0].shape,
    )

    # =========================================================
    # Infer policies
    # =========================================================

    q_pi, G = agent.infer_policies(
        qs_for_policy
    )

    print("\n===== POLICY INFERENCE =====")
    print(
        "q_pi:",
        q_pi,
    )
    print(
        "q_pi.shape:",
        q_pi.shape,
    )
    print(
        "G:",
        G,
    )

    # =========================================================
    # Direct stochastic sampling from q_pi
    # =========================================================
    #
    # IMPORTANT:
    #
    # We do NOT:
    # - use top-k
    # - apply an additional sampling temperature
    # - modify q_pi before action selection
    #
    # Therefore:
    #
    # P(action = a) = q_pi[a]
    #
    # =========================================================

    # ---------------------------------------------------------
    # Initialize RNG only if no RNG key was supplied.
    #
    # For formal experiments, it is better to initialize
    # rng_key once outside run_agent(), for example:
    #
    # rng_key = jax.random.PRNGKey(RANDOM_SEED)
    #
    # and then pass the returned rng_key into the next step.
    # ---------------------------------------------------------
    if rng_key is None:
        rng_key = jax.random.PRNGKey(25)

    # q_pi shape:
    #
    # (batch_size, num_policies)
    #
    # policy_len = 1, therefore each policy corresponds
    # directly to one action in this experiment.
    probs = jnp.asarray(
        q_pi[0],
        dtype=jnp.float32,
    )

    # Numerical safety
    probs = jnp.clip(
        probs,
        min=0.0,
    )

    probs_sum = jnp.sum(probs)

    probs = jnp.where(
        probs_sum > 0.0,
        probs / probs_sum,
        jnp.ones_like(probs) / probs.shape[0],
    )

    # ---------------------------------------------------------
    # Split the RNG key exactly once for this action sample.
    #
    # rng_key:
    # carried forward to the next interaction step
    #
    # action_sample_key:
    # used only for the current random draw
    # ---------------------------------------------------------
    rng_key, action_sample_key = jax.random.split(
        rng_key
    )

    chosen_action_idx = int(
        jax.random.choice(
            action_sample_key,
            probs.shape[0],
            p=probs,
        )
    )

    chosen_action = agent_actions[
        chosen_action_idx
    ]

    print(
        "\n===== STOCHASTIC ACTION SELECTED "
        "DIRECTLY FROM q_pi ====="
    )

    print(
        "Sampling probabilities:",
        probs,
    )

    print(
        f"Action chosen: "
        f"{chosen_action}"
    )

    print(
        "\n===== ACTION PROBABILITIES ====="
    )

    for i, action in enumerate(agent_actions):
        print(
            f"{i}: {action:6s} "
            f"| q_pi={float(q_pi[0][i]):.4f} "
            f"| sample_p={float(probs[i]):.4f} "
            f"| G={float(G[0][i]):.4f}"
        )

    # =========================================================
    # Execute action a_t in the environment
    # =========================================================
    #
    # The environment transition is driven by:
    #
    # temperature_observed + chosen_action
    #
    # The MAP comfort state is diagnostic only.
    # It is NOT treated as the true hidden state.
    # =========================================================

    print("\n===== EXECUTE ACTION =====")

    print(
        f"Current inferred comfort "
        f"(MAP estimate only): "
        f"{current_position_map}"
    )

    print(
        f"Executed action a_t: "
        f"{chosen_action}"
    )

    (
        label_next_position,
        next_position_index,
    ) = environment_maze_step(
        action_input=chosen_action,
        current_position=position_observed,
    )

    print("\n===== ENVIRONMENT RESULT =====")

    print(
        f"New position observation: "
        f"{label_next_position}"
    )

    # =========================================================
    # Predict next prior q^-(s_{t+1})
    # using B and selected action
    # =========================================================
    #
    # q^-(s_{t+1})
    #     =
    # B[a_t] @ q(s_t)
    #
    # IMPORTANT:
    # We use the FULL SOFT posterior comfort_belief here,
    # not its argmax/MAP state.
    # =========================================================

    qs_prior_next = predict_next_state_belief(
        model=model_agent,
        qs_current=position_belief,
        action_input=chosen_action,
        position_input=position_states,
    )

    print(
        "\n===== PREDICTED PRIOR AFTER ACTION ====="
    )

    for i, state in enumerate(position_states):
        print(
            f"Prior q(s_t+1={state}) "
            f"before new observation: "
            f"{float(qs_prior_next[i]):.4f}"
        )

    # =========================================================
    # Do not infer the next observation here.
    #
    # qs_prior_next is the predicted prior for the next
    # timestep:
    #
    # q^-(s_t+1)
    #     =
    # B[a_t] @ q(s_t)
    #
    # It will be passed into the next run_agent() call.
    #
    # At that point, the new observation is incorporated
    # exactly once:
    #
    # q(s_t+1)
    #     ∝
    # P(o_t+1 | s_t+1)
    # *
    # q^-(s_t+1)
    #
    # =========================================================

    return {
        "current_position": position_observed,

        # Full soft posterior:
        "current_belief": position_belief,

        # MAP estimate for diagnostics only:
        "current_position_map_idx": current_position_idx,
        "current_position_map": current_position_map,

        # Environment output:
        "next_position": label_next_position,
        "next_position_idx": next_position_index,

        # Predicted prior for next inference step:
        "predicted_prior_next": qs_prior_next,

        # Selected action:
        "chosen_action_idx": chosen_action_idx,
        "chosen_action": chosen_action,
        "lst_obj_belief": lst_obj_belief,

        # Policy inference:
        "q_pi": q_pi,
        "G": G,

        # Actual sampling distribution:
        "sampling_probs": probs,

        # RNG state to carry to next step:
        "rng_key": rng_key,
    }


init_element(maze_size)
print("position_observes: ")
print(position_observes)
print("position_states: ")
print(position_states)

test_model = extend_action_space("SWIM_UP")

print("XXX_current_model: " + str(test_model.B))


