from env.agent import infer_belief_once, extend_action_space
from env.elements import comforts

agent_model = extend_action_space("ACN1")
agent_model = extend_action_space("ACN2")

# Danh sách observation truyền vào liên tiếp theo thời gian
observation_sequence = [
    ("T2", "L1", "H2"),
    ("T2", "L1", "H2"),
    ("T2", "L1", "H2"),
    ("T2", "L1", "H2"),
    ("T2", "L1", "H2"),

    # Có thể đổi observation để test phản ứng belief
    ("T2", "L1", "H1"),
    ("T2", "L0", "H2"),
    ("T3", "L1", "H2"),
    ("T2", "L2", "H2"),
    ("T2", "L1", "H2"),
]

qs_prior = None
belief_history = []

for t, (temperature_observed, light_observed) in enumerate(
    observation_sequence,
    start=1,
):
    print(f"\n================ BELIEF TEST {t} ================")

    posterior_belief = infer_belief_once(
        model_agent=agent_model,
        temperature_observed=temperature_observed,
        light_observed=light_observed,
        qs_prior_input=qs_prior,
    )

    belief_history.append({
        "step": t,
        "temperature": temperature_observed,
        "light": light_observed,
        "belief": posterior_belief,
    })

    # Posterior of current step becomes prior of next step
    qs_prior = posterior_belief


def debug_observation_likelihood(
    model_agent,
    temperature_observed,
    light_observed,
):
    print("\n===== OBSERVATION LIKELIHOOD DEBUG =====")
    print(f"Observation: {temperature_observed}, {light_observed}")

    for state in comforts:
        p_temp = model_agent.A["temperature"][temperature_observed, state]
        p_light = model_agent.A["light"][light_observed, state]

        joint_likelihood = p_temp * p_light

        print(
            f"{state:13s} | "
            f"P(T={temperature_observed}|s)={float(p_temp):.4f} | "
            f"P(L={light_observed}|s)={float(p_light):.4f} | "
            f"Joint={float(joint_likelihood):.6f}"
        )