from ssa.agents.prompt import (
    build_ssa_ablation_system_prompt,
    build_system_prompt,
)


def test_react_prompt_uses_shared_base_and_react_sections():
    prompt = build_system_prompt(
        agent_type="react",
        agent_id="agent_0",
        num_jobs=12,
        num_tasks=3,
        task_ids=["SK-A", "SK-B", "SK-C"],
        format_instructions="<FMT>",
    )

    assert "You are agent_0, an AI agent competing in a freelancer marketplace." in prompt
    assert "By default, payment is performance-adjusted" in prompt
    assert "**RECENT REASONING/ACTION/OUTCOME**" in prompt
    assert "REASONING STRATEGY:" not in prompt
    assert "<FMT>" in prompt


def test_payment_mechanics_can_be_overridden_per_agent():
    prompt = build_system_prompt(
        agent_type="react",
        agent_id="agent_0",
        num_jobs=12,
        num_tasks=3,
        task_ids=["SK-A", "SK-B", "SK-C"],
        format_instructions="<FMT>",
        payment_mechanics_id="full_payment",
    )

    assert "you will be paid in full as per your bidding price" in prompt
    assert "performance-adjusted" not in prompt


def test_cot_prompt_uses_cot_sections():
    prompt = build_system_prompt(
        agent_type="cot",
        agent_id="agent_cot",
        num_jobs=6,
        num_tasks=2,
        task_ids=["SK-A", "SK-B"],
        format_instructions="<FMT>",
    )

    assert "**RECENT ACTIONS**" in prompt
    assert "**PREVIOUS REASONING**" in prompt
    assert "<FMT>" in prompt


def test_ssa_prompt_includes_ssa_specific_reasoning_modules():
    prompt = build_system_prompt(
        agent_type="ssa_default",
        agent_id="agent_1",
        num_jobs=10,
        num_tasks=4,
        task_ids=["SK-A", "SK-B"],
        format_instructions="<FMT>",
    )

    assert "You are agent_1, an AI agent competing in a freelancer marketplace." in prompt
    assert "Your job performance affects payment - poor performance results in partial payment" in prompt
    assert "REASONING STRATEGY:" in prompt
    assert "**META-COGNITION:**" in prompt
    assert "<FMT>" in prompt


def test_ssa_ablation_prompt_injects_custom_reasoning_sections():
    prompt = build_ssa_ablation_system_prompt(
        agent_id="agent_2",
        num_jobs=8,
        num_tasks=2,
        task_ids=["SK-A"],
        ssa_description="1. DESC A\n\n2. DESC B",
        ssa_format="- **A**\n- **B**",
        format_instructions="<FMT>",
    )

    assert "YOUR COGNITIVE ARCHITECTURE:" in prompt
    assert "1. DESC A" in prompt
    assert "- **A**" in prompt
    assert "<FMT>" in prompt


def test_build_system_prompt_supports_custom_agent_type_file():
    prompt = build_system_prompt(
        agent_type="config_1",
        agent_id="agent_cfg",
        num_jobs=5,
        num_tasks=2,
        task_ids=["SK-A", "SK-B"],
        format_instructions="<FMT>",
    )

    assert "You are agent_cfg, an AI agent competing in a freelancer marketplace." in prompt
    assert "Use a concise three-step loop each round" in prompt
    assert "<FMT>" in prompt
