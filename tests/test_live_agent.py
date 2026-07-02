import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from returns_agent.graph import ReturnsAgent
from returns_agent.models import Category, Decision, FinalDecision
from returns_agent.state import AgentState

GOLDEN_CASES_PATH = Path(__file__).with_name("golden_cases.jsonl")
REQUIRED_FINAL_KEYS = {"order_id", "decision", "customer_reply", "category"}
TRUTHY = {"1", "true", "yes"}
DAMAGE_CATEGORY_FAMILY = {
    Category.DAMAGE_OR_BROKEN_ITEM,
    Category.DAMAGE_IN_TRANSIT,
    Category.MAJOR_DEFECT,
}


@pytest.fixture
def live_agent() -> Iterator[ReturnsAgent]:
    load_dotenv()

    if os.getenv("RUN_LIVE_LLM_TESTS", "").casefold() not in TRUTHY:
        pytest.skip("Set RUN_LIVE_LLM_TESTS=true to run live LLM smoke tests.")

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("Set ANTHROPIC_API_KEY to run live LLM smoke tests.")

    yield ReturnsAgent()


def assert_strict_final_payload(result: FinalDecision) -> None:
    payload = result.model_dump(mode="json")

    assert set(payload.keys()) == REQUIRED_FINAL_KEYS
    assert payload["customer_reply"]


def assert_live_category_is_acceptable(
    actual: Category,
    expected: Category,
) -> None:
    if actual == expected:
        return

    if actual in DAMAGE_CATEGORY_FAMILY and expected in DAMAGE_CATEGORY_FAMILY:
        return

    raise AssertionError(f"Expected category {expected!r}, got {actual!r}")


def load_golden_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for line in GOLDEN_CASES_PATH.read_text().splitlines():
        if line.strip():
            cases.append(json.loads(line))

    return cases


def test_live_agent_required_assessment_cases(live_agent: ReturnsAgent) -> None:
    cases = [
        (
            "live-case-1",
            "Hi, my order 101 arrived but it doesn't fit in my living room. "
            "I want to return it.",
            "101",
            Decision.ESCALATE,
        ),
        (
            "live-case-2",
            "I bought a vase (Order 102) a while ago and just opened it, "
            "it's broken. Can I get my money back?",
            "102",
            Decision.REJECT,
        ),
        (
            "live-case-3",
            "Where is my outdoor dining set?! Order 103.",
            "103",
            Decision.REJECT,
        ),
    ]

    for ticket_id, message, expected_order_id, expected_decision in cases:
        result, _ = live_agent.run(message, ticket_id=ticket_id)

        assert result.order_id == expected_order_id
        assert result.decision == expected_decision
        assert_strict_final_payload(result)


def test_live_agent_ask_for_info_resume(live_agent: ReturnsAgent) -> None:
    first_result, state = live_agent.run(
        "I want to return the rug I bought, it's the wrong color.",
        ticket_id="live-case-4",
    )

    assert first_result.order_id is None
    assert first_result.decision == Decision.ASK_FOR_INFO
    assert_strict_final_payload(first_result)

    final_result, _ = live_agent.run(
        "Order 101",
        prior_state=state,
        ticket_id="live-case-4",
    )

    assert final_result.order_id == "101"
    assert final_result.decision == Decision.ESCALATE
    assert_strict_final_payload(final_result)


def test_live_golden_conversation_suite(live_agent: ReturnsAgent) -> None:
    for case in load_golden_cases():
        result: FinalDecision | None = None
        state: AgentState | None = None

        for message in case["messages"]:
            result, state = live_agent.run(
                message,
                prior_state=state,
                ticket_id=f"live-golden-{case['id']}",
            )

        assert result is not None
        assert state is not None

        expected = case["expected"]
        assert result.order_id == expected["order_id"]
        assert result.decision == Decision(expected["decision"])
        assert_live_category_is_acceptable(
            actual=result.category,
            expected=Category(expected["category"]),
        )
        assert getattr(state["status"], "value", state["status"]) == expected["status"]
        assert_strict_final_payload(result)
