import json
from pathlib import Path
from typing import Any

from returns_agent.graph import ReturnsAgent
from returns_agent.models import Category, Decision, FinalDecision
from returns_agent.state import AgentState
from returns_agent.trace import render_markdown_trace
from tests.deterministic_llm import (
    DeterministicReplyModel,
    DeterministicResponseModel,
    DeterministicTestModel,
)

GOLDEN_CASES_PATH = Path(__file__).with_name("golden_cases.jsonl")
REQUIRED_FINAL_KEYS = {"order_id", "decision", "customer_reply", "category"}


def _load_golden_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for line in GOLDEN_CASES_PATH.read_text().splitlines():
        if line.strip():
            cases.append(json.loads(line))

    return cases


def _build_agent() -> ReturnsAgent:
    return ReturnsAgent(
        agent_model=DeterministicTestModel(),
        response_model=DeterministicResponseModel(),
        reply_model=DeterministicReplyModel(),
    )


def test_golden_conversation_suite() -> None:
    for case in _load_golden_cases():
        _run_golden_case(case)


def _run_golden_case(case: dict[str, Any]) -> None:
    agent = _build_agent()
    result: FinalDecision | None = None
    state: AgentState | None = None

    for message in case["messages"]:
        result, state = agent.run(message, prior_state=state, ticket_id=case["id"])

    assert result is not None
    assert state is not None

    expected = case["expected"]
    _assert_final_payload(result, expected)
    _assert_state(state, expected)
    _assert_customer_reply(result.customer_reply, expected)
    _assert_trace(render_markdown_trace(state), expected)


def _assert_final_payload(
    result: FinalDecision,
    expected: dict[str, Any],
) -> None:
    payload = result.model_dump(mode="json")

    assert set(payload.keys()) == REQUIRED_FINAL_KEYS
    assert result.order_id == expected["order_id"]
    assert result.decision == Decision(expected["decision"])
    assert result.category == Category(expected["category"])
    assert result.customer_reply.strip()


def _assert_state(state: AgentState, expected: dict[str, Any]) -> None:
    status = state["status"]

    assert getattr(status, "value", status) == expected["status"]

    if expected.get("policy_reason"):
        policy_outcome = state.get("policy_outcome")
        assert policy_outcome is not None
        assert policy_outcome.reason_code == expected["policy_reason"]


def _assert_customer_reply(reply: str, expected: dict[str, Any]) -> None:
    normalized_reply = reply.casefold()

    for phrase in expected.get("reply_contains", []):
        assert phrase.casefold() in normalized_reply

    for phrase in expected.get("reply_excludes", []):
        assert phrase.casefold() not in normalized_reply


def _assert_trace(trace: str, expected: dict[str, Any]) -> None:
    for phrase in expected.get("trace_contains", []):
        assert phrase in trace
