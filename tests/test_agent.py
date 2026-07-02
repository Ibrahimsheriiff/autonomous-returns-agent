from returns_agent.graph import ReturnsAgent
from returns_agent.models import Category, Decision, FinalDecision
from tests.deterministic_llm import (
    DeterministicReplyModel,
    DeterministicResponseModel,
    DeterministicTestModel,
)

REQUIRED_FINAL_KEYS = {"order_id", "decision", "customer_reply", "category"}


def build_test_agent() -> ReturnsAgent:
    return ReturnsAgent(
        agent_model=DeterministicTestModel(),
        response_model=DeterministicResponseModel(),
        reply_model=DeterministicReplyModel(),
    )


def assert_strict_final_payload(result: FinalDecision) -> None:
    payload = result.model_dump(mode="json")

    assert set(payload.keys()) == REQUIRED_FINAL_KEYS
    assert isinstance(payload["customer_reply"], str)
    assert payload["customer_reply"]


def test_assessment_case_1_high_value_sofa_escalates() -> None:
    agent = build_test_agent()

    result, _ = agent.run(
        "Hi, my order 101 arrived but it doesn't fit in my living room. "
        "I want to return it."
    )

    assert result.order_id == "101"
    assert result.decision == Decision.ESCALATE
    assert result.category == Category.CHANGE_OF_MIND
    assert_strict_final_payload(result)


def test_assessment_case_2_broken_vase_outside_window_rejects() -> None:
    agent = build_test_agent()

    result, _ = agent.run(
        "I bought a vase (Order 102) a while ago and just opened it, "
        "it's broken. Can I get my money back?"
    )

    assert result.order_id == "102"
    assert result.decision == Decision.REJECT
    assert result.category == Category.DAMAGE_OR_BROKEN_ITEM
    assert_strict_final_payload(result)


def test_assessment_case_3_in_transit_order_rejects_with_wait_reply() -> None:
    agent = build_test_agent()

    result, _ = agent.run("Where is my outdoor dining set?! Order 103.")

    assert result.order_id == "103"
    assert result.decision == Decision.REJECT
    assert result.category == Category.DELIVERY_STATUS_ENQUIRY
    assert "wait until delivery" in result.customer_reply
    assert_strict_final_payload(result)


def test_assessment_case_4_missing_order_id_asks_for_info() -> None:
    agent = build_test_agent()

    result, state = agent.run("I want to return the rug I bought, it's the wrong color.")

    assert result.order_id is None
    assert result.decision == Decision.ASK_FOR_INFO
    assert result.category == Category.WRONG_COLOUR_OR_PRODUCT_MISMATCH
    assert "order number" in result.customer_reply
    assert state["status"] == "WAITING_FOR_CUSTOMER"
    assert_strict_final_payload(result)


def test_agent_resumes_after_ask_for_info_without_losing_context() -> None:
    agent = build_test_agent()

    first_result, state = agent.run(
        "I want to return the rug I bought, it's the wrong color."
    )
    assert first_result.decision == Decision.ASK_FOR_INFO

    final_result, resumed_state = agent.run("Order 101", prior_state=state)

    assert final_result.order_id == "101"
    assert final_result.decision == Decision.ESCALATE
    assert final_result.category == Category.WRONG_COLOUR_OR_PRODUCT_MISMATCH
    assert len(resumed_state["messages"]) >= 4
    assert_strict_final_payload(final_result)


def test_order_id_alone_asks_for_issue_before_lookup() -> None:
    agent = build_test_agent()

    result, state = agent.run("Order 101")

    assert result.order_id == "101"
    assert result.decision == Decision.ASK_FOR_INFO
    assert result.category == Category.UNKNOWN
    assert "what happened" in result.customer_reply
    assert "order" not in state
    assert_strict_final_payload(result)


def test_invalid_order_id_asks_customer_to_check_number() -> None:
    agent = build_test_agent()

    result, state = agent.run("Order 999 arrived broken.")

    assert result.order_id == "999"
    assert result.decision == Decision.ASK_FOR_INFO
    assert result.category == Category.DAMAGE_OR_BROKEN_ITEM
    assert "could not find an order" in result.customer_reply
    assert state.get("order") is None
    assert_strict_final_payload(result)


def test_graph_can_render_mermaid_for_architecture_walkthrough() -> None:
    agent = build_test_agent()

    mermaid = agent.draw_mermaid()

    assert "agent" in mermaid
    assert "tools" in mermaid
    assert "policy" in mermaid
    assert "validate" in mermaid
