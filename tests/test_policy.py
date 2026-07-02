from returns_agent.models import Decision, Order
from returns_agent.policy import apply_returns_policy
from returns_agent.tools import get_order


def test_order_101_high_value_item_escalates() -> None:
    order = get_order("101")
    assert order is not None

    outcome = apply_returns_policy(order)

    assert outcome.decision == Decision.ESCALATE
    assert outcome.reason_code == "HIGH_VALUE_ITEM"
    assert "human review" in outcome.explanation


def test_order_102_outside_return_window_rejects() -> None:
    order = get_order("102")
    assert order is not None

    outcome = apply_returns_policy(order)

    assert outcome.decision == Decision.REJECT
    assert outcome.reason_code == "OUTSIDE_30_DAY_WINDOW"
    assert "outside the 30-day return window" in outcome.explanation


def test_order_103_in_transit_rejects() -> None:
    order = get_order("103")
    assert order is not None

    outcome = apply_returns_policy(order)

    assert outcome.decision == Decision.REJECT
    assert outcome.reason_code == "ITEM_IN_TRANSIT"
    assert "wait until delivery" in outcome.explanation


def test_low_value_item_inside_window_refunds() -> None:
    order = Order(
        order_id="200",
        item="Table Lamp",
        price=120.00,
        status="Delivered 3 days ago",
        delivered_days_ago=3,
    )

    outcome = apply_returns_policy(order)

    assert outcome.decision == Decision.REFUND
    assert outcome.reason_code == "WITHIN_WINDOW_AUTO_REFUND"