from langsmith import traceable

from returns_agent.models import Decision, Order, PolicyOutcome

REFUND_WINDOW_DAYS = 30
AUTO_REFUND_PRICE_LIMIT = 500.00


# This is intentionally boring Python. The LLM does not decide money outcomes.
@traceable(run_type="chain", name="apply_returns_policy")
def apply_returns_policy(order: Order) -> PolicyOutcome:
    # In-transit comes first because return/refund should wait for delivery.
    if _is_in_transit(order):
        return PolicyOutcome(
            decision=Decision.REJECT,
            reason_code="ITEM_IN_TRANSIT",
            explanation=(
                "The item is still in transit, so the customer should wait until "
                "delivery before a return or refund can be processed."
            ),
        )

    # Outside the return window means reject, regardless of item price.
    if _is_outside_refund_window(order):
        return PolicyOutcome(
            decision=Decision.REJECT,
            reason_code="OUTSIDE_30_DAY_WINDOW",
            explanation=(
                f"The item was delivered {order.delivered_days_ago} days ago, "
                "which is outside the 30-day return window."
            ),
        )

    # High-value orders need a human, even if they are inside the window.
    if _requires_human_review(order):
        return PolicyOutcome(
            decision=Decision.ESCALATE,
            reason_code="HIGH_VALUE_ITEM",
            explanation=(
                "The item price is over $500, so it cannot be automatically "
                "refunded and must be escalated for human review."
            ),
        )

    # Only low-value, delivered, inside-window orders get automatic refund.
    return PolicyOutcome(
        decision=Decision.REFUND,
        reason_code="WITHIN_WINDOW_AUTO_REFUND",
        explanation=(
            "The item is within the 30-day return window and is not above the "
            "$500 automatic refund threshold."
        ),
    )


# Status from trusted order data decides this, not customer wording.
def _is_in_transit(order: Order) -> bool:
    return order.status.casefold() == "in transit"


# Delivery age is the 30-day policy check.
def _is_outside_refund_window(order: Order) -> bool:
    return (
        order.delivered_days_ago is not None
        and order.delivered_days_ago > REFUND_WINDOW_DAYS
    )


# Anything above the threshold becomes human review.
def _requires_human_review(order: Order) -> bool:
    return order.price > AUTO_REFUND_PRICE_LIMIT
