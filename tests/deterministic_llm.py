import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from returns_agent.models import Category, Decision


class DeterministicTestModel:
    """Small chat-model double for graph tests."""

    def invoke(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> AIMessage:
        state = _extract_graph_state(messages)
        category = _known_category(state)
        issue_summary = _issue_summary(state)
        order_id = _known_order_id(state)
        order = state.get("trusted_order")
        policy_outcome = state.get("policy_outcome")

        if policy_outcome:
            return _final_response(order, policy_outcome, category, issue_summary)

        if order:
            return _json_response(
                {
                    "route": "RUN_POLICY",
                    "order_id": order["order_id"],
                    "category": category.value,
                    "customer_reply": None,
                    "issue_summary": issue_summary,
                }
            )

        if _last_tool_failed(messages):
            return _json_response(
                {
                    "route": "ASK_CUSTOMER",
                    "order_id": order_id,
                    "category": category.value,
                    "customer_reply": (
                        "I could not find an order with that number. Could you "
                        "please check it and send it through again?"
                    ),
                    "issue_summary": issue_summary,
                }
            )

        if not _has_known_issue(state, category):
            return _json_response(
                {
                    "route": "ASK_CUSTOMER",
                    "order_id": order_id,
                    "category": Category.UNKNOWN.value,
                    "customer_reply": (
                        "Hi, I can help with returns or order issues. Could you "
                        "tell me what happened and share your order number if you have it?"
                    ),
                    "issue_summary": "Customer has not described a return or order issue yet.",
                }
            )

        if not order_id:
            return _json_response(
                {
                    "route": "ASK_CUSTOMER",
                    "order_id": None,
                    "category": category.value,
                    "customer_reply": "Could you please provide your order number so I can look this up?",
                    "issue_summary": issue_summary,
                }
            )

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_order",
                    "args": {"order_id": order_id},
                    "id": f"get_order_{order_id}",
                    "type": "tool_call",
                }
            ],
        )


class DeterministicResponseModel:
    """Structured AgentResponse double used after the tool-choice turn."""

    def invoke(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> dict[str, Any]:
        response = DeterministicTestModel().invoke(messages)

        if response.tool_calls:
            raise AssertionError("Response model should return AgentResponse JSON.")

        return json.loads(_message_content(response))


class DeterministicReplyModel:
    """Reply-only double used after deterministic policy has run."""

    def invoke(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> dict[str, str]:
        state = _extract_reply_state(messages)
        order = state.get("trusted_order")
        policy_outcome = state["policy_outcome"]
        item = order["item"] if order else "your item"
        decision = policy_outcome["decision"]
        reason_code = policy_outcome["reason_code"]

        if decision == Decision.REFUND.value:
            reply = f"Thanks for reaching out. {item} is eligible for an automatic refund."
        elif decision == Decision.ESCALATE.value:
            reply = (
                f"Thanks for reaching out. Because {item} requires human review, "
                "I will escalate this to our Customer Care team."
            )
        elif reason_code == "ITEM_IN_TRANSIT":
            reply = (
                f"Thanks for reaching out. {item} is still in transit, so please "
                "wait until delivery before starting a return."
            )
        else:
            reply = f"Thanks for reaching out. {policy_outcome['explanation']}"

        return {"customer_reply": reply}


def _json_response(payload: dict[str, Any]) -> AIMessage:
    return AIMessage(content=json.dumps(payload))


def _final_response(
    order: dict[str, Any] | None,
    policy_outcome: dict[str, Any],
    category: Category,
    issue_summary: str,
) -> AIMessage:
    item = order["item"] if order else "your item"
    decision = policy_outcome["decision"]
    reason_code = policy_outcome["reason_code"]

    if decision == Decision.REFUND.value:
        reply = f"Thanks for reaching out. {item} is eligible for an automatic refund."
    elif decision == Decision.ESCALATE.value:
        reply = (
            f"Thanks for reaching out. Because {item} requires human review, "
            "I will escalate this to our Customer Care team."
        )
    elif reason_code == "ITEM_IN_TRANSIT":
        reply = (
            f"Thanks for reaching out. {item} is still in transit, so please "
            "wait until delivery before starting a return."
        )
    else:
        reply = f"Thanks for reaching out. {policy_outcome['explanation']}"

    return _json_response(
        {
            "route": "FINAL",
            "order_id": order["order_id"] if order else None,
            "category": category.value,
            "customer_reply": reply,
            "issue_summary": issue_summary,
        }
    )


def _extract_graph_state(messages: list[BaseMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        content = _message_content(message)
        if not content.startswith("Trusted graph state:"):
            continue

        raw_state = content.removeprefix("Trusted graph state:").split(
            "\n\nChoose the next step:",
            maxsplit=1,
        )[0]
        return json.loads(raw_state)

    return {}


def _extract_reply_state(messages: list[BaseMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        content = _message_content(message)
        if "Trusted state:" not in content:
            continue

        raw_state = content.split("Trusted state:", maxsplit=1)[1].split(
            "\n\nRules:",
            maxsplit=1,
        )[0]
        return json.loads(raw_state)

    raise AssertionError("Reply prompt did not include trusted policy state.")


def _known_order_id(state: dict[str, Any]) -> str | None:
    if state.get("known_order_id"):
        return str(state["known_order_id"])

    latest_message_order_id = _extract_order_id(_latest_customer_text(state))
    if latest_message_order_id:
        return latest_message_order_id

    return _extract_order_id(_conversation_text(state))


def _known_category(state: dict[str, Any]) -> Category:
    category = state.get("category")
    if category and category != Category.UNKNOWN.value:
        return Category(category)

    return _classify_category(_conversation_text(state)) or Category.UNKNOWN


def _issue_summary(state: dict[str, Any]) -> str:
    if state.get("issue_summary"):
        return str(state["issue_summary"])

    return _conversation_text(state)


def _has_known_issue(state: dict[str, Any], category: Category) -> bool:
    return category != Category.UNKNOWN or _mentions_customer_issue(_conversation_text(state))


def _conversation_text(state: dict[str, Any]) -> str:
    messages = state.get("messages", [])
    return " ".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") in {"human", "user"}
    )


def _latest_customer_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        if message.get("role") in {"human", "user"}:
            return str(message.get("content", ""))

    return ""


def _last_tool_failed(messages: list[BaseMessage]) -> bool:
    for message in reversed(messages):
        if message.type != "tool":
            continue

        content = _message_content(message)
        return '"found": false' in content.casefold() or "'found': False" in content

    return False


def _message_content(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _extract_order_id(text: str) -> str | None:
    match = re.search(
        r"\border(?:\s+number)?\s*#?\s*(\d+)\b|#\s*(\d+)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return next(group for group in match.groups() if group)

    bare_number = re.fullmatch(r"\s*(\d{3,})\s*", text)
    return bare_number.group(1) if bare_number else None


ISSUE_KEYWORDS = [
    "return",
    "refund",
    "money back",
    "broken",
    "damaged",
    "cracked",
    "wrong color",
    "wrong colour",
    "wrong product",
    "wrong item",
    "assembly",
    "assemble",
    "put together",
    "defect",
    "faulty",
    "doesn't fit",
    "does not fit",
    "not fit",
    "change of mind",
    "where is",
    "tracking",
    "in transit",
    "delivery",
    "delivered",
    "not arrived",
    "not delivered",
    "late",
]


def _mentions_customer_issue(text: str) -> bool:
    lower_text = text.casefold()
    return any(keyword in lower_text for keyword in ISSUE_KEYWORDS)


def _classify_category(text: str) -> Category | None:
    lower_text = text.casefold()

    if any(
        term in lower_text
        for term in [
            "where is",
            "tracking",
            "in transit",
            "not delivered",
            "not arrived",
            "delivery",
            "late",
        ]
    ):
        return Category.DELIVERY_STATUS_ENQUIRY
    if any(term in lower_text for term in ["broken", "damaged", "cracked"]):
        return Category.DAMAGE_OR_BROKEN_ITEM
    if any(
        term in lower_text
        for term in ["wrong color", "wrong colour", "wrong product", "wrong item"]
    ):
        return Category.WRONG_COLOUR_OR_PRODUCT_MISMATCH
    if any(term in lower_text for term in ["assembly", "assemble", "put together"]):
        return Category.ASSEMBLY_ISSUE
    if any(term in lower_text for term in ["defect", "faulty", "major issue"]):
        return Category.MAJOR_DEFECT
    if any(
        term in lower_text
        for term in ["doesn't fit", "does not fit", "not fit", "change of mind"]
    ):
        return Category.CHANGE_OF_MIND

    return None
