from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langsmith import traceable
from pydantic import ValidationError

from returns_agent.llm import LLMResponseError
from returns_agent.models import (
    AgentResponse,
    AgentRoute,
    Category,
    CustomerReplyDraft,
    Decision,
    FinalDecision,
    Order,
)
from returns_agent.policy import apply_returns_policy
from returns_agent.prompts import (
    build_agent_state_prompt,
    build_agent_system_prompt,
    build_policy_reply_prompt,
)
from returns_agent.state import (
    DEFAULT_MAX_CLARIFICATION_ATTEMPTS,
    AgentState,
    WorkflowStatus,
)

RouteAfterAgent = Literal["tools", "parse_response"]
RouteAfterParse = Literal["tools", "ask_customer", "policy"]
REQUIRED_FINAL_KEYS = {"order_id", "decision", "customer_reply", "category"}


# First LLM turn. It can either talk in structured form or call a tool.
def make_agent_node(agent_model: Runnable[Any, AIMessage]):
    @traceable(run_type="chain", name="agent_node")
    def agent_node(state: AgentState) -> AgentState:
        response = agent_model.invoke(
            [
                SystemMessage(content=build_agent_system_prompt()),
                *state.get("messages", []),
                HumanMessage(content=build_agent_state_prompt(state)),
            ]
        )

        return {"messages": [response]}

    return agent_node


# After the agent speaks, the graph decides whether to run tools or parse text.
def route_after_agent(state: AgentState) -> RouteAfterAgent:
    latest_ai = _last_ai_message(state)
    tool_calls = getattr(latest_ai, "tool_calls", None)

    # Let the model call the order tool only while we still need trusted data.
    if tool_calls and state.get("order") is None and not _latest_lookup_failed(state):
        return "tools"

    # If the model did not call a tool, or is trying to repeat one, parse it.
    return "parse_response"


# Turns the raw tool output into trusted graph state.
@traceable(run_type="parser", name="observe_order_tool")
def observe_tool_node(state: AgentState) -> AgentState:
    payload = _parse_tool_payload(_last_tool_message(state))

    if not payload.get("found"):
        return {"order_id": payload.get("order_id") or state.get("order_id")}

    order = Order.model_validate(payload["order"])

    return {
        "order": order,
        "order_id": order.order_id,
        "clarification_attempts": 0,
    }


# Second structured pass. This makes the model choose a route we can validate.
def make_parse_response_node(response_model: Runnable[Any, dict[str, Any]]):
    @traceable(run_type="parser", name="parse_agent_response")
    def parse_response_node(state: AgentState) -> AgentState:
        # Remove the latest free-form/tool-call AI message before asking for JSON.
        prompt_state: AgentState = {
            **state,
            "messages": _messages_before_latest_agent_reply(state),
        }
        response = response_model.invoke(
            [
                SystemMessage(content=build_agent_system_prompt()),
                *prompt_state.get("messages", []),
                HumanMessage(content=build_agent_state_prompt(prompt_state)),
            ]
        )

        try:
            agent_response = AgentResponse.model_validate(response)
        except ValidationError as exc:
            raise LLMResponseError(
                f"Invalid AgentResponse payload: {response}"
            ) from exc

        return _agent_response_update(state, agent_response)

    return parse_response_node


# Main routing checkpoint after the model has returned AgentResponse JSON.
def route_after_parse(state: AgentState) -> RouteAfterParse:
    agent_response = state.get("agent_response")
    if agent_response is None:
        raise RuntimeError("Parsed route missing AgentResponse.")

    # Trusted order alone is not always enough. We still need the actual issue.
    if state.get("order") is not None and state.get("policy_outcome") is None:
        return "policy" if _has_policy_context(state, agent_response) else "ask_customer"

    # If the model extracted order + issue but forgot the tool call, force lookup.
    if _should_lookup_order(state, agent_response):
        return "tools"

    if agent_response.route == AgentRoute.ASK_CUSTOMER:
        return "ask_customer"

    if agent_response.route == AgentRoute.RUN_POLICY:
        return "policy" if _has_policy_context(state, agent_response) else "ask_customer"

    raise RuntimeError(f"Unsupported agent route: {agent_response.route}")


# Builds the ASK_FOR_INFO response for this turn.
@traceable(run_type="chain", name="ask_customer_node")
def ask_customer_node(state: AgentState) -> AgentState:
    agent_response = _require_agent_response(state)
    attempts = state.get("clarification_attempts", 0) + 1
    max_attempts = state.get(
        "max_clarification_attempts",
        DEFAULT_MAX_CLARIFICATION_ATTEMPTS,
    )

    # Do not keep the customer trapped in an automation loop forever.
    if attempts > max_attempts:
        return _clarification_limit_escalation(state, attempts)

    reply = agent_response.customer_reply or _fallback_customer_question(
        state,
        agent_response,
    )
    final_decision = FinalDecision(
        order_id=agent_response.order_id or state.get("order_id"),
        decision=Decision.ASK_FOR_INFO,
        customer_reply=reply,
        category=_current_category(state, agent_response),
    )

    return {
        "status": WorkflowStatus.WAITING_FOR_CUSTOMER,
        "clarification_attempts": attempts,
        "final_decision": final_decision,
        "messages": [_replace_last_ai_message(state, reply)],
    }


# This is the deterministic decision boundary. No LLM decides refund/reject here.
@traceable(run_type="chain", name="policy_node")
def policy_node(state: AgentState) -> AgentState:
    order = state.get("order")
    if order is None:
        return _operational_escalation(
            state,
            "I need to pass this to our Customer Care team so they can review the order details.",
        )

    policy_outcome = apply_returns_policy(order)
    update: AgentState = {
        "policy_outcome": policy_outcome,
        "human_review_required": policy_outcome.decision == Decision.ESCALATE,
    }

    # If the only clear fact is "in transit", classify it as delivery related.
    if (
        state.get("category", Category.UNKNOWN) == Category.UNKNOWN
        and policy_outcome.reason_code == "ITEM_IN_TRANSIT"
    ):
        update["category"] = Category.DELIVERY_STATUS_ENQUIRY

    return update


# After policy, the LLM only writes the customer-facing wording.
def make_draft_reply_node(reply_model: Runnable[Any, dict[str, Any]]):
    @traceable(run_type="chain", name="draft_reply_node")
    def draft_reply_node(state: AgentState) -> AgentState:
        policy_outcome = state.get("policy_outcome")

        if policy_outcome is None:
            return _operational_escalation(
                state,
                "I need to pass this to our Customer Care team for manual review.",
            )

        response = reply_model.invoke(
            [
                SystemMessage(content=build_agent_system_prompt()),
                HumanMessage(content=build_policy_reply_prompt(state)),
            ]
        )

        try:
            draft = CustomerReplyDraft.model_validate(response)
        except ValidationError as exc:
            raise LLMResponseError(f"Invalid CustomerReplyDraft payload: {response}") from exc

        order = state.get("order")
        final_decision = FinalDecision(
            order_id=order.order_id if order else state.get("order_id"),
            # Decision comes from policy_outcome, not the reply model.
            decision=policy_outcome.decision,
            customer_reply=draft.customer_reply,
            category=state.get("category", Category.UNKNOWN),
        )

        return {
            "final_decision": final_decision,
            "messages": [AIMessage(content=draft.customer_reply)],
        }

    return draft_reply_node


# Every exit path goes through this so the API always returns the same shape.
@traceable(run_type="parser", name="validate_final_json")
def validate_final_json_node(state: AgentState) -> AgentState:
    final_decision = state.get("final_decision")
    if final_decision is None:
        raise RuntimeError("Validation node received no final decision.")

    payload = final_decision.model_dump(mode="json")
    if set(payload.keys()) != REQUIRED_FINAL_KEYS:
        raise RuntimeError(f"Final payload has invalid keys: {sorted(payload.keys())}")

    validated = FinalDecision.model_validate(payload)

    # Status is internal workflow state; decision is the external JSON field.
    if state.get("status") == WorkflowStatus.FAILED:
        status = WorkflowStatus.FAILED
    elif state.get("human_review_required") or validated.decision == Decision.ESCALATE:
        status = WorkflowStatus.HUMAN_REVIEW
    elif validated.decision == Decision.ASK_FOR_INFO:
        status = WorkflowStatus.WAITING_FOR_CUSTOMER
    else:
        status = WorkflowStatus.COMPLETED

    return {
        "status": status,
        "final_decision": validated,
    }


# Save the structured model response into the graph state.
def _agent_response_update(state: AgentState, agent_response: AgentResponse) -> AgentState:
    update: AgentState = {"agent_response": agent_response}

    if agent_response.order_id:
        update["order_id"] = agent_response.order_id

    previous_category = state.get("category", Category.UNKNOWN)
    if agent_response.category != Category.UNKNOWN:
        update["category"] = agent_response.category
        if previous_category == Category.UNKNOWN:
            update["clarification_attempts"] = 0

    # Keep a short issue summary so later turns do not lose context.
    if agent_response.issue_summary:
        update["issue_summary"] = agent_response.issue_summary

    # A validated order id + known issue is enough to make the tool call.
    if _should_lookup_order({**state, **update}, agent_response):
        order_id = agent_response.order_id or state.get("order_id")
        if order_id:
            update["messages"] = [_order_lookup_message(order_id)]

    return update


# Decide whether we have enough to look up the order now.
def _should_lookup_order(state: AgentState, agent_response: AgentResponse) -> bool:
    order_id = agent_response.order_id or state.get("order_id")
    category = _current_category(state, agent_response)

    return (
        bool(order_id)
        and state.get("order") is None
        and category != Category.UNKNOWN
        and not _latest_lookup_failed(state)
    )


# Creates the LangChain tool-call message when the graph forces a lookup.
def _order_lookup_message(order_id: str) -> AIMessage:
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


# Fail closed into human review when the graph cannot continue safely.
def _operational_escalation(state: AgentState, reply: str) -> AgentState:
    final_decision = FinalDecision(
        order_id=state.get("order_id"),
        decision=Decision.ESCALATE,
        customer_reply=reply,
        category=state.get("category", Category.UNKNOWN),
    )

    return {
        "status": WorkflowStatus.FAILED,
        "human_review_required": True,
        "final_decision": final_decision,
        "messages": [AIMessage(content=reply)],
    }


# Too many unclear replies means a person should take over.
def _clarification_limit_escalation(
    state: AgentState,
    attempts: int,
) -> AgentState:
    reply = (
        "Thanks for sticking with me. I am not able to confirm the order with "
        "the details available in this prototype, so I will pass this to our "
        "Customer Care team for a person to help."
    )
    final_decision = FinalDecision(
        order_id=state.get("order_id"),
        decision=Decision.ESCALATE,
        customer_reply=reply,
        category=state.get("category", Category.UNKNOWN),
    )

    return {
        "status": WorkflowStatus.HUMAN_REVIEW,
        "human_review_required": True,
        "clarification_attempts": attempts,
        "final_decision": final_decision,
        "messages": [_replace_last_ai_message(state, reply)],
    }


# Small helper so missing AgentResponse fails loudly.
def _require_agent_response(state: AgentState) -> AgentResponse:
    agent_response = state.get("agent_response")
    if agent_response is None:
        raise RuntimeError("AgentResponse is missing from state.")

    return agent_response


# Prefer the latest model category, otherwise keep what we already knew.
def _current_category(state: AgentState, agent_response: AgentResponse) -> Category:
    if agent_response.category != Category.UNKNOWN:
        return agent_response.category

    return state.get("category", Category.UNKNOWN)


# Stops policy from running when we only know an order id but not the problem.
def _has_policy_context(state: AgentState, agent_response: AgentResponse) -> bool:
    category = _current_category(state, agent_response)

    if category != Category.UNKNOWN:
        return True

    order = state.get("order")
    return order is not None and order.status.casefold() == "in transit"


# Backup wording for when the model route is right but the question is missing.
def _fallback_customer_question(
    state: AgentState,
    agent_response: AgentResponse,
) -> str:
    category = _current_category(state, agent_response)

    if category != Category.UNKNOWN and not (
        agent_response.order_id or state.get("order_id")
    ):
        return "Could you please provide your order number so I can look this up?"

    if category == Category.UNKNOWN:
        if agent_response.order_id or state.get("order_id"):
            return (
                "Thanks for the order number. Could you tell me what happened "
                "or what you need help with for this order?"
            )

        return (
            "Hi, I can help with Temple & Webster returns, delivery, or order "
            "issues. What can I help you with today?"
        )

    return "I can help with that. Could you please share a little more detail?"


# Find the latest AI message LangGraph added to message history.
def _last_ai_message(state: AgentState) -> AIMessage:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message

    raise RuntimeError("Expected an AIMessage in graph state.")


# Find the latest tool result after ToolNode runs.
def _last_tool_message(state: AgentState) -> ToolMessage:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            return message

    raise RuntimeError("Expected a ToolMessage in graph state.")


# Used to avoid retrying the same failed order lookup forever.
def _latest_lookup_failed(state: AgentState) -> bool:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage):
            return not bool(_parse_tool_payload(message).get("found"))

    return False


# Replace the last AI message so message history shows the actual customer reply.
def _replace_last_ai_message(state: AgentState, content: str) -> AIMessage:
    last_message = _last_ai_message(state)
    return AIMessage(content=content, id=last_message.id)


# Parse structured output without feeding the model its own previous tool-call text.
def _messages_before_latest_agent_reply(state: AgentState) -> list[AnyMessage]:
    messages = list(state.get("messages", []))

    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            return messages[:index]

    return messages


# ToolNode can return dicts or JSON-looking strings, so normalize that here.
def _parse_tool_payload(message: ToolMessage) -> dict[str, Any]:
    content = message.content

    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = _extract_json_object(content)

        if isinstance(parsed, dict):
            return parsed

    raise RuntimeError(f"Unexpected tool payload: {content}")


# Last-resort JSON extraction for tool payloads that arrive wrapped as text.
def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object from model.")

    return parsed
