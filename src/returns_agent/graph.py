from __future__ import annotations

from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from returns_agent.llm import (
    AnthropicError,
    LLMProviderError,
    bind_order_tools,
    build_agent_response_model,
    build_customer_reply_model,
    provider_error_message,
)
from returns_agent.models import FinalDecision
from returns_agent.nodes import (
    ask_customer_node,
    make_agent_node,
    make_draft_reply_node,
    make_parse_response_node,
    observe_tool_node,
    policy_node,
    route_after_agent,
    route_after_parse,
    validate_final_json_node,
)
from returns_agent.state import AgentState, WorkflowStatus, build_initial_state
from returns_agent.tools import ORDER_TOOLS

DEFAULT_RECURSION_LIMIT = 12
TERMINAL_STATUSES = {
    WorkflowStatus.COMPLETED,
    WorkflowStatus.FAILED,
    WorkflowStatus.HUMAN_REVIEW,
}


# Small wrapper around the compiled LangGraph workflow.
class ReturnsAgent:
    # Models are injectable so tests can use deterministic doubles.
    def __init__(
        self,
        agent_model: Runnable[Any, AIMessage] | None = None,
        response_model: Runnable[Any, dict[str, Any]] | None = None,
        reply_model: Runnable[Any, dict[str, Any]] | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._agent_model = agent_model or bind_order_tools()
        self._response_model = response_model or build_agent_response_model()
        self._reply_model = reply_model or build_customer_reply_model()
        self._graph: Any = build_graph(
            self._agent_model,
            self._response_model,
            self._reply_model,
            checkpointer=checkpointer,
        )

    # One run handles one customer turn. ASK_FOR_INFO resumes with prior_state.
    @traceable(run_type="chain", name="returns_agent_run")
    def run(
        self,
        customer_message: str,
        prior_state: AgentState | None = None,
        ticket_id: str = "local-demo",
    ) -> tuple[FinalDecision, AgentState]:
        # Once a ticket is terminal, do not let the LLM reopen or rewrite it.
        if prior_state is not None and _is_terminal_state(prior_state):
            final_decision = prior_state.get("final_decision")
            if final_decision is None:
                raise RuntimeError("Terminal ticket is missing final decision.")
            return final_decision, prior_state

        state = _prepare_state_for_turn(customer_message, ticket_id, prior_state)
        try:
            result_state = cast(
                AgentState,
                self._graph.invoke(
                    state,
                    config={
                        "configurable": {"thread_id": ticket_id},
                        "recursion_limit": DEFAULT_RECURSION_LIMIT,
                    },
                ),
            )
        except AnthropicError as exc:
            raise LLMProviderError(provider_error_message(exc)) from exc

        # The graph should always leave through validate with a final_decision.
        final_decision = result_state.get("final_decision")
        if final_decision is None:
            raise RuntimeError("Graph ended without a final decision.")

        return final_decision, result_state

    # Useful for me to produce graph and see it after every change
    def draw_mermaid(self) -> str:
        return cast(str, self._graph.get_graph().draw_mermaid())


# The graph shape: LLM/tool loop, structured parse, policy, reply, validate.
def build_graph(
    agent_model: Runnable[Any, AIMessage],
    response_model: Runnable[Any, dict[str, Any]],
    reply_model: Runnable[Any, dict[str, Any]],
    checkpointer: Any | None = None,
) -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", make_agent_node(agent_model))
    workflow.add_node("tools", ToolNode(ORDER_TOOLS, handle_tool_errors=True))
    workflow.add_node("observe_tool", observe_tool_node)
    workflow.add_node("parse_response", make_parse_response_node(response_model))
    workflow.add_node("ask_customer", ask_customer_node)
    workflow.add_node("policy", policy_node)
    workflow.add_node("draft_reply", make_draft_reply_node(reply_model))
    workflow.add_node("validate", validate_final_json_node)

    workflow.add_edge(START, "agent")
    # First branch: direct model tool call or structured parsing.
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "parse_response": "parse_response",
        },
    )
    workflow.add_edge("tools", "observe_tool")
    # After observing the tool result, let the agent reason with trusted data.
    workflow.add_edge("observe_tool", "agent")

    # Second branch: force tool lookup, ask the customer, or run policy.
    workflow.add_conditional_edges(
        "parse_response",
        route_after_parse,
        {
            "tools": "tools",
            "ask_customer": "ask_customer",
            "policy": "policy",
        },
    )
    workflow.add_edge("ask_customer", "validate")
    workflow.add_edge("policy", "draft_reply")
    workflow.add_edge("draft_reply", "validate")
    workflow.add_edge("validate", END)

    return workflow.compile(checkpointer=checkpointer)


# New message plus old state is how the ASK_FOR_INFO resume works.
def _prepare_state_for_turn(
    customer_message: str,
    ticket_id: str,
    prior_state: AgentState | None,
) -> AgentState:
    if prior_state is None:
        return build_initial_state(customer_message, ticket_id)

    state: AgentState = {
        **prior_state,
        "ticket_id": ticket_id,
        "status": WorkflowStatus.RUNNING,
        "messages": [
            *prior_state.get("messages", []),
            HumanMessage(content=customer_message),
        ],
    }

    state["agent_response"] = None
    state["final_decision"] = None
    state["policy_outcome"] = None
    state["human_review_required"] = False

    return state


# Terminal tickets should not go back into the automated graph.
def _is_terminal_state(state: AgentState) -> bool:
    return state.get("status") in TERMINAL_STATUSES
