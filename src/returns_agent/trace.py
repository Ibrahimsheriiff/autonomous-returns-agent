from __future__ import annotations

from returns_agent.models import AgentRoute
from returns_agent.state import AgentState


# Small local trace for demos when LangSmith is not open.
def render_markdown_trace(state: AgentState) -> str:
    path = _graph_path(state)
    agent_response = state.get("agent_response")
    final_decision = state.get("final_decision")
    policy_outcome = state.get("policy_outcome")

    lines = [
        "### Graph trace",
        "",
        f"- Path: `{' -> '.join(path)}`",
        f"- Status: `{state.get('status', 'UNKNOWN')}`",
        f"- Route: `{agent_response.route if agent_response else 'n/a'}`",
        f"- Category: `{state.get('category', 'Unknown')}`",
        f"- Order ID: `{state.get('order_id') or 'missing'}`",
        (
            "- Clarification attempts: "
            f"`{state.get('clarification_attempts', 0)} / "
            f"{state.get('max_clarification_attempts', 0)}`"
        ),
    ]

    if policy_outcome is not None:
        lines.append(f"- Policy: `{policy_outcome.decision}` / `{policy_outcome.reason_code}`")

    if final_decision is not None:
        lines.append(f"- Final decision: `{final_decision.decision}`")

    return "\n".join(lines)


# Reconstruct the high-level path from state for a readable CLI summary.
def _graph_path(state: AgentState) -> list[str]:
    path = ["agent"]

    # If order/policy exists, the run went through the tool-observation loop.
    if state.get("order") is not None or state.get("policy_outcome") is not None:
        path.extend(["tools", "observe_tool", "agent"])

    path.append("parse_response")

    agent_response = state.get("agent_response")
    if (
        (agent_response and agent_response.route == AgentRoute.RUN_POLICY)
        or state.get("policy_outcome") is not None
    ):
        path.extend(["policy", "draft_reply"])
    else:
        path.append("ask_customer")

    path.append("validate")
    return path
